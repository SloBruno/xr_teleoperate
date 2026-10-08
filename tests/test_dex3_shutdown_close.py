"""DEX3_SHUTDOWN_HAND: on q / Ctrl+C / SIGTERM / error the Dex3 closes (default).

Fakes only: no DDS, no actuators. Compares the ``open`` mode against the base
commit 0c5ba31 (sources loaded with ``git show``).
"""

import contextlib
import importlib
import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace as ns

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import teleop.utils.dex3_protection as dp  # noqa: E402
from teleop.utils import dex3_shutdown_hand as sh  # noqa: E402
from teleop.utils.arm_graceful_shutdown import run_graceful_arm_shutdown  # noqa: E402
from teleop.utils.dex3_protection import Dex3HandProtector, ProtectionWarner  # noqa: E402

BASE = "0c5ba31"


# --------------------------------------------------------------- fixtures
def _sdk_stubs():
    sdk_channel = types.ModuleType("unitree_sdk2py.core.channel")
    sdk_channel.ChannelPublisher = sdk_channel.ChannelSubscriber = sdk_channel.ChannelFactoryInitialize = object
    sdk_hand = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    sdk_hand.HandCmd_ = sdk_hand.HandState_ = object
    sdk_default = types.ModuleType("unitree_sdk2py.idl.default")
    sdk_default.unitree_hg_msg_dds__HandCmd_ = sdk_default.unitree_go_msg_dds__MotorCmd_ = object
    go = types.ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_")
    go.MotorCmds_ = go.MotorStates_ = object
    mods = {name: types.ModuleType(name) for name in (
        "unitree_sdk2py", "unitree_sdk2py.core", "unitree_sdk2py.idl", "unitree_sdk2py.idl.unitree_hg",
        "unitree_sdk2py.idl.unitree_hg.msg", "unitree_sdk2py.idl.unitree_go", "unitree_sdk2py.idl.unitree_go.msg")}
    mods.update({
        "unitree_sdk2py.core.channel": sdk_channel,
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": sdk_hand,
        "unitree_sdk2py.idl.default": sdk_default,
        "unitree_sdk2py.idl.unitree_go.msg.dds_": go,
    })
    return mods


@pytest.fixture
def module(monkeypatch):
    for name, mod in _sdk_stubs().items():
        monkeypatch.setitem(sys.modules, name, mod)
    logs = []
    lm = types.ModuleType("logging_mp")
    lm.getLogger = lambda name: ns(warning=lambda m, *a, **k: logs.append(("warning", m)),
                                   info=lambda m, *a, **k: logs.append(("info", m)),
                                   error=lambda m, *a, **k: logs.append(("error", m)))
    monkeypatch.setitem(sys.modules, "logging_mp", lm)
    sys.modules.pop("teleop.robot_control.robot_hand_unitree", None)
    try:
        mod = importlib.import_module("teleop.robot_control.robot_hand_unitree")
    finally:
        sys.modules.pop("teleop.robot_control.robot_hand_unitree", None)
    mod._test_logs = logs
    return mod


@pytest.fixture
def base_module(monkeypatch, tmp_path):
    """robot_hand_unitree.py exactly as in the base commit."""
    try:
        source = subprocess.run(["git", "show", f"{BASE}:teleop/robot_control/robot_hand_unitree.py"],
                                cwd=ROOT, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("base commit not available")
    for name, mod in _sdk_stubs().items():
        monkeypatch.setitem(sys.modules, name, mod)
    lm = types.ModuleType("logging_mp")
    lm.getLogger = lambda name: ns(warning=lambda *a, **k: None, info=lambda *a, **k: None,
                                   error=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "logging_mp", lm)
    path = tmp_path / "base_robot_hand_unitree.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("base_robot_hand_unitree", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_controller(module):
    c = module.Dex3_1_Controller.__new__(module.Dex3_1_Controller)
    c._telemetry_lock = threading.Lock()
    c._protectors = {"left": Dex3HandProtector(module.Dex3_Open_Pose),
                     "right": Dex3HandProtector(module.Dex3_Open_Pose)}
    c._protection_warner = ProtectionWarner()
    written = {"left": [], "right": []}

    def msg():
        return ns(motor_cmd=[ns(q=0.0, dq=0.0, tau=0.0, kp=1.5, kd=0.2, mode=1) for _ in range(7)])

    c.left_msg, c.right_msg = msg(), msg()
    for side in written:
        setattr(c, f"{side.capitalize()}HandCmb_publisher", ns(Write=lambda m, side=side: written[side].append(
            [(x.q, x.kp, x.kd, x.tau) for x in m.motor_cmd])))
    return c, written


class Sample:
    def __init__(self, trigger, clock=None):
        self.trigger = trigger
        # resolved at read time so a monkeypatched time.monotonic is honoured
        self.clock = clock if clock is not None else (lambda: time.monotonic())

    def get_lock(self):
        return contextlib.nullcontext()

    def __getitem__(self, k):
        return [self.trigger, self.clock()][k]


def feed(c, module, side, q, *, temp=40, mode=1, ms=0, dq=1.0, tau=0.0, now=None):
    ids = module.Dex3_1_Left_JointIndex if side == "left" else module.Dex3_1_Right_JointIndex
    motors = [ns(q=float(q[i]), dq=dq, tau_est=tau, temperature=[temp, temp], mode=mode, motorstate=ms)
              for i in range(7)]
    c._record_protection_state(side, ns(motor_state=motors), ids,
                               module.time.monotonic() if now is None else now)


def last_q(written, side):
    return np.array([q for q, *_ in written[side][-1]])


def step(c, trig_l=0.0, trig_r=0.0):
    out = c.control_step(None, None, left_ctrl_sample_in=Sample(trig_l), right_ctrl_sample_in=Sample(trig_r))
    with c._telemetry_lock:
        c._left_action, c._right_action = out[0].copy(), out[1].copy()
        c._left_action_valid = c._right_action_valid = True
    return out


@pytest.fixture
def clock(module, monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: t[0])
    return t


def run_follow(c, module, clock, seconds, *, dt=0.01, trig=(0.0, 0.0), follow=("left", "right"),
               state_overrides=None):
    """Ideal servo: measured q = last published command (free fingers)."""
    end = clock[0] + seconds
    while clock[0] < end:
        for side in ("left", "right"):
            if side in follow:
                q = last_q(c._written, side) if c._written[side] else np.zeros(7)
                kw = (state_overrides or {}).get(side, {})
                feed(c, module, side, kw.pop("q", q) if kw else q, **kw)
        step(c, *trig)
        clock[0] += dt


def make(module):
    c, written = make_controller(module)
    c._written = written
    return c, written


# ------------------------------------------------------------------- pure
def test_mode_default_is_close_and_invalid_falls_back_to_open_with_pt_br_warning():
    assert sh.resolve_mode(None) == ("close", None)
    assert sh.resolve_mode("") == ("close", None)
    assert sh.resolve_mode(" HOLD ") == ("hold", None)
    assert sh.resolve_mode("open") == ("open", None)
    mode, warning = sh.resolve_mode("fechar")
    assert mode == "open" and "inválido" in warning
    assert "FECHA" in sh.describe_mode("close") and "ABRE" in sh.describe_mode("open")


def test_thresholds_shared_with_existing_protection():
    assert sh.CUTOFF_TEMP_C == dp.DERATE_OPEN_C == 80.0
    assert sh.STATE_STALE_S == dp.STATE_STALE_S
    # ramp of ~0.8-1.0 s for the full open->closed travel
    full = sh.plan_duration(np.zeros(7), np.array([0.0, 1.05, 1.75, -1.37, -1.53, -1.37, -1.53]))
    assert 0.8 <= full <= 1.0


def test_close_blocker_rules():
    ok = {"timestamp": 10.0, "q": [0.0] * 7, "temp": [40] * 7, "motorstate": [0] * 7}
    assert sh.close_blocker(ok, 10.1) is None
    assert sh.close_blocker(None, 10.1) == "state_missing"
    assert sh.close_blocker(ok, 10.6) == "state_stale"
    assert sh.close_blocker({**ok, "temp": [40] * 6 + [80]}, 10.1) == "hot"
    assert sh.close_blocker({**ok, "motorstate": [0] * 6 + [512]}, 10.1) == "fault"
    assert sh.close_blocker(ok, 10.1, fault_latched=True) == "fault"
    assert sh.close_blocker(ok, 10.1, hot_latched=True) == "hot"


# ------------------------------------------------------- controller close
def test_q_closes_with_smooth_ramp_to_the_trigger_one_pose_same_gains(module, clock):
    c, written = make(module)
    run_follow(c, module, clock, 0.3)                         # trigger released: open
    np.testing.assert_allclose(last_q(written, "left"), np.zeros(7))
    c._shutdown_hand_mode = "close"                           # what begin_shutdown_hand sets
    t0 = clock[0]
    traj = []
    while clock[0] < t0 + 1.5:
        for side in ("left", "right"):
            feed(c, module, side, last_q(written, side))
        step(c, 0.0, 0.0)                                     # trigger released: still closes
        traj.append((clock[0], last_q(written, "left"), last_q(written, "right")))
        clock[0] += 0.01
    np.testing.assert_allclose(traj[-1][1], module.Dex3_Left_Closed_Pose, atol=1e-9)
    np.testing.assert_allclose(traj[-1][2], module.Dex3_Right_Closed_Pose, atol=1e-9)
    ts = np.array([t for t, *_ in traj])
    left = np.array([q for _, q, _ in traj])
    velocity = np.abs(np.diff(left, axis=0)) / np.diff(ts)[:, None]
    assert velocity.max() <= sh.CLOSE_MAX_JOINT_VELOCITY + 1e-6
    reached = next(t for t, q, _ in traj if np.allclose(q, module.Dex3_Left_Closed_Pose, atol=1e-6))
    assert 0.8 <= reached - t0 <= 1.05
    dist = np.abs(left - module.Dex3_Left_Closed_Pose)        # monotonic, no overshoot
    assert np.all(np.diff(dist, axis=0) <= 1e-9)
    for side in written:
        for frame in written[side]:
            assert all(kp == module.Dex3_Kp and kd == module.Dex3_Kd and tau == 0.0 for _, kp, kd, tau in frame)


def test_begin_and_finish_shutdown_hand_end_to_end_with_command_thread(module):
    c, written = make(module)
    c.running = True

    def loop():
        while c.running:
            for side in ("left", "right"):
                feed(c, module, side, last_q(written, side) if written[side] else np.zeros(7))
            step(c, 0.0, 0.0)
            time.sleep(0.005)

    c.hand_control_process = threading.Thread(target=loop, daemon=True)
    c.hand_control_process.start()
    c.outputs_activated = True
    time.sleep(0.05)
    t0 = time.monotonic()
    summary = c.begin_shutdown_hand("close")
    elapsed = time.monotonic() - t0
    assert 0.8 <= elapsed <= 1.5
    assert summary["left"]["done"] and summary["left"]["blocked_reason"] is None
    assert c.hand_control_process.is_alive()                  # still holding closed
    time.sleep(0.05)
    np.testing.assert_allclose(last_q(written, "left"), module.Dex3_Left_Closed_Pose, atol=1e-9)
    assert c.finish_shutdown_hand() is True
    assert not c.hand_control_process.is_alive() and c.outputs_activated is False
    # the last command left in the firmware = closed
    np.testing.assert_allclose(last_q(written, "left"), module.Dex3_Left_Closed_Pose, atol=1e-9)
    np.testing.assert_allclose(last_q(written, "right"), module.Dex3_Right_Closed_Pose, atol=1e-9)
    assert any("fechando a mão" in m for _, m in module._test_logs)


def test_idempotent_begin_does_not_restart_the_ramp(module, clock):
    c, written = make(module)
    run_follow(c, module, clock, 0.1)
    c.begin_shutdown_hand("close")                            # no thread: no wait
    run_follow(c, module, clock, 0.5)
    plan = c._shutdown_plans["left"]
    c.begin_shutdown_hand("close")
    c.begin_shutdown_hand("hold")                             # mode cannot change mid-shutdown
    assert c._shutdown_plans["left"] is plan and c._shutdown_hand_mode == "close"
    assert sum("fechando" in m for _, m in module._test_logs) == 1


# ------------------------------------------------------------ safety blocks
def test_left_hand_without_state_is_not_closed_and_reason_logged(module, clock):
    c, written = make(module)
    c._shutdown_hand_mode = "close"
    for _ in range(150):                                      # left never publishes state
        feed(c, module, "right", last_q(written, "right") if written["right"] else np.zeros(7))
        step(c)
        clock[0] += 0.01
    np.testing.assert_allclose(last_q(written, "left"), np.zeros(7))
    np.testing.assert_allclose(last_q(written, "right"), module.Dex3_Right_Closed_Pose, atol=1e-9)
    msgs = [m for _, m in module._test_logs if "NÃO fecha" in m]
    assert len(msgs) == 1 and "left" in msgs[0] and "ausente" in msgs[0]


def test_stale_state_is_not_closed(module, clock):
    c, written = make(module)
    feed(c, module, "left", np.zeros(7))
    feed(c, module, "right", np.zeros(7))
    clock[0] += 0.6                                           # > 0.5 s, stale
    c._shutdown_hand_mode = "close"
    for _ in range(120):
        step(c)
        clock[0] += 0.01
    np.testing.assert_allclose(last_q(written, "left"), np.zeros(7))
    np.testing.assert_allclose(last_q(written, "right"), np.zeros(7))
    assert any("antigo" in m and "NÃO fecha" in m for _, m in module._test_logs)


def test_hot_80c_is_not_closed_even_if_it_heats_mid_ramp(module, clock):
    c, written = make(module)
    c._shutdown_hand_mode = "close"
    for i in range(150):
        hot = i >= 30
        for side in ("left", "right"):
            q = last_q(written, side) if written[side] else np.zeros(7)
            feed(c, module, side, q, temp=80 if (hot and side == "left") else 40)
        step(c)
        clock[0] += 0.01
    np.testing.assert_allclose(last_q(written, "left"), np.zeros(7))   # latched open
    np.testing.assert_allclose(last_q(written, "right"), module.Dex3_Right_Closed_Pose, atol=1e-9)
    assert c._shutdown_plans["left"].blocked_reason == "hot"


def test_fault_is_not_closed_and_faulted_motor_keeps_zero_gain(module, clock):
    c, written = make(module)
    c._shutdown_hand_mode = "close"
    for _ in range(150):
        for side in ("left", "right"):
            q = last_q(written, side) if written[side] else np.zeros(7)
            feed(c, module, side, q, ms=512 if side == "right" else 0)
        step(c)
        clock[0] += 0.01
    assert c._shutdown_plans["right"].blocked_reason == "fault"
    assert all(kp == 0.0 and kd == 0.0 for _, kp, kd, _ in written["right"][-1])
    assert np.max(np.abs(last_q(written, "right"))) < 1e-9
    np.testing.assert_allclose(last_q(written, "left"), module.Dex3_Left_Closed_Pose, atol=1e-9)


# ------------------------------------------------------ force never larger
def test_holding_object_closing_never_exceeds_normal_full_trigger_grip(module, clock):
    """Same blocked finger: shutdown close vs. the operator pulling trigger=1."""
    blocked = np.array([0.0, 0.4, 0.6, -0.7, -0.7, -0.7, -0.7])

    def errors(shutdown):
        c, written = make(module)
        run_follow(c, module, clock, 0.05, trig=(0.6, 0.0))
        errs = []
        for i in range(400):
            feed(c, module, "left", blocked, dq=0.0, tau=8e5)
            feed(c, module, "right", np.zeros(7))
            if shutdown and i == 100:
                c._shutdown_hand_mode = "close"
            step(c, 1.0 if not shutdown or i < 100 else 0.0, 0.0)
            errs.append(np.abs(last_q(written, "left") - blocked))
            clock[0] += 0.01
        return np.max(errs, axis=0), written

    normal, _ = errors(False)
    closing, written = errors(True)
    assert np.all(closing <= normal + 1e-9)
    assert np.all(dp.DEX3_KP * closing <= np.maximum(dp.CLOSE_TORQUE_CEILING_NM,
                                                      np.asarray(dp.GRIP_HOLD_TORQUE_NM)) + 1e-9)
    assert all(kp == 1.5 and kd == 0.2 for f in written["left"] for _, kp, kd, _ in f)


def test_final_frame_has_no_squeeze_on_blocked_joints(module, clock):
    c, written = make(module)
    blocked = np.array([0.0, 1.05, 1.75, -0.7, -0.7, -0.7, -0.7])   # thumb free, fingers on an object
    c._shutdown_hand_mode = "close"
    for _ in range(150):
        feed(c, module, "left", blocked, dq=0.0, tau=8e5)
        feed(c, module, "right", last_q(written, "right") if written["right"] else np.zeros(7))
        step(c)
        clock[0] += 0.01
    c._shutdown_finalize = True
    feed(c, module, "left", blocked, dq=0.0, tau=8e5)
    feed(c, module, "right", last_q(written, "right"))
    step(c)
    final = last_q(written, "left")
    np.testing.assert_allclose(final[3:], blocked[3:])                 # zero implicit torque
    np.testing.assert_allclose(final[:3], module.Dex3_Left_Closed_Pose[:3], atol=1e-9)
    np.testing.assert_allclose(last_q(written, "right"), module.Dex3_Right_Closed_Pose, atol=1e-9)
    assert c._shutdown_final_published


def test_hold_mode_keeps_last_trigger_target(module, clock):
    c, written = make(module)
    run_follow(c, module, clock, 0.5, trig=(0.5, 0.0))
    expected = 0.5 * module.Dex3_Left_Closed_Pose
    np.testing.assert_allclose(last_q(written, "left"), expected, atol=1e-9)
    c._shutdown_hand_mode = "hold"
    run_follow(c, module, clock, 1.5, trig=(1.0, 1.0))                # triggers ignored
    np.testing.assert_allclose(last_q(written, "left"), expected, atol=1e-9)
    np.testing.assert_allclose(last_q(written, "right"), np.zeros(7), atol=1e-9)


def test_force_open_still_wins_over_close_mode(module, clock):
    c, written = make(module)
    c._shutdown_hand_mode = "close"
    run_follow(c, module, clock, 0.5)
    c._force_open = True
    run_follow(c, module, clock, 0.02)
    np.testing.assert_allclose(last_q(written, "left"), np.zeros(7))


# ---------------------------------------------------- open mode regression
def test_open_mode_dex3_frames_identical_to_base(module, base_module, monkeypatch):
    def frames(mod):
        t = [1000.0]
        monkeypatch.setattr(mod.time, "monotonic", lambda: t[0])
        c, written = make_controller(mod)
        seq = [1.0] * 60 + [0.4] * 30
        for i, trig in enumerate(seq):
            for side in ("left", "right"):
                q = np.array([q for q, *_ in written[side][-1]]) if written[side] else np.zeros(7)
                motors = [ns(q=float(q[j]), dq=1.0, tau_est=0.0, temperature=[40, 40], mode=1, motorstate=0)
                          for j in range(7)]
                ids = mod.Dex3_1_Left_JointIndex if side == "left" else mod.Dex3_1_Right_JointIndex
                c._record_protection_state(side, ns(motor_state=motors), ids, t[0])
            c.control_step(None, None, left_ctrl_sample_in=Sample(trig, lambda: t[0]),
                           right_ctrl_sample_in=Sample(trig, lambda: t[0]))
            t[0] += 0.01
        c.outputs_activated, c.hand_control_process = False, None
        c.open_and_deactivate(open_hold_s=0.0)          # mode open: the previous call, unchanged
        for _ in range(5):
            c.control_step(None, None, left_ctrl_sample_in=Sample(1.0, lambda: t[0]),
                           right_ctrl_sample_in=Sample(1.0, lambda: t[0]))
            t[0] += 0.01
        return written

    new, old = frames(module), frames(base_module)
    assert new == old
    assert [q for q, *_ in new["left"][-1]] == [0.0] * 7


def _base_shutdown_module(tmp_path):
    try:
        source = subprocess.run(["git", "show", f"{BASE}:teleop/utils/arm_graceful_shutdown.py"],
                                cwd=ROOT, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("base commit not available")
    path = tmp_path / "base_arm_graceful_shutdown.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("base_arm_graceful_shutdown", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += float(seconds)


class SimArm:
    arm_joint_split = (7, 7)

    def __init__(self, clock, log):
        self.clock, self.log = clock, log
        self.motion_mode = True
        self.q = np.linspace(-0.8, 0.9, 14)
        self.weight = 1.0
        self.active = True

    def get_dual_arm_q_snapshot(self):
        return self.q.copy(), 0.0

    def get_arm_command(self):
        return self.q.copy(), np.full(14, 2.0)

    def ctrl_dual_arm(self, q, tau):
        assert self.active
        self.log.append(("arm", round(self.clock(), 6), tuple(np.round(q, 12)), tuple(np.round(tau, 12))))
        self.q = np.asarray(q, float).copy()

    def set_motion_authority_weight(self, w):
        assert self.active
        self.weight = float(w)
        self.log.append(("weight", round(self.clock(), 6), self.weight))

    def get_publication_status(self):
        return {"active": self.active, "last_publish_monotonic": self.clock(),
                "last_published_weight": self.weight}

    def deactivate(self):
        self.active = False
        self.log.append(("arm_deactivate",))


class FakeHand:
    """Dex3-like API: records calls; deactivate/open_and_deactivate as in the controller."""

    def __init__(self, log, clock):
        self.log, self.clock = log, clock

    def open_and_deactivate(self, *a, **k):
        self.log.append(("hand_open", round(self.clock(), 6)))

    def deactivate(self):
        self.log.append(("hand_deactivate",))

    def begin_shutdown_hand(self, mode):
        self.log.append(("hand_close", mode, round(self.clock(), 6)))
        self.clock.sleep(0.9)                                  # bounded ramp
        return {"left": {"done": True, "blocked_reason": None}}

    def finish_shutdown_hand(self):
        self.log.append(("hand_release", round(self.clock(), 6)))
        return True


def _run(shutdown_fn, mode, **kw):
    clock, log = FakeClock(), []
    hand = FakeHand(log, clock)
    arm = SimArm(clock, log)
    if mode is None:
        cb = {"open_hands": hand.open_and_deactivate}
    else:
        _, open_hands, close_hands, release_hands = sh.hand_shutdown_callbacks(hand, mode)
        cb = {"open_hands": open_hands, "close_hands": close_hands, "release_hands": release_hands}
    result = shutdown_fn(arm, clock=clock, sleep=clock.sleep, **cb, **kw)
    return log, result


def test_open_mode_arm_and_hand_sequence_identical_to_base(tmp_path):
    base = _base_shutdown_module(tmp_path)
    old_log, old = _run(base.run_graceful_arm_shutdown, None)
    new_log, new = _run(run_graceful_arm_shutdown, "open")
    assert new_log == old_log
    assert [e["event"] for e in new.events] == [e["event"] for e in old.events]
    assert new.hands_opened and not new.hands_closed


def test_close_order_close_then_arms_then_weight_then_release_then_deactivate():
    log, result = _run(run_graceful_arm_shutdown, "close")
    kinds = [entry[0] for entry in log]
    close_i = kinds.index("hand_close")
    first_arm = kinds.index("arm")
    last_arm = len(kinds) - 1 - kinds[::-1].index("arm")
    first_w = kinds.index("weight")
    last_w = len(kinds) - 1 - kinds[::-1].index("weight")
    release_i = kinds.index("hand_release")
    assert close_i < first_arm <= last_arm < first_w <= last_w < release_i < kinds.index("arm_deactivate")
    assert "hand_open" not in kinds and log[close_i][1] == "close"
    assert log[first_arm][1] >= log[close_i][2] + 0.9            # arms start after the ramp
    assert result.hands_closed and result.hands_released and result.release_confirmed


def test_ctrl_c_during_close_falls_back_to_open_and_still_releases_arms():
    clock, log = FakeClock(), []
    arm = SimArm(clock, log)

    def interrupted():
        log.append(("hand_close",))
        raise KeyboardInterrupt

    result = run_graceful_arm_shutdown(arm, clock=clock, sleep=clock.sleep,
                                       open_hands=lambda: log.append(("hand_open",)),
                                       close_hands=interrupted,
                                       release_hands=lambda: log.append(("hand_release",)))
    kinds = [e[0] for e in log]
    assert "hand_open" in kinds and "hand_release" not in kinds
    assert result.hands_opened and result.deactivated and result.weight_released


def test_shutdown_twice_closes_once():
    clock, log = FakeClock(), []
    hand, arm = FakeHand(log, clock), SimArm(clock, log)
    _, open_hands, close_hands, release_hands = sh.hand_shutdown_callbacks(hand, "close")
    for _ in range(2):
        run_graceful_arm_shutdown(arm, clock=clock, sleep=clock.sleep, open_hands=open_hands,
                                  close_hands=close_hands, release_hands=release_hands)
    assert [e[0] for e in log].count("hand_close") == 1


def test_callbacks_env_default_close_and_non_dex3_hands_keep_open():
    hand = FakeHand([], FakeClock())
    mode, open_hands, close_hands, release_hands = sh.hand_shutdown_callbacks(hand, None, environ={})
    assert mode == "close" and close_hands is not None and release_hands == hand.finish_shutdown_hand
    logs = []
    mode, open_hands, close_hands, _ = sh.hand_shutdown_callbacks(
        hand, None, environ={"DEX3_SHUTDOWN_HAND": "x"}, log=logs.append)
    assert mode == "open" and close_hands is None and open_hands == hand.open_and_deactivate and logs
    inspire = ns(deactivate=lambda: None)                     # no Dex3 close API
    _, open_hands, close_hands, _ = sh.hand_shutdown_callbacks(inspire, "close")
    assert close_hands is None and open_hands is inspire.deactivate


# ------------------------------------------------------------------ SIGTERM
def test_sigterm_raises_keyboard_interrupt_once_then_ignored_during_shutdown():
    stopping = [False]
    logs = []
    handler = sh.make_sigterm_handler(lambda: stopping[0], logs.append)
    previous = signal.signal(signal.SIGTERM, handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)                  # second one: shutdown keeps going
        time.sleep(0.05)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert any("mesmo caminho do q" in m for m in logs)
    assert any("continuando o encerramento" in m for m in logs)
    stopping[0] = True
    sh.make_sigterm_handler(lambda: stopping[0])(signal.SIGTERM, None)  # STOP already set: no raise


def test_teleop_installs_sigterm_and_logs_mode():
    source = (ROOT / "teleop" / "teleop_hand_and_arm.py").read_text(encoding="utf-8")
    main = source[source.index("if __name__ == '__main__':"):]
    assert main.index("install_sigterm_handler()") < main.index("    try:\n        # setup dds")
    assert "dex3_shutdown_hand.describe_mode(" in main
    helper = source[source.index("def graceful_g1_29_shutdown"):source.index("def _pose_stream_hands")]
    assert "hand_shutdown_callbacks(" in helper and "close_hands=close_hands" in helper


# ----------------------------------------------------------------- launcher
LAUNCHER = ROOT / "teleop" / "run_g1_quest_dex3.sh"


def test_launcher_prints_mode_default_close_and_rejects_invalid():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "DEX3_SHUTDOWN_HAND=${DEX3_SHUTDOWN_HAND:-close}" in source
    assert "export DEX3_SHUTDOWN_HAND" in source
    assert 'echo "$dex3_shutdown_msg"' in source
    assert source.index("DEX3_SHUTDOWN_HAND=${DEX3_SHUTDOWN_HAND:-close}") < source.index("ensure_teleimager()")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["DEX3_SHUTDOWN_HAND"] = "fechar"
    proc = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 2 and "close|open|hold" in proc.stderr
