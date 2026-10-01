import glob
import importlib
import json
import os
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace as ns

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import teleop.utils.dex3_protection as dp
from teleop.utils.dex3_protection import Dex3HandProtector, ProtectionWarner, derate_factor

OPEN = np.zeros(7)
CLOSED = np.array([0.0, 1.05, 1.75, -0.785, -0.873, -0.785, -0.873])


def state(t, q=None, dq=None, tau=None, temp=None, mode=None, ms=None):
    n = 7
    return {
        "timestamp": t,
        "q": list(q) if q is not None else [0.0] * n,
        "dq": list(dq) if dq is not None else [1.0] * n,
        "tau": list(tau) if tau is not None else [0.0] * n,
        "temp": list(temp) if temp is not None else [40.0] * n,
        "mode": list(mode) if mode is not None else [1] * n,
        "motorstate": list(ms) if ms is not None else [0] * n,
    }


def run(prot, t0, dur, target, make_state, dt=0.01):
    res = None
    t = t0
    while t < t0 + dur:
        res = prot.update(t, target, make_state(t))
        t += dt
    return res


# ---- (1) torque ceiling --------------------------------------------------
def test_free_finger_small_error_passes_through_unchanged():
    p = Dex3HandProtector(OPEN)
    target = CLOSED * 0.1  # tiny step, err 0.175 rad << cap
    res = p.update(0.0, target, state(0.0))
    np.testing.assert_allclose(res.q_cmd, target)
    assert not any(res.torque_limited)


def test_full_trigger_error_clamped_to_ceiling_not_old_2p6_nm():
    p = Dex3HandProtector(OPEN)
    res = p.update(0.0, CLOSED, state(0.0))
    for i in (1, 2):
        assert res.torque_limited[i]
        implicit = dp.DEX3_KP * abs(res.q_cmd[i] - 0.0)
        assert implicit == pytest.approx(dp.CLOSE_TORQUE_CEILING_NM[i])
    assert dp.DEX3_KP * CLOSED[2] > 2.5  # what was published before


def test_finger_follows_and_reaches_full_closure_if_free():
    """Simulated free first-order finger: still closes fully (ceiling does not cap reach)."""
    p = Dex3HandProtector(OPEN)
    q = np.zeros(7)
    t = 0.0
    for _ in range(600):
        res = p.update(t, CLOSED, state(t, q=q, dq=np.full(7, 1.0)))
        q = q + 0.3 * (res.q_cmd - q) * 0.1 + 0.0  # move toward (limited) command
        t += 0.01
    np.testing.assert_allclose(q[1:3], CLOSED[1:3], atol=0.05)


def test_opening_is_not_limited_by_close_ceiling_and_never_exceeds_target():
    p = Dex3HandProtector(OPEN)
    q = CLOSED.copy()
    res = p.update(0.0, OPEN, state(0.0, q=q))
    # opening cap is larger than closing cap
    assert abs(res.q_cmd[2] - q[2]) > dp.CLOSE_TORQUE_CEILING_NM[2] / dp.DEX3_KP
    assert np.all(np.abs(res.q_cmd) <= np.abs(q) + 1e-9)


def test_protection_never_commands_beyond_trigger_target():
    p = Dex3HandProtector(OPEN)
    rng = np.random.default_rng(0)
    for k in range(300):
        trig = rng.random()
        tgt = trig * CLOSED
        q = rng.random(7) * CLOSED
        res = p.update(k * 0.01, tgt, state(k * 0.01, q=q, dq=rng.random(7) * 2))
        # the command always lies between the measured position and the target
        lo, hi = np.minimum(q, tgt), np.maximum(q, tgt)
        assert np.all(res.q_cmd >= lo - 1e-9) and np.all(res.q_cmd <= hi + 1e-9)


# ---- (2) stall ----------------------------------------------------------
def blocked(t, q=(0, -0.3, -1.0, 0, 0, 0, 0)):
    return state(t, q=q, dq=np.zeros(7), tau=np.full(7, 8e5))


def test_stall_detected_after_time_and_relaxes_to_measured():
    p = Dex3HandProtector(OPEN)
    tgt = -CLOSED.copy()
    q = np.array([0, -0.3, -1.0, 0, 0, 0, 0.0])
    res = run(p, 0.0, dp.STALL_TIME_S - 0.1, tgt, lambda t: blocked(t))
    assert not res.stall[1] and not res.stall[2]
    res = run(p, dp.STALL_TIME_S - 0.1, 0.3, tgt, lambda t: blocked(t))
    assert res.stall[1] and res.stall[2]
    assert res.q_cmd[1] == pytest.approx(q[1]) and res.q_cmd[2] == pytest.approx(q[2])  # thumb relaxes
    assert ("stall", 1) in res.active


def test_stall_holds_until_trigger_reduced_or_released():
    p = Dex3HandProtector(OPEN)
    tgt = -CLOSED.copy()
    run(p, 0.0, 1.0, tgt, lambda t: blocked(t))
    res = p.update(1.01, tgt, blocked(1.01))
    assert res.stall[2]
    res = p.update(1.02, tgt * 0.95, blocked(1.02))  # tiny reduction: still held
    assert res.stall[2]
    res = p.update(1.03, tgt * 0.5, blocked(1.03))   # big reduction, debounced
    assert res.stall[2]
    res = p.update(1.03 + dp.STALL_RELEASE_DEBOUNCE_S + 0.01, tgt * 0.5, blocked(1.03 + dp.STALL_RELEASE_DEBOUNCE_S + 0.01))  # sustained: released
    assert not res.stall[2]
    # the next full press re-detects only after the stall time again
    res = p.update(1.35, tgt, blocked(1.35))
    assert not res.stall[2]


def test_moving_finger_with_large_error_is_not_a_stall():
    p = Dex3HandProtector(OPEN)
    res = run(p, 0.0, 2.0, CLOSED, lambda t: state(t, dq=np.full(7, 5.0), tau=np.full(7, 9e5)))
    assert not any(res.stall)


def test_low_tau_large_error_is_not_a_stall():
    p = Dex3HandProtector(OPEN)
    res = run(p, 0.0, 2.0, CLOSED, lambda t: state(t, dq=np.zeros(7), tau=np.full(7, 1e4)))
    assert not any(res.stall)


# ---- (3) thermal -------------------------------------------------------------
def test_derate_factor_ramp_and_nonfinite():
    assert derate_factor(40) == 1.0
    assert derate_factor(dp.DERATE_START_C) == 1.0
    assert derate_factor(dp.DERATE_OPEN_C) == 0.0
    assert derate_factor(0.5 * (dp.DERATE_START_C + dp.DERATE_OPEN_C)) == pytest.approx(0.5)
    assert derate_factor(None) == 1.0 and derate_factor(float("nan")) == 1.0


def test_hot_joint_reduces_target_and_torque_ceiling():
    p = Dex3HandProtector(OPEN)
    cold = p.update(0.0, CLOSED, state(0.0))
    hot = Dex3HandProtector(OPEN).update(0.0, CLOSED, state(0.0, temp=[72.5] * 7))
    assert hot.derate[1] == pytest.approx(0.5)
    assert abs(hot.q_cmd[2]) < abs(cold.q_cmd[2])
    # implied ceiling also scaled
    assert dp.DEX3_KP * abs(hot.q_cmd[2]) <= dp.CLOSE_TORQUE_CEILING_NM[2] * 0.5 + 1e-9


def test_open_at_80_with_hysteresis_until_60():
    p = Dex3HandProtector(OPEN)
    t = 0.0
    res = p.update(t, CLOSED, state(t, temp=[81] * 7))
    assert res.q_cmd[2] == pytest.approx(0.0)  # relaxed/opened (q_meas == 0)
    for temp in (75, 65, 61):  # cooler but above resume: stays latched
        t += 0.1
        res = p.update(t, CLOSED, state(t, temp=[temp] * 7))
        assert res.derate[2] == 0.0 and res.q_cmd[2] == pytest.approx(0.0)
    t += 0.1
    res = p.update(t, CLOSED, state(t, temp=[59] * 7))
    assert res.derate[2] > 0.0  # unlatched, ramps back (slow recovery)
    assert res.derate[2] < 0.1
    res = run(p, t, 5.0, CLOSED, lambda tt: state(tt, temp=[50] * 7))
    assert res.derate[2] == pytest.approx(1.0)


def test_uses_hotter_of_two_temperature_sensors_via_extract():
    hs = ns(motor_state=[ns(q=0.1, dq=0.0, tau_est=1.0, temperature=[44, 83], mode=1, motorstate=0)] * 7)
    st = dp.extract_protection_state(hs, range(7), 1.0)
    assert st["temp"][0] == 83


def test_implausible_temperature_is_ignored():
    hs = ns(motor_state=[ns(q=0.1, dq=0.0, tau_est=1.0, temperature=[5000, -3], mode=1, motorstate=0)] * 7)
    assert dp.extract_protection_state(hs, range(7), 1.0)["temp"][0] is None


# ---- (4) fault ---------------------------------------------------------------
def test_motorstate_nonzero_latches_fault_disables_and_never_reenables():
    p = Dex3HandProtector(OPEN)
    ms = [0] * 7
    ms[2] = 512
    mode = [1] * 7
    mode[2] = 0
    q = np.array([0, -0.3, -0.5, 0, 0, 0, 0.0])
    res = run(p, 0.0, 0.1, -CLOSED, lambda t: state(t, q=q, mode=mode, ms=ms))
    assert res.fault[2] and not res.enable[2]
    assert res.q_cmd[2] == pytest.approx(-0.5)
    assert ("fault", 2) in res.active
    # motor reports healthy again: still latched
    res = run(p, 0.1, 0.2, -CLOSED, lambda t: state(t, q=q))
    assert res.fault[2] and not res.enable[2]
    assert res.enable[1] and not res.fault[1]


def test_mode_zero_before_ever_enabled_is_not_fault():
    p = Dex3HandProtector(OPEN)
    res = run(p, 0.0, 0.1, CLOSED, lambda t: state(t, mode=[0] * 7))
    assert not any(res.fault)


def test_single_glitch_sample_is_debounced():
    p = Dex3HandProtector(OPEN)
    p.update(0.0, CLOSED, state(0.0))
    res = p.update(0.01, CLOSED, state(0.01, ms=[512] * 7))
    assert not any(res.fault)
    res = p.update(0.02, CLOSED, state(0.02))
    assert not any(res.fault)


# ---- (6) fail-safe ------------------------------------------------------------
def test_missing_or_stale_state_commands_open_pose():
    p = Dex3HandProtector(OPEN)
    res = p.update(10.0, CLOSED, None)
    np.testing.assert_allclose(res.q_cmd, OPEN)
    assert res.state_stale
    res = p.update(10.0, CLOSED, state(9.0))  # 1 s old
    np.testing.assert_allclose(res.q_cmd, OPEN)
    res = p.update(10.0, CLOSED, state(float("nan")))
    np.testing.assert_allclose(res.q_cmd, OPEN)


def test_garbage_target_and_state_do_not_raise_and_fail_open():
    p = Dex3HandProtector(OPEN)
    res = p.update(1.0, [float("nan")] * 7, state(1.0))
    np.testing.assert_allclose(res.q_cmd, OPEN)
    res = p.update(1.0, "bad", state(1.0))
    np.testing.assert_allclose(res.q_cmd, OPEN)
    res = p.update(1.0, CLOSED, {"timestamp": 1.0})
    np.testing.assert_allclose(res.q_cmd, OPEN)


def test_deterministic():
    def go():
        p = Dex3HandProtector(OPEN)
        out = []
        for k in range(200):
            t = k * 0.01
            out.append(p.update(t, CLOSED * (k % 7) / 7, blocked(t, q=(0, -0.2, -0.4, 0, 0, 0, 0))).q_cmd.copy())
        return np.array(out)
    np.testing.assert_array_equal(go(), go())


def test_protection_module_has_no_io():
    src = (ROOT / "teleop" / "utils" / "dex3_protection.py").read_text()
    for banned in ("open(", "import json", "time.sleep", "time.time", "monotonic(", "socket", "subprocess"):
        assert banned not in src


# ---- (5) warnings ----------------------------------------------------------
def test_warner_rate_limits_per_kind_and_joint():
    w = ProtectionWarner(period_s=2.0)
    active = {("stall", 1): "a", ("derate", 2): "b"}
    assert len(w.messages(0.0, "right", active)) == 2
    assert w.messages(1.0, "right", active) == []
    assert len(w.messages(1.0, "left", active)) == 2  # other side independent
    assert len(w.messages(2.1, "right", active)) == 2
    assert "right" in w.messages(10.0, "right", {("fault", 2): "x"})[0]


# ---- constants / gains unchanged --------------------------------------------
def test_gains_unchanged_and_ceilings_conservative():
    src = (ROOT / "teleop" / "robot_control" / "robot_hand_unitree.py").read_text()
    assert "kp = 1.5" in src and "kd = 0.2" in src
    assert dp.DEX3_KP == 1.5
    assert max(dp.CLOSE_TORQUE_CEILING_NM[1:3]) <= 1.0
    assert dp.DERATE_START_C < dp.DERATE_OPEN_C and dp.DERATE_RESUME_C < dp.DERATE_START_C
    assert dp.DERATE_START_C <= 70 and dp.DERATE_OPEN_C <= 80


# ---- controller integration (fakes, no DDS) ----------------------------------
@pytest.fixture
def module(monkeypatch):
    sdk_channel = types.ModuleType("unitree_sdk2py.core.channel")
    sdk_channel.ChannelPublisher = sdk_channel.ChannelSubscriber = sdk_channel.ChannelFactoryInitialize = object
    sdk_hand = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    sdk_hand.HandCmd_ = sdk_hand.HandState_ = object
    sdk_default = types.ModuleType("unitree_sdk2py.idl.default")
    sdk_default.unitree_hg_msg_dds__HandCmd_ = sdk_default.unitree_go_msg_dds__MotorCmd_ = object
    go = types.ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_")
    go.MotorCmds_ = go.MotorStates_ = object
    for name, mod in {
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": sdk_channel,
        "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.unitree_hg": types.ModuleType("unitree_sdk2py.idl.unitree_hg"),
        "unitree_sdk2py.idl.unitree_hg.msg": types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg"),
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": sdk_hand,
        "unitree_sdk2py.idl.default": sdk_default,
        "unitree_sdk2py.idl.unitree_go": types.ModuleType("unitree_sdk2py.idl.unitree_go"),
        "unitree_sdk2py.idl.unitree_go.msg": types.ModuleType("unitree_sdk2py.idl.unitree_go.msg"),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": go,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    warnings = []
    lm = types.ModuleType("logging_mp")
    lm.getLogger = lambda name: ns(warning=lambda m, *a, **k: warnings.append(m), info=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "logging_mp", lm)
    sys.modules.pop("teleop.robot_control.robot_hand_unitree", None)
    mod = importlib.import_module("teleop.robot_control.robot_hand_unitree")
    mod._test_warnings = warnings
    return mod


def make_controller(module, now):
    import time
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
    def __init__(self, trigger):
        import time
        self.v = [trigger, time.monotonic()]

    def get_lock(self):
        import contextlib
        return contextlib.nullcontext()

    def __getitem__(self, k):
        return self.v[k]


def feed_state(c, module, side, q, temp=40, mode=1, ms=0, dq=0.0, tau=8e5):
    import time
    ids = module.Dex3_1_Left_JointIndex if side == "left" else module.Dex3_1_Right_JointIndex
    motors = [ns(q=q[i], dq=dq, tau_est=tau, temperature=[temp, temp], mode=mode, motorstate=ms) for i in range(7)]
    c._record_protection_state(side, ns(motor_state=motors), ids, time.monotonic())


def test_control_step_limits_error_keeps_gains_and_trigger_semantics(module):
    c, written = make_controller(module, 0)
    zeros = [0.0] * 7
    feed_state(c, module, "left", zeros, dq=1.0, tau=0)
    feed_state(c, module, "right", zeros, dq=1.0, tau=0)
    l, r = c.control_step(None, None, left_ctrl_sample_in=Sample(1.0), right_ctrl_sample_in=Sample(0.0))
    # trigger 1 -> closing but error-limited; trigger 0 -> open
    assert 0 < l[2] < module.Dex3_Left_Closed_Pose[2]
    assert dp.DEX3_KP * l[2] == pytest.approx(dp.CLOSE_TORQUE_CEILING_NM[2])
    np.testing.assert_allclose(r, np.zeros(7))
    assert all(kp == 1.5 and kd == 0.2 for _, kp, kd, _ in written["left"][-1])
    assert all(t == 0.0 for *_, t in written["left"][-1])


def test_control_step_without_state_publishes_open(module):
    c, written = make_controller(module, 0)
    l, r = c.control_step(None, None, left_ctrl_sample_in=Sample(1.0), right_ctrl_sample_in=Sample(1.0))
    np.testing.assert_allclose(l, np.zeros(7))
    np.testing.assert_allclose(r, np.zeros(7))
    assert any("ausente" in w for w in module._test_warnings)


def test_control_step_faulted_motor_gets_zero_gain_and_warning_and_flags(module):
    c, written = make_controller(module, 0)
    q = [0, -0.3, -0.5, 0, 0, 0, 0]
    for _ in range(4):
        feed_state(c, module, "right", q, dq=1.0, tau=0, mode=0, ms=512)
        feed_state(c, module, "left", [0] * 7, dq=1.0, tau=0)
        # mode must have been seen healthy once for mode==0; motorstate!=0 alone suffices
        c.control_step(None, None, left_ctrl_sample_in=Sample(0.0), right_ctrl_sample_in=Sample(1.0))
    last = written["right"][-1]
    # every fake joint reports motorstate 512 -> all faulted -> no gain, no tau
    assert all(kp == 0.0 and kd == 0.0 and tau == 0.0 for _, kp, kd, tau in last)
    flags = c.get_extended_samples()["right"]["protection"]
    assert set(flags) >= {"torque_limited", "stall", "derate", "fault"}


def test_control_step_fault_only_one_joint(module):
    c, written = make_controller(module, 0)
    import time
    ids = module.Dex3_1_Right_JointIndex
    for _ in range(4):
        motors = [ns(q=-0.4, dq=1.0, tau_est=0, temperature=[40, 40], mode=1, motorstate=0) for _ in range(7)]
        motors[2] = ns(q=-0.5, dq=0.0, tau_est=0, temperature=[40, 40], mode=0, motorstate=512)
        c._record_protection_state("right", ns(motor_state=motors), ids, time.monotonic())
        feed_state(c, module, "left", [0] * 7, dq=1.0, tau=0)
        c.control_step(None, None, left_ctrl_sample_in=Sample(0.0), right_ctrl_sample_in=Sample(1.0))
    last = written["right"][-1]
    assert last[2][1] == 0.0 and last[2][2] == 0.0 and last[2][3] == 0.0
    assert last[1][1] == 1.5 and last[1][2] == 0.2
    assert c.get_extended_samples()["right"]["protection"]["fault"][2] is True
    assert any("Thumb2" in w and "desligado" in w for w in module._test_warnings)


def test_protection_flags_survive_full_pose_telemetry_sanitizer():
    from teleop.utils.full_pose_telemetry import _extended_side
    item = {"state": None, "protection": {"fault": [False] * 7, "derate": [float("nan")] * 7}}
    out = _extended_side(item, 10.0)
    assert out["protection"]["fault"] == [False] * 7
    assert out["protection"]["derate"] == [None] * 7


def test_bare_controller_without_protectors_bypasses(module):
    c = module.Dex3_1_Controller.__new__(module.Dex3_1_Controller)
    c._telemetry_lock = threading.Lock()
    out, enable = c._apply_protection("left", 0.0, CLOSED)
    np.testing.assert_allclose(out, CLOSED)
    assert enable is None


# ---- (7) replay of the real session (estimate) --------------------------------------
SESSION = glob.glob(os.path.expanduser(
    "~/g1_dex3_thumb_final_review/pose-telemetry-20260930T190939*.jsonl"))


@pytest.mark.skipif(not SESSION, reason="recorded session not present")
def test_replay_real_session_torque_and_thermal():
    sys.path.insert(0, str(ROOT / "tools"))
    import replay_dex3_protection as rp
    stats = rp.replay(SESSION[0])
    for side in ("left", "right"):
        for i in (1, 2):
            s = stats[side][i]
            # implicit closing torque never above the configured ceiling
            assert s["new_close_max_nm"] <= dp.CLOSE_TORQUE_CEILING_NM[i] + 1e-6
            assert s["old_close_max_nm"] > 1.4  # unprotected controller did exceed it
    r1, r2 = stats["right"][1], stats["right"][2]
    # stall on the blocked right thumb and derate begun before recorded 70 C
    assert r1["first_stall"] is not None and r2["first_stall"] is not None
    assert r1["first_derate"] < r1["first_temp70"] and r2["first_derate"] < r2["first_temp70"]
    # recorded motor-off event is flagged as fault for Thumb2 only
    assert r2["first_fault"] is not None and r1["first_fault"] is None
    # healthy left hand: no stall, no derate, no fault
    for i in range(7):
        l = stats["left"][i]
        assert l["first_stall"] is None and l["first_derate"] is None and l["first_fault"] is None


# ---- (8) grip hold under load (fix/dex3-grip-hold) -----------------------------
# Long fingers (index/middle, slots 3-6). Left-hand sign convention (negative = closed).
FQ = (0, 0, 0, -0.4, -0.4, -0.4, -0.4)


def fblocked(t, temp=40.0):
    return state(t, q=FQ, dq=np.zeros(7), tau=np.full(7, 8e5), temp=np.full(7, temp))


def test_ceilings_bounded_by_rated_effort_thumb_unchanged():
    assert all(c <= 1.8 + 1e-9 for c in dp.CLOSE_TORQUE_CEILING_NM[1:])
    assert dp.CLOSE_TORQUE_CEILING_NM[:3] == (0.75, 0.75, 0.75)
    assert all(h == 0.0 for h in dp.GRIP_HOLD_TORQUE_NM[:3])
    assert dp.DEX3_KP == 1.5


def test_thumb_stall_still_relaxes_to_measured():
    p = Dex3HandProtector(OPEN)
    tgt = -CLOSED.copy()
    mk = lambda t: state(t, q=(0, -0.3, -1.0, 0, 0, 0, 0), dq=np.zeros(7), tau=np.full(7, 8e5))
    res = run(p, 0.0, dp.STALL_TIME_S + 0.3, tgt, mk)
    assert res.stall[2] and not res.grip_hold[2]
    assert res.q_cmd[2] == pytest.approx(-1.0)


def test_finger_stall_holds_pose_with_reduced_force():
    p = Dex3HandProtector(OPEN)
    tgt = np.array([0, 0, 0, 1.15, 1.3, 1.15, 1.3]) * -1
    res = run(p, 0.0, dp.STALL_TIME_S + 0.3, tgt, fblocked)
    for i in range(3, 7):
        assert res.stall[i] and res.grip_hold[i]
        implicit = dp.DEX3_KP * abs(res.q_cmd[i] - FQ[i])
        assert implicit == pytest.approx(dp.GRIP_HOLD_TORQUE_NM[i])
        assert res.q_cmd[i] < FQ[i]                      # still squeezing
        assert abs(res.q_cmd[i]) <= dp.GRIP_HOLD_CMD_LIMIT_RAD[i] + 1e-9
    assert dp.GRIP_HOLD_TORQUE_NM[3] < dp.CLOSE_TORQUE_CEILING_NM[3]


def test_grip_hold_is_time_limited_then_relaxes_until_release():
    p = Dex3HandProtector(OPEN)
    tgt = np.array([0, 0, 0, 1.15, 1.3, 1.15, 1.3]) * -1
    run(p, 0.0, dp.STALL_TIME_S + 0.3, tgt, fblocked)
    t_late = dp.STALL_TIME_S + 0.3 + dp.GRIP_HOLD_MAX_S + 0.5
    res = p.update(t_late, tgt, fblocked(t_late))
    assert res.stall[3] and not res.grip_hold[3]
    assert res.q_cmd[3] == pytest.approx(FQ[3])
    p.update(t_late + 0.01, tgt * 0.4, fblocked(t_late + 0.01))
    res = p.update(t_late + 0.3, tgt * 0.4, fblocked(t_late + 0.3))
    assert not res.stall[3]


def test_grip_hold_fades_with_temperature_and_stops_when_hot():
    FQ2 = (0, 0, 0, -0.1, -0.1, -0.1, -0.1)  # shallow block so the derated target still exceeds STALL_ERR
    tgt = np.array([0, 0, 0, 1.15, 1.3, 1.15, 1.3]) * -1
    for temp, frac in ((65.0, 1.0), (72.5, 0.5), (80.0, 0.0)):
        p = Dex3HandProtector(OPEN)
        res = run(p, 0.0, 1.5, tgt, lambda t, temp=temp: state(t, q=FQ2, dq=np.zeros(7), tau=np.full(7, 8e5), temp=np.full(7, temp)))
        implicit = dp.DEX3_KP * abs(res.q_cmd[3] - FQ2[3])
        assert res.derate[3] == pytest.approx(frac, abs=1e-6)
        if frac == 0.0:
            assert not res.grip_hold[3] and res.q_cmd[3] >= FQ2[3] - 1e-9
        else:
            assert res.grip_hold[3] and implicit <= dp.GRIP_HOLD_TORQUE_NM[3] * frac + 1e-6


def test_grip_hold_never_when_fault():
    p = Dex3HandProtector(OPEN)
    tgt = np.array([0, 0, 0, 1.15, 1.3, 1.15, 1.3]) * -1
    p.update(0.0, tgt, fblocked(0.0))
    mk = lambda t: state(t, q=FQ, dq=np.zeros(7), tau=np.full(7, 8e5),
                         mode=[1, 1, 1, 0, 1, 1, 1], ms=[0, 0, 0, 512, 0, 0, 0])
    res = run(p, 0.01, 2.0, tgt, mk)
    assert res.fault[3] and not res.grip_hold[3] and not res.enable[3]


def test_flags_report_grip_hold():
    p = Dex3HandProtector(OPEN)
    res = p.update(0.0, OPEN, state(0.0))
    assert res.flags()["grip_hold"] == [False] * 7


# ---- (9) no open/close oscillation while holding (fix/dex3-grip-no-oscillation) --
FT = np.array([0, 0, 0, 1.15, 1.3, 1.15, 1.3]) * -1


def _engage(p):
    return run(p, 0.0, dp.STALL_TIME_S + 0.3, FT, fblocked)


def test_hold_torque_stable_without_drop_for_whole_hold():
    p = Dex3HandProtector(OPEN)
    _engage(p)
    t = dp.STALL_TIME_S + 0.3
    vals = []
    while t < dp.STALL_TIME_S + 0.3 + dp.GRIP_HOLD_MAX_S - 1.0:
        res = p.update(t, FT, fblocked(t))
        assert res.stall[3] and res.grip_hold[3]
        vals.append(dp.DEX3_KP * abs(res.q_cmd[3] - FQ[3]))
        t += 0.05
    assert max(vals) - min(vals) < 1e-9
    assert vals[0] == pytest.approx(dp.GRIP_HOLD_TORQUE_NM[3])
    assert dp.GRIP_HOLD_TORQUE_NM[3] >= 1.2 - 1e-9


def test_hold_survives_one_sample_trigger_dropout_and_finger_recoil():
    p = Dex3HandProtector(OPEN)
    _engage(p)
    t = 1.2
    res = p.update(t, FT * 0.0, fblocked(t))      # one 10 ms trigger dropout
    assert res.stall[3] and res.grip_hold[3]
    res = p.update(t + 0.01, FT, state(t + 0.01, q=(0, 0, 0, -0.1, -0.1, -0.1, -0.1),
                                       dq=np.full(7, 5.0), tau=np.full(7, 8e5)))   # recoil + moving
    assert res.stall[3] and res.grip_hold[3]
    assert dp.DEX3_KP * abs(res.q_cmd[3] + 0.1) == pytest.approx(dp.GRIP_HOLD_TORQUE_NM[3])


def test_small_trigger_reduction_does_not_release_but_clear_drop_does():
    p = Dex3HandProtector(OPEN)
    _engage(p)
    t = 1.2
    for k in range(60):                              # 25% reduction for 0.6 s
        res = p.update(t + 0.01 * k, FT * 0.75, fblocked(t + 0.01 * k))
        assert res.stall[3] and res.grip_hold[3]
    t2 = t + 0.6
    for k in range(30):                              # 40% reduction, sustained
        res = p.update(t2 + 0.01 * k, FT * 0.6, fblocked(t2 + 0.01 * k))
    assert not res.stall[3] and not res.grip_hold[3]


def test_hold_command_never_pulls_back_past_limit_and_stays_inside_urdf():
    p = Dex3HandProtector(OPEN)
    q = (0, 0, 0, -0.75, -0.85, -0.75, -0.85)
    mk = lambda t: state(t, q=q, dq=np.zeros(7), tau=np.full(7, 8e5))
    res = run(p, 0.0, dp.STALL_TIME_S + 1.5, FT, mk)
    for i in range(3, 7):
        assert abs(res.q_cmd[i]) <= dp.GRIP_HOLD_CMD_LIMIT_RAD[i] + 1e-9
        assert abs(res.q_cmd[i]) >= abs(q[i]) - 1e-9    # never loosens
    assert max(dp.GRIP_HOLD_CMD_LIMIT_RAD) < 1.745


def test_hold_cut_by_temperature_and_fault_still_win():
    for temp, frac in ((72.5, 0.5), (80.0, 0.0)):
        p = Dex3HandProtector(OPEN)
        mk = lambda t, temp=temp: state(t, q=(0, 0, 0, -0.1, -0.1, -0.1, -0.1), dq=np.zeros(7),
                                        tau=np.full(7, 8e5), temp=np.full(7, temp))
        res = run(p, 0.0, 2.0, FT, mk)
        implicit = dp.DEX3_KP * abs(res.q_cmd[3] + 0.1)
        if frac > 0:
            assert res.grip_hold[3] and implicit <= dp.GRIP_HOLD_TORQUE_NM[3] * frac + 1e-6
        else:
            assert not res.grip_hold[3] and abs(res.q_cmd[3]) < 0.1   # opens toward rest
        assert ("temp_warn", 3) in res.active


def test_no_temp_warning_below_70():
    p = Dex3HandProtector(OPEN)
    res = run(p, 0.0, 1.5, FT, lambda t: state(t, q=FQ, dq=np.zeros(7), tau=np.full(7, 8e5), temp=np.full(7, 69.0)))
    assert ("temp_warn", 3) not in res.active


def test_closed_loop_box_model_no_cycle_old_vs_new():
    """Finger blocked by a box (stiff spring, contact at |q|=0.72); trigger held at 1.0.
    Count stall on->off transitions: new latch must have none after engage."""
    def sim(release_debounce, hold_nm):
        p = Dex3HandProtector(OPEN)
        old = dp.STALL_RELEASE_DEBOUNCE_S
        t, q, trans, prev, torques = 0.0, 0.0, 0, False, []
        qc = 0.72
        while t < 12.0:
            q = min(q + 0.05, qc) if q < qc and True else q
            st = state(t, q=(0, 0, 0, -q, -q, -q, -q), dq=np.zeros(7) if q >= qc else np.full(7, 2.0),
                       tau=np.full(7, 8e5 if q >= qc else 0.0), temp=np.full(7, 45.0))
            r = p.update(t, FT, st)
            if prev and not r.stall[3]:
                trans += 1
            prev = bool(r.stall[3])
            if t > 3.0:
                torques.append(dp.DEX3_KP * abs(r.q_cmd[3] + q))
            t += 0.01
        return trans, torques
    trans, tq = sim(None, None)
    assert trans == 0
    assert min(tq) >= 1.1 and max(tq) <= dp.CLOSE_TORQUE_CEILING_NM[3] + 1e-9
