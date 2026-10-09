"""Dex3 hand-state reception is independent per hand.

Incident: both sides were read by one thread with blocking ``Read()``; when the
right Dex3 driver went silent the thread hung in the right ``Read`` and the
LEFT state timestamp froze too, so shutdown close-on-q refused BOTH hands.

Fake SDK channels only: no DDS participant, no publisher, no actuator.
"""

import contextlib
import importlib
import importlib.util
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

from teleop.utils.dex3_state_grace import state_is_fresh  # noqa: E402

BASE = "418e14a"
LEFT_TOPIC, RIGHT_TOPIC = "rt/dex3/left/state", "rt/dex3/right/state"


# ------------------------------------------------------------- fake SDK
class Hub:
    """Fake DDS bus: per-topic subscribers, a background publisher per side."""

    def __init__(self):
        self.subs = {}
        self.publishers_created = 0
        self.active = {LEFT_TOPIC, RIGHT_TOPIC}
        self.q_source = {LEFT_TOPIC: lambda: np.zeros(7), RIGHT_TOPIC: lambda: np.zeros(7)}
        self.closed = threading.Event()
        self.counts = {LEFT_TOPIC: 0, RIGHT_TOPIC: 0}
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self.closed.is_set():
            for topic in (LEFT_TOPIC, RIGHT_TOPIC):
                sub = self.subs.get(topic)
                if sub is not None and topic in self.active:
                    sub.deliver(make_msg(self.q_source[topic]()))
                    self.counts[topic] += 1
            time.sleep(0.002)

    def stop(self):
        self.closed.set()
        self._thread.join(timeout=1.0)
        for sub in self.subs.values():
            with sub.cond:
                sub.cond.notify_all()


def make_msg(q, temp=40):
    motors = [ns(q=float(q[i % 7]), dq=0.0, tau_est=0.0, temperature=[temp, temp], mode=1, motorstate=0)
              for i in range(7)]
    return ns(motor_state=motors, press_sensor_state=[])


def sdk_stubs(hub):
    class Subscriber:
        def __init__(self, name, typ):
            self.name = name
            self.handler = None
            self.queue_len = None
            self.read_calls = 0
            self.close_calls = 0
            self.pending = None
            self.cond = threading.Condition()
            hub.subs[name] = self

        def Init(self, handler=None, queueLen=0):
            self.handler, self.queue_len = handler, queueLen

        def deliver(self, msg):
            if self.close_calls:
                return
            if self.handler is not None:
                self.handler(msg)            # SDK listener path (DDS thread)
            else:
                with self.cond:
                    self.pending = msg
                    self.cond.notify_all()

        def Read(self, timeout=None):
            # SDK Read(None) == take_one() with a near-infinite wait.
            self.read_calls += 1
            deadline = None if timeout is None else time.monotonic() + timeout
            with self.cond:
                while self.pending is None and not hub.closed.is_set():
                    if deadline is not None and time.monotonic() >= deadline:
                        return None
                    self.cond.wait(0.05)
                msg, self.pending = self.pending, None
                return msg

        def Close(self):
            self.close_calls += 1

    class Publisher:
        def __init__(self, *a, **k):
            hub.publishers_created += 1

        def Init(self):
            pass

        def Write(self, *a, **k):
            raise AssertionError("no DDS publication in these tests")

    sdk_channel = types.ModuleType("unitree_sdk2py.core.channel")
    sdk_channel.ChannelPublisher = Publisher
    sdk_channel.ChannelSubscriber = Subscriber
    sdk_channel.ChannelFactoryInitialize = lambda *a, **k: None
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


def _logger(logs):
    lm = types.ModuleType("logging_mp")
    lm.getLogger = lambda name: ns(warning=lambda m, *a, **k: logs.append(("warning", m)),
                                   info=lambda m, *a, **k: logs.append(("info", m)),
                                   error=lambda m, *a, **k: logs.append(("error", m)))
    return lm


@pytest.fixture
def hub():
    h = Hub()
    h.start()
    yield h
    h.stop()


@pytest.fixture
def module(monkeypatch, hub):
    for name, mod in sdk_stubs(hub).items():
        monkeypatch.setitem(sys.modules, name, mod)
    logs = []
    monkeypatch.setitem(sys.modules, "logging_mp", _logger(logs))
    sys.modules.pop("teleop.robot_control.robot_hand_unitree", None)
    try:
        mod = importlib.import_module("teleop.robot_control.robot_hand_unitree")
    finally:
        sys.modules.pop("teleop.robot_control.robot_hand_unitree", None)
    mod._test_logs = logs
    return mod


@pytest.fixture
def base_module(monkeypatch, hub, tmp_path):
    try:
        source = subprocess.run(["git", "show", f"{BASE}:teleop/robot_control/robot_hand_unitree.py"],
                                cwd=ROOT, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("base commit not available")
    for name, mod in sdk_stubs(hub).items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, "logging_mp", _logger([]))
    path = tmp_path / "base_robot_hand_unitree.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("base_robot_hand_unitree", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def construct(module):
    c = module.Dex3_1_Controller(None, None)
    return c


def left_right_ages(c):
    now = time.monotonic()
    _, _, meta = c.get_pose_samples()
    return now - meta["left"]["state_timestamp"], now - meta["right"]["state_timestamp"]


def protection_fresh(c, side):
    with c._telemetry_lock:
        state = c._protection_state.get(side)
    return state_is_fresh(time.monotonic(), state)


def close_readers(c):
    closer = getattr(c, "close_state_readers", None)
    if closer is not None:
        closer()


def _wait_until(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


# ------------------------------------------------------------------ tests
def test_base_reproduces_left_freeze_when_right_is_silent(base_module, hub):
    """Documents the incident on the base commit (single blocking reader)."""
    c = construct(base_module)
    time.sleep(0.1)
    hub.active.discard(RIGHT_TOPIC)
    time.sleep(0.8)
    left_age, right_age = left_right_ages(c)
    assert left_age > 0.5 and right_age > 0.5          # left frozen with right


def test_each_hand_uses_its_own_sdk_callback_and_never_blocks_in_read(module, hub):
    c = construct(module)
    try:
        for topic in (LEFT_TOPIC, RIGHT_TOPIC):
            sub = hub.subs[topic]
            assert callable(sub.handler) and sub.queue_len == 0
            assert sub.read_calls == 0
        assert hub.subs[LEFT_TOPIC].handler is not hub.subs[RIGHT_TOPIC].handler
        assert hub.publishers_created == 0                  # passive pre-arm unchanged
        assert c.hand_control_process is None and c.outputs_activated is False
    finally:
        close_readers(c)


def test_right_silent_left_stays_fresh(module, hub):
    c = construct(module)
    try:
        time.sleep(0.1)
        hub.active.discard(RIGHT_TOPIC)
        time.sleep(0.8)
        left_age, right_age = left_right_ages(c)
        assert left_age < 0.2, left_age
        assert right_age > 0.6, right_age
        assert protection_fresh(c, "left") and not protection_fresh(c, "right")
        (lp, lts), (rp, rts) = c.get_pressure_samples()
        assert time.monotonic() - lts < 0.2 and time.monotonic() - rts > 0.6
    finally:
        close_readers(c)


def test_left_silent_right_stays_fresh(module, hub):
    c = construct(module)
    try:
        time.sleep(0.1)
        hub.active.discard(LEFT_TOPIC)
        time.sleep(0.8)
        left_age, right_age = left_right_ages(c)
        assert right_age < 0.2 and left_age > 0.6
        assert protection_fresh(c, "right") and not protection_fresh(c, "left")
    finally:
        close_readers(c)


def test_silent_hand_resuming_recovers(module, hub):
    c = construct(module)
    try:
        hub.active.discard(RIGHT_TOPIC)
        time.sleep(0.7)
        assert not protection_fresh(c, "right")
        hub.active.add(RIGHT_TOPIC)
        assert _wait_until(lambda: protection_fresh(c, "right"), 1.0)
        assert left_right_ages(c)[1] < 0.2
    finally:
        close_readers(c)


def test_state_array_written_per_side(module, hub):
    hub.q_source[LEFT_TOPIC] = lambda: np.full(7, 0.25)
    hub.q_source[RIGHT_TOPIC] = lambda: np.full(7, -0.5)
    c = construct(module)
    try:
        assert _wait_until(lambda: np.allclose(c.left_hand_state_array[:], 0.25)
                           and np.allclose(c.right_hand_state_array[:], -0.5), 1.0)
    finally:
        close_readers(c)


# --------------------------------------------------- shutdown close-on-q
class Sample:
    def get_lock(self):
        return contextlib.nullcontext()

    def __getitem__(self, k):
        return [0.0, time.monotonic()][k]


def attach_fake_writers(c):
    written = {"left": [], "right": []}

    def msg():
        return ns(motor_cmd=[ns(q=0.0, dq=0.0, tau=0.0, kp=1.5, kd=0.2, mode=1) for _ in range(7)])

    c.left_msg, c.right_msg = msg(), msg()
    for side in written:
        setattr(c, f"{side.capitalize()}HandCmb_publisher", ns(Write=lambda m, side=side: written[side].append(
            np.array([x.q for x in m.motor_cmd]))))
    return written


def run_shutdown_close(module, c, hub, written, seconds=1.4):
    # ideal servo: published state follows the last command written
    hub.q_source[LEFT_TOPIC] = lambda: written["left"][-1] if written["left"] else np.zeros(7)
    hub.q_source[RIGHT_TOPIC] = lambda: written["right"][-1] if written["right"] else np.zeros(7)
    for _ in range(20):                                     # a few normal cycles first
        out = c.control_step(None, None, left_ctrl_sample_in=Sample(), right_ctrl_sample_in=Sample())
        time.sleep(0.005)
    c._shutdown_hand_mode = "close"                          # what begin_shutdown_hand sets
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        out = c.control_step(None, None, left_ctrl_sample_in=Sample(), right_ctrl_sample_in=Sample())
        with c._telemetry_lock:
            c._left_action, c._right_action = out[0].copy(), out[1].copy()
            c._left_action_valid = c._right_action_valid = True
        time.sleep(0.01)
    return c.shutdown_hand_summary()


def test_shutdown_left_closes_while_right_silent_is_refused(module, hub):
    c = construct(module)
    written = attach_fake_writers(c)
    try:
        hub.active.discard(RIGHT_TOPIC)
        time.sleep(0.7)
        summary = run_shutdown_close(module, c, hub, written)
        assert summary["left"]["blocked_reason"] is None and summary["left"]["done"]
        assert summary["right"]["blocked_reason"] == "state_stale"
        np.testing.assert_allclose(written["left"][-1], module.Dex3_Left_Closed_Pose, atol=1e-6)
        np.testing.assert_allclose(written["right"][-1], module.Dex3_Open_Pose, atol=1e-9)
        assert any("NÃO fecha" in m and "right" in m for _, m in module._test_logs)
        assert not any("NÃO fecha" in m and "left" in m for _, m in module._test_logs)
    finally:
        close_readers(c)


def test_shutdown_right_closes_while_left_silent_is_refused(module, hub):
    c = construct(module)
    written = attach_fake_writers(c)
    try:
        hub.active.discard(LEFT_TOPIC)
        time.sleep(0.7)
        summary = run_shutdown_close(module, c, hub, written)
        assert summary["right"]["blocked_reason"] is None and summary["right"]["done"]
        assert summary["left"]["blocked_reason"] == "state_stale"
        np.testing.assert_allclose(written["right"][-1], module.Dex3_Right_Closed_Pose, atol=1e-6)
        np.testing.assert_allclose(written["left"][-1], module.Dex3_Open_Pose, atol=1e-9)
    finally:
        close_readers(c)


def test_shutdown_both_publishing_closes_both_unchanged(module, hub):
    c = construct(module)
    written = attach_fake_writers(c)
    try:
        summary = run_shutdown_close(module, c, hub, written)
        assert summary["left"]["blocked_reason"] is None and summary["right"]["blocked_reason"] is None
        np.testing.assert_allclose(written["left"][-1], module.Dex3_Left_Closed_Pose, atol=1e-6)
        np.testing.assert_allclose(written["right"][-1], module.Dex3_Right_Closed_Pose, atol=1e-6)
        assert hub.publishers_created == 0
    finally:
        close_readers(c)


# ------------------------------------------------------ absence warning
def test_absent_hand_warning_is_rate_limited_and_side_specific(module, hub, monkeypatch):
    monkeypatch.setattr(module, "DEX3_STATE_ABSENT_WARN_S", 0.3)
    monkeypatch.setattr(module, "DEX3_STATE_ABSENT_REPEAT_S", 10.0)
    c = construct(module)
    try:
        hub.active.discard(RIGHT_TOPIC)
        time.sleep(0.9)
        absent = [m for lvl, m in module._test_logs if lvl == "warning" and "sem estado DDS" in m]
        assert len(absent) == 1, absent
        assert "mão direita" in absent[0] and "reinicie" in absent[0]
        assert not any("mão esquerda sem estado DDS" in m for _, m in module._test_logs)
        hub.active.add(RIGHT_TOPIC)
        assert _wait_until(lambda: any("mão direita" in m and "recuperado" in m
                                       for _, m in module._test_logs), 1.0)
    finally:
        close_readers(c)


def test_absence_warner_pure_rules(module):
    w = module.Dex3StateAbsenceWarner("right", warn_after_s=1.0, repeat_s=5.0)
    assert w.update(10.0, 9.5) is None                       # fresh: silent
    msg = w.update(11.0, 9.5)
    assert msg and "mão direita sem estado DDS há 1.5 s" in msg
    assert w.update(12.0, 9.5) is None                       # rate-limited
    assert "há 6.5 s" in w.update(16.0, 9.5)
    assert "recuperado" in w.update(16.1, 16.05)
    assert w.update(16.2, 16.15) is None
    never = module.Dex3StateAbsenceWarner("left", warn_after_s=1.0, repeat_s=5.0)
    assert never.update(5.0, None, started_at=3.0).startswith("[Dex3] mão esquerda sem estado DDS")
    assert never.update(5.1, float("nan"), started_at=3.0) is None   # rate-limited, never raises


# ------------------------------------------------------------- lifecycle
def test_close_state_readers_is_bounded_and_idempotent(module, hub):
    c = construct(module)
    hub.active.discard(RIGHT_TOPIC)                          # one side silent at close
    threads = list(c._state_reader_threads.values())
    assert len(threads) == 2 and all(t.is_alive() and t.daemon for t in threads)
    t0 = time.monotonic()
    c.close_state_readers(timeout_s=1.0)
    assert time.monotonic() - t0 < 1.0
    assert not any(t.is_alive() for t in threads)
    assert hub.subs[LEFT_TOPIC].close_calls == 1 and hub.subs[RIGHT_TOPIC].close_calls == 1
    frozen = c.get_pose_samples()[2]["left"]["state_timestamp"]
    c._on_hand_state("left", make_msg(np.ones(7)))           # late DDS callback after close
    time.sleep(0.05)
    assert c.get_pose_samples()[2]["left"]["state_timestamp"] == frozen
    c.close_state_readers(timeout_s=1.0)                     # idempotent
    assert hub.subs[LEFT_TOPIC].close_calls == 1
    assert hub.publishers_created == 0


def test_callback_is_nonblocking_and_never_raises(module, hub):
    c = construct(module)
    try:
        t0 = time.perf_counter()
        for _ in range(1000):
            c._on_hand_state("right", make_msg(np.zeros(7)))
        assert (time.perf_counter() - t0) / 1000 < 0.001
        c._on_hand_state("right", None)                      # malformed sample
        c._on_hand_state("bogus", make_msg(np.zeros(7)))
        time.sleep(0.05)
        assert c._state_reader_threads["right"].is_alive()
    finally:
        close_readers(c)


def test_teleop_closes_dex3_state_readers_on_exit():
    source = (ROOT / "teleop" / "teleop_hand_and_arm.py").read_text(encoding="utf-8")
    assert "close_state_readers" in source
