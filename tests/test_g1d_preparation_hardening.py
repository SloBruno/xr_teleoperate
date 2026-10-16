"""Behavioural tests for the hardened G1D arm preparation (fake clock/state/publisher)."""
import ast
import hashlib
import logging
import subprocess
import threading
from enum import IntEnum
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "teleop" / "robot_control" / "robot_arm.py"
SRC = ARM.read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def _cls(name):
    import copy
    return copy.deepcopy(next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == name))


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _build(clock, errors=None):
    ns = {"np": np, "IntEnum": IntEnum, "threading": threading, "logger_mp": logging.getLogger("t"),
          "time": type("T", (), {"monotonic": staticmethod(clock.monotonic), "sleep": staticmethod(clock.sleep)})}
    exec(compile(ast.Module(body=[_cls("G1_29_JointArmIndex")], type_ignores=[]), str(ARM), "exec"), ns)
    body = [n for n in _cls("G1D_ArmController").body if not (isinstance(n, ast.FunctionDef) and n.name == "__init__")]
    node = _cls("G1D_ArmController")
    node.body = body
    node.bases = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(ARM), "exec"), ns)
    return ns["G1D_ArmController"], ns["G1_29_JointArmIndex"]


class Cmd:
    def __init__(self, kp, kd):
        self.kp, self.kd, self.q, self.tau = kp, kd, 0.0, 0.0


class Msg:
    def __init__(self):
        self.motor_cmd = {i: Cmd(80.0 if i < 22 else 40.0, 3.0 if i < 22 else 1.5) for i in range(29)}
        for i in (12, 13, 14):
            self.motor_cmd[i] = Cmd(0.0, 0.0)


def make_arm(q_fn, clock, age=0.0):
    cls, _ = _build(clock)

    class Arm(cls):
        def __init__(self):
            self.ctrl_lock = threading.Lock()
            self.q_target = np.full(14, 9.0)
            self.tauff_target = np.zeros(14)
            self._command_request_id = 0
            self.output_enabled = threading.Event()
            self.output_enabled.set()
            self.msg = Msg()
            self.history = []  # (q_target, kp, kd) captured at each sleep
            self.q_fn, self.age = q_fn, age

        def get_dual_arm_q_snapshot(self):
            a = self.age(clock.t) if callable(self.age) else self.age
            return self.q_fn(clock.t), a

        def get_current_dual_arm_q(self):
            return self.q_fn(clock.t)

        def deactivate(self):
            self.deactivated_at = clock.t
            self.output_enabled.clear()
            self.final_target = self.q_target.copy()

    return Arm()


def test_timeout_holds_measured_q_with_low_gain_then_deactivates():
    clock = Clock()
    q = np.zeros(14); q[3] = 1.1
    arm = make_arm(lambda t: q, clock)
    assert arm.ctrl_dual_arm_go_home(cancelled=lambda: False, require_fresh_lowstate=True) is False
    assert arm.last_preparation_status == "preparation_timeout"
    np.testing.assert_array_equal(arm.final_target, q)  # hold target = measured q, not zero
    assert not arm.output_enabled.is_set()
    for i in range(15, 29):
        assert arm.msg.motor_cmd[i].kp <= 40.0
        assert arm.msg.motor_cmd[i].kd in (1.5, arm.prep_hold_kd)
    assert arm.deactivated_at >= 10.0 + arm.prep_hold_seconds - 0.1  # held a few seconds first


def test_q_cancel_and_lowstate_failure_hold_measured_q():
    for kind in ("cancel", "stale"):
        clock = Clock()
        q = np.linspace(-0.3, 0.3, 14)
        arm = make_arm(lambda t: q, clock, age=(lambda t: 0.0 if t < 0.5 else 1.0) if kind == "stale" else 0.0)
        cancelled = (lambda: clock.t >= 0.5) if kind == "cancel" else (lambda: False)
        assert not arm.ctrl_dual_arm_go_home(cancelled=cancelled, require_fresh_lowstate=True)
        np.testing.assert_array_equal(arm.final_target, q)
        assert arm.msg.motor_cmd[15].kp <= 40.0
        assert arm.last_preparation_status == ("preparation_cancelled" if kind == "cancel" else "preparation_lowstate_stale")
        assert not arm.output_enabled.is_set()


def test_per_joint_error_logged_periodically_and_at_timeout_with_worst_joint(caplog):
    clock = Clock()
    q = np.zeros(14); q[3] = 1.1; q[6] = -0.35
    events = []
    arm = make_arm(lambda t: q, clock)
    with caplog.at_level(logging.INFO):
        arm.ctrl_dual_arm_go_home(cancelled=lambda: False, require_fresh_lowstate=True,
                                  progress=lambda e, t: events.append((e, t)))
    progress = [e for e in events if e[0] == "preparation_progress"]
    assert 9 <= len(progress) <= 12  # ~1 per second over 10 s
    final = events[-1]
    assert final[0] == "preparation_timeout"
    assert "worst_joint=18:" in final[1] and "max_abs_err=1.100" in final[1]
    for joint in range(15, 29):
        assert f"{joint}:" in final[1]
    assert "q_measured=" in final[1]
    assert "worst_joint" in caplog.text and arm.last_preparation_report == final[1]


def test_timeout_scales_with_largest_initial_error():
    clock = Clock()
    q = np.zeros(14); q[0] = 5.0  # 3*5/2.1 = 7.14 -> floor 10
    arm = make_arm(lambda t: q, clock)
    arm.ctrl_dual_arm_go_home(cancelled=lambda: False)
    assert abs(arm.deactivated_at - arm.prep_hold_seconds - 10.0) < 0.2

    clock = Clock()
    q = np.zeros(14); q[0] = 8.4  # 3*8.4/2.1 = 12
    arm = make_arm(lambda t: q, clock)
    arm.ctrl_dual_arm_go_home(cancelled=lambda: False)
    assert abs(arm.deactivated_at - arm.prep_hold_seconds - 12.0) < 0.2


def test_arrival_requires_stable_window_inside_same_tolerance():
    clock = Clock()
    # touches tolerance briefly (0.1 s), leaves, then stays 0.4 s inside
    def q_fn(t):
        q = np.zeros(14)
        if not ((1.0 <= t < 1.1) or t >= 3.0):
            q[0] = 0.2
        if t >= 3.0:
            q[5] = 0.05  # boundary is inside (<=)
        return q
    arm = make_arm(q_fn, clock)
    assert arm.ctrl_dual_arm_go_home(cancelled=lambda: False, require_fresh_lowstate=True)
    assert arm.last_preparation_status == "preparation_ready"
    assert 3.3 <= clock.t <= 3.5  # not at the 1.0 s brief touch
    np.testing.assert_array_equal(arm.q_target, np.zeros(14))
    assert arm.output_enabled.is_set()  # success does not hold/deactivate
    # just outside tolerance never arrives
    clock = Clock()
    arm = make_arm(lambda t: np.full(14, 0.051), clock)
    assert not arm.ctrl_dual_arm_go_home(cancelled=lambda: False)


def test_waist_slots_never_commanded_during_hold():
    clock = Clock()
    arm = make_arm(lambda t: np.full(14, 0.5), clock)
    arm.ctrl_dual_arm_go_home(cancelled=lambda: False)
    for i in (12, 13, 14):
        c = arm.msg.motor_cmd[i]
        assert (c.kp, c.kd, c.tau) == (0.0, 0.0, 0.0)
    assert arm.q_target.shape == (14,)


def test_normal_g1_path_is_byte_identical_to_base_commit():
    base = subprocess.run(["git", "show", "1bee958:teleop/robot_control/robot_arm.py"], cwd=ROOT,
                          capture_output=True, check=True).stdout.decode()
    def seg(text):
        tree = ast.parse(text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "G1_29_ArmController")
        return ast.get_source_segment(text, cls)
    assert hashlib.sha256(seg(base).encode()).digest() == hashlib.sha256(seg(SRC).encode()).digest()
    assert "preparation_progress" not in seg(SRC)
