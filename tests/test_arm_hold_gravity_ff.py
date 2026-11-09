"""C1: tracking hold = LAST PUBLISHED q + model gravity feed-forward at MEASURED q.

Root cause (``/home/bruno/g1d_kin_study/RELATORIO.md`` 3(a)/(b)): the shared
hold froze the measured q with ``tauff = 0``.  Every hold therefore stepped
back by the tracking lag, sagged by ``|tau_g| / kp`` and then jumped when the
first fresh target and full feed-forward returned.  These tests pin the new
contract with fake arm controllers that record every published ``(q, tauff)``.
"""

import threading
import types
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from teleop.utils.arm_command_gate import publish_arm_command
from teleop.utils.arm_tracking_orchestration import run_arm_tracking_cycle
from teleop.utils.arm_enable_ramp import ArmEnableRamp


def gravity(q):
    """Deterministic, pose-dependent, non-zero 'model' gravity."""
    q = np.asarray(q, dtype=float)
    return 2.0 * np.cos(q) + 0.5


class FakeArm:
    def __init__(self, q=None):
        self.measured_q = np.linspace(-0.4, 0.6, 14) if q is None else np.asarray(q, float)
        self.commands = []
        self.snapshot_age = None  # None -> no snapshot method semantics (fresh)

    def get_current_dual_arm_q(self):
        return self.measured_q.copy()

    def ctrl_dual_arm(self, q, tau):
        self.commands.append((np.asarray(q, float).copy(), np.asarray(tau, float).copy()))
        return len(self.commands)


class SnapshotArm(FakeArm):
    def __init__(self, q=None, age=0.0):
        super().__init__(q)
        self.age = age

    def get_dual_arm_q_snapshot(self):
        return self.measured_q.copy(), self.age


LOCK = threading.Lock()


def gate(arm, q, tau, *, accepted=True, fresh=True, started=True, stopped=False, **kw):
    kw.setdefault("hold_gravity_tauff", gravity)
    return publish_arm_command(
        arm, q, tau, target_accepted=accepted, sample_fresh=fresh,
        lifecycle_lock=LOCK, is_started=lambda: started, is_stopped=lambda: stopped, **kw,
    )


Q_A = np.full(14, 0.3)
TAU_A = np.full(14, 1.7)


def test_hold_publishes_last_published_q_not_measured_and_gravity_at_measured():
    arm = FakeArm()
    gate(arm, Q_A, TAU_A)
    decision = gate(arm, np.full(14, 9.0), np.ones(14), accepted=False)
    assert decision.hold and not decision.published
    np.testing.assert_array_equal(arm.commands[-1][0], Q_A)
    assert not np.allclose(arm.commands[-1][0], arm.measured_q)
    np.testing.assert_allclose(arm.commands[-1][1], gravity(arm.measured_q))
    assert np.all(np.abs(arm.commands[-1][1]) > 0.0)
    np.testing.assert_array_equal(decision.selected_q, Q_A)
    np.testing.assert_allclose(decision.selected_tauff, gravity(arm.measured_q))


def test_stale_sample_and_nonfinite_ik_branches_use_the_same_gravity_hold():
    for kwargs in ({"fresh": False}, {"accepted": True, "q": np.full(14, np.nan)}):
        arm = FakeArm()
        gate(arm, Q_A, TAU_A)
        q = kwargs.pop("q", np.full(14, 9.0))
        gate(arm, q, np.ones(14), **kwargs)
        np.testing.assert_array_equal(arm.commands[-1][0], Q_A)
        np.testing.assert_allclose(arm.commands[-1][1], gravity(arm.measured_q))


def test_hold_reference_is_frozen_across_consecutive_holds_but_gravity_follows_measured():
    arm = FakeArm()
    gate(arm, Q_A, TAU_A)
    refs, taus = [], []
    for k in range(5):
        arm.measured_q = arm.measured_q - 0.05  # arm sags / moves under the hold
        gate(arm, np.full(14, 9.0), np.ones(14), fresh=False)
        refs.append(arm.commands[-1][0])
        taus.append(arm.commands[-1][1])
        np.testing.assert_allclose(taus[-1], gravity(arm.measured_q))
    for ref in refs:
        np.testing.assert_array_equal(ref, Q_A)  # never ratchets/resamples
    assert not np.allclose(taus[0], taus[-1])


def test_invalid_measured_state_still_fails_closed_without_publication():
    for bad in (np.full(14, np.nan), np.full(14, np.inf), np.array([]), np.zeros((2, 7))):
        arm = FakeArm()
        gate(arm, Q_A, TAU_A)
        arm.measured_q = bad
        n = len(arm.commands)
        decision = gate(arm, np.full(14, 9.0), np.ones(14), fresh=False)
        assert decision.hold and decision.publication is None
        assert len(arm.commands) == n


def test_first_ever_cycle_hold_uses_measured_q_with_gravity():
    arm = FakeArm()
    gate(arm, np.full(14, 9.0), np.ones(14), accepted=False)
    np.testing.assert_array_equal(arm.commands[-1][0], arm.measured_q)
    np.testing.assert_allclose(arm.commands[-1][1], gravity(arm.measured_q))


def test_hold_to_fresh_transition_has_no_extra_step_versus_last_published():
    arm = FakeArm()
    gate(arm, Q_A, TAU_A)
    for _ in range(3):
        arm.measured_q = arm.measured_q + 0.02
        gate(arm, np.full(14, 9.0), np.ones(14), fresh=False)
    q_b = Q_A + 0.01
    gate(arm, q_b, TAU_A)
    sent = [q for q, _ in arm.commands]
    # Joint reference sequence: A, A, A, A, B -> the only step is A->B.
    for q in sent[:4]:
        np.testing.assert_array_equal(q, Q_A)
    np.testing.assert_array_equal(sent[4], q_b)
    np.testing.assert_allclose(np.max(np.abs(sent[4] - sent[3])), 0.01)
    # The next hold freezes the NEW last published q (B), not the old A.
    gate(arm, np.full(14, 9.0), np.ones(14), fresh=False)
    np.testing.assert_array_equal(arm.commands[-1][0], q_b)


def test_new_tracking_epoch_and_stop_reset_the_last_published_reference():
    arm = FakeArm()
    gate(arm, Q_A, TAU_A)
    gate(arm, Q_A, TAU_A, started=False)  # not started: measured, zero tau (unchanged)
    np.testing.assert_array_equal(arm.commands[-1][0], arm.measured_q)
    np.testing.assert_array_equal(arm.commands[-1][1], np.zeros(14))
    gate(arm, np.full(14, 9.0), np.ones(14), fresh=False)
    np.testing.assert_array_equal(arm.commands[-1][0], arm.measured_q)
    np.testing.assert_allclose(arm.commands[-1][1], gravity(arm.measured_q))

    arm = FakeArm()
    gate(arm, Q_A, TAU_A)
    n = len(arm.commands)
    stop = gate(arm, Q_A, TAU_A, stopped=True)
    assert len(arm.commands) == n and stop.publication is None  # STOP unchanged
    gate(arm, np.full(14, 9.0), np.ones(14), fresh=False)
    np.testing.assert_array_equal(arm.commands[-1][0], arm.measured_q)


def test_gravity_is_bounded_by_the_torque_limit():
    arm = FakeArm()
    gate(arm, Q_A, TAU_A)
    limit = np.full(14, 1.0)
    gate(arm, Q_A, TAU_A, accepted=False, hold_tau_limit=limit)
    tau = arm.commands[-1][1]
    assert np.all(np.abs(tau) <= 1.0 + 1e-12)
    np.testing.assert_allclose(tau, np.clip(gravity(arm.measured_q), -1.0, 1.0))


def test_gravity_unavailable_invalid_or_stale_state_falls_back_to_zero_tau_same_reference():
    def raising(q):
        raise ValueError("model failure")

    cases = [
        (FakeArm(), {"hold_gravity_tauff": raising}),
        (FakeArm(), {"hold_gravity_tauff": lambda q: np.full(14, np.nan)}),
        (FakeArm(), {"hold_gravity_tauff": lambda q: np.ones(13)}),
        (SnapshotArm(age=0.5), {}),          # stale lowstate -> no gravity
        (SnapshotArm(age=float("nan")), {}),
    ]
    for arm, kw in cases:
        gate(arm, Q_A, TAU_A)
        gate(arm, np.full(14, 9.0), np.ones(14), fresh=False, **kw)
        np.testing.assert_array_equal(arm.commands[-1][0], Q_A)
        np.testing.assert_array_equal(arm.commands[-1][1], np.zeros(14))
    arm = SnapshotArm(age=0.01)
    gate(arm, Q_A, TAU_A)
    gate(arm, np.full(14, 9.0), np.ones(14), fresh=False)
    np.testing.assert_allclose(arm.commands[-1][1], gravity(arm.measured_q))


def test_gate_without_gravity_provider_keeps_legacy_measured_zero_hold():
    arm = FakeArm()
    publish_arm_command(
        arm, Q_A, TAU_A, target_accepted=True, sample_fresh=True, lifecycle_lock=LOCK,
        is_started=lambda: True, is_stopped=lambda: False)
    publish_arm_command(
        arm, Q_A, TAU_A, target_accepted=False, sample_fresh=True, lifecycle_lock=LOCK,
        is_started=lambda: True, is_stopped=lambda: False)
    np.testing.assert_array_equal(arm.commands[-1][0], arm.measured_q)
    np.testing.assert_array_equal(arm.commands[-1][1], np.zeros(14))


def test_hold_gravity_scale_follows_the_enable_ramp_feedforward_scale():
    arm = FakeArm()
    gate(arm, Q_A, TAU_A)
    gate(arm, Q_A, TAU_A, fresh=False, hold_gravity_scale=0.25)
    np.testing.assert_allclose(arm.commands[-1][1], 0.25 * gravity(arm.measured_q))


# --------------------------------------------------------------- orchestration


class GravityIK:
    """IK with zero FK residual and a gravity model, like G1_29_ArmIK."""

    def __init__(self, q=0.3):
        self.q = q
        self.gravity_calls = []

    def solve_ik(self, left, right, current_q, current_dq):
        self.last = (left.copy(), right.copy())
        return np.full(14, self.q), gravity(np.full(14, self.q))

    def forward_kinematics(self, q):
        return self.last

    def gravity_tauff(self, q):
        self.gravity_calls.append(np.asarray(q).copy())
        return gravity(q)


class Calib:
    calibrated = True

    def __init__(self, target):
        self.target = target

    def targets(self, poses, ts, now=None):
        return self.target


def target_pair(dx=0.0):
    left, right = np.eye(4), np.eye(4)
    left[0, 3], right[0, 3] = 0.3 + dx, -0.3
    left[2, 3] = right[2, 3] = 0.1
    return left, right


def cycle(arm, ik, ts, now, *, ramp=None, limiter=None, started=True):
    return run_arm_tracking_cycle(
        arm_ctrl=arm, arm_ik=ik, calibrator=Calib(target_pair()), controller_poses=target_pair(),
        sample_timestamp=ts, now=now, current_q=arm.measured_q, current_dq=np.zeros(14),
        lifecycle_lock=threading.Lock(), is_started=lambda: started, is_stopped=lambda: False,
        rate_limiter=limiter, enable_ramp=ramp)


def test_orchestrated_stale_hold_uses_last_published_q_and_model_gravity_at_measured():
    arm, ik = FakeArm(), GravityIK(q=0.3)
    first = cycle(arm, ik, 10.0, 10.0)
    assert first.published
    arm.measured_q = np.full(14, 0.25)  # lagging the 0.3 command
    held = cycle(arm, ik, 10.0, 10.5)   # 0.5 s old sample -> stale hold
    assert held.hold and not held.published
    np.testing.assert_allclose(arm.commands[-1][0], 0.3)
    np.testing.assert_allclose(arm.commands[-1][1], gravity(arm.measured_q))
    np.testing.assert_allclose(held.selected_tauff, gravity(arm.measured_q))


def test_orchestrated_hold_during_enable_ramp_scales_gravity_like_the_ramp():
    arm, ik = FakeArm(), GravityIK(q=0.3)
    ramp = ArmEnableRamp(duration_s=1.0)
    cycle(arm, ik, 10.0, 10.0, ramp=ramp)          # ramp begins at alpha 0
    cycle(arm, ik, 10.5, 10.5, ramp=ramp)          # alpha = 0.5
    alpha = ramp.alpha
    assert 0.0 < alpha < 1.0
    last_q = arm.commands[-1][0].copy()
    cycle(arm, ik, 10.5, 10.9, ramp=ramp)          # stale -> hold mid-ramp
    np.testing.assert_allclose(arm.commands[-1][0], last_q)
    np.testing.assert_allclose(arm.commands[-1][1], alpha * gravity(arm.measured_q))


def test_orchestrated_hold_after_ramp_complete_uses_full_gravity():
    arm, ik = FakeArm(), GravityIK(q=0.3)
    ramp = ArmEnableRamp(duration_s=0.5)
    cycle(arm, ik, 10.0, 10.0, ramp=ramp)
    cycle(arm, ik, 11.0, 11.0, ramp=ramp)
    assert ramp.complete
    cycle(arm, ik, 11.0, 11.5, ramp=ramp)
    np.testing.assert_allclose(arm.commands[-1][0], 0.3)
    np.testing.assert_allclose(arm.commands[-1][1], gravity(arm.measured_q))


def test_orchestrated_ik_without_gravity_model_keeps_legacy_hold():
    class NoGravityIK(GravityIK):
        gravity_tauff = None

    arm, ik = FakeArm(), NoGravityIK(q=0.3)
    cycle(arm, ik, 10.0, 10.0)
    cycle(arm, ik, 10.0, 10.5)
    np.testing.assert_array_equal(arm.commands[-1][0], arm.measured_q)
    np.testing.assert_array_equal(arm.commands[-1][1], np.zeros(14))


def test_orchestrated_hold_to_fresh_target_is_limited_against_last_published_target():
    from teleop.utils.ee_rate_limiter import DualEePoseRateLimiter, EeRateLimiterConfig

    limiter = DualEePoseRateLimiter(EeRateLimiterConfig())
    arm, ik = FakeArm(), GravityIK(q=0.3)
    cycle(arm, ik, 10.0, 10.0, limiter=limiter)
    committed = [p.copy() for p in limiter.last_committed_targets]
    cycle(arm, ik, 10.0, 10.6, limiter=limiter)  # hold, limiter not advanced
    np.testing.assert_allclose(limiter.last_committed_targets[0], committed[0])
    np.testing.assert_allclose(arm.commands[-1][0], 0.3)  # == last published q


# ------------------------------------------------- G1D waist through real writer


def _load_robot_arm(monkeypatch):
    logging_mp = types.ModuleType("logging_mp")
    logging_mp.getLogger = lambda name: types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "logging_mp", logging_mp)
    channel = types.ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelPublisher = object
    channel.ChannelSubscriber = object
    channel.ChannelFactoryInitialize = lambda *args: None
    for name in ("unitree_sdk2py", "unitree_sdk2py.core", "unitree_sdk2py.idl", "unitree_sdk2py.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", channel)
    for name in ("unitree_hg", "unitree_go"):
        msg = types.ModuleType(f"unitree_sdk2py.idl.{name}.msg.dds_")
        msg.LowCmd_ = object
        msg.LowState_ = object
        monkeypatch.setitem(sys.modules, f"unitree_sdk2py.idl.{name}.msg.dds_", msg)
    default = types.ModuleType("unitree_sdk2py.idl.default")
    default.unitree_hg_msg_dds__LowCmd_ = object
    default.unitree_go_msg_dds__LowCmd_ = object
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.default", default)
    crc = types.ModuleType("unitree_sdk2py.utils.crc")
    crc.CRC = object
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.utils.crc", crc)
    monkeypatch.delitem(sys.modules, "teleop.robot_control.robot_arm", raising=False)
    from teleop.robot_control import robot_arm
    return robot_arm


def test_g1d_writer_hold_with_gravity_never_touches_waist_slots(monkeypatch):
    module = _load_robot_arm(monkeypatch)
    try:
        cls = module.G1D_ArmController
        arm = cls.__new__(cls)
        arm.ctrl_lock = threading.Lock()
        arm._init_arm_publication_state()
        arm.q_target = np.zeros(14)
        arm.tauff_target = np.zeros(14)
        arm.motion_mode = False
        arm.simulation_mode = True
        arm._speed_gradual_max = False
        arm.control_dt = 0.0
        arm._motion_authority_weight = 1.0
        arm._last_publish_monotonic = None
        arm._last_published_weight = None
        arm.output_enabled = threading.Event()
        arm.output_enabled.set()

        class Cmd:
            def __init__(self, i):
                waist = i in (12, 13, 14)
                self.q, self.dq = 0.123 * i, 0.0
                self.kp = 0.0 if waist else 80.0
                self.kd = 0.0 if waist else 3.0
                self.tau = 0.0

        arm.msg = types.SimpleNamespace(motor_cmd={i: Cmd(i) for i in range(29)}, crc=0)
        arm.crc = types.SimpleNamespace(Crc=lambda msg: 0)

        def write(msg):
            arm.output_enabled.clear()  # one writer iteration only

        arm.lowcmd_publisher = types.SimpleNamespace(Write=write)
        measured = np.linspace(-0.2, 0.5, 14)
        arm.get_current_dual_arm_q = lambda: measured.copy()
        arm.get_dual_arm_q_snapshot = lambda: (measured.copy(), 0.004)
        before = {i: vars(arm.msg.motor_cmd[i]).copy() for i in (12, 13, 14)}

        gate(arm, Q_A, TAU_A)
        gate(arm, np.full(14, 9.0), np.ones(14), fresh=False)
        arm._ctrl_motor_state()

        for i in (12, 13, 14):
            assert vars(arm.msg.motor_cmd[i]) == before[i]
            assert arm.msg.motor_cmd[i].kp == 0.0 and arm.msg.motor_cmd[i].kd == 0.0
            assert arm.msg.motor_cmd[i].tau == 0.0
        arm_ids = [int(j) for j in module.G1_29_JointArmIndex]
        assert arm_ids == list(range(15, 29))
        np.testing.assert_allclose([arm.msg.motor_cmd[i].q for i in arm_ids], Q_A)
        np.testing.assert_allclose([arm.msg.motor_cmd[i].tau for i in arm_ids], gravity(measured))
    finally:
        sys.modules.pop("teleop.robot_control.robot_arm", None)


# ----------------------------------------------- real model (tv env, optional)


def test_real_g1_29_model_gravity_hold_is_nonzero_for_a_loaded_pose():
    pin = pytest.importorskip("pinocchio")
    pytest.importorskip("casadi")
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
    import os

    cwd = os.getcwd()
    os.chdir(Path(__file__).parents[1] / "teleop")
    try:
        ik = G1_29_ArmIK()
    finally:
        os.chdir(cwd)
    q = np.zeros(14)
    q[0] = q[7] = -0.8  # shoulders pitched forward: arm out in front
    q[3] = q[10] = 0.3
    arm = FakeArm(q)
    gate(arm, Q_A, TAU_A, hold_gravity_tauff=ik.gravity_tauff)
    gate(arm, Q_A, TAU_A, fresh=False, hold_gravity_tauff=ik.gravity_tauff)
    tau = arm.commands[-1][1]
    np.testing.assert_allclose(tau, ik.gravity_tauff(q))
    assert abs(tau[0]) > 1.0 and abs(tau[7]) > 1.0  # shoulder pitch carries load
    np.testing.assert_array_equal(arm.commands[-1][0], Q_A)
