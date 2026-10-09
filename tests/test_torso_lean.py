"""Torso lean (G1_TORSO_LEAN): head displacement -> waist pitch/roll via rt/arm_sdk.

Fakes only (no DDS, no actuators). pinocchio is used for the URDF sign
convention and the IK/FK geometry checks (same model as the teleop IK).
"""
import importlib
import importlib.util
import math
import os
import runpy
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teleop.utils import torso_lean as tl  # noqa: E402
from teleop.utils import pose_stream as ps  # noqa: E402
from teleop.utils.arm_graceful_shutdown import run_graceful_arm_shutdown  # noqa: E402

DEG = math.pi / 180.0
CFG = tl.TorsoLeanConfig()
URDF = ROOT / "assets" / "g1" / "g1_body29_hand14.urdf"


# ----------------------------------------------------------------- helpers
def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def head(pivot=(0.0, 0.0, 1.6), yaw=0.0, pitch=0.0, roll=0.0):
    """Head pose (robot basis) whose neck pivot is at ``pivot``."""
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pivot, float) + R @ tl.NECK_TO_EYE
    return T


def run_tracker(poses, dt=1 / 30.0, cfg=CFG):
    tr = tl.HeadLeanTracker(cfg)
    t = 0.0
    assert tr.engage(poses[0], t)
    out = []
    for p in poses[1:]:
        t += dt
        out.append(tr.update(p, t))
    return tr, out


# ------------------------------------------------- sign convention (URDF)
def _full_model():
    pin = pytest.importorskip("pinocchio")
    robot = pin.RobotWrapper.BuildFromURDF(str(URDF), str(URDF.parent))
    return pin, robot.model, robot.model.createData()


def _waist_q(model, yaw=0.0, roll=0.0, pitch=0.0, base=None):
    q = np.zeros(model.nq) if base is None else base.copy()
    for name, v in (("waist_yaw_joint", yaw), ("waist_roll_joint", roll), ("waist_pitch_joint", pitch)):
        q[model.joints[model.getJointId(name)].idx_q] = v
    return q


def test_urdf_sign_convention_pitch_forward_roll_positive_is_right():
    pin, model, data = _full_model()
    fid = model.getFrameId("torso_link")
    top = np.array([0.0, 0.0, 0.4])  # a point 40 cm above the torso origin (head/chest)

    def point(**kw):
        pin.framesForwardKinematics(model, data, _waist_q(model, **kw))
        return data.oMf[fid].act(top)

    p0 = point()
    fwd = point(pitch=0.1) - p0
    side = point(roll=0.1) - p0
    assert fwd[0] > 0.03 and abs(fwd[1]) < 1e-6        # +pitch: torso tilts FORWARD (+x)
    assert side[1] < -0.03 and abs(side[0]) < 1e-6     # +roll: torso tilts RIGHT (-y)
    # joint axes as declared in the URDF
    src = URDF.read_text()
    for name, axis in (("waist_yaw_joint", "0 0 1"), ("waist_roll_joint", "1 0 0"), ("waist_pitch_joint", "0 1 0")):
        block = src[src.index(f'<joint name="{name}"'):]
        assert f'<axis xyz="{axis}"/>' in block[:block.index("</joint>")]


def test_mapping_signs_follow_urdf_forward_pitch_left_negative_roll():
    pitch, roll = tl.lean_from_displacement(0.10, 0.0, CFG)
    assert pitch > 0 and roll == 0.0
    pitch, roll = tl.lean_from_displacement(0.0, 0.10, CFG)   # operator moves LEFT
    assert roll < 0 and pitch == 0.0                         # torso tilts LEFT = negative roll


def test_waist_indices_match_enums_and_official_example():
    module = _import_robot_arm()
    J = module.G1_29_JointIndex
    assert (J.kWaistYaw, J.kWaistRoll, J.kWaistPitch) == (12, 13, 14)
    assert tuple(int(i) for i in module.G1_29_WAIST_INDICES) == tl.WAIST_MOTOR_INDICES == (12, 13, 14)
    assert J.kNotUsedJoint0 == 29


def test_waist_rotation_matches_pinocchio_and_urdf_limits():
    pin, model, data = _full_model()
    rng = np.random.default_rng(3)
    for _ in range(5):
        y, r, p = rng.uniform(-0.5, 0.5, 3)
        pin.framesForwardKinematics(model, data, _waist_q(model, y, r, p))
        R = data.oMf[model.getFrameId("torso_link")].rotation
        np.testing.assert_allclose(tl.waist_rotation((y, r, p)), R, atol=1e-12)
    for k, name in enumerate(("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")):
        jid = model.getJointId(name)
        iq = model.joints[jid].idx_q
        assert model.lowerPositionLimit[iq] == pytest.approx(tl.URDF_WAIST_LOWER[k])
        assert model.upperPositionLimit[iq] == pytest.approx(tl.URDF_WAIST_UPPER[k])


# ------------------------------------------------------------- mapping
def test_deadband():
    assert tl.lean_from_displacement(0.029, -0.029, CFG) == (0.0, 0.0)
    assert tl.lean_from_displacement(0.03, 0.03, CFG) == (0.0, 0.0)


def test_gain_15cm_beyond_deadband_is_10deg():
    pitch, roll = tl.lean_from_displacement(0.03 + 0.075, -(0.03 + 0.15), CFG)
    assert pitch == pytest.approx(5.0 * DEG)
    assert roll == pytest.approx(10.0 * DEG)   # operator moved right -> positive roll


@pytest.mark.parametrize("fwd,left", [(1e3, 0), (-1e3, 0), (0, 1e3), (0, -1e3), (5.0, -7.0), (-0.4, 0.4)])
def test_saturation_10deg_all_directions_even_for_huge_input(fwd, left):
    pitch, roll = tl.lean_from_displacement(fwd, left, CFG)
    assert abs(pitch) <= 10.0 * DEG + 1e-12 and abs(roll) <= 10.0 * DEG + 1e-12
    cfg3 = tl.TorsoLeanConfig(max_deg=3.0)
    pitch, roll = tl.lean_from_displacement(fwd, left, cfg3)
    assert abs(pitch) <= 3.0 * DEG + 1e-12 and abs(roll) <= 3.0 * DEG + 1e-12


def test_max_deg_above_hard_ceiling_20_is_clamped_even_if_config_built_directly():
    assert tl.HARD_MAX_LEAN_DEG == 20.0
    assert tl.TorsoLeanConfig(max_deg=45.0).max_rad == pytest.approx(20.0 * DEG)
    assert tl.TorsoLeanConfig(max_deg=20.0).max_rad == pytest.approx(20.0 * DEG)


def test_nan_displacement_gives_zero():
    assert tl.lean_from_displacement(float("nan"), float("nan"), CFG) == (0.0, 0.0)


# -------------------------------------------------- head -> displacement
def test_looking_down_or_around_does_not_lean():
    poses = [head()] + [head(pitch=a * DEG) for a in range(0, 61, 3)] + [head(yaw=a * DEG) for a in range(0, 41, 4)] \
        + [head(roll=a * DEG) for a in range(0, 31, 3)]
    tr, out = run_tracker(poses)
    assert all(o == (0.0, 0.0) for o in out)


def test_forward_displacement_leans_forward_side_displacement_rolls():
    tr, out = run_tracker([head(), head(pivot=(0.20, 0.0, 1.6))])
    assert out[-1][0] == pytest.approx(10.0 * DEG) and out[-1][1] == 0.0
    tr, out = run_tracker([head(), head(pivot=(0.0, 0.105, 1.6))])
    assert out[-1][1] == pytest.approx(-5.0 * DEG)


def test_height_is_ignored():
    tr, out = run_tracker([head(), head(pivot=(0.0, 0.0, 1.2))])
    assert out[-1] == (0.0, 0.0)


def test_displacement_in_neutral_yaw_frame_operator_facing_90deg():
    yaw0 = 90 * DEG  # operator faces world +y at r
    tr, out = run_tracker([head(yaw=yaw0), head(pivot=(0.0, 0.20, 1.6), yaw=yaw0)])
    assert out[-1][0] == pytest.approx(10.0 * DEG) and out[-1][1] == 0.0   # forward for the operator
    tr, out = run_tracker([head(yaw=yaw0), head(pivot=(-0.105, 0.0, 1.6), yaw=yaw0)])
    assert out[-1][0] == 0.0 and out[-1][1] == pytest.approx(-5.0 * DEG)  # world -x = operator's left


def test_invalid_nan_and_fallback_head_pose_hold_then_decay():
    cfg = CFG
    tr = tl.HeadLeanTracker(cfg)
    assert not tr.engage(tl.FALLBACK_HEAD_POSE, 0.0)           # TeleVuer "no data" pose
    assert not tr.engage(np.full((4, 4), np.nan), 0.0)
    assert tr.engage(head(), 0.0)
    target = tr.update(head(pivot=(0.15, 0, 1.6)), 0.03)
    assert target[0] > 0
    bad = head(pivot=(0.3, 0, 1.6))
    bad[0, 0] = np.nan
    t = 0.03
    while t < 1.0:
        t += 0.033
        assert tr.update(bad, t) == target                      # hold last target
    t += 0.1
    assert tr.update(bad, t) == (0.0, 0.0) and tr.status == "lost_decay"
    # non-rigid matrix is rejected too
    assert tl.head_pivot_and_yaw(np.diag([2.0, 1, 1, 1]))[0] is None


def test_stale_repeated_sample_holds_then_decays():
    tr = tl.HeadLeanTracker(CFG)
    assert tr.engage(head(), 0.0)
    p = head(pivot=(0.15, 0, 1.6))
    target = tr.update(p, 0.03)
    t = 0.03
    statuses = []
    while t < 1.5:
        t += 0.033
        out = tr.update(p.copy(), t)        # same values: no new XR sample
        statuses.append(tr.status)
        if t < 0.03 + 0.2 + 1.0 - 0.05:
            assert out == target
    assert "hold" in statuses and tr.status == "lost_decay" and tr.target == (0.0, 0.0)


def test_jump_is_ignored_and_reanchored():
    tr = tl.HeadLeanTracker(CFG)
    assert tr.engage(head(), 0.0)
    a = tr.update(head(pivot=(0.10, 0, 1.6)), 0.03)
    b = tr.update(head(pivot=(0.10 + 0.6, 0.3, 1.6)), 0.06)    # Quest recentre: 0.67 m in one frame
    assert b == a and tr.jumps == 1 and tr.status == "jump"
    c = tr.update(head(pivot=(0.10 + 0.6 + 0.05, 0.3, 1.6)), 0.09)  # continues from the last accepted
    assert c[0] == pytest.approx(tl.lean_from_displacement(0.15, 0.0, CFG)[0])
    # recentre that also rotates the world yaw by 90 deg keeps "forward" = operator forward
    tr = tl.HeadLeanTracker(CFG)
    assert tr.engage(head(), 0.0)
    tr.update(head(pivot=(0.05, 0, 1.6)), 0.03)
    tr.update(head(pivot=(1.0, 1.0, 1.6), yaw=90 * DEG), 0.06)
    out = tr.update(head(pivot=(1.0, 1.10, 1.6), yaw=90 * DEG), 0.09)   # +0.10 along new forward
    assert tr.jumps == 1 and out[0] == pytest.approx(tl.lean_from_displacement(0.15, 0, CFG)[0]) and out[1] == 0.0


# ---------------------------------------------------- command shaping
def test_command_has_no_lowpass_and_no_rate_limit_writer_is_the_single_limiter():
    """Latency: the tracker no longer low-passes or rate-limits (the arm_sdk
    writer slews at cfg.rate_dps, 250 Hz). The command only clamps to the box."""
    neutral = np.array([0.02, -0.01, 0.03])
    assert CFG.lowpass_tau_s == 0.0 and CFG.accel_dps2 == 0.0
    cmd = tl.WaistLeanCommand(CFG, neutral, neutral, 0.0)
    q = cmd.step((8 * DEG, -5 * DEG), 1 / 15.0)          # one cycle: no lag at all
    np.testing.assert_allclose(q, neutral + [0, -5 * DEG, 8 * DEG], atol=1e-12)
    q = cmd.step((1.0, -1.0), 2 / 15.0)                  # absurd target: clamped to +-max
    np.testing.assert_allclose(q, neutral + [0, -10 * DEG, 10 * DEG], atol=1e-12)
    assert q[0] == neutral[0]                             # yaw stays at neutral
    q = cmd.step((0.0, 0.0), 3 / 15.0)                    # release: straight back, writer slews
    np.testing.assert_allclose(q, neutral, atol=1e-12)


def test_box_20deg_and_joint_limit_margin():
    cfg = tl.TorsoLeanConfig(max_deg=20.0)
    cmd = tl.WaistLeanCommand(cfg, np.zeros(3), np.zeros(3), 0.0)
    np.testing.assert_allclose(cmd.upper[1:], [20 * DEG, 20 * DEG])
    q = cmd.step((1.0, 1.0), 0.1)
    np.testing.assert_allclose(q[1:], [20 * DEG, 20 * DEG])
    # neutral close to the URDF limit: the lean box stops WAIST_LIMIT_MARGIN_RAD inside it
    neutral = np.array([0.0, 0.30, -0.30])
    cmd = tl.WaistLeanCommand(cfg, neutral, neutral, 0.0)
    assert cmd.upper[1] == pytest.approx(tl.URDF_WAIST_UPPER[1] - tl.WAIST_LIMIT_MARGIN_RAD)
    assert cmd.lower[2] == pytest.approx(tl.URDF_WAIST_LOWER[2] + tl.WAIST_LIMIT_MARGIN_RAD)
    assert tl.WAIST_LIMIT_MARGIN_RAD >= 0.03
    q = cmd.step((-1.0, 1.0), 0.1)
    assert q[1] <= tl.URDF_WAIST_UPPER[1] - tl.WAIST_LIMIT_MARGIN_RAD + 1e-12
    assert q[2] >= tl.URDF_WAIST_LOWER[2] + tl.WAIST_LIMIT_MARGIN_RAD - 1e-12


def test_latency_20deg_step_through_single_writer_limiter():
    """Head step -> written waist q: 20 deg in <= 0.40 s at the 10-15 Hz loop
    measured on the robot (was ~2 s with tau 0.3 s + 15 deg/s twice)."""
    cfg = tl.TorsoLeanConfig(max_deg=20.0)
    neutral = np.zeros(3)
    cmd = tl.WaistLeanCommand(cfg, neutral, neutral, 0.0)
    rate = cfg.rate_dps * DEG
    written, t, loop_dt, w_dt, next_loop, target_q = neutral.copy(), 0.0, 1 / 12.0, 1 / 250.0, 0.0, neutral
    reached = None
    while t < 1.0:
        if t >= next_loop - 1e-12:
            target_q = cmd.step((20 * DEG, 0.0), t + 1e-9)
            next_loop += loop_dt
        written = written + np.clip(target_q - written, -rate * w_dt, rate * w_dt)
        t += w_dt
        if reached is None and written[2] >= 20 * DEG - 1e-9:
            reached = t
    assert reached is not None and reached <= 0.40


def test_accel_limit_optional():
    # opt-in profile (default off), validated as before: with the 0.3 s low-pass
    cfg = tl.TorsoLeanConfig(accel_dps2=30.0, rate_dps=15.0, lowpass_tau_s=0.3)
    cmd = tl.WaistLeanCommand(cfg, np.zeros(3), np.zeros(3), 0.0)
    t, v_prev, q_prev = 0.0, 0.0, 0.0
    dt = 1 / 30.0
    for _ in range(200):
        t += dt
        q = cmd.step((10 * DEG, 0.0), t)[2]
        v = (q - q_prev) / dt
        assert abs(v - v_prev) <= 30 * DEG * dt + 1e-9 and abs(v) <= cfg.rate_dps * DEG + 1e-9
        v_prev, q_prev = v, q
    assert q_prev == pytest.approx(10 * DEG, abs=0.2 * DEG)


def test_urdf_limits_respected_when_neutral_near_limit():
    neutral = np.array([0.0, 0.50, -0.50])
    cmd = tl.WaistLeanCommand(CFG, neutral, neutral, 0.0)
    assert np.all(cmd.upper <= tl.URDF_WAIST_UPPER) and np.all(cmd.lower >= tl.URDF_WAIST_LOWER)
    assert cmd.upper[1] == pytest.approx(0.50) and cmd.lower[2] == pytest.approx(-0.50)  # never deeper than start
    t = 0.0
    for _ in range(200):
        t += 0.033
        q = cmd.step((-1.0, 1.0), t)
    assert q[1] <= 0.52 and q[2] >= -0.52


def _lean_session(neutral=(0.0, 0.0, 0.0)):
    cfg = tl.TorsoLeanConfig(max_deg=10.0, lowpass_tau_s=0.0, rate_dps=30.0)
    tracker = tl.HeadLeanTracker(cfg)
    assert tracker.engage(head(), 0.0)
    command = tl.WaistLeanCommand(cfg, neutral, neutral, 0.0)
    return tl.TorsoLeanSession(cfg, tracker, command, np.asarray(neutral, float))


def test_ik_compensation_rotation_uses_measured_waist_never_commanded_waist():
    session = _lean_session()
    session.waist_cmd = np.array([0.0, -8 * DEG, 9 * DEG])
    measured = np.array([0.0, 1.5 * DEG, -2.0 * DEG])

    rotation = session.observe_measured_waist(measured, age=0.01, now=1.0)

    np.testing.assert_allclose(rotation, tl.waist_rotation(measured), atol=1e-12)
    assert not np.allclose(rotation, tl.waist_rotation(session.waist_cmd))


def test_persistent_roll_pitch_tracking_error_warns_degraded_but_keeps_limited_tilt():
    logs = []
    session = _lean_session()
    session.log = types.SimpleNamespace(warning=logs.append)
    session.waist_cmd = np.array([0.0, 4 * DEG, -4 * DEG])
    session.command.cmd = session.waist_cmd.copy()
    session.command.last_t = 1.0
    measured = np.zeros(3)

    assert session.observe_measured_waist(measured, age=0.01, now=1.0) is not None
    assert session.observe_measured_waist(measured, age=0.01, now=1.49) is not None
    assert session.observe_measured_waist(measured, age=0.01, now=1.51) is not None

    assert session.watchdog_tripped is False
    assert session.enabled is True
    assert session.telemetry()["status"] == "waist_tracking_degraded"
    before = session.waist_cmd.copy()
    after = session.step(head(pivot=(0.20, 0.0, 1.6)), 1.61)
    assert np.all(np.abs(after[1:] - session.command.neutral[1:]) <= 10 * DEG + 1e-12)
    assert np.any(np.abs(after[1:] - before[1:]) > 0.0)
    assert session.tracker.target != (0.0, 0.0)
    assert len(logs) == 1 and "não acompanhou" in logs[0] and "continuando" in logs[0]

    status = tl.status_telemetry(session)
    assert status["status"] == "waist_tracking_degraded"
    assert status["enabled"] is True and status["watchdog_tripped"] is False
    assert status["waist_tracking"]["duration_s"] == pytest.approx(0.51)
    assert status["waist_tracking"]["error"][1:] == pytest.approx([4 * DEG, -4 * DEG])


def test_invalid_or_stale_waist_telemetry_fails_neutral_without_raising():
    session = _lean_session()
    session.waist_cmd = np.array([0.0, 3 * DEG, 3 * DEG])

    assert session.observe_measured_waist(None, age=float("inf"), now=1.0) is None
    assert session.observe_measured_waist(np.full(3, np.nan), age=0.0, now=1.1) is None
    assert session.observe_measured_waist(np.zeros(3), age=1.0, now=1.2) is None
    assert session.enabled is False
    assert session.telemetry()["status"] == "waist_telemetry_lost"


def test_status_telemetry_reports_watchdog_and_measured_compensation_source():
    session = _lean_session()
    measured = np.array([0.0, 1 * DEG, -1 * DEG])
    session.observe_measured_waist(measured, age=0.02, now=1.0)

    status = tl.status_telemetry(session)

    assert status["configured"] is True and status["enabled"] is True
    assert status["watchdog_tripped"] is False
    assert status["compensation_source"] == "measured_waist"
    assert status["waist_measured"] == pytest.approx(measured.tolist())


# ------------------------------------------------------------ env config
def test_env_default_off_and_values():
    assert tl.config_from_env({}) is None
    assert tl.config_from_env({"G1_TORSO_LEAN": "0"}) is None
    cfg = tl.config_from_env({"G1_TORSO_LEAN": "1"})
    assert cfg.max_deg == 10.0 and cfg.deadband_m == 0.03 and cfg.rate_dps == 60.0
    assert cfg.lowpass_tau_s == 0.0
    assert tl.config_from_env({"G1_TORSO_LEAN": "1", "G1_TORSO_LEAN_MAX_DEG": "20"}).max_deg == 20.0
    assert tl.config_from_env({"G1_TORSO_LEAN": "1", "G1_TORSO_LEAN_RATE_DPS": "90"}).rate_dps == 90.0
    assert cfg.gain_deg_per_m == pytest.approx(10.0 / 0.15)
    cfg = tl.config_from_env({"G1_TORSO_LEAN": "1", "G1_TORSO_LEAN_MAX_DEG": "3", "G1_TORSO_LEAN_RATE_DPS": "5",
                              "G1_TORSO_LEAN_GAIN_DEG_PER_M": "40", "G1_TORSO_LEAN_DEADBAND_M": "0.05"})
    assert (cfg.max_deg, cfg.rate_dps, cfg.gain_deg_per_m, cfg.deadband_m) == (3.0, 5.0, 40.0, 0.05)
    assert "LIGADA, máx 3°" in cfg.describe()


@pytest.mark.parametrize("key,value", [("G1_TORSO_LEAN_MAX_DEG", "20.5"), ("G1_TORSO_LEAN_MAX_DEG", "21"),
                                       ("G1_TORSO_LEAN_MAX_DEG", "30"),
                                       ("G1_TORSO_LEAN_MAX_DEG", "0"), ("G1_TORSO_LEAN_MAX_DEG", "nan"),
                                       ("G1_TORSO_LEAN_MAX_DEG", "abc"), ("G1_TORSO_LEAN_RATE_DPS", "91"),
                                       ("G1_TORSO_LEAN_GAIN_DEG_PER_M", "-1")])
def test_env_out_of_range_rejected(key, value):
    with pytest.raises(tl.TorsoLeanConfigError):
        tl.config_from_env({"G1_TORSO_LEAN": "1", key: value})


# ------------------------------------------------------ IK geometry (FK)
def _reduced_fk():
    pytest.importorskip("pinocchio")
    from teleop.utils.arm_fk import G1_29_WristFK
    return G1_29_WristFK()


def test_ik_frame_is_torso_frame_and_retarget_keeps_operator_vector():
    """Numerical FK check of the IK decision (docs/torso_lean.md, section IK).

    1. The reduced IK model (waist locked at 0) is rigidly the TORSO frame:
       full-model FK with waist q = pelvis_T_torso(q) * pelvis_T_torso(0)^-1 * reduced FK.
    2. Operator leans rigidly by R (head->hand vector v rotated by R): the
       retargeted IK target equals the un-leaned target, so the IK solution is
       unchanged and, in the pelvis frame, the robot reproduces the operator's
       head->hand vector R v from its own (torso-fixed) head point.
    3. Without retargeting the target would be off by centimetres.
    """
    pin, model, data = _full_model()
    fk = _reduced_fk()
    fid_t = model.getFrameId("torso_link")
    rng = np.random.default_rng(7)
    arm_names = [model.names[fk.model.joints[j].id] for j in range(1, fk.model.njoints)]
    for _ in range(4):
        q_arm = np.clip(rng.normal(0, 0.3, 14), fk.model.lowerPositionLimit, fk.model.upperPositionLimit)
        lean = (0.0, rng.uniform(-10, 10) * DEG, rng.uniform(-10, 10) * DEG)
        q_full = _waist_q(model, *lean)
        for k, name in enumerate(fk.model.names[1:]):
            q_full[model.joints[model.getJointId(name)].idx_q] = q_arm[k]
        q_zero = _waist_q(model, base=q_full)
        pin.framesForwardKinematics(model, data, q_zero)
        P0 = data.oMf[fid_t].homogeneous.copy()
        pin.framesForwardKinematics(model, data, q_full)
        P = data.oMf[fid_t].homogeneous.copy()
        L_full = data.oMi[model.getJointId("left_wrist_yaw_joint")].homogeneous.copy()
        l_red, _ = fk.points_xyz(q_arm)
        L_red = np.eye(4)
        pin_d = fk.model.createData()
        pin.forwardKinematics(fk.model, pin_d, q_arm)
        L_red = pin_d.oMi[fk.model.getJointId("left_wrist_yaw_joint")].homogeneous.copy()
        np.testing.assert_allclose(L_full, P @ np.linalg.inv(P0) @ L_red, atol=1e-9)        # (1)

        # (2) operator target in the neutral IK frame (= what TeleVuer gives with no lean)
        h = tl.HEAD_POINT_IN_WAIST
        t0 = L_red.copy()
        R = tl.waist_rotation(lean)
        t1 = t0.copy()                                   # TeleVuer target after a rigid operator lean
        t1[:3, 3] = h + R @ (t0[:3, 3] - h)
        t1[:3, :3] = R @ t0[:3, :3]
        np.testing.assert_allclose(tl.retarget_to_torso(t1, R), t0, atol=1e-12)
        head_world = (P @ np.linalg.inv(P0) @ np.append(h, 1.0))[:3]   # torso-fixed head point, pelvis frame
        np.testing.assert_allclose(L_full[:3, 3] - head_world, R @ (t0[:3, 3] - h), atol=1e-9)
        np.testing.assert_allclose(L_full[:3, :3], R @ t0[:3, :3], atol=1e-9)
        # (3) the naive (non-retargeted) target would be wrong by several cm
        if max(abs(lean[1]), abs(lean[2])) > 4 * DEG:
            assert np.linalg.norm(t1[:3, 3] - t0[:3, 3]) > 0.01


def test_ik_gravity_feedforward_matches_full_model_with_lean():
    pin, model, data = _full_model()
    fk = _reduced_fk()
    mod = _load_ik_module()
    ik_like = types.SimpleNamespace(reduced_robot=types.SimpleNamespace(model=fk.model, data=fk.model.createData()))
    rng = np.random.default_rng(11)
    q_arm = np.clip(rng.normal(0, 0.3, 14), fk.model.lowerPositionLimit, fk.model.upperPositionLimit)
    lean = (0.0, 6 * DEG, -8 * DEG)
    mod.G1_29_ArmIK.set_torso_rotation(ik_like, tl.waist_rotation(lean))
    zeros = np.zeros(14)
    tau_red = pin.rnea(fk.model, ik_like.reduced_robot.data, q_arm, zeros, zeros)
    q_full = _waist_q(model, *lean)
    for k, name in enumerate(fk.model.names[1:]):
        q_full[model.joints[model.getJointId(name)].idx_q] = q_arm[k]
    tau_full = pin.rnea(model, data, q_full, np.zeros(model.nv), np.zeros(model.nv))
    idx = [model.joints[model.getJointId(n)].idx_v for n in fk.model.names[1:]]
    np.testing.assert_allclose(tau_red, tau_full[idx], atol=1e-9)
    mod.G1_29_ArmIK.set_torso_rotation(ik_like, np.eye(3))
    np.testing.assert_allclose(fk.model.gravity.linear, [0, 0, -9.81])
    with pytest.raises(ValueError):
        mod.G1_29_ArmIK.set_torso_rotation(ik_like, np.full((3, 3), np.nan))


def _load_ik_module():
    pytest.importorskip("casadi")
    for name in ("meshcat", "meshcat.geometry"):
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)
    if "logging_mp" not in sys.modules:
        import logging
        lm = types.ModuleType("logging_mp")
        lm.getLogger = logging.getLogger
        sys.modules["logging_mp"] = lm
    from teleop.robot_control import robot_arm_ik
    return robot_arm_ik


# ------------------------------------------ real G1_29 writer (fake DDS)
class _Cmd:
    __slots__ = ("mode", "q", "dq", "tau", "kp", "kd")

    def __init__(self, i):
        self.mode, self.q, self.dq, self.tau, self.kp, self.kd = 1, 0.01 * i, 0.0, 0.0, 300.0 if 12 <= i <= 14 else 80.0, 3.0


def _stub_sdk():
    saved = {}

    def put(name, module):
        saved.setdefault(name, sys.modules.get(name))
        sys.modules[name] = module

    lm = types.ModuleType("logging_mp")
    lm.getLogger = lambda name: types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None,
                                                       error=lambda *a: None, debug=lambda *a: None)
    put("logging_mp", lm)
    channel = types.ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelPublisher = channel.ChannelSubscriber = object
    channel.ChannelFactoryInitialize = lambda *a: None
    for name in ("unitree_sdk2py", "unitree_sdk2py.core", "unitree_sdk2py.idl", "unitree_sdk2py.utils"):
        put(name, types.ModuleType(name))
    put("unitree_sdk2py.core.channel", channel)
    for name in ("unitree_hg", "unitree_go"):
        msg = types.ModuleType(f"unitree_sdk2py.idl.{name}.msg.dds_")
        msg.LowCmd_ = msg.LowState_ = object
        put(f"unitree_sdk2py.idl.{name}.msg.dds_", msg)
    default = types.ModuleType("unitree_sdk2py.idl.default")
    default.unitree_hg_msg_dds__LowCmd_ = default.unitree_go_msg_dds__LowCmd_ = object
    put("unitree_sdk2py.idl.default", default)
    crc = types.ModuleType("unitree_sdk2py.utils.crc")
    crc.CRC = object
    put("unitree_sdk2py.utils.crc", crc)
    return saved


def _restore(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_module_from(path, name):
    saved = _stub_sdk()
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        _restore(saved)


def _import_robot_arm():
    return _load_module_from(ROOT / "teleop" / "robot_control" / "robot_arm.py", "robot_arm_under_test")


def _writer(module, frames, stop_after=None, motion=True):
    """Real G1_29 _ctrl_motor_state with a fake publisher recording every field of every motor."""
    ctrl = module.G1_29_ArmController.__new__(module.G1_29_ArmController)
    ctrl.motion_mode = motion
    ctrl.simulation_mode = True
    ctrl.control_dt = 0.001
    ctrl._speed_gradual_max = False
    ctrl.ctrl_lock = threading.Lock()
    ctrl.q_target = np.linspace(-0.3, 0.3, 14)
    ctrl.tauff_target = np.linspace(0.5, -0.5, 14)
    ctrl._motion_authority_weight = 1.0
    ctrl._last_publish_monotonic = None
    ctrl._last_published_weight = None
    ctrl.output_enabled = threading.Event()
    ctrl.output_enabled.set()
    ctrl.msg = types.SimpleNamespace(motor_cmd=[_Cmd(i) for i in range(35)], crc=0)
    ctrl.crc = types.SimpleNamespace(Crc=lambda msg: 0)

    def write(msg):
        frames.append(tuple((c.mode, c.q, c.dq, c.tau, c.kp, c.kd) for c in msg.motor_cmd))
        if stop_after is not None and len(frames) >= stop_after:
            ctrl.output_enabled.clear()

    ctrl.lowcmd_publisher = types.SimpleNamespace(Write=write)
    if hasattr(ctrl, "_init_arm_publication_state"):     # Dex3 line: publication receipts
        ctrl._init_arm_publication_state()
    ctrl.outputs_activated = True
    ctrl.publish_thread = threading.Thread(target=ctrl._ctrl_motor_state, daemon=True)
    return ctrl


def _baseline_source(tmp_path):
    """robot_arm.py as of the base commit (before this feature)."""
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", "ec07bcf:teleop/robot_control/robot_arm.py"],
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip("base commit ec07bcf (Dex3 line) not available in this checkout")
    path = tmp_path / "robot_arm_base.py"
    path.write_text(src)
    return path


def test_feature_off_writer_frames_identical_to_base_commit(tmp_path):
    base = _load_module_from(_baseline_source(tmp_path), "robot_arm_base")
    new = _import_robot_arm()
    frames_base, frames_new = [], []
    for module, frames in ((base, frames_base), (new, frames_new)):
        ctrl = _writer(module, frames, stop_after=20)
        ctrl.publish_thread.start()
        ctrl.publish_thread.join(timeout=2.0)
        assert not ctrl.publish_thread.is_alive()
    assert len(frames_base) == len(frames_new) == 20
    assert frames_base == frames_new                       # all 35 motors, every field
    for frame in frames_new:                               # waist untouched: constructor values
        for i in (12, 13, 14):
            assert frame[i] == (1, 0.01 * i, 0.0, 0.0, 300.0, 3.0)


def test_feature_on_writer_final_clamp_and_rate_limit_and_gains_unchanged():
    module = _import_robot_arm()
    frames = []
    ctrl = _writer(module, frames)
    neutral = np.array([0.12, 0.13, 0.14])                 # = constructor q of 12..14
    lo = neutral - [0.0, 10 * DEG, 10 * DEG]
    hi = neutral + [0.0, 10 * DEG, 10 * DEG]
    ctrl.configure_waist_command(lo, hi, 15 * DEG, neutral)
    ctrl.set_waist_target(neutral + [1.0, 1.0, -1.0])      # far outside the box: clamped
    ctrl.publish_thread.start()
    time.sleep(0.3)
    ctrl.deactivate()
    w = np.array([[f[i][1] for i in (12, 13, 14)] for f in frames])
    assert len(w) > 20
    assert np.max(np.abs(np.diff(w, axis=0))) <= 15 * DEG * 0.001 + 1e-12   # per-frame rate limit
    assert np.all(w >= lo - 1e-12) and np.all(w <= hi + 1e-12)              # final clamp
    assert np.all(w[:, 0] == neutral[0])                                     # yaw held
    np.testing.assert_allclose(w[0], neutral, atol=15 * DEG * 0.001 + 1e-12)  # continuous handover
    for f in frames:
        for i in (12, 13, 14):
            assert f[i][4] == 300.0 and f[i][5] == 3.0 and f[i][0] == 1 and f[i][2] == 0 and f[i][3] == 0
    with pytest.raises(ValueError):
        ctrl.set_waist_target([np.nan, 0, 0])
    with pytest.raises(ValueError):
        ctrl.configure_waist_command(hi, lo, 15 * DEG, neutral)
    with pytest.raises(ValueError):                         # hard max rate also at the final writer
        ctrl.configure_waist_command(lo, hi, 91 * DEG, neutral)
    ctrl.configure_waist_command(lo, hi, 90 * DEG, neutral)


# ------------------------------------------------- session / handover
class FakeWaistArm:
    arm_joint_split = (7, 7)

    def __init__(self, held=(0.0, 0.01, -0.02), meas=None, age=0.0):
        self.held = np.array(held, float)
        self.meas = self.held.copy() if meas is None else np.array(meas, float)
        self.age = age
        self.configured = None
        self.targets = []

    def get_waist_q_snapshot(self):
        return self.meas.copy(), self.age

    def get_waist_command_written(self):
        return self.held.copy()

    def configure_waist_command(self, lower, upper, rate, initial):
        self.configured = (np.array(lower), np.array(upper), rate, np.array(initial))

    def set_waist_target(self, q):
        self.targets.append(np.array(q))


def test_engage_neutral_is_held_command_and_refuses_far_or_stale_waist():
    arm = FakeWaistArm(meas=(0.0, 0.03, -0.01))
    s = tl.TorsoLeanSession.try_engage(CFG, arm, head(), 0.0)
    assert isinstance(s, tl.TorsoLeanSession)
    np.testing.assert_allclose(arm.configured[3], arm.held)          # neutral = held, not re-sampled
    np.testing.assert_allclose(arm.configured[1] - arm.held, [0, 10 * DEG, 10 * DEG])
    assert arm.configured[2] == pytest.approx(60 * DEG)       # the writer is the single rate limiter
    assert tl.TorsoLeanSession.try_engage(CFG, FakeWaistArm(meas=(0.0, 0.2, 0.0)), head(), 0.0) == tl.REFUSED
    assert tl.TorsoLeanSession.try_engage(CFG, FakeWaistArm(age=1.0), head(), 0.0) is None
    assert tl.TorsoLeanSession.try_engage(CFG, FakeWaistArm(), tl.FALLBACK_HEAD_POSE, 0.0) is None


# --------------------------------------------------------- shutdown
class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += float(s)


class SimArmWaist:
    """Ideal arms + a waist writer that slews at max_rate (like the real writer)."""

    arm_joint_split = (7, 7)
    motion_mode = True

    def __init__(self, clock, waist=True):
        self.clock = clock
        self.q = np.full(14, 0.3)
        self.log = []
        self.weight = 1.0
        self.active = True
        self.waist = waist
        self.neutral = np.array([0.0, 0.01, -0.02])
        self.written = self.neutral + [0.0, -8 * DEG, 9 * DEG]
        self.target = self.written.copy()
        self.rate = 15 * DEG
        self._t = clock()

    def _advance(self):
        dt = self.clock() - self._t
        self._t = self.clock()
        self.written = self.written + np.clip(self.target - self.written, -self.rate * dt, self.rate * dt)

    def get_dual_arm_q_snapshot(self):
        return self.q.copy(), 0.0

    def get_arm_command(self):
        return self.q.copy(), np.zeros(14)

    def ctrl_dual_arm(self, q, tau):
        self._advance()
        self.q = np.asarray(q, float).copy()
        self.log.append(("cmd", self.written.copy()))

    def set_motion_authority_weight(self, w):
        self._advance()
        self.weight = float(w)
        self.log.append(("weight", float(w), self.written.copy()))

    def get_publication_status(self):
        self._advance()
        return {"active": self.active, "last_publish_monotonic": self.clock(), "last_published_weight": self.weight}

    def deactivate(self, join_timeout=1.0):
        self.active = False
        self.log.append(("deactivate",))

    def get_waist_command(self):
        self._advance()
        if not self.waist:
            return {"enabled": False}
        return {"enabled": True, "target": self.target.copy(), "written": self.written.copy(),
                "neutral": self.neutral.copy(), "max_rate": self.rate}

    def waist_return_to_neutral(self):
        self.log.append(("waist_neutral",))
        self.target = self.neutral.copy()
        return True

    def get_waist_q_snapshot(self):
        return self.written.copy(), 0.0


def test_shutdown_returns_waist_to_neutral_before_weight_ramp():
    clock = FakeClock()
    arm = SimArmWaist(clock)
    result = run_graceful_arm_shutdown(arm, clock=clock, sleep=clock.sleep)
    assert result.waist_return_requested and result.waist_neutral_reached and result.release_confirmed
    kinds = [e[0] for e in arm.log]
    assert kinds[0] == "waist_neutral"
    first_w = kinds.index("weight")
    assert "cmd" in kinds[:first_w]                                   # arms returned before release
    for e in arm.log:
        if e[0] == "weight":
            np.testing.assert_allclose(e[2], arm.neutral, atol=1e-4)    # waist at neutral during the whole ramp
    assert kinds[-1] == "deactivate"


def test_shutdown_without_waist_is_unchanged():
    clock = FakeClock()
    arm = SimArmWaist(clock, waist=False)
    result = run_graceful_arm_shutdown(arm, clock=clock, sleep=clock.sleep)
    assert not result.waist_return_requested and "waist_neutral" not in [e[0] for e in arm.log]
    assert result.release_confirmed


# --------------------------------------------------------- pose stream
def test_pose_stream_old_and_new_packets():
    q = np.linspace(0, 1, 14)
    old = ps.pack_sample(1, 2.0, 3.0, True, True, (1, 2, 3), (4, 5, 6), q, q)
    s = ps.unpack_sample(old)
    assert s["lean_target"] is None and s["lean_active"] is False and len(old) == ps.PACKET_SIZE
    new = ps.pack_sample_lean(2, 2.0, 3.0, True, False, (1, 2, 3), (4, 5, 6), q, q,
                              True, (0.1, -0.05), (0.08, -0.04), (0.0, -0.04, 0.08), (0.0, -0.03, 0.07))
    s = ps.unpack_sample(new)
    assert s["lean_active"] and s["seq"] == 2 and s["q_cmd"] == pytest.approx(list(q))
    assert s["lean_target"] == pytest.approx([0.1, -0.05]) and s["waist_meas"] == pytest.approx([0.0, -0.03, 0.07])
    assert ps.unpack_sample(new[:-1]) is None and ps.unpack_sample(b"XPS2" + old[4:]) is None


def test_pose_stream_sender_feature_off_bytes_unchanged():
    import socket
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(1.0)
    tx = ps.PoseStreamSender(port=rx.getsockname()[1], rate_hz=0)
    eye = np.eye(4)
    assert tx.maybe_send(True, eye, eye, np.zeros(14), np.zeros(14), now=1.0)
    data, _ = rx.recvfrom(4096)
    assert len(data) == ps.PACKET_SIZE and data[:4] == ps.MAGIC
    assert tx.maybe_send(True, eye, eye, np.zeros(14), np.zeros(14), now=2.0,
                         lean={"active": True, "target": (0.1, 0), "cmd": (0.1, 0), "waist_cmd": (0, 0, 0.1),
                               "waist_meas": None})
    data, _ = rx.recvfrom(4096)
    s = ps.unpack_sample(data)
    assert len(data) == ps.PACKET_SIZE2 and math.isnan(s["waist_meas"][0])
    rx.close()
    tx.close()


def test_pose_web_hub_accepts_old_and_new_packets_and_reports_lean():
    sys.path.insert(0, str(ROOT / "tools"))
    import pose_compare_web as web
    t = [0.0]
    hub = web.PoseHub(fk=None, clock=lambda: t[0])
    q = np.zeros(14)
    hub.ingest(ps.pack_sample(0, 1.0, 1.7e9, True, True, (0, 0, 0), (0, 0, 0), q, q))
    assert hub.status()["lean"] is None
    hub.ingest(ps.pack_sample_lean(1, 1.02, 1.7e9, True, True, (0, 0, 0), (0, 0, 0), q, q,
                                   True, (0.1, -0.05), (0.08, -0.04), (0.0, -0.04, 0.08), (0.0, -0.03, 0.07)))
    lean = hub.status()["lean"]
    assert lean["active"] and lean["target_deg"][0] == pytest.approx(5.73, abs=0.01)
    assert lean["meas_deg"] == pytest.approx([math.degrees(0.07), math.degrees(-0.03)], abs=0.01)
    assert hub.n_rx == 2 and hub.n_bad == 0

# The real teleop_hand_and_arm.py main-loop tests for the Dex3 line live in
# tests/test_dex3_torso_lean.py (Dex3 architecture: deferred activate(),
# controller calibration, Dex3 graceful shutdown).
