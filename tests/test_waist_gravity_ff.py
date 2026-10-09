"""Waist gravity feed-forward (G1_TORSO_LEAN_WAIST_FF) for the Dex3 torso lean.

Fakes only (no DDS, no actuators). The model tests use pinocchio on the same
URDF as the teleop IK (skipped where pinocchio is not installed); the shaper,
config and writer tests are pure numpy.
"""
import importlib.util
import math
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

from teleop.utils import waist_gravity_ff as wff  # noqa: E402
from teleop.utils.arm_graceful_shutdown import run_graceful_arm_shutdown  # noqa: E402

DEG = math.pi / 180.0
URDF = ROOT / "assets" / "g1" / "g1_body29_hand14.urdf"
UPRIGHT = (1.0, 0.0, 0.0, 0.0)          # pelvis IMU quaternion wxyz, upright


# ------------------------------------------------------------------ config
def test_kill_switch_default_on_and_values():
    assert wff.ff_enabled_from_env({}) is True
    assert wff.ff_enabled_from_env({"G1_TORSO_LEAN_WAIST_FF": ""}) is True
    assert wff.ff_enabled_from_env({"G1_TORSO_LEAN_WAIST_FF": "1"}) is True
    assert wff.ff_enabled_from_env({"G1_TORSO_LEAN_WAIST_FF": "0"}) is False
    for bad in ("2", "yes", "off", "-1", "nan"):
        with pytest.raises(wff.WaistFFConfigError):
            wff.ff_enabled_from_env({"G1_TORSO_LEAN_WAIST_FF": bad})


def test_cap_is_below_absolute_ceiling_well_below_urdf_effort():
    assert wff.URDF_WAIST_EFFORT_NM == 50.0                      # waist_roll/pitch effort in the URDF
    assert wff.HARD_CEILING_NM <= 0.6 * wff.URDF_WAIST_EFFORT_NM
    assert 0.0 < wff.TAU_CAP_NM <= wff.HARD_CEILING_NM
    assert wff.RAMP_S == pytest.approx(0.5)


# ------------------------------------------------------------------ shaper
def _run(shaper, raw, seconds, dt=0.004):
    out = []
    for _ in range(int(round(seconds / dt))):
        out.append(shaper.step(raw, dt).copy())
    return np.array(out)


def test_ramp_in_takes_half_a_second_and_is_monotonic():
    s = wff.WaistFFShaper()
    raw = np.array([4.0, -10.0])
    out = _run(s, raw, 0.6)
    assert np.all(np.diff(np.abs(out[:, 1])) >= -1e-12)
    np.testing.assert_allclose(out[0], raw * 0.004 / 0.5, rtol=1e-9)
    assert abs(out[61, 1]) == pytest.approx(10.0 * 62 * 0.004 / 0.5, rel=1e-9)   # linear: ~half way at 0.25 s
    assert np.all(np.abs(out[: int(0.5 / 0.004) - 1, 1]) < 10.0)
    np.testing.assert_allclose(out[-1], raw)


def test_cap_per_axis_never_exceeded_and_sign_kept():
    s = wff.WaistFFShaper()
    out = _run(s, np.array([-500.0, 500.0]), 1.0)
    assert np.max(np.abs(out)) <= wff.TAU_CAP_NM + 1e-12
    np.testing.assert_allclose(out[-1], [-wff.TAU_CAP_NM, wff.TAU_CAP_NM])
    with pytest.raises(ValueError):
        wff.WaistFFShaper(cap_nm=wff.HARD_CEILING_NM + 1.0)


@pytest.mark.parametrize("bad", [None, np.array([np.nan, 1.0]), np.array([1.0, np.inf]), np.array([1.0]), "x"])
def test_invalid_raw_gives_zero_immediately_and_ramps_in_again(bad):
    s = wff.WaistFFShaper()
    _run(s, np.array([3.0, -8.0]), 1.0)
    out = s.step(bad, 0.004, reason="state_stale")
    np.testing.assert_array_equal(out, [0.0, 0.0])
    assert s.gain == 0.0 and s.reason == ("state_stale" if bad is None else "invalid_model_output")
    again = s.step(np.array([3.0, -8.0]), 0.004)
    np.testing.assert_allclose(again, np.array([3.0, -8.0]) * 0.004 / 0.5)     # ramp-in again, no step


def test_ramp_out_reaches_zero_in_half_a_second():
    s = wff.WaistFFShaper()
    raw = np.array([2.0, -12.0])
    _run(s, raw, 1.0)
    s.ramp_out()
    out = _run(s, raw, 0.6)
    assert np.all(np.diff(np.abs(out[:, 1])) <= 1e-12)
    assert np.all(np.abs(out[: int(0.5 / 0.004) - 2, 1]) > 0.0)        # a ramp, not a step
    np.testing.assert_array_equal(out[-1], [0.0, 0.0])
    assert s.gain == 0.0 and s.finished


def test_bad_dt_never_advances_ramp():
    s = wff.WaistFFShaper()
    for dt in (float("nan"), -1.0, float("inf")):
        np.testing.assert_array_equal(s.step(np.array([1.0, 1.0]), dt), [0.0, 0.0])
    s.step(np.array([1.0, 1.0]), 10.0)                                 # long gap: one bounded step only
    assert s.gain <= wff.MAX_DT_S / wff.RAMP_S + 1e-12


# ------------------------------------------------------------------ model
def _pin():
    return pytest.importorskip("pinocchio")


def _independent_tau(pin, roll, pitch, arm_q=None, R_pelvis=np.eye(3)):
    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()
    q = np.zeros(model.nq)
    q[model.joints[model.getJointId("waist_roll_joint")].idx_q] = roll
    q[model.joints[model.getJointId("waist_pitch_joint")].idx_q] = pitch
    if arm_q is not None:
        for name, value in zip(wff.ARM_JOINT_NAMES, arm_q):
            q[model.joints[model.getJointId(name)].idx_q] = value
    model.gravity.linear = R_pelvis.T @ np.array([0.0, 0.0, -9.81])
    tau = pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv))
    return np.array([tau[model.joints[model.getJointId(n)].idx_v] for n in ("waist_roll_joint", "waist_pitch_joint")])


def _quat_from_R(R):
    pin = _pin()
    c = pin.Quaternion(R).coeffs()            # x, y, z, w
    return (c[3], c[0], c[1], c[2])


def test_model_matches_independent_rnea_at_0_10_20_deg_and_roll_10():
    pin = _pin()
    model = wff.WaistGravityModel()
    arms = np.zeros(14)
    for pitch_deg in (-20, -10, 0, 10, 20):
        got = model.waist_tau([0.0, 0.0, pitch_deg * DEG], arms, UPRIGHT)
        np.testing.assert_allclose(got, _independent_tau(pin, 0.0, pitch_deg * DEG), atol=1e-9)
    for roll_deg in (-10, 10):
        got = model.waist_tau([0.0, roll_deg * DEG, 0.0], arms, UPRIGHT)
        np.testing.assert_allclose(got, _independent_tau(pin, roll_deg * DEG, 0.0), atol=1e-9)
    rng = np.random.default_rng(3)
    arm_q = rng.uniform(-0.6, 0.6, 14)
    got = model.waist_tau([0.0, 7 * DEG, -12 * DEG], arm_q, UPRIGHT)
    np.testing.assert_allclose(got, _independent_tau(pin, 7 * DEG, -12 * DEG, arm_q), atol=1e-9)


def test_model_numbers_pinned_for_zero_arm_pose():
    _pin()
    model = wff.WaistGravityModel()
    pitch = [model.waist_tau([0, 0, d * DEG], np.zeros(14), UPRIGHT)[1] for d in (-20, -10, 0, 10, 20)]
    np.testing.assert_allclose(pitch, [0.258, -3.957, -8.051, -11.900, -15.388], atol=0.01)


def test_sign_forward_lean_torque_opposes_forward_fall_and_roll_opposes_tip():
    _pin()
    model = wff.WaistGravityModel()
    arms = np.zeros(14)
    t0 = model.waist_tau([0, 0, 0.0], arms, UPRIGHT)
    t10 = model.waist_tau([0, 0, 10 * DEG], arms, UPRIGHT)
    t20 = model.waist_tau([0, 0, 20 * DEG], arms, UPRIGHT)
    # pitch+ = forward. Gravity pulls a forward-leaning torso further forward,
    # so the holding torque is NEGATIVE (backwards) and grows with the lean.
    assert t20[1] < t10[1] < t0[1] < 0.0
    # roll+ = tips right: holding torque must be negative (to the left); roll- positive.
    assert model.waist_tau([0, 10 * DEG, 0], arms, UPRIGHT)[0] < -3.0
    assert model.waist_tau([0, -10 * DEG, 0], arms, UPRIGHT)[0] > 3.0
    # Physically: the PD sag (measured - cmd) = -tau/kp; with this torque
    # added the static error of a kp=300 servo vanishes in the model.


def test_measured_arms_change_the_torque():
    _pin()
    model = wff.WaistGravityModel()
    down = model.waist_tau([0, 0, 10 * DEG], np.zeros(14), UPRIGHT)
    arms_fwd = np.zeros(14)
    arms_fwd[0] = arms_fwd[7] = -1.57          # both shoulders pitched forward (arms straight ahead)
    fwd = model.waist_tau([0, 0, 10 * DEG], arms_fwd, UPRIGHT)
    assert fwd[1] < down[1] - 3.0


def test_pelvis_imu_orientation_is_used():
    pin = _pin()
    model = wff.WaistGravityModel()
    c, s = math.cos(10 * DEG), math.sin(10 * DEG)
    R_fwd = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])      # pelvis pitched 10 deg forward
    got = model.waist_tau([0, 0, 0.0], np.zeros(14), _quat_from_R(R_fwd))
    np.testing.assert_allclose(got, _independent_tau(pin, 0.0, 0.0, None, R_fwd), atol=1e-9)
    # same torso orientation in the world as waist pitch 10 deg on an upright pelvis
    np.testing.assert_allclose(got[1], model.waist_tau([0, 0, 10 * DEG], np.zeros(14), UPRIGHT)[1], atol=1e-6)


@pytest.mark.parametrize("quat", [None, (0, 0, 0, 0), (np.nan, 0, 0, 1), (1, 0, 0), (0.7071068, 0.7071068, 0, 0)])
def test_invalid_or_fallen_imu_raises(quat):
    _pin()
    model = wff.WaistGravityModel()
    with pytest.raises(ValueError):
        model.waist_tau([0, 0, 0], np.zeros(14), quat)


def test_invalid_joints_raise():
    _pin()
    model = wff.WaistGravityModel()
    with pytest.raises(ValueError):
        model.waist_tau([0, 0, np.nan], np.zeros(14), UPRIGHT)
    with pytest.raises(ValueError):
        model.waist_tau([0, 0, 0], np.zeros(13), UPRIGHT)


def test_cap_justified_by_model_at_20_deg_any_arm_pose():
    pin = _pin()
    model = wff.WaistGravityModel()
    full = pin.buildModelFromUrdf(str(URDF))
    lo = np.array([full.lowerPositionLimit[full.joints[full.getJointId(n)].idx_q] for n in wff.ARM_JOINT_NAMES])
    hi = np.array([full.upperPositionLimit[full.joints[full.getJointId(n)].idx_q] for n in wff.ARM_JOINT_NAMES])
    rng = np.random.default_rng(7)
    worst = np.zeros(2)
    for _ in range(3000):
        lean = rng.uniform(-20, 20, 2) * DEG
        worst = np.maximum(worst, np.abs(model.waist_tau([0, lean[0], lean[1]], rng.uniform(lo, hi), UPRIGHT)))
    assert np.all(worst <= wff.TAU_CAP_NM)
    assert np.max(worst) >= 0.75 * wff.TAU_CAP_NM          # the cap is not arbitrarily loose


# --------------------------------------------- real G1_29 writer (fake DDS)
def _load_robot_arm(path, name):
    tests_dir = str(ROOT / "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import test_torso_lean as ttl
    return ttl._load_module_from(path, name), ttl


class FakeModel:
    """Duck-typed WaistGravityModel: torque = f(measured pitch/roll, arms, imu)."""

    def __init__(self):
        self.calls = []

    def waist_tau(self, waist_q, arm_q, quat):
        waist_q = np.asarray(waist_q, float)
        arm_q = np.asarray(arm_q, float)
        if quat is None or not np.all(np.isfinite(quat)):
            raise ValueError("imu")
        self.calls.append((waist_q.copy(), arm_q.copy(), tuple(quat)))
        return np.array([-20.0 * waist_q[1], -8.0 - 20.0 * waist_q[2] - 0.1 * arm_q[0]])


def _lowstate(module, waist=(0.12, 0.13, 0.14), quat=UPRIGHT, motorstate=0):
    ls = module.G1_29_LowState()
    for i in range(module.G1_29_Num_Motors):
        ls.motor_state[i].q = 0.01 * i
        ls.motor_state[i].dq = 0.0
    for k, i in enumerate((12, 13, 14)):
        ls.motor_state[i].q = waist[k]
    for i in (12, 13, 14):
        ls.motor_state[i].motorstate = motorstate
    ls.imu_quaternion = quat
    return ls


def _ff_writer(tmp_path=None, *, configure_ff=True, quat=UPRIGHT, motorstate=0):
    module, ttl = _load_robot_arm(ROOT / "teleop" / "robot_control" / "robot_arm.py", "robot_arm_ff_under_test")
    frames = []
    ctrl = ttl._writer(module, frames)
    ctrl.control_dt = 0.004
    ctrl.lowstate_buffer = module.DataBuffer()
    ctrl.lowstate_buffer.SetData(_lowstate(module, quat=quat, motorstate=motorstate))
    neutral = np.array([0.12, 0.13, 0.14])
    lo, hi = neutral - [0, 10 * DEG, 10 * DEG], neutral + [0, 10 * DEG, 10 * DEG]
    ctrl.configure_waist_command(lo, hi, 60 * DEG, neutral)
    model = FakeModel()
    if configure_ff:
        ctrl.configure_waist_gravity_ff(model)
    return module, ctrl, frames, model


def _tau(frames):
    return np.array([[f[i][3] for i in (12, 13, 14)] for f in frames])


def _keep_fresh(module, ctrl, stop, **kw):
    while not stop.is_set():
        ctrl.lowstate_buffer.SetData(_lowstate(module, **kw))
        time.sleep(0.002)


def test_writer_applies_model_tau_on_roll_pitch_only_with_ramp_and_unchanged_gains():
    module, ctrl, frames, model = _ff_writer()
    stop = threading.Event()
    feeder = threading.Thread(target=_keep_fresh, args=(module, ctrl, stop), daemon=True)
    feeder.start()
    ctrl.publish_thread.start()
    time.sleep(0.9)
    ctrl.deactivate()
    stop.set()
    ctrl.publish_thread.join(timeout=2.0)
    tau = _tau(frames)
    assert len(tau) > 50
    assert np.all(tau[:, 0] == 0.0)                                   # yaw tau stays 0
    expected = np.array([-20.0 * 0.13, -8.0 - 20.0 * 0.14 - 0.1 * 0.15])   # arm_q[0] = motor 15 q
    np.testing.assert_allclose(tau[-1, 1:], expected, rtol=1e-9)
    assert abs(tau[0, 2]) < 0.2 * abs(expected[1])                     # ramped in, not stepped
    assert np.all(np.diff(np.abs(tau[:, 2])) >= -1e-9)
    for f in frames:
        for i in (12, 13, 14):
            assert f[i][4] == 300.0 and f[i][5] == 3.0 and f[i][0] == 1   # kp/kd/mode unchanged
    status = ctrl.get_waist_gravity_ff()
    assert status["configured"] and status["gain"] == pytest.approx(1.0)
    np.testing.assert_allclose(status["tau_nm"], [0.0, *expected], rtol=1e-9)


def test_writer_stale_lowstate_gives_zero_tau_immediately():
    module, ctrl, frames, _ = _ff_writer()
    stop = threading.Event()
    feeder = threading.Thread(target=_keep_fresh, args=(module, ctrl, stop), daemon=True)
    feeder.start()
    ctrl.publish_thread.start()
    time.sleep(0.7)
    stop.set()                                     # telemetry stops: sample ages past the limit
    feeder.join()
    n_fresh = len(frames)
    time.sleep(wff.STATE_MAX_AGE_S + 0.1)
    ctrl.deactivate()
    ctrl.publish_thread.join(timeout=2.0)
    tau = _tau(frames)
    assert np.max(np.abs(tau[n_fresh - 5:n_fresh, 1:])) > 5.0
    # within STATE_MAX_AGE_S of the last sample + one frame, tau is exactly 0
    first_zero = next(i for i in range(n_fresh, len(tau)) if np.all(tau[i] == 0.0))
    assert np.all(tau[first_zero:] == 0.0)
    assert ctrl.get_waist_gravity_ff()["reason"] == "state_stale"


@pytest.mark.parametrize("kw,reason", [({"quat": None}, "model_rejected"),
                                       ({"motorstate": 1}, "motor_fault")])
def test_writer_invalid_imu_or_motor_fault_gives_zero_tau(kw, reason):
    module, ctrl, frames, _ = _ff_writer(**kw)
    stop = threading.Event()
    threading.Thread(target=_keep_fresh, args=(module, ctrl, stop), kwargs=kw, daemon=True).start()
    ctrl.publish_thread.start()
    time.sleep(0.2)
    ctrl.deactivate()
    stop.set()
    ctrl.publish_thread.join(timeout=2.0)
    assert np.all(_tau(frames) == 0.0)
    assert ctrl.get_waist_gravity_ff()["reason"] == reason


def test_writer_ramp_out_reaches_zero_before_weight_release():
    module, ctrl, frames, _ = _ff_writer()
    stop = threading.Event()
    threading.Thread(target=_keep_fresh, args=(module, ctrl, stop), daemon=True).start()
    ctrl.publish_thread.start()
    time.sleep(0.8)
    assert ctrl.waist_gravity_ff_ramp_out() is True
    time.sleep(0.7)
    assert ctrl.get_waist_gravity_ff()["finished"] is True
    n = len(frames)
    time.sleep(0.05)
    ctrl.deactivate()
    stop.set()
    ctrl.publish_thread.join(timeout=2.0)
    tau = _tau(frames)
    assert np.all(tau[n - 1:] == 0.0)
    peak = int(np.argmax(np.abs(tau[:, 2])))
    assert np.abs(tau[peak, 2]) > 5.0
    # between the peak and zero there are intermediate values (a ramp, not a step)
    assert np.sum((np.abs(tau[peak:, 2]) > 0.0) & (np.abs(tau[peak:, 2]) < 0.9 * np.abs(tau[peak, 2]))) > 20


def test_configure_ff_requires_configured_waist_and_valid_model():
    module, ttl = _load_robot_arm(ROOT / "teleop" / "robot_control" / "robot_arm.py", "robot_arm_ff_under_test2")
    ctrl = ttl._writer(module, [])
    with pytest.raises(ValueError):
        ctrl.configure_waist_gravity_ff(FakeModel())            # waist not configured
    neutral = np.array([0.12, 0.13, 0.14])
    ctrl.configure_waist_command(neutral - 0.1, neutral + 0.1, 60 * DEG, neutral)
    with pytest.raises(ValueError):
        ctrl.configure_waist_gravity_ff(object())               # no waist_tau
    assert ctrl.get_waist_gravity_ff() == {"configured": False}
    assert ctrl.waist_gravity_ff_ramp_out() is False


def test_lean_on_without_ff_writer_frames_identical_to_base_aeccc09(tmp_path):
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", "aeccc09:teleop/robot_control/robot_arm.py"],
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip("base commit aeccc09 not available in this checkout")
    path = tmp_path / "robot_arm_aeccc09.py"
    path.write_text(src)
    base, ttl = _load_robot_arm(path, "robot_arm_aeccc09")
    new, _ = _load_robot_arm(ROOT / "teleop" / "robot_control" / "robot_arm.py", "robot_arm_ff_new")
    out = []
    for module in (base, new):
        frames = []
        ctrl = ttl._writer(module, frames, stop_after=40)
        neutral = np.array([0.12, 0.13, 0.14])
        ctrl.configure_waist_command(neutral - [0, 0.1, 0.1], neutral + [0, 0.1, 0.1], 60 * DEG, neutral)
        ctrl.set_waist_target(neutral + [0, 0.05, -0.05])
        ctrl.publish_thread.start()
        ctrl.publish_thread.join(timeout=2.0)
        out.append(frames)
    assert len(out[0]) == len(out[1]) == 40
    assert out[0] == out[1]                     # every field of all 35 motors, waist tau = 0


# ------------------------------------------------------------- shutdown
class SimArmFF:
    arm_joint_split = (7, 7)
    motion_mode = True

    def __init__(self, clock, ff=True):
        self.clock = clock
        self.q = np.full(14, 0.3)
        self.log = []
        self.weight = 1.0
        self.active = True
        self.ff = ff
        self.gain = 1.0
        self.ramping_out = False
        self._t = clock()

    def _advance(self):
        dt = self.clock() - self._t
        self._t = self.clock()
        if self.ramping_out:
            self.gain = max(0.0, self.gain - dt / wff.RAMP_S)

    def get_dual_arm_q_snapshot(self):
        return self.q.copy(), 0.0

    def get_arm_command(self):
        return self.q.copy(), np.zeros(14)

    def ctrl_dual_arm(self, q, tau):
        self._advance()
        self.q = np.asarray(q, float).copy()
        self.log.append(("cmd", self.gain))

    def set_motion_authority_weight(self, w):
        self._advance()
        self.weight = float(w)
        self.log.append(("weight", float(w), self.gain))

    def get_publication_status(self):
        self._advance()
        return {"active": self.active, "last_publish_monotonic": self.clock(), "last_published_weight": self.weight}

    def deactivate(self, join_timeout=1.0):
        self.active = False
        self.log.append(("deactivate",))

    def get_waist_gravity_ff(self):
        self._advance()
        if not self.ff:
            return {"configured": False}
        return {"configured": True, "gain": self.gain, "finished": self.ramping_out and self.gain == 0.0}

    def waist_gravity_ff_ramp_out(self):
        if not self.ff:
            return False
        self.log.append(("ff_ramp_out",))
        self.ramping_out = True
        return True


def test_shutdown_ramps_ff_to_zero_before_weight_release():
    class Clock:
        now = 100.0

        def __call__(self):
            return self.now

        def sleep(self, s):
            self.now += float(s)

    clock = Clock()
    arm = SimArmFF(clock)
    result = run_graceful_arm_shutdown(arm, clock=clock, sleep=clock.sleep)
    kinds = [e[0] for e in arm.log]
    assert "ff_ramp_out" in kinds and kinds.index("ff_ramp_out") < kinds.index("weight")
    assert result.waist_ff_zeroed is True
    for e in arm.log:
        if e[0] == "weight":
            assert e[2] == 0.0                     # ff fully out during the whole weight ramp
    assert result.release_confirmed and kinds[-1] == "deactivate"
    assert any(ev["event"] == "shutdown_waist_ff_ramp_out_finished" for ev in result.events)


def test_shutdown_without_ff_is_unchanged():
    class Clock:
        now = 100.0

        def __call__(self):
            return self.now

        def sleep(self, s):
            self.now += float(s)

    clock = Clock()
    arm = SimArmFF(clock, ff=False)
    result = run_graceful_arm_shutdown(arm, clock=clock, sleep=clock.sleep)
    assert "ff_ramp_out" not in [e[0] for e in arm.log]
    assert result.waist_ff_zeroed is False and result.release_confirmed
    assert not any("waist_ff" in ev["event"] for ev in result.events)


# ------------------------------------------------------------- telemetry
def test_status_block_adds_per_axis_ff_and_never_raises():
    base = {"configured": True, "enabled": True, "status": "ok"}
    arm = types.SimpleNamespace(get_waist_gravity_ff=lambda: {
        "configured": True, "gain": 0.5, "tau_nm": [0.0, -1.0, -6.0], "raw_nm": [0.0, -2.0, -12.0],
        "reason": "ok", "finished": False})
    out = wff.status_block(base, arm, enabled=True)
    assert out["waist_ff"]["tau_nm"] == {"yaw": 0.0, "roll": -1.0, "pitch": -6.0}
    assert out["waist_ff"]["model_nm"] == {"roll": -2.0, "pitch": -12.0}
    assert base == {"configured": True, "enabled": True, "status": "ok"}       # not mutated
    off = {"configured": False, "enabled": False, "status": "off"}
    assert wff.status_block(off, arm, enabled=True) == off                    # torso lean off: unchanged
    killed = wff.status_block(base, arm, enabled=False)
    assert killed["waist_ff"] == {"enabled": False, "reason": "kill_switch"}
    broken = types.SimpleNamespace(get_waist_gravity_ff=lambda: 1 / 0)
    assert wff.status_block(base, broken, enabled=True)["waist_ff"]["reason"] == "telemetry_error"
