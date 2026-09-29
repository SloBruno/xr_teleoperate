import math
import time
import types

import numpy as np
import pytest

from teleop.utils import human_arm_calibration as hac
from teleop.utils.controller_wrist_calibration import ControllerWristCalibrator


W0 = {
    "left": np.array([0.2498, 0.1487, 0.0952]),
    "right": np.array([0.2498, -0.1486, 0.0952]),
}


def pose(position, rotation=None):
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


class Operator:
    """Synthetic operator in the stable XR (robot-basis) world frame."""

    def __init__(self, yaw_deg=20.0, length=0.62, origin=(0.35, -0.10, 1.40), half_width=0.19):
        self.yaw = math.radians(yaw_deg)
        self.R = hac.yaw_rotation(self.yaw)
        self.length = length
        origin = np.asarray(origin, dtype=float)
        self.shoulder = {
            "left": origin + self.R @ np.array([0.0, half_width, 0.0]),
            "right": origin + self.R @ np.array([0.0, -half_width, 0.0]),
        }

    def straight(self, side, body_direction):
        body_direction = np.asarray(body_direction, dtype=float)
        body_direction = body_direction / np.linalg.norm(body_direction)
        return self.shoulder[side] + self.length * (self.R @ body_direction)

    def sweep(self, side, rng, noise=0.005, start_deg=-90.0, end_deg=0.0, count=90, wobble_deg=4.0):
        elevation = np.radians(np.linspace(start_deg, end_deg, count))
        wobble = np.radians(rng.normal(0.0, wobble_deg, count))
        body = np.stack([np.cos(elevation) * np.cos(wobble), np.cos(elevation) * np.sin(wobble), np.sin(elevation)], 1)
        return self.shoulder[side] + self.length * (body @ self.R.T) + rng.normal(0.0, noise, (count, 3))

    def l_pose_controller(self, calibration, side):
        """Controller position that the calibration maps exactly onto W0."""
        fit = calibration.fits[side]
        return fit.shoulder + calibration.rotation.T @ (W0[side] - hac.G1_29_SHOULDER_ORIGINS_M[side]) / fit.scale


def calibrated(operator, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    return hac.calibrate_human_arms(operator.sweep("left", rng, **kwargs), operator.sweep("right", rng, **kwargs))


# ---------------------------------------------------------------- sphere fit

@pytest.mark.parametrize("seed", range(8))
def test_arc_fit_recovers_shoulder_and_length_with_noise(seed):
    operator = Operator(yaw_deg=25.0, length=0.64)
    calibration = calibrated(operator, seed=seed, noise=0.005, wobble_deg=4.0)
    assert calibration.accepted, calibration.reason
    for side in hac.SIDES:
        fit = calibration.fits[side]
        assert abs(fit.length_m / operator.length - 1.0) < 0.03
        assert np.linalg.norm(fit.shoulder - operator.shoulder[side]) < 0.03
        assert fit.rms_m < 0.015
        assert fit.scale == pytest.approx(hac.G1_29_STRAIGHT_ARM_REACH_M / fit.length_m)
    assert abs(math.degrees(calibration.yaw_rad) - 25.0) < 5.0


def test_lateral_wobble_keeps_shoulder_error_within_study_bounds():
    operator = Operator(yaw_deg=25.0, length=0.64)
    errors = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        fit = hac.fit_arm_arc(operator.sweep("left", rng, noise=0.005, wobble_deg=4.0), "left")
        errors.append(np.linalg.norm(fit.shoulder - operator.shoulder["left"]))
    assert np.percentile(errors, 50) < 0.015
    assert np.percentile(errors, 95) < 0.03


def test_one_centimetre_noise_is_still_accurate():
    operator = Operator(yaw_deg=0.0, length=0.60)
    errors = []
    for seed in range(20):
        calibration = calibrated(operator, seed=seed, noise=0.01)
        assert calibration.accepted, calibration.reason
        errors.append(abs(calibration.fits["left"].length_m / operator.length - 1.0))
    assert np.median(errors) < 0.03


def test_high_noise_is_rejected_by_rms_gate():
    operator = Operator()
    calibration = calibrated(operator, noise=0.03)
    assert not calibration.accepted
    assert calibration.reason.endswith("fit_rms_too_high")


def test_partial_arc_below_sixty_degrees_is_rejected():
    rng = np.random.default_rng(1)
    fit = hac.fit_arm_arc(Operator().sweep("left", rng, start_deg=-90.0, end_deg=-45.0), "left")
    assert not fit.accepted
    assert fit.reason == "arc_too_short"


def test_arc_that_does_not_reach_forward_is_rejected():
    rng = np.random.default_rng(1)
    fit = hac.fit_arm_arc(Operator().sweep("left", rng, start_deg=-90.0, end_deg=-40.0 + 0.0, count=90), "left")
    assert not fit.accepted


def test_arc_without_the_hanging_pose_is_rejected():
    rng = np.random.default_rng(2)
    fit = hac.fit_arm_arc(Operator().sweep("left", rng, start_deg=-40.0, end_deg=40.0), "left")
    assert not fit.accepted
    assert fit.reason == "arc_missing_down_pose"


def test_extended_arc_above_horizontal_is_accepted():
    rng = np.random.default_rng(3)
    fit = hac.fit_arm_arc(Operator().sweep("left", rng, start_deg=-90.0, end_deg=30.0), "left")
    assert fit.accepted


@pytest.mark.parametrize("points,reason", [
    (np.zeros((60, 3)), "degenerate_geometry"),
    (np.c_[np.linspace(0, 1, 60), np.zeros(60), np.zeros(60)], "degenerate_geometry"),
    (np.ones((5, 3)), "too_few_samples"),
    (np.full((60, 3), np.nan), "non_finite_samples"),
    (np.zeros((60, 2)), "invalid_samples"),
    ("garbage", "invalid_samples"),
])
def test_degenerate_inputs_fail_closed(points, reason):
    fit = hac.fit_arm_arc(points)
    assert not fit.accepted
    assert fit.reason == reason


def test_implausible_radius_is_rejected():
    rng = np.random.default_rng(4)
    fit = hac.fit_arm_arc(Operator(length=0.30).sweep("left", rng, noise=0.002), "left")
    assert not fit.accepted
    assert fit.reason == "implausible_arm_length"


def test_scale_is_clamped_to_reviewed_range():
    rng = np.random.default_rng(5)
    fit = hac.fit_arm_arc(Operator(length=0.84).sweep("left", rng, noise=0.002), "left")
    assert fit.accepted
    assert hac.MIN_TRANSLATION_SCALE <= fit.scale <= hac.MAX_TRANSLATION_SCALE


def test_sweep_that_is_not_vertical_is_rejected():
    # Arm swung in a plane tilted 45 deg away from vertical (sideways-and-up).
    operator = Operator(yaw_deg=0.0)
    rng = np.random.default_rng(9)
    elevation = np.radians(np.linspace(-90.0, 0.0, 90))
    tilt = math.radians(45.0)
    direction = np.stack([np.cos(elevation), math.sin(tilt) * np.sin(elevation), math.cos(tilt) * np.sin(elevation)], 1)
    points = operator.shoulder["left"] + 0.62 * direction + rng.normal(0, 0.003, (90, 3))
    fit = hac.fit_arm_arc(points, "left")
    assert not fit.accepted


def test_one_failed_side_rejects_the_whole_calibration():
    operator = Operator()
    rng = np.random.default_rng(6)
    calibration = hac.calibrate_human_arms(operator.sweep("left", rng), operator.sweep("right", rng, noise=0.04))
    assert not calibration.accepted
    assert calibration.reason.startswith("right_")
    telemetry = calibration.telemetry()
    assert telemetry["accepted"] is False
    assert telemetry["sides"]["left"]["accepted"] is True


# ---------------------------------------------------------- directional map

@pytest.mark.parametrize("direction", [(1, 0, 0), (0, 0, -1), (1, 0.3, 0.2), (0.6, 0, 0.8), (0.2, 1.0, 0)])
@pytest.mark.parametrize("yaw_deg", [0.0, 35.0, -60.0])
def test_straight_human_arm_maps_to_straight_robot_arm_in_same_direction(direction, yaw_deg):
    operator = Operator(yaw_deg=yaw_deg, length=0.62)
    calibration = calibrated(operator, noise=0.0, wobble_deg=0.0)
    assert calibration.accepted, calibration.reason
    unit = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    for side in hac.SIDES:
        mapped = hac.map_controller_position(calibration, side, operator.straight(side, unit))
        offset = mapped - hac.G1_29_SHOULDER_ORIGINS_M[side]
        assert np.linalg.norm(offset) == pytest.approx(hac.G1_29_STRAIGHT_ARM_REACH_M, abs=2e-3)
        assert math.degrees(hac._angle_between(offset, unit)) < 1.0


def test_mapping_uses_per_side_scale():
    operator = Operator(length=0.62)
    calibration = calibrated(operator, noise=0.0, wobble_deg=0.0)
    left = calibration.fits["left"].scale
    shifted = calibration.fits["left"].shoulder + np.array([0.1, 0.0, 0.0])
    mapped = hac.map_controller_position(calibration, "left", shifted)
    expected = hac.G1_29_SHOULDER_ORIGINS_M["left"] + left * (calibration.rotation @ np.array([0.1, 0.0, 0.0]))
    assert np.allclose(mapped, expected)


# ------------------------------------------------------------- L-pose gate

def test_l_pose_gate_accepts_within_five_centimetres_and_refuses_beyond():
    operator = Operator(yaw_deg=15.0)
    calibration = calibrated(operator)
    wrists = (pose(W0["left"]), pose(W0["right"]))
    controllers = [pose(operator.l_pose_controller(calibration, side)) for side in hac.SIDES]
    accepted = hac.evaluate_l_pose_gate(calibration, controllers, wrists)
    assert accepted.accepted and accepted.reason == "operator_in_l_pose"
    assert max(accepted.errors_m.values()) < 1e-9

    # Move the left controller so the mapped wrist moves 6 cm.
    fit = calibration.fits["left"]
    controllers[0] = pose(controllers[0][:3, 3] + calibration.rotation.T @ np.array([0.0, 0.0, 0.06 / fit.scale]))
    refused = hac.evaluate_l_pose_gate(calibration, controllers, wrists)
    assert not refused.accepted
    assert refused.reason == "operator_not_in_l_pose"
    assert refused.errors_m["left"] == pytest.approx(0.06, abs=1e-6)
    assert "cm" in refused.message()


def test_l_pose_gate_refuses_arm_up_at_chest_like_incident():
    operator = Operator()
    calibration = calibrated(operator)
    wrists = (pose(W0["left"]), pose(W0["right"]))
    controllers = [pose(operator.straight(side, (1.0, 0.0, 0.0))) for side in hac.SIDES]
    assert not hac.evaluate_l_pose_gate(calibration, controllers, wrists).accepted


def test_l_pose_gate_without_calibration_keeps_legacy_behaviour_and_rejects_invalid_pose():
    wrists = (pose(W0["left"]), pose(W0["right"]))
    assert hac.evaluate_l_pose_gate(None, (np.eye(4), np.eye(4)), wrists).accepted
    calibration = calibrated(Operator())
    bad = np.full((4, 4), np.nan)
    decision = hac.evaluate_l_pose_gate(calibration, (bad, np.eye(4)), wrists)
    assert not decision.accepted and decision.reason == "invalid_pose_sample"


# ------------------------------------------------------------ sweep session

def tele(left, right, timestamp):
    return types.SimpleNamespace(left_wrist_pose=pose(left), right_wrist_pose=pose(right), controller_sample_timestamp=timestamp)


def test_sweep_cannot_start_during_tracking_or_before_preparation():
    sweep = hac.HumanArmSweep()
    assert not sweep.request(1.0, preparation_ready=True, tracking_active=True)
    assert not sweep.request(1.0, preparation_ready=False, tracking_active=False)
    assert not sweep.active
    assert sweep.request(1.0, preparation_ready=True, tracking_active=False)
    assert not sweep.request(1.1, preparation_ready=True, tracking_active=False)


def test_sweep_aborts_if_tracking_becomes_active():
    sweep = hac.HumanArmSweep()
    now = time.monotonic()
    sweep.request(now, preparation_ready=True, tracking_active=False)
    result = sweep.observe(tele([0, 0, 0], [0, 0, 0], now + 0.01), now + 0.01, tracking_active=True)
    assert result is not None and not result.accepted
    assert result.reason == "aborted_tracking_active"
    assert not sweep.active


def test_sweep_collects_fresh_unique_samples_and_calibrates():
    operator = Operator()
    rng = np.random.default_rng(7)
    left, right = operator.sweep("left", rng, count=90), operator.sweep("right", rng, count=90)
    sweep = hac.HumanArmSweep(duration_s=3.0)
    start = 100.0
    sweep.request(start, preparation_ready=True, tracking_active=False)
    result = None
    for index in range(91):
        now = start + index * (3.0 / 90)
        sample = tele(left[min(index, 89)], right[min(index, 89)], now - 0.005)
        # duplicate (non-increasing) and stale samples must be ignored
        first = sweep.observe(sample, now, tracking_active=False)
        assert first is None or index == 90
        result = first or result
        result = sweep.observe(tele(left[0], right[0], now - 5.0), now + 1e-4, tracking_active=False) or result
    assert result is not None
    assert result.accepted, result.reason
    assert sweep.stale_or_invalid_count >= 90


def test_sweep_with_no_samples_fails_closed():
    sweep = hac.HumanArmSweep(duration_s=0.1)
    sweep.request(10.0, preparation_ready=True, tracking_active=False)
    result = sweep.observe(tele([0, 0, 0], [0, 0, 0], 0.0), 10.2, tracking_active=False)
    assert result is not None and not result.accepted
    assert "too_few_samples" in result.reason


# ------------------------------------------- wrapper + real base calibrator

class LegacyBase:
    """Base without the translation-scale API (current deployed contract)."""

    def __init__(self):
        self.inner = ControllerWristCalibrator()
        # On the scale branch the inner calibrator defaults to k=0.7; pin the
        # legacy k=1 law so this fixture emulates the deployed contract.
        if hasattr(self.inner, "set_translation_scale"):
            self.inner.set_translation_scale(1.0)

    def __getattr__(self, name):
        if name in ("translation_scale", "set_translation_scale"):
            raise AttributeError(name)
        return getattr(self.inner, name)


class ScaledBase(LegacyBase):
    """Emulates the calib-projection branch API: p = W0 + k (p_C - p_C0)."""

    def __init__(self, k=0.7):
        super().__init__()
        self.translation_scale = k
        self.set_calls = []

    def set_translation_scale(self, k):
        if self.inner.calibrated:
            raise RuntimeError("after calibrate")
        self.set_calls.append(k)
        self.translation_scale = k

    def calibrate(self, controllers, wrists, *args, **kwargs):
        self._c0 = [np.asarray(c, dtype=float).copy() for c in controllers]
        self._w0 = [np.asarray(w, dtype=float).copy() for w in wrists]
        return self.inner.calibrate(controllers, wrists, *args, **kwargs)

    def targets(self, controllers, timestamp, now=None):
        base = self.inner.targets(controllers, timestamp, now=now)
        if base is None:
            return None
        out = []
        for target, controller, c0, w0 in zip(base, controllers, self._c0, self._w0):
            scaled = target.copy()
            scaled[:3, 3] = w0[:3, 3] + self.translation_scale * (np.asarray(controller)[:3, 3] - c0[:3, 3])
            out.append(scaled)
        return tuple(out)


@pytest.mark.parametrize("base_factory", [LegacyBase, ScaledBase])
def test_wrapper_first_target_is_w0_and_offset_blends_into_directional_mapping(base_factory):
    operator = Operator(yaw_deg=10.0)
    calibration = calibrated(operator)
    base = base_factory()
    wrapper = hac.HumanCalibratedWristCalibrator(base)
    wrapper.set_human_calibration(calibration)
    rotation = np.eye(3)
    offset = np.array([0.02, -0.01, 0.015])
    controllers = [pose(operator.l_pose_controller(calibration, side) + calibration.rotation.T @ offset / calibration.fits[side].scale, rotation) for side in hac.SIDES]
    wrists = (pose(W0["left"]), pose(W0["right"]))
    request = time.monotonic()
    t0 = request + 0.001
    assert wrapper.calibrate(controllers, wrists, t0, request, now=t0 + 0.01)
    assert np.allclose(wrapper.consume_first_target()[0][:3, 3], W0["left"])
    if isinstance(base, ScaledBase):
        assert base.set_calls and base.set_calls[-1] == pytest.approx(np.mean([calibration.scale(s) for s in hac.SIDES]))

    # Same controller sample right after calibration: still at W0.
    first = wrapper.targets(controllers, t0 + 1e-3, now=t0 + 0.01)
    assert np.allclose(first[0][:3, 3], W0["left"], atol=2e-4)
    # After the blend window, the target is exactly the directional mapping.
    later_time = t0 + hac.OFFSET_BLEND_S + 0.1
    later = wrapper.targets(controllers, later_time, now=later_time + 0.01)
    expected = hac.map_controller_position(calibration, "left", controllers[0][:3, 3])
    assert np.allclose(later[0][:3, 3], expected, atol=1e-9)
    assert np.allclose(later[0][:3, 3] - W0["left"], offset, atol=1e-9)


def test_wrapper_orientation_stays_relative_one_to_one():
    operator = Operator()
    calibration = calibrated(operator)
    wrapper = hac.HumanCalibratedWristCalibrator(LegacyBase())
    wrapper.set_human_calibration(calibration)
    controllers = [pose(operator.l_pose_controller(calibration, side)) for side in hac.SIDES]
    wrist_rotation = hac.yaw_rotation(0.3)
    wrists = (pose(W0["left"], wrist_rotation), pose(W0["right"], wrist_rotation))
    request = time.monotonic()
    t0 = request + 0.001
    assert wrapper.calibrate(controllers, wrists, t0, request, now=t0)
    turn = np.array([[1, 0, 0], [0, math.cos(0.2), -math.sin(0.2)], [0, math.sin(0.2), math.cos(0.2)]])
    rotated = [pose(c[:3, 3], turn) for c in controllers]
    target = wrapper.targets(rotated, t0 + 0.01, now=t0 + 0.02)
    assert np.allclose(target[0][:3, :3], turn @ wrist_rotation)


def test_wrapper_refuses_calibration_outside_l_pose_and_base_stays_uncalibrated():
    operator = Operator()
    calibration = calibrated(operator)
    wrapper = hac.HumanCalibratedWristCalibrator(LegacyBase())
    wrapper.set_human_calibration(calibration)
    controllers = [pose(operator.straight(side, (1, 0, 0))) for side in hac.SIDES]
    request = time.monotonic()
    assert not wrapper.calibrate(controllers, (pose(W0["left"]), pose(W0["right"])), request + 1e-3, request, now=request + 0.01)
    assert not wrapper.calibrated
    assert wrapper.last_gate_decision.reason == "operator_not_in_l_pose"


def test_wrapper_without_human_calibration_is_transparent():
    base = LegacyBase()
    wrapper = hac.HumanCalibratedWristCalibrator(base)
    assert wrapper.human_calibration is None
    controllers = (pose([0.4, 0.2, 1.5]), pose([0.4, -0.2, 1.5]))
    wrists = (pose(W0["left"]), pose(W0["right"]))
    request = time.monotonic()
    t0 = request + 1e-3
    assert wrapper.calibrate(controllers, wrists, t0, request, now=t0)
    moved = (pose([0.45, 0.2, 1.5]), pose([0.4, -0.2, 1.5]))
    target = wrapper.targets(moved, t0 + 0.01, now=t0 + 0.02)
    assert np.allclose(target[0][:3, 3], W0["left"] + [0.05, 0, 0])


def test_failed_calibration_cannot_be_installed_and_change_is_refused_after_calibrate():
    wrapper = hac.HumanCalibratedWristCalibrator(LegacyBase())
    wrapper.set_human_calibration(hac.HumanArmCalibration(False, "x"))
    assert wrapper.human_calibration is None
    request = time.monotonic()
    assert wrapper.calibrate((pose([0.4, 0.2, 1.5]), pose([0.4, -0.2, 1.5])), (pose(W0["left"]), pose(W0["right"])), request + 1e-3, request, now=request + 0.01)
    with pytest.raises(RuntimeError):
        wrapper.set_human_calibration(calibrated(Operator()))


def test_wrapper_targets_still_pass_through_base_gates():
    operator = Operator()
    calibration = calibrated(operator)
    wrapper = hac.HumanCalibratedWristCalibrator(LegacyBase())
    wrapper.set_human_calibration(calibration)
    controllers = [pose(operator.l_pose_controller(calibration, side)) for side in hac.SIDES]
    request = time.monotonic()
    t0 = request + 1e-3
    assert wrapper.calibrate(controllers, (pose(W0["left"]), pose(W0["right"])), t0, request, now=t0)
    # stale sample
    assert wrapper.targets(controllers, t0 - 1.0, now=t0 + 5.0) is None
    # non-finite controller
    assert wrapper.targets((np.full((4, 4), np.nan), controllers[1]), t0 + 0.01, now=t0 + 0.02) is None
    # teleport jump
    jumped = [pose(c[:3, 3] + [0, 0, 0.5]) for c in controllers]
    assert wrapper.targets(jumped, t0 + 0.02, now=t0 + 0.03) is None


def test_wrapper_with_real_base_restores_default_scale_without_human_calibration():
    base = ControllerWristCalibrator()
    if not hasattr(base, "set_translation_scale"):
        pytest.skip("scale API lands with the calib-projection branch")
    wrapper = hac.HumanCalibratedWristCalibrator(base)
    default = base.translation_scale
    operator = Operator()
    calibration = calibrated(operator)
    wrapper.set_human_calibration(calibration)
    controllers = [pose(operator.l_pose_controller(calibration, side)) for side in hac.SIDES]
    request = time.monotonic()
    t0 = request + 1e-3
    assert wrapper.calibrate(controllers, (pose(W0["left"]), pose(W0["right"])), t0, request, now=t0)
    assert base.translation_scale == pytest.approx(np.mean([calibration.scale(s) for s in hac.SIDES]))
    later = t0 + hac.OFFSET_BLEND_S + 0.1
    moved = [pose(c[:3, 3] + calibration.rotation.T @ np.array([0.03, 0, 0])) for c in controllers]
    target = wrapper.targets(moved, later, now=later + 0.01)
    assert np.allclose(target[0][:3, 3], hac.map_controller_position(calibration, "left", moved[0][:3, 3]), atol=1e-6)
    wrapper.reset_for_start_request(later)
    wrapper.set_human_calibration(None)
    assert wrapper.calibrate(controllers, (pose(W0["left"]), pose(W0["right"])), later + 1e-3, later, now=later + 0.01)
    assert base.translation_scale == pytest.approx(default)
