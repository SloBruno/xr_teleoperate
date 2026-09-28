import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "teleop" / "utils" / "controller_wrist_calibration.py"
sys.path.insert(0, str(MODULE_PATH.parents[2]))


def calibration_api():
    if not MODULE_PATH.exists():
        pytest.fail("controller wrist calibration module is missing")
    import importlib

    return importlib.import_module("teleop.utils.controller_wrist_calibration")


def pose(x=0.0, y=0.0, z=0.0, angle=0.0):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )


def roll_pose(x=0.0, y=0.0, z=0.0, angle=0.0):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0, x], [0.0, c, -s, y], [0.0, s, c, z], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )


# Provenance: independently composed from the zero-angle joint origins and
# fixed waist chain in assets/g1/g1_body29_hand14.urdf, including the L_ee/R_ee
# Pinocchio operational-frame +[0.05, 0, 0] offset in robot_arm_ik.py.
G1_ZERO_FK = (
    pose(0.24977428, 0.14865212467, 0.09523008),
    # Right shoulder origin is y=-0.10021 in the URDF (left is +0.10022),
    # so the independently composed pelvis-root FK is y=-0.14864212467.
    pose(0.24977428, -0.14864212467, 0.09523008),
)


def measured_zero_fk():
    return tuple(p.copy() for p in G1_ZERO_FK)


def _rpy_matrix(rpy):
    roll, pitch, yaw = map(float, rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _urdf_zero_fk(link_name):
    urdf = Path(__file__).parents[1] / "assets/g1/g1_body29_hand14.urdf"
    root = ET.parse(urdf).getroot()
    joints = {
        joint.find("child").attrib["link"]: joint
        for joint in root.findall("joint")
        if joint.find("child") is not None
    }
    chain = []
    while link_name != "pelvis":
        joint = joints[link_name]
        chain.append(joint.find("origin"))
        link_name = joint.find("parent").attrib["link"]

    transform = np.eye(4)
    for origin in reversed(chain):
        parent_rotation = transform[:3, :3].copy()
        transform[:3, :3] = parent_rotation @ _rpy_matrix(
            origin.attrib.get("rpy", "0 0 0").split()
        )
        transform[:3, 3] += parent_rotation @ np.fromstring(
            origin.attrib.get("xyz", "0 0 0"), sep=" "
        )
    operational_frame = np.eye(4)
    operational_frame[0, 3] = 0.05
    return transform @ operational_frame


def test_right_zero_fk_fixture_matches_independent_urdf_composition():
    right_fk = _urdf_zero_fk("right_wrist_yaw_link")
    np.testing.assert_allclose(right_fk[:3, 3], G1_ZERO_FK[1][:3, 3], atol=1e-11)
    assert right_fk[1, 3] == pytest.approx(-0.14864212467, abs=1e-11)


def test_preheld_controller_pose_maps_first_target_to_measured_fk_pose():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    left_controller = pose(1.0, 2.0, 3.0, 0.2)
    right_controller = pose(-1.0, 2.0, 3.0, -0.3)
    measured_left, measured_right = measured_zero_fk()
    measured_left[:3, :3] = pose(angle=0.1)[:3, :3]
    measured_right[:3, :3] = pose(angle=-0.1)[:3, :3]

    assert calibrator.calibrate(
        (left_controller, right_controller),
        (measured_left, measured_right),
        sample_timestamp=10.1,
        request_timestamp=10.0,
        now=10.1,
    )
    targets = calibrator.targets(
        (left_controller, right_controller), sample_timestamp=10.1, now=10.1
    )

    assert np.allclose(targets[0], measured_left)
    assert np.allclose(targets[1], measured_right)


def test_independently_derived_all_zero_fk_reference_is_calibratable_with_tolerance():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    assert calibrator.calibrate(
        (pose(1.0), pose(-1.0)), measured_zero_fk(), 10.1, 10.0, 10.1
    )
    first = calibrator.consume_first_target()
    np.testing.assert_allclose(first[0][:3, 3], [0.2498, 0.1487, 0.0952], atol=5e-4)
    np.testing.assert_allclose(first[1][:3, 3], [0.24977428, -0.14864212467, 0.09523008], atol=1e-11)


def test_absolute_workspace_check_rejects_forward_lateral_vertical_poses():
    api = calibration_api()
    measured = measured_zero_fk()
    workspace = api.ControllerWristCalibrator._within_absolute_workspace
    assert workspace(measured)
    assert not workspace((pose(0.95, 0.15, 0.1), measured[1]))
    assert not workspace((pose(0.25, 0.8, 0.1), measured[1]))
    assert not workspace((pose(0.25, 0.15, 0.9), measured[1]))


def test_slow_drift_is_projected_onto_envelope_instead_of_held():
    """Old contract: slow drift eventually returned None forever (hold).

    New contract: every sample yields a target; the target stops on the
    envelope boundary and never exceeds it.
    """
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    measured = measured_zero_fk()
    assert calibrator.calibrate((pose(1.0), pose(-1.0)), measured, 10.1, 10.0, 10.1)
    shoulder = api.G1_29_SHOULDER_ORIGINS_M["left"]
    for index in range(1, 40):
        result = calibrator.targets(
            (pose(1.0 + index * 0.02), pose(-1.0)), 10.1 + index / 10, 10.1 + index / 10
        )
        assert result is not None
        left = result[0][:3, 3]
        assert np.linalg.norm(left - shoulder) <= api.G1_29_MAX_SHOULDER_REACH_M + 1e-6
        assert np.linalg.norm(left - measured[0][:3, 3]) <= api.MAX_TARGET_TRANSLATION_FROM_CALIBRATION_M + 1e-6
    assert calibrator.last_projected[0]
    assert calibrator.projection_count > 0


def test_relative_controller_translation_and_rotation_maps_consistently_per_side():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator(translation_scale=1.0)
    controllers = (pose(), pose())
    measured = measured_zero_fk()
    measured[0][:3, :3] = pose(angle=0.1)[:3, :3]
    measured[1][:3, :3] = pose(angle=-0.1)[:3, :3]
    assert calibrator.calibrate(controllers, measured, 10.1, 10.0, 10.1)

    moved = (pose(0.1, 0.0, 0.0, 0.2), pose(-0.1, 0.0, 0.0, -0.2))
    targets = calibrator.targets(moved, sample_timestamp=10.2, now=10.2)
    assert targets[0][0, 3] == pytest.approx(measured[0][0, 3] + 0.1)
    assert targets[1][0, 3] == pytest.approx(measured[1][0, 3] - 0.1)
    np.testing.assert_allclose(
        targets[0][:3, :3], moved[0][:3, :3] @ measured[0][:3, :3]
    )
    np.testing.assert_allclose(
        targets[1][:3, :3], moved[1][:3, :3] @ measured[1][:3, :3]
    )


def test_controller_rotation_in_place_does_not_translate_wrist_target():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    measured = measured_zero_fk()
    controllers = (
        roll_pose(0.15, 0.25, 0.55),
        roll_pose(0.15, -0.25, 0.55),
    )
    assert calibrator.calibrate(controllers, measured, 10.1, 10.0, 10.1)

    rotated = (
        roll_pose(0.15, 0.25, 0.55, math.radians(10.0)),
        roll_pose(0.15, -0.25, 0.55, math.radians(-10.0)),
    )
    targets = calibrator.targets(rotated, 10.2, 10.2)

    assert targets is not None
    np.testing.assert_allclose(targets[0][:3, 3], measured[0][:3, 3], atol=1e-9)
    np.testing.assert_allclose(targets[1][:3, 3], measured[1][:3, 3], atol=1e-9)


def test_left_and_right_calibration_offsets_are_independent():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator(translation_scale=1.0)
    controllers = (pose(), pose())
    measured = measured_zero_fk()
    assert calibrator.calibrate(controllers, measured, 10.1, 10.0, 10.1)
    targets = calibrator.targets((pose(0.05), pose()), 10.2, 10.2)
    assert targets[0][0, 3] == pytest.approx(measured[0][0, 3] + 0.05)
    assert targets[1][1, 3] == pytest.approx(measured[1][1, 3])


@pytest.mark.parametrize(
    "sample_timestamp, now, controller_pair",
    [
        (9.9, 10.1, (pose(1.0), pose(-1.0))),
        (10.0, 10.1, (pose(1.0), pose(-1.0))),
        (10.1, 10.1, (np.full((4, 4), np.nan), pose(-1.0))),
    ],
)
def test_invalid_or_stale_calibration_is_rejected_without_targets(sample_timestamp, now, controller_pair):
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    measured = measured_zero_fk()
    assert not calibrator.calibrate(controller_pair, measured, sample_timestamp, 10.0, now)
    assert not calibrator.calibrated


def test_jump_and_out_of_workspace_targets_are_rejected_and_never_returned():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    controllers = (pose(1.0), pose(-1.0))
    measured = measured_zero_fk()
    assert calibrator.calibrate(controllers, measured, 10.1, 10.0, 10.1)

    assert calibrator.targets((pose(1.3), pose(-1.0)), 10.2, 10.2) is None
    assert calibrator.targets((pose(1.0), pose(-1.0, 0.0, 1.0)), 10.21, 10.21) is None
    assert calibrator.targets((pose(1.0, 0.0, 0.0, math.pi), pose(-1.0)), 10.22, 10.22) is None


def test_workspace_rejection_is_distinct_from_sample_jump_rejection():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    controllers = (pose(1.0), pose(-1.0))
    measured = measured_zero_fk()
    assert calibrator.calibrate(controllers, measured, 10.1, 10.0, 10.1)

    # The shoulder-relative gate rejects a forward target independently of
    # the controller-sample jump gate.
    assert not api.ControllerWristCalibrator._within_absolute_workspace(
        (pose(0.8, 0.15, 0.1), measured[1])
    )

    calibrator = api.ControllerWristCalibrator()
    assert calibrator.calibrate(controllers, measured, 10.1, 10.0, 10.1)
    # The target stays in the absolute envelope, but the controller sample
    # itself jumps too far and must be rejected independently.
    assert calibrator.targets((pose(1.0, 0.0, 0.0, math.radians(100.0)), pose(-1.0)), 10.2, 10.2) is None


def _calibrated(api, controllers=None, measured=None, scale=1.0):
    calibrator = api.ControllerWristCalibrator(translation_scale=scale)
    controllers = controllers if controllers is not None else (pose(1.0), pose(-1.0))
    measured = measured if measured is not None else measured_zero_fk()
    assert calibrator.calibrate(controllers, measured, 10.1, 10.0, 10.1)
    return calibrator, measured


class _Clock:
    def __init__(self, start=10.1):
        self.t = start

    def tick(self):
        self.t += 0.02
        return self.t


def test_single_jump_rejection_does_not_latch_into_permanent_hold():
    """RED on the old calibrator: comparing against the last ACCEPTED sample
    turned every later sample into a 'jump' after one rejection."""
    api = calibration_api()
    calibrator, _ = _calibrated(api)
    clock = _Clock()
    t = clock.tick()
    assert calibrator.targets((pose(1.0), pose(-1.0)), t, t) is not None
    t = clock.tick()
    # 0.2 m discontinuity between consecutive samples: rejected this cycle.
    assert calibrator.targets((pose(1.2), pose(-1.0)), t, t) is None
    assert calibrator.last_rejection_reason == "controller_sample_jump"
    # Controller stays at the new place and keeps moving slowly: tracking
    # must resume on the very next sample.
    for step in range(1, 6):
        t = clock.tick()
        targets = calibrator.targets((pose(1.2 + 0.01 * step), pose(-1.0)), t, t)
        assert targets is not None, step
    assert calibrator.reanchor_count == 1


def test_reanchor_continues_from_last_emitted_target_without_teleport():
    api = calibration_api()
    calibrator, measured = _calibrated(api)
    clock = _Clock()
    t = clock.tick()
    before = calibrator.targets((pose(1.03), pose(-1.0)), t, t)
    np.testing.assert_allclose(before[0][:3, 3], measured[0][:3, 3] + [0.03, 0.0, 0.0])
    t = clock.tick()
    # Glitch: controller teleports 0.3 m and 60 degrees.
    assert calibrator.targets((pose(1.33, 0.0, 0.0, math.radians(60.0)), pose(-1.0)), t, t) is None
    t = clock.tick()
    after = calibrator.targets((pose(1.34, 0.0, 0.0, math.radians(60.0)), pose(-1.0)), t, t)
    assert after is not None
    # Continues from the last emitted target plus only the new 1 cm delta.
    np.testing.assert_allclose(after[0][:3, 3], before[0][:3, 3] + [0.01, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(after[0][:3, :3], before[0][:3, :3], atol=1e-9)
    # The untouched right side keeps its original mapping.
    np.testing.assert_allclose(after[1], measured[1], atol=1e-12)


def test_jump_reference_always_advances_to_previous_sample():
    api = calibration_api()
    calibrator, _ = _calibrated(api)
    clock = _Clock()
    # Fast but continuous motion: 0.1 m per sample never trips the 0.15 m
    # consecutive-sample limit even though the total exceeds it.
    x = 1.0
    for _ in range(3):
        x += 0.1
        t = clock.tick()
        calibrator.targets((pose(x), pose(-1.0)), t, t)
    assert calibrator.reanchor_count == 0
    assert "controller_sample_jump" not in calibrator.rejection_counts


def test_translation_projection_stops_at_boundary_then_follows_back():
    api = calibration_api()
    calibrator, measured = _calibrated(api)
    clock = _Clock()
    # Push the left wrist straight down (-z) far beyond the 0.45 m ball.
    z = 0.0
    last = None
    for _ in range(70):
        z -= 0.01
        t = clock.tick()
        last = calibrator.targets((pose(1.0, 0.0, z), pose(-1.0)), t, t)
        assert last is not None
    offset = last[0][:3, 3] - measured[0][:3, 3]
    assert np.linalg.norm(offset) <= api.MAX_TARGET_TRANSLATION_FROM_CALIBRATION_M + 1e-6
    assert calibrator.last_projected[0]
    # Coming back inside the envelope immediately tracks the raw mapping
    # (z = -0.10 m is inside the reviewed 0.42 m shoulder reach).
    for _ in range(60):
        z += 0.01
        t = clock.tick()
        last = calibrator.targets((pose(1.0, 0.0, z), pose(-1.0)), t, t)
        assert last is not None
    np.testing.assert_allclose(last[0][:3, 3], measured[0][:3, 3] + [0.0, 0.0, z], atol=1e-9)
    assert not calibrator.last_projected[0]


def test_rotation_projection_caps_angle_from_calibration_and_follows_back():
    api = calibration_api()
    calibrator, measured = _calibrated(api)
    clock = _Clock()
    angle = 0.0
    for _ in range(35):  # 175 degrees in 5 degree steps
        angle += math.radians(5.0)
        t = clock.tick()
        targets = calibrator.targets((pose(1.0, 0.0, 0.0, angle), pose(-1.0)), t, t)
        assert targets is not None
        rotation = api._rotation_distance(measured[0], targets[0])
        assert rotation <= api.MAX_TARGET_ROTATION_FROM_CALIBRATION_RAD + 1e-6
        # Rotation projection must never translate the wrist.
        np.testing.assert_allclose(targets[0][:3, 3], measured[0][:3, 3], atol=1e-9)
    assert rotation == pytest.approx(api.MAX_TARGET_ROTATION_FROM_CALIBRATION_RAD, abs=1e-6)
    for _ in range(35):
        angle -= math.radians(5.0)
        t = clock.tick()
        targets = calibrator.targets((pose(1.0, 0.0, 0.0, angle), pose(-1.0)), t, t)
        assert targets is not None
    np.testing.assert_allclose(targets[0], measured[0], atol=1e-9)


def test_midline_and_shoulder_annulus_are_projected_not_held():
    api = calibration_api()
    calibrator, measured = _calibrated(api)
    clock = _Clock()
    y = 0.0
    for _ in range(40):  # left hand sweeps 0.4 m toward/over the midline
        y -= 0.01
        t = clock.tick()
        targets = calibrator.targets((pose(1.0, y, 0.0), pose(-1.0)), t, t)
        assert targets is not None
        assert targets[0][1, 3] >= -api.MIDLINE_CROSSING_MARGIN_M - 1e-6
        assert api.ControllerWristCalibrator._within_absolute_workspace(targets)
    # Now pull straight back toward the shoulder (inner 0.18 m radius).
    calibrator, measured = _calibrated(api)
    shoulder = api.G1_29_SHOULDER_ORIGINS_M["left"]
    direction = (shoulder - measured[0][:3, 3]) / np.linalg.norm(shoulder - measured[0][:3, 3])
    for step in range(1, 40):
        t = clock.tick()
        controller = pose(1.0, 0.0, 0.0)
        controller[:3, 3] += direction * 0.01 * step
        targets = calibrator.targets((controller, pose(-1.0)), t, t)
        assert targets is not None
        reach = np.linalg.norm(targets[0][:3, 3] - shoulder)
        assert reach >= api.G1_29_MIN_SHOULDER_REACH_M - 1e-6


def test_random_targets_always_project_inside_envelope():
    api = calibration_api()
    rng = np.random.default_rng(1234)
    measured = measured_zero_fk()
    for side_index, side in enumerate(("left", "right")):
        center = measured[side_index][:3, 3]
        for _ in range(3000):
            raw = center + rng.uniform(-1.2, 1.2, size=3)
            projected, _ = api._project_position(raw, side, center)
            assert projected is not None
            assert api._position_constraints_hold(projected, side, center)


def _random_rotation(rng, scale):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = rng.normal(scale=scale)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def test_random_walk_output_step_never_exceeds_input_step_plus_slack():
    """Continuity across the non-convex envelope (shoulder inner radius,
    rotation cap wrap-around): no emitted target may teleport."""
    api = calibration_api()
    rng = np.random.default_rng(99)
    calibrator, _ = _calibrated(api, controllers=(pose(0.0), pose(0.0)))
    controllers = [pose(0.0), pose(0.0)]
    last = calibrator.consume_first_target()
    clock = _Clock()
    accepted = 0
    for _ in range(4000):
        previous_controllers = [c.copy() for c in controllers]
        for c in controllers:
            c[:3, 3] += rng.normal(scale=0.015, size=3)
            c[:3, :3] = _random_rotation(rng, math.radians(4.0)) @ c[:3, :3]
        t = clock.tick()
        targets = calibrator.targets(tuple(controllers), t, t)
        if targets is None:
            continue
        accepted += 1
        assert api.ControllerWristCalibrator._within_absolute_workspace(targets)
        for side_index in range(2):
            input_step = np.linalg.norm(controllers[side_index][:3, 3] - previous_controllers[side_index][:3, 3])
            input_rot = api._rotation_distance(previous_controllers[side_index], controllers[side_index])
            assert np.linalg.norm(targets[side_index][:3, 3] - last[side_index][:3, 3]) <= input_step + api.CONTINUITY_SLACK_M + 1e-9
            assert api._rotation_distance(last[side_index], targets[side_index]) <= input_rot + api.CONTINUITY_SLACK_RAD + 1e-6
            assert api._rotation_distance(calibrator._measured_wrist_poses[side_index], targets[side_index]) <= api.MAX_TARGET_ROTATION_FROM_CALIBRATION_RAD + 1e-6
        last = targets
    assert accepted > 3900
    assert calibrator.projection_count > 0


@pytest.mark.parametrize(
    "bad_pose",
    [
        np.full((4, 4), np.nan),
        np.diag([np.inf, 1.0, 1.0, 1.0]),
        np.diag([2.0, 1.0, 1.0, 1.0]),  # non-rigid (scaled)
        np.diag([-1.0, 1.0, 1.0, 1.0]),  # reflection
        np.eye(3),
    ],
)
def test_invalid_controller_pose_fails_closed_and_does_not_advance_jump_reference(bad_pose):
    api = calibration_api()
    calibrator, _ = _calibrated(api)
    assert calibrator.targets((bad_pose, pose(-1.0)), 10.2, 10.2) is None
    assert calibrator.last_rejection_reason == "invalid_controller_pose"
    assert calibrator.targets((pose(1.01), pose(-1.0)), 10.21, 10.21) is not None
    assert calibrator.reanchor_count == 0


def test_stale_pre_request_and_uncalibrated_samples_fail_closed():
    api = calibration_api()
    fresh = api.ControllerWristCalibrator()
    assert fresh.targets((pose(1.0), pose(-1.0)), 10.2, 10.2) is None
    assert fresh.last_rejection_reason == "not_calibrated"
    calibrator, _ = _calibrated(api)
    assert calibrator.targets((pose(1.0), pose(-1.0)), 10.2, 10.6) is None  # stale
    assert calibrator.last_rejection_reason == "stale_or_pre_request_sample"
    assert calibrator.targets((pose(1.0), pose(-1.0)), 9.9, 10.0) is None  # pre-request
    assert calibrator.targets((pose(1.0), pose(-1.0)), float("nan"), 10.0) is None


def test_calibration_outside_absolute_workspace_is_refused():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    measured = measured_zero_fk()
    measured[0][:3, 3] = [0.9, 0.15, 0.1]
    assert not calibrator.calibrate((pose(1.0), pose(-1.0)), measured, 10.1, 10.0, 10.1)
    assert not calibrator.calibrated
    measured = measured_zero_fk()
    measured[0][1, 3] = -0.1  # left wrist on the right side
    assert not calibrator.calibrate((pose(1.0), pose(-1.0)), measured, 10.1, 10.0, 10.1)


def test_new_start_request_resets_counters_and_reanchored_offsets():
    api = calibration_api()
    calibrator, measured = _calibrated(api)
    assert calibrator.targets((pose(1.5), pose(-1.0)), 10.2, 10.2) is None
    assert calibrator.reanchor_count == 1
    assert calibrator.calibrate((pose(1.5), pose(-1.0)), measured, 20.1, 20.0, 20.1)
    assert calibrator.reanchor_count == 0
    assert calibrator.projection_count == 0
    assert calibrator.rejection_counts == {}
    np.testing.assert_allclose(calibrator.targets((pose(1.5), pose(-1.0)), 20.2, 20.2)[0], measured[0])


def test_calibration_preserves_and_consumes_exact_first_measured_target():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    measured = measured_zero_fk()
    sample = (pose(1.0), pose(-1.0))
    assert calibrator.calibrate(sample, measured, 10.1, 10.0, 10.1)
    first = calibrator.consume_first_target()
    assert np.array_equal(first[0], measured[0])
    assert np.array_equal(first[1], measured[1])
    assert calibrator.consume_first_target() is None


def test_each_new_start_request_resets_calibration():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    first = (pose(1.0), pose(-1.0))
    second = (pose(2.0), pose(-2.0))
    measured = measured_zero_fk()
    assert calibrator.calibrate(first, measured, 10.1, 10.0, 10.1)
    calibrator.reset_for_start_request(20.0)
    assert not calibrator.calibrated
    assert calibrator.targets(second, 20.1, 20.1) is None
    assert calibrator.calibrate(second, measured, 20.1, 20.0, 20.1)
    assert np.allclose(calibrator.targets(second, 20.2, 20.2)[0], measured[0])


@pytest.mark.parametrize("angle", [math.pi, math.pi - 1e-7, math.pi - 5e-4, 3.0, 1e-10])
def test_rotation_log_exp_roundtrip_including_near_pi(angle):
    api = calibration_api()
    axis = np.array([0.3, -0.5, 0.8])
    axis /= np.linalg.norm(axis)
    rotation = api._rotation_exp(axis, angle)
    recovered_axis, recovered_angle = api._rotation_log(rotation)
    assert np.all(np.isfinite(recovered_axis))
    assert recovered_angle == pytest.approx(angle, abs=1e-6)
    np.testing.assert_allclose(api._rotation_exp(recovered_axis, recovered_angle), rotation, atol=1e-6)


def test_default_translation_scale_is_0_7_and_validated():
    api = calibration_api()
    assert api.DEFAULT_TRANSLATION_SCALE == pytest.approx(0.7)
    assert api.ControllerWristCalibrator().translation_scale == pytest.approx(0.7)
    for bad in (0.29, 1.21, float("nan"), float("inf"), -0.7, "x", None, True):
        with pytest.raises(ValueError):
            api.ControllerWristCalibrator(translation_scale=bad)
    calibrator = api.ControllerWristCalibrator()
    calibrator.set_translation_scale(0.3)
    calibrator.set_translation_scale(1.2)
    with pytest.raises(ValueError):
        calibrator.set_translation_scale(1.5)
    assert calibrator.translation_scale == pytest.approx(1.2)


def test_set_translation_scale_is_refused_after_calibrate():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    assert calibrator.calibrate((pose(1.0), pose(-1.0)), measured_zero_fk(), 10.1, 10.0, 10.1)
    with pytest.raises(RuntimeError):
        calibrator.set_translation_scale(1.0)
    assert calibrator.translation_scale == pytest.approx(0.7)
    # A new start request clears calibration, after which k may change again.
    calibrator.reset_for_start_request(20.0)
    calibrator.set_translation_scale(1.0)
    assert calibrator.translation_scale == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [0.3, 0.7, 1.2])
def test_translation_scale_affects_only_position_delta(scale):
    api = calibration_api()
    measured = measured_zero_fk()
    measured[0][:3, :3] = pose(angle=0.1)[:3, :3]
    calibrator = api.ControllerWristCalibrator(translation_scale=scale)
    assert calibrator.calibrate((pose(), pose()), measured, 10.1, 10.0, 10.1)
    moved = (pose(0.05, -0.03, 0.04, 0.3), pose(-0.02, 0.01, 0.03, -0.2))
    targets = calibrator.targets(moved, 10.12, 10.12)
    assert targets is not None
    for side in range(2):
        np.testing.assert_allclose(
            targets[side][:3, 3], measured[side][:3, 3] + scale * moved[side][:3, 3], atol=1e-12
        )
        # Orientation is 1:1 regardless of k.
        np.testing.assert_allclose(targets[side][:3, :3], moved[side][:3, :3] @ measured[side][:3, :3], atol=1e-12)


def test_scaled_rotation_in_place_does_not_translate_wrist_target():
    api = calibration_api()
    measured = measured_zero_fk()
    calibrator = api.ControllerWristCalibrator(translation_scale=0.7)
    controllers = (roll_pose(0.15, 0.25, 0.55), roll_pose(0.15, -0.25, 0.55))
    assert calibrator.calibrate(controllers, measured, 10.1, 10.0, 10.1)
    rotated = (
        roll_pose(0.15, 0.25, 0.55, math.radians(20.0)),
        roll_pose(0.15, -0.25, 0.55, math.radians(-20.0)),
    )
    targets = calibrator.targets(rotated, 10.12, 10.12)
    assert targets is not None
    for side in range(2):
        np.testing.assert_allclose(targets[side][:3, 3], measured[side][:3, 3], atol=1e-9)
        assert api._rotation_distance(measured[side], targets[side]) == pytest.approx(math.radians(20.0), abs=1e-9)


def test_scaled_reanchor_after_jump_is_continuous_and_keeps_k():
    api = calibration_api()
    calibrator, measured = _calibrated(api, scale=0.7)
    clock = _Clock()
    t = clock.tick()
    # Move backward (toward the torso) so every target stays inside the
    # reviewed 0.42 m shoulder reach and no projection interferes.
    before = calibrator.targets((pose(0.90), pose(-1.0)), t, t)
    np.testing.assert_allclose(before[0][:3, 3], measured[0][:3, 3] + [-0.07, 0.0, 0.0], atol=1e-12)
    t = clock.tick()
    assert calibrator.targets((pose(0.50, 0.2, 0.0), pose(-1.0)), t, t) is None
    t = clock.tick()
    resumed = calibrator.targets((pose(0.50, 0.2, 0.0), pose(-1.0)), t, t)
    # No teleport: exactly the last emitted target.
    np.testing.assert_allclose(resumed[0], before[0], atol=1e-12)
    t = clock.tick()
    after = calibrator.targets((pose(0.40, 0.2, 0.0), pose(-1.0)), t, t)
    # Same k after re-anchoring: -0.10 controller -> -0.07 wrist.
    np.testing.assert_allclose(after[0][:3, 3], before[0][:3, 3] + [-0.07, 0.0, 0.0], atol=1e-12)


def test_reviewed_reach_and_rotation_jump_constants():
    """Replay of the 2026-09-28 sessions: 0.42 m matches the real G1_29 arm
    reach (0.424 m); 90 deg cuts spurious re-anchors from 13 to 1."""
    api = calibration_api()
    assert api.G1_29_MAX_SHOULDER_REACH_M == pytest.approx(0.42)
    assert api.MAX_SAMPLE_ROTATION_JUMP_RAD == pytest.approx(math.radians(90.0))
    assert api.MAX_SAMPLE_TRANSLATION_JUMP_M == pytest.approx(0.15)
    # The preparation (all-zero FK) pose must stay calibratable.
    shoulder = api.G1_29_SHOULDER_ORIGINS_M["left"]
    reach = np.linalg.norm(measured_zero_fk()[0][:3, 3] - shoulder)
    assert reach < api.G1_29_MAX_SHOULDER_REACH_M


def test_sixty_degree_consecutive_rotation_is_not_a_jump_but_hundred_is():
    api = calibration_api()
    calibrator, _ = _calibrated(api)
    clock = _Clock()
    t = clock.tick()
    assert calibrator.targets((pose(1.0, 0.0, 0.0, math.radians(60.0)), pose(-1.0)), t, t) is not None
    assert calibrator.reanchor_count == 0
    t = clock.tick()
    assert calibrator.targets((pose(1.0, 0.0, 0.0, math.radians(160.0)), pose(-1.0)), t, t) is None
    assert calibrator.last_rejection_reason == "controller_sample_jump"


def test_extended_target_is_projected_inside_042_reach():
    api = calibration_api()
    calibrator, measured = _calibrated(api)
    shoulder = api.G1_29_SHOULDER_ORIGINS_M["left"]
    clock = _Clock()
    last = None
    for index in range(1, 30):
        t = clock.tick()
        result = calibrator.targets((pose(1.0 + 0.01 * index), pose(-1.0)), t, t)
        assert result is not None
        last = result[0][:3, 3]
        assert np.linalg.norm(last - shoulder) <= 0.42 + 1e-6
    assert np.linalg.norm(last - shoulder) > 0.40
