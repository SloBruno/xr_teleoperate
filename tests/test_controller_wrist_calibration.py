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


def test_shoulder_relative_workspace_rejects_forward_lateral_vertical_and_accepts_drift_until_limit():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    assert calibrator.calibrate((pose(1.0), pose(-1.0)), measured_zero_fk(), 10.1, 10.0, 10.1)
    assert calibrator.targets((pose(1.7), pose(-1.0)), 10.2, 10.2) is None
    assert calibrator.targets((pose(1.0, 0.8), pose(-1.0)), 10.3, 10.3) is None
    assert calibrator.targets((pose(1.0, 0.0, 0.8), pose(-1.0)), 10.4, 10.4) is None

    calibrator = api.ControllerWristCalibrator()
    assert calibrator.calibrate((pose(1.0), pose(-1.0)), measured_zero_fk(), 10.1, 10.0, 10.1)
    accepted = 0
    for index in range(1, 20):
        result = calibrator.targets((pose(1.0 + index * 0.02), pose(-1.0)), 10.1 + index / 10, 10.1 + index / 10)
        if result is None:
            break
        accepted += 1
    assert accepted > 1
    assert result is None


def test_relative_controller_translation_and_rotation_maps_consistently_per_side():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    controllers = (pose(), pose())
    measured = measured_zero_fk()
    measured[0][:3, :3] = pose(angle=0.1)[:3, :3]
    measured[1][:3, :3] = pose(angle=-0.1)[:3, :3]
    assert calibrator.calibrate(controllers, measured, 10.1, 10.0, 10.1)

    moved = (pose(0.1, 0.0, 0.0, 0.2), pose(-0.1, 0.0, 0.0, -0.2))
    targets = calibrator.targets(moved, sample_timestamp=10.2, now=10.2)
    assert np.allclose(targets[0], moved[0] @ np.linalg.inv(controllers[0]) @ measured[0])
    assert np.allclose(targets[1], moved[1] @ np.linalg.inv(controllers[1]) @ measured[1])


def test_left_and_right_calibration_offsets_are_independent():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
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
    assert calibrator.targets((pose(1.0, 0.0, 0.0, math.radians(50.0)), pose(-1.0)), 10.2, 10.2) is None


def test_slow_cumulative_drift_eventually_hits_absolute_envelope():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    measured = measured_zero_fk()
    assert calibrator.calibrate((pose(1.0), pose(-1.0)), measured, 10.1, 10.0, 10.1)

    accepted = 0
    for index in range(1, 20):
        x = 1.0 + index * 0.02
        result = calibrator.targets((pose(x), pose(-1.0)), 10.1 + index / 10, 10.1 + index / 10)
        if result is None:
            break
        accepted += 1
    assert accepted > 1
    assert result is None


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
