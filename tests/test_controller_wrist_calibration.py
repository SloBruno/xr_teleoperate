import math
from pathlib import Path
import sys

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


def test_preheld_controller_pose_maps_first_target_to_measured_fk_pose():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    left_controller = pose(1.0, 2.0, 3.0, 0.2)
    right_controller = pose(-1.0, 2.0, 3.0, -0.3)
    measured_left = pose(0.4, 0.2, 0.7, 0.1)
    measured_right = pose(0.4, -0.2, 0.7, -0.1)

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


def test_relative_controller_translation_and_rotation_maps_consistently_per_side():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    controllers = (pose(), pose())
    measured = (pose(0.4, 0.2, 0.7, 0.1), pose(0.4, -0.2, 0.7, -0.1))
    assert calibrator.calibrate(controllers, measured, 10.1, 10.0, 10.1)

    moved = (pose(0.1, 0.0, 0.0, 0.2), pose(-0.1, 0.0, 0.0, -0.2))
    targets = calibrator.targets(moved, sample_timestamp=10.2, now=10.2)
    assert np.allclose(targets[0], moved[0] @ np.linalg.inv(controllers[0]) @ measured[0])
    assert np.allclose(targets[1], moved[1] @ np.linalg.inv(controllers[1]) @ measured[1])


def test_left_and_right_calibration_offsets_are_independent():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    controllers = (pose(), pose())
    measured = (pose(0.4, 0.2, 0.7), pose(0.4, -0.2, 0.7))
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
    measured = (pose(0.4, 0.2, 0.7), pose(0.4, -0.2, 0.7))
    assert not calibrator.calibrate(controller_pair, measured, sample_timestamp, 10.0, now)
    assert not calibrator.calibrated


def test_jump_and_out_of_workspace_targets_are_rejected_and_never_returned():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    controllers = (pose(1.0), pose(-1.0))
    measured = (pose(0.4, 0.2, 0.7), pose(0.4, -0.2, 0.7))
    assert calibrator.calibrate(controllers, measured, 10.1, 10.0, 10.1)

    assert calibrator.targets((pose(1.3), pose(-1.0)), 10.2, 10.2) is None
    assert calibrator.targets((pose(1.0), pose(-1.0, 0.0, 1.0)), 10.21, 10.21) is None
    assert calibrator.targets((pose(1.0, 0.0, 0.0, math.pi), pose(-1.0)), 10.22, 10.22) is None


def test_each_new_start_request_resets_calibration():
    api = calibration_api()
    calibrator = api.ControllerWristCalibrator()
    first = (pose(1.0), pose(-1.0))
    second = (pose(2.0), pose(-2.0))
    measured = (pose(0.4, 0.2, 0.7), pose(0.4, -0.2, 0.7))
    assert calibrator.calibrate(first, measured, 10.1, 10.0, 10.1)
    calibrator.reset_for_start_request(20.0)
    assert not calibrator.calibrated
    assert calibrator.targets(second, 20.1, 20.1) is None
    assert calibrator.calibrate(second, measured, 20.1, 20.0, 20.1)
    assert np.allclose(calibrator.targets(second, 20.2, 20.2)[0], measured[0])
