"""Fail-closed Quest-controller to G1 wrist pose calibration.

The absolute envelope is expressed in the G1 torso/waist reference frame used
by the arm IK targets.  It is deliberately smaller than the reachable volume:
the supplied G1_29 URDF places the shoulder origins at x~=0, y=+/-0.10 m,
z~=0.25 m from ``torso_link`` and the arm chain adds about 0.19 m of nominal
forward wrist-link offsets before the hand.  The bounds below retain margin
from the torso, shoulder, and full joint-limit reach.  They are rejection
limits, never clipping limits.  Runtime FK validation against the deployed
Pinocchio model remains required before deployment; Pinocchio is optional in
this test environment.
"""

import math
import time

import numpy as np

from teleop.utils.quest_safety import controller_sample_is_fresh


# Conservative limits are deliberately checked before IK; bad poses are held,
# not clipped into a different target.
MAX_TARGET_TRANSLATION_FROM_CALIBRATION_M = 0.35
MAX_TARGET_ROTATION_FROM_CALIBRATION_RAD = math.radians(90.0)
MAX_SAMPLE_TRANSLATION_JUMP_M = 0.15
MAX_SAMPLE_ROTATION_JUMP_RAD = math.radians(45.0)

# Conservative per-side wrist-center bounds in the robot torso/waist frame.
# Source geometry: assets/g1/g1_body29_hand14.urdf, shoulder origins
# (x=0.0039563, y=+/-0.10022, z=0.24778), elbow origin x=0.015783,
# wrist-roll origin x=0.100, wrist-pitch x=0.038, wrist-yaw x=0.046.
# The x lower bound keeps the wrist in front of the torso; the side-specific
# y bounds keep each wrist on its own side; z excludes hip/neck-level targets.
G1_29_WRIST_WORKSPACE_M = {
    "left": {"x": (0.05, 0.48), "y": (0.02, 0.62), "z": (0.45, 1.30)},
    "right": {"x": (0.05, 0.48), "y": (-0.62, -0.02), "z": (0.45, 1.30)},
}


def _is_rigid_se3(transform):
    try:
        matrix = np.asarray(transform, dtype=float)
    except (TypeError, ValueError):
        return False
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return False
    rotation = matrix[:3, :3]
    return (
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    )


def _rotation_distance(first, second):
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.acos(float(cosine))


class ControllerWristCalibrator:
    """Calibrate and validate a paired absolute controller pose stream."""

    def __init__(self):
        self.reset_for_start_request(0.0)

    @property
    def calibrated(self):
        return self._offsets is not None

    def reset_for_start_request(self, request_timestamp):
        self.request_timestamp = float(request_timestamp)
        self._offsets = None
        self._measured_wrist_poses = None
        self._last_controller_poses = None
        self._first_target = None

    def calibrate(self, controller_poses, measured_wrist_poses, sample_timestamp, request_timestamp, now=None):
        self.reset_for_start_request(request_timestamp)
        if not self._fresh_post_request(sample_timestamp, now):
            return False
        if not self._valid_pair(controller_poses) or not self._valid_pair(measured_wrist_poses):
            return False

        measured_wrist_poses = tuple(np.asarray(pose, dtype=float) for pose in measured_wrist_poses)
        if not self._within_absolute_workspace(measured_wrist_poses):
            return False

        self._offsets = tuple(
            np.linalg.inv(np.asarray(controller, dtype=float)) @ np.asarray(measured, dtype=float)
            for controller, measured in zip(controller_poses, measured_wrist_poses)
        )
        self._measured_wrist_poses = tuple(pose.copy() for pose in measured_wrist_poses)
        self._last_controller_poses = tuple(np.asarray(pose, dtype=float).copy() for pose in controller_poses)
        self._first_target = tuple(pose.copy() for pose in measured_wrist_poses)
        return True

    def consume_first_target(self):
        """Return the exact measured FK sample once for the first command."""
        if self._first_target is None:
            return None
        first_target = self._first_target
        self._first_target = None
        return first_target

    def targets(self, controller_poses, sample_timestamp, now=None):
        if not self.calibrated or not self._fresh_post_request(sample_timestamp, now):
            return None
        if not self._valid_pair(controller_poses):
            return None

        controller_poses = tuple(np.asarray(pose, dtype=float) for pose in controller_poses)
        for current, previous in zip(controller_poses, self._last_controller_poses):
            if np.linalg.norm(current[:3, 3] - previous[:3, 3]) > MAX_SAMPLE_TRANSLATION_JUMP_M:
                return None
            if _rotation_distance(previous, current) > MAX_SAMPLE_ROTATION_JUMP_RAD:
                return None

        targets = tuple(current @ offset for current, offset in zip(controller_poses, self._offsets))
        if not self._valid_pair(targets):
            return None
        for target, measured in zip(targets, self._measured_wrist_poses):
            if np.linalg.norm(target[:3, 3] - measured[:3, 3]) > MAX_TARGET_TRANSLATION_FROM_CALIBRATION_M:
                return None
        if not self._within_absolute_workspace(targets):
            return None

        for target, measured in zip(targets, self._measured_wrist_poses):
            if _rotation_distance(measured, target) > MAX_TARGET_ROTATION_FROM_CALIBRATION_RAD:
                return None

        self._last_controller_poses = tuple(pose.copy() for pose in controller_poses)
        return targets

    def _fresh_post_request(self, sample_timestamp, now):
        if now is None:
            now = time.monotonic()
        return (
            sample_timestamp > self.request_timestamp
            and controller_sample_is_fresh(sample_timestamp, now)
        )

    @staticmethod
    def _valid_pair(poses):
        return len(poses) == 2 and all(_is_rigid_se3(pose) for pose in poses)

    @staticmethod
    def _within_absolute_workspace(poses):
        for side, pose in zip(("left", "right"), poses):
            bounds = G1_29_WRIST_WORKSPACE_M[side]
            position = pose[:3, 3]
            for axis, value in zip(("x", "y", "z"), position):
                lower, upper = bounds[axis]
                if not lower <= float(value) <= upper:
                    return False
        return True
