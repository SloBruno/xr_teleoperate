"""Fail-closed Quest-controller to G1 wrist pose calibration.

Frame contract: ``G1_29_ArmIK.forward_kinematics`` returns the exact
``reduced_robot.data.oMf[L_ee/R_ee]`` operational frames, and ``solve_ik``
consumes targets in that same reduced Pinocchio model root frame.  For the
checked-in URDF, the reduced root is the pelvis; this module deliberately
does not call it a waist frame.  ``tv_wrapper`` produces incoming controller
poses in a synthetic head-relative frame with +[0.15, 0, 0.45] origin
translation, but calibration maps those poses to the measured Pinocchio FK
sample before validating every resulting target.

The workspace is a shoulder-relative spherical annulus in the pelvis-root
frame.  Shoulder origins are taken from the URDF fixed zero waist chain;
0.18 m is a conservative torso-clearance inner radius and 0.50 m is below
the loose sum of the checked-in arm-link lengths plus margin.  The left/right
half-space constraints keep a target on its own URDF shoulder side.  These
are rejection limits, never clipping limits.
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

# Derived from pelvis->waist_yaw(0)->waist_roll origin [-.0039635,0,.044]
# -> torso_link -> shoulder_pitch origins in assets/g1/g1_body29_hand14.urdf.
G1_29_SHOULDER_ORIGINS_M = {
    "left": np.array([-0.0000072, 0.10022, 0.29178]),
    "right": np.array([-0.0000072, -0.10021, 0.29178]),
}
G1_29_MIN_SHOULDER_REACH_M = 0.18
G1_29_MAX_SHOULDER_REACH_M = 0.50


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

        # Position and orientation must be calibrated independently because the
        # controller poses and Pinocchio wrist poses use different origins.
        # A full ``inv(controller) @ wrist`` SE(3) offset would make an
        # in-place controller rotation orbit the wrist around the controller.
        self._offsets = tuple(
            (
                np.asarray(measured, dtype=float)[:3, 3]
                - np.asarray(controller, dtype=float)[:3, 3],
                np.asarray(controller, dtype=float)[:3, :3].T
                @ np.asarray(measured, dtype=float)[:3, :3],
            )
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

        offsets = self._offsets
        assert offsets is not None
        targets = []
        for current, (translation_offset, rotation_offset) in zip(controller_poses, offsets):
            target = np.eye(4)
            target[:3, :3] = current[:3, :3] @ rotation_offset
            target[:3, 3] = current[:3, 3] + translation_offset
            targets.append(target)
        targets = tuple(targets)
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
            position = pose[:3, 3]
            shoulder = G1_29_SHOULDER_ORIGINS_M[side]
            reach = float(np.linalg.norm(position - shoulder))
            if not G1_29_MIN_SHOULDER_REACH_M <= reach <= G1_29_MAX_SHOULDER_REACH_M:
                return False
            if side == "left" and position[1] < 0.0:
                return False
            if side == "right" and position[1] > 0.0:
                return False
        return True
