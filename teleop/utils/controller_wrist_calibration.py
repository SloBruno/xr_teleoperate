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
half-space constraints keep a target on its own URDF shoulder side (with a
small midline margin).  During tracking these limits, plus a translation ball
and a rotation cap around the calibration wrist pose, are enforced by
continuous PROJECTION onto the envelope boundary; the calibration pose itself
must lie inside the envelope or calibration is refused.
"""

import math
import time

import numpy as np

from teleop.utils.quest_safety import controller_sample_is_fresh


# Per-sample discontinuity limits between CONSECUTIVE controller samples.  A
# larger step is a tracking glitch/teleport: the cycle yields no target and
# the mapping is re-anchored so the next target continues from the last
# emitted target instead of jumping (a clutch, never a latch).
MAX_SAMPLE_TRANSLATION_JUMP_M = 0.15
MAX_SAMPLE_ROTATION_JUMP_RAD = math.radians(45.0)

# Fixed translation scale k applied only to the controller POSITION delta:
# p_target = p_anchor_W + k * (p_C - p_anchor_C).  Orientation stays 1:1.
DEFAULT_TRANSLATION_SCALE = 0.7
MIN_TRANSLATION_SCALE = 0.3
MAX_TRANSLATION_SCALE = 1.2

# Workspace envelope.  Targets outside it are continuously PROJECTED onto its
# boundary (the robot stops at the edge and resumes tracking as soon as the
# controller comes back), not rejected into an indefinite hold.
MAX_TARGET_TRANSLATION_FROM_CALIBRATION_M = 0.45
MAX_TARGET_ROTATION_FROM_CALIBRATION_RAD = math.radians(120.0)
# Each wrist may cross the sagittal (y=0) plane by at most this margin.
MIDLINE_CROSSING_MARGIN_M = 0.02
# Alternating projections onto the convex/annular sets converge quickly from
# any start because the calibration pose lies strictly inside every set.
PROJECTION_ITERATIONS = 8
PROJECTION_TOLERANCE_M = 1e-4
# The envelope is not convex (inner shoulder radius), so a raw projection can
# flip sides.  Emitted targets may therefore never move more than the raw
# (pre-projection) target moved this cycle plus this slack; otherwise the
# target catches up toward the projection at that bounded rate.
CONTINUITY_SLACK_M = 0.01
CONTINUITY_SLACK_RAD = math.radians(2.0)
CONTINUITY_SUBSTEPS = 8

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


def _orthonormalize(rotation):
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _rotation_log(rotation):
    """Return (unit axis, angle in [0, pi]) of a rotation matrix."""
    skew_axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    sine_twice = float(np.linalg.norm(skew_axis))
    cosine_twice = float(np.trace(rotation) - 1.0)
    angle = math.atan2(sine_twice, cosine_twice)
    if angle < 1e-9 or not math.isfinite(angle):
        return np.array([1.0, 0.0, 0.0]), 0.0
    if math.pi - angle < 1e-3:
        # Near pi the skew part vanishes; the axis is the eigenvector of the
        # symmetric part of the rotation with eigenvalue +1.
        eigenvalues, eigenvectors = np.linalg.eigh((rotation + rotation.T) / 2.0)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if float(axis @ skew_axis) < 0.0:
            axis = -axis
        return axis / np.linalg.norm(axis), angle
    if sine_twice < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return skew_axis / sine_twice, angle


def _rotation_exp(axis, angle):
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _project_rotation(rotation, center_rotation, max_angle):
    """Clamp ``rotation`` to a geodesic ball around ``center_rotation``."""
    axis, angle = _rotation_log(center_rotation.T @ rotation)
    if angle <= max_angle:
        return rotation, False
    return _orthonormalize(center_rotation @ _rotation_exp(axis, max_angle)), True


def _position_constraints_hold(position, side, center, tolerance=PROJECTION_TOLERANCE_M):
    if np.linalg.norm(position - center) > MAX_TARGET_TRANSLATION_FROM_CALIBRATION_M + tolerance:
        return False
    reach = float(np.linalg.norm(position - G1_29_SHOULDER_ORIGINS_M[side]))
    if not G1_29_MIN_SHOULDER_REACH_M - tolerance <= reach <= G1_29_MAX_SHOULDER_REACH_M + tolerance:
        return False
    if side == "left":
        return position[1] >= -MIDLINE_CROSSING_MARGIN_M - tolerance
    return position[1] <= MIDLINE_CROSSING_MARGIN_M + tolerance


def _project_position(position, side, center):
    """Project a wrist position onto the workspace envelope.

    The envelope is the intersection of a ball around the calibration wrist
    position, the own-side half-space (with a small midline margin), and the
    shoulder-centred spherical annulus.  Alternating projections are used; if
    they do not converge, fall back to the farthest feasible point on the
    segment from the (feasible) calibration position toward the request.
    Returns ``(position, projected)`` or ``(None, True)`` if nothing feasible
    was found (fail closed).
    """
    if _position_constraints_hold(position, side, center, tolerance=0.0):
        return position, False
    shoulder = G1_29_SHOULDER_ORIGINS_M[side]
    point = np.asarray(position, dtype=float).copy()
    for _ in range(PROJECTION_ITERATIONS):
        offset = point - center
        distance = float(np.linalg.norm(offset))
        if distance > MAX_TARGET_TRANSLATION_FROM_CALIBRATION_M:
            point = center + offset * (MAX_TARGET_TRANSLATION_FROM_CALIBRATION_M / distance)
        if side == "left":
            point[1] = max(point[1], -MIDLINE_CROSSING_MARGIN_M)
        else:
            point[1] = min(point[1], MIDLINE_CROSSING_MARGIN_M)
        radial = point - shoulder
        reach = float(np.linalg.norm(radial))
        if reach < 1e-9:
            radial, reach = center - shoulder, float(np.linalg.norm(center - shoulder))
        if reach > G1_29_MAX_SHOULDER_REACH_M:
            point = shoulder + radial * (G1_29_MAX_SHOULDER_REACH_M / reach)
        elif reach < G1_29_MIN_SHOULDER_REACH_M:
            point = shoulder + radial * (G1_29_MIN_SHOULDER_REACH_M / reach)
        if _position_constraints_hold(point, side, center):
            return point, True
    for fraction in np.linspace(1.0, 0.0, 65):
        candidate = center + fraction * (np.asarray(position, dtype=float) - center)
        if _position_constraints_hold(candidate, side, center, tolerance=0.0):
            return candidate, True
    return None, True


def _limit_position_step(previous, candidate, budget, side, center):
    """Move from feasible ``previous`` toward ``candidate`` by <= ``budget``.

    Every substep must stay inside the envelope; otherwise the last feasible
    substep is kept (never a jump through the torso-clearance hole).
    """
    delta = candidate - previous
    distance = float(np.linalg.norm(delta))
    if distance <= budget:
        return candidate
    result = previous
    for fraction in np.linspace(0.0, budget / distance, CONTINUITY_SUBSTEPS + 1)[1:]:
        point = previous + fraction * delta
        if not _position_constraints_hold(point, side, center):
            break
        result = point
    return result.copy()


def _limit_rotation_step(previous, candidate, budget, center, max_angle):
    axis, angle = _rotation_log(previous.T @ candidate)
    if angle <= budget:
        return candidate
    step = _orthonormalize(previous @ _rotation_exp(axis, budget))
    if _rotation_distance_matrix(center, step) > max_angle + 1e-9:
        return previous.copy()
    return step


def _rotation_distance_matrix(first, second):
    cosine = np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0)
    return math.acos(float(cosine))


class ControllerWristCalibrator:
    """Calibrate and bound a paired absolute controller pose stream.

    Contract of :meth:`targets`:

    * fail closed (``None``) for: not calibrated, stale or pre-request sample,
      non-finite/non-rigid controller pose, a discontinuity between
      consecutive controller samples, or an unprojectable target;
    * the discontinuity reference is ALWAYS the previous valid fresh
      controller sample, never the last accepted one, so one rejection can
      never latch into a permanent hold;
    * after a discontinuity the jumped side is re-anchored so the next target
      continues from the last emitted target (no robot teleport);
    * targets leaving the workspace envelope are projected onto its boundary,
      and an emitted target never moves more than the raw mapped target moved
      this cycle plus a small slack (continuity across the non-convex
      envelope); telemetry counters: ``reanchor_count``,
      ``projection_count``, ``continuity_limit_count``, ``rejection_counts``,
      ``last_rejection_reason``, ``last_projected``.
    """

    def __init__(self, translation_scale=DEFAULT_TRANSLATION_SCALE):
        self._translation_scale = self._validated_scale(translation_scale)
        self.reset_for_start_request(0.0)

    @property
    def translation_scale(self):
        return self._translation_scale

    def set_translation_scale(self, translation_scale):
        """Change k; only allowed while uncalibrated (before ``calibrate``)."""
        if self.calibrated:
            raise RuntimeError("translation scale can only be changed before calibrate()")
        self._translation_scale = self._validated_scale(translation_scale)

    @staticmethod
    def _validated_scale(translation_scale):
        if isinstance(translation_scale, bool):
            raise ValueError("translation scale must be a finite number")
        try:
            value = float(translation_scale)
        except (TypeError, ValueError):
            raise ValueError("translation scale must be a finite number") from None
        if not math.isfinite(value) or not MIN_TRANSLATION_SCALE <= value <= MAX_TRANSLATION_SCALE:
            raise ValueError(
                f"translation scale must be finite and within "
                f"[{MIN_TRANSLATION_SCALE}, {MAX_TRANSLATION_SCALE}]"
            )
        return value

    @property
    def calibrated(self):
        return self._offsets is not None

    def reset_for_start_request(self, request_timestamp):
        self.request_timestamp = float(request_timestamp)
        self._offsets = None
        self._measured_wrist_poses = None
        self._previous_controller_poses = None
        self._last_emitted_targets = None
        self._previous_raw_targets = None
        self._first_target = None
        self.reanchor_count = 0
        self.continuity_limit_count = 0
        self.projection_count = 0
        self.rejection_counts = {}
        self.last_rejection_reason = None
        self.last_projected = (False, False)

    def calibrate(self, controller_poses, measured_wrist_poses, sample_timestamp, request_timestamp, now=None):
        self.reset_for_start_request(request_timestamp)
        if not self._fresh_post_request(sample_timestamp, now):
            return False
        if not self._valid_pair(controller_poses) or not self._valid_pair(measured_wrist_poses):
            return False

        measured_wrist_poses = tuple(np.asarray(pose, dtype=float) for pose in measured_wrist_poses)
        if not self._within_absolute_workspace(measured_wrist_poses, tolerance=0.0):
            return False

        # Position and orientation must be calibrated independently because the
        # controller poses and Pinocchio wrist poses use different origins.
        # A full ``inv(controller) @ wrist`` SE(3) offset would make an
        # in-place controller rotation orbit the wrist around the controller.
        controller_poses = tuple(np.asarray(pose, dtype=float) for pose in controller_poses)
        self._offsets = [
            self._offset_mapping(controller, measured)
            for controller, measured in zip(controller_poses, measured_wrist_poses)
        ]
        self._measured_wrist_poses = tuple(pose.copy() for pose in measured_wrist_poses)
        self._previous_controller_poses = tuple(pose.copy() for pose in controller_poses)
        self._last_emitted_targets = tuple(pose.copy() for pose in measured_wrist_poses)
        self._previous_raw_targets = [pose.copy() for pose in measured_wrist_poses]
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
        if not self.calibrated:
            return self._reject("not_calibrated")
        if not self._fresh_post_request(sample_timestamp, now):
            return self._reject("stale_or_pre_request_sample")
        if not self._valid_pair(controller_poses):
            return self._reject("invalid_controller_pose")

        controller_poses = tuple(np.asarray(pose, dtype=float).copy() for pose in controller_poses)
        previous = self._previous_controller_poses
        # Always advance the discontinuity reference to this valid sample.
        self._previous_controller_poses = controller_poses
        jumped = [
            self._exceeds_step(prev, current)
            for prev, current in zip(previous, controller_poses)
        ]
        if any(jumped):
            self._reanchor(controller_poses, jumped)
            return self._reject("controller_sample_jump")

        targets = []
        raw_targets = []
        projected = []
        continuity_limited = False
        for side_index, side in enumerate(("left", "right")):
            current = controller_poses[side_index]
            anchor_controller, anchor_wrist, rotation_offset = self._offsets[side_index]
            center = self._measured_wrist_poses[side_index]
            last = self._last_emitted_targets[side_index]
            previous_raw = self._previous_raw_targets[side_index]
            raw_rotation = current[:3, :3] @ rotation_offset
            raw_position = anchor_wrist + self._translation_scale * (current[:3, 3] - anchor_controller)
            rotation, rotation_projected = _project_rotation(
                raw_rotation, center[:3, :3], MAX_TARGET_ROTATION_FROM_CALIBRATION_RAD
            )
            position, position_projected = _project_position(raw_position, side, center[:3, 3])
            if position is None:
                return self._reject("workspace_projection_failed")
            position_budget = float(np.linalg.norm(raw_position - previous_raw[:3, 3])) + CONTINUITY_SLACK_M
            rotation_budget = _rotation_distance_matrix(previous_raw[:3, :3], raw_rotation) + CONTINUITY_SLACK_RAD
            limited_position = _limit_position_step(last[:3, 3], position, position_budget, side, center[:3, 3])
            limited_rotation = _limit_rotation_step(
                last[:3, :3], rotation, rotation_budget, center[:3, :3], MAX_TARGET_ROTATION_FROM_CALIBRATION_RAD
            )
            side_limited = limited_position is not position or limited_rotation is not rotation
            continuity_limited = continuity_limited or side_limited
            target = np.eye(4)
            target[:3, :3] = limited_rotation
            target[:3, 3] = limited_position
            targets.append(target)
            raw = np.eye(4)
            raw[:3, :3] = raw_rotation
            raw[:3, 3] = raw_position
            raw_targets.append(raw)
            projected.append(rotation_projected or position_projected or side_limited)
        targets = tuple(targets)
        if not self._valid_pair(targets) or not self._within_absolute_workspace(targets):
            return self._reject("invalid_target")

        # Defense in depth: the emitted target itself must be continuous.
        output_jump = [
            self._exceeds_step(last, target)
            for last, target in zip(self._last_emitted_targets, targets)
        ]
        if any(output_jump):
            self._reanchor(controller_poses, output_jump)
            return self._reject("target_step_jump")

        self._previous_raw_targets = raw_targets
        if continuity_limited:
            self.continuity_limit_count += 1
        self.last_projected = tuple(projected)
        if any(projected):
            self.projection_count += 1
        self.last_rejection_reason = None
        self._last_emitted_targets = tuple(target.copy() for target in targets)
        return targets

    @staticmethod
    def _offset_mapping(controller, wrist):
        """Anchor pair for position (scaled delta) and the 1:1 rotation offset."""
        return (
            controller[:3, 3].copy(),
            wrist[:3, 3].copy(),
            _orthonormalize(controller[:3, :3].T @ wrist[:3, :3]),
        )

    def _reanchor(self, controller_poses, sides):
        """Make the current controller sample map onto the last emitted target."""
        for side_index, jumped in enumerate(sides):
            if jumped:
                self._offsets[side_index] = self._offset_mapping(
                    controller_poses[side_index], self._last_emitted_targets[side_index]
                )
                self._previous_raw_targets[side_index] = self._last_emitted_targets[side_index].copy()
        self.reanchor_count += 1

    def _reject(self, reason):
        self.last_rejection_reason = reason
        self.rejection_counts[reason] = self.rejection_counts.get(reason, 0) + 1
        return None

    @staticmethod
    def _exceeds_step(previous, current):
        return (
            np.linalg.norm(current[:3, 3] - previous[:3, 3]) > MAX_SAMPLE_TRANSLATION_JUMP_M
            or _rotation_distance(previous, current) > MAX_SAMPLE_ROTATION_JUMP_RAD
        )

    def _fresh_post_request(self, sample_timestamp, now):
        if now is None:
            now = time.monotonic()
        try:
            sample_timestamp = float(sample_timestamp)
        except (TypeError, ValueError):
            return False
        return (
            sample_timestamp > self.request_timestamp
            and controller_sample_is_fresh(sample_timestamp, now)
        )

    @staticmethod
    def _valid_pair(poses):
        try:
            return len(poses) == 2 and all(_is_rigid_se3(pose) for pose in poses)
        except TypeError:
            return False

    @staticmethod
    def _within_absolute_workspace(poses, tolerance=PROJECTION_TOLERANCE_M):
        """Absolute shoulder annulus + own-side half-space (no calibration ball)."""
        for side, pose in zip(("left", "right"), poses):
            position = np.asarray(pose, dtype=float)[:3, 3]
            reach = float(np.linalg.norm(position - G1_29_SHOULDER_ORIGINS_M[side]))
            if not G1_29_MIN_SHOULDER_REACH_M - tolerance <= reach <= G1_29_MAX_SHOULDER_REACH_M + tolerance:
                return False
            if side == "left" and position[1] < -MIDLINE_CROSSING_MARGIN_M - tolerance:
                return False
            if side == "right" and position[1] > MIDLINE_CROSSING_MARGIN_M + tolerance:
                return False
        return True
