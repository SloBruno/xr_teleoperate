"""Phase-2 human-arm calibration for G1_29 controller teleoperation.

Everything here is pure numpy: no DDS, no publisher, no IK, no I/O.

1. **Sweep.**  While the robot is prepared and NOT tracking (``READY and not
   START``), the operator sweeps each straight arm from hanging down to
   pointing forward (~3 s).  Both controller positions are collected from
   fresh, strictly-increasing, finite atomic controller samples.

2. **Shoulder / length fit.**  Per side a sphere is fitted (Gauss-Newton on
   the 3-D residual ``|p_i - c| - r``, Kasa initial guess) with its centre
   constrained to the best-fit VERTICAL plane of the samples (a down->forward
   sweep contains gravity).  The centre is the estimated shoulder ``S_h`` and
   the radius the shoulder->controller length ``L_h``.  The constraint is
   deliberate: a ~90 deg arc does not determine the out-of-plane coordinate
   of a free sphere centre, and a free PCA plane is tilted by lateral wobble
   (≈1 cm of shoulder error per degree).  The fit is
   rejected when RMS > 1.5 cm, arc span < 60 deg, radius outside
   0.40-0.85 m, the arc does not include the hanging (down) or the forward
   (near horizontal) direction, too few samples, or degenerate geometry.
   The head pose is NOT used: in production telemetry it is the constant
   ``CONST_HEAD_POSE`` fallback.

3. **Scale.**  ``k_side = L_robot / L_h`` with ``L_robot = 0.424 m`` (G1_29
   straight-arm shoulder->wrist distance by FK), clamped to [0.3, 1.2].  If
   the calibration fails, nothing changes: the base calibrator keeps its
   fixed scale (0.7 in the scale branch) and the failure is logged.

4. **Body alignment ``R_align``.**  Pure yaw about +z (controller frame is
   the stable, gravity-aligned robot-basis world frame produced by
   ``tv_wrapper``).  The lateral body axis is the mean of the two sweep-plane
   normals (sign-aligned to ``S_L - S_R``), projected on the horizontal
   plane; it is cross-checked against the shoulder line and the swept
   forward direction.  ``R_align = Rz(-yaw)`` maps the operator's forward to
   robot +x.  Roll/pitch are never estimated.

5. **Directional mapping.**  ``p_target = S_robot + k * R_align (p_C - S_h)``:
   a straight human arm (``|p_C - S_h| = L_h``) maps to a straight robot arm
   (``0.424 m``) in the same body-relative direction.  Orientation stays
   relative 1:1 until the final 4 cm of shoulder reach; there it is smoothly
   blended toward the measured calibration wrist orientation.  This keeps the
   unchanged downstream residual gate fail-closed while avoiding a known
   incompatible controller orientation at the 0.42 m workspace boundary.

6. **L-pose gate on ``r``.**  With an accepted calibration, ``r`` only
   authorizes tracking if the directionally-mapped controller position of
   each side is within 5 cm of the measured robot wrist ``W0`` (FK of the
   all-zero preparation pose).  Otherwise the start is refused and the robot
   stays in its preparation hold.

7. **Integration.**  :class:`HumanCalibratedWristCalibrator` wraps the
   existing ``ControllerWristCalibrator`` (duck-typed; it may or may not
   expose ``translation_scale``/``set_translation_scale``).  It feeds the base
   a *virtual controller position* chosen so that the base's own relative
   law ``W0 + k_base (p_virtual - p_virtual0)`` equals the directional target,
   so all base gates (jump/re-anchor, workspace projection, fail-closed) and
   the downstream IK-residual gate keep applying unchanged.  The small
   calibration offset ``e0 = map(p_C0) - W0`` (< 5 cm by the gate) is blended
   out over ``OFFSET_BLEND_S`` so the first target is exactly W0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from teleop.utils.quest_safety import controller_sample_is_fresh


SIDES = ("left", "right")

# G1_29 geometry in the reduced Pinocchio pelvis-root frame (see
# controller_wrist_calibration.G1_29_SHOULDER_ORIGINS_M and the FK probe).
G1_29_SHOULDER_ORIGINS_M = {
    "left": np.array([-0.0000072, 0.10022, 0.29178]),
    "right": np.array([-0.0000072, -0.10021, 0.29178]),
}
G1_29_STRAIGHT_ARM_REACH_M = 0.424
# The workspace projects at 0.42 m.  Begin the feasibility transition only in
# its final 4 cm; below this radius orientation is exactly the controller's
# current 1:1 target.  The downstream EE rate limiter bounds every emitted
# angular step after this pure target policy.
EXTENSION_ORIENTATION_BLEND_START_M = 0.38
EXTENSION_ORIENTATION_BLEND_END_M = 0.42

DEFAULT_TRANSLATION_SCALE = 0.7
MIN_TRANSLATION_SCALE = 0.3
MAX_TRANSLATION_SCALE = 1.2

# Sweep acceptance.
SWEEP_DURATION_S = 3.0
SWEEP_MIN_SAMPLES = 30
MAX_FIT_RMS_M = 0.015
MIN_ARC_SPAN_RAD = math.radians(60.0)
MIN_HUMAN_ARM_LENGTH_M = 0.40
MAX_HUMAN_ARM_LENGTH_M = 0.85
# The arc must contain the hanging pose and reach near horizontal forward.
MAX_DOWN_ANGLE_RAD = math.radians(35.0)
MIN_TOP_ELEVATION_RAD = math.radians(-35.0)
MIN_PLANE_SPREAD_RATIO = 0.03
FIT_ITERATIONS = 20

# Plausible human shoulder pair (sphere centres).
MIN_SHOULDER_SEPARATION_M = 0.20
MAX_SHOULDER_SEPARATION_M = 0.60
MAX_SHOULDER_HEIGHT_DIFFERENCE_M = 0.10
MAX_LATERAL_AXIS_DISAGREEMENT_RAD = math.radians(30.0)
MAX_FORWARD_AXIS_DISAGREEMENT_RAD = math.radians(45.0)

# L-pose start gate and first-target blending.
L_POSE_TOLERANCE_M = 0.05
OFFSET_BLEND_S = 1.0

UP = np.array([0.0, 0.0, 1.0])


def _rigid(pose):
    try:
        matrix = np.asarray(pose, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    rotation = matrix[:3, :3]
    if not (
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        return None
    return matrix


def _rigid_pair(poses):
    try:
        if len(poses) != 2:
            return None
    except TypeError:
        return None
    pair = tuple(_rigid(pose) for pose in poses)
    return None if any(pose is None for pose in pair) else pair


def yaw_rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _angle_between(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm < 1e-12:
        return math.pi
    return math.acos(float(np.clip(first @ second / norm, -1.0, 1.0)))


# --------------------------------------------------------------------------
# Arc / sphere fit
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ArmFit:
    side: str
    accepted: bool
    reason: str
    sample_count: int
    shoulder: np.ndarray | None = None
    length_m: float | None = None
    rms_m: float | None = None
    arc_span_rad: float | None = None
    plane_normal: np.ndarray | None = None
    forward_direction: np.ndarray | None = None
    down_angle_rad: float | None = None
    top_elevation_rad: float | None = None
    scale: float | None = None

    def telemetry(self):
        def vec(value):
            return None if value is None else [float(x) for x in value]

        def num(value):
            return None if value is None else float(value)

        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "sample_count": int(self.sample_count),
            "shoulder_estimate_m": vec(self.shoulder),
            "human_arm_length_m": num(self.length_m),
            "fit_rms_m": num(self.rms_m),
            "arc_span_deg": None if self.arc_span_rad is None else math.degrees(self.arc_span_rad),
            "down_angle_deg": None if self.down_angle_rad is None else math.degrees(self.down_angle_rad),
            "top_elevation_deg": None if self.top_elevation_rad is None else math.degrees(self.top_elevation_rad),
            "plane_normal": vec(self.plane_normal),
            "translation_scale": num(self.scale),
        }


def _circle_fit_2d(points_2d):
    """Algebraic (Kasa) circle fit; returns (centre, radius) or None."""
    design = np.c_[2.0 * points_2d, np.ones(len(points_2d))]
    rhs = (points_2d ** 2).sum(axis=1)
    try:
        solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    centre = solution[:2]
    radius_squared = float(solution[2] + centre @ centre)
    if not np.all(np.isfinite(centre)) or not math.isfinite(radius_squared) or radius_squared <= 0.0:
        return None
    return centre, math.sqrt(radius_squared)


def _refine_3d_circle(points, origin, basis_u, basis_v, centre_2d, radius):
    """Gauss-Newton on d_i = |p_i - c| - r with c constrained to the plane."""
    params = np.array([centre_2d[0], centre_2d[1], radius], dtype=float)
    for _ in range(FIT_ITERATIONS):
        centre = origin + params[0] * basis_u + params[1] * basis_v
        offsets = points - centre
        distances = np.linalg.norm(offsets, axis=1)
        if np.any(distances < 1e-9):
            break
        residual = distances - params[2]
        unit = offsets / distances[:, None]
        jacobian = np.c_[-(unit @ basis_u), -(unit @ basis_v), -np.ones(len(points))]
        try:
            step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        params = params + step
        if float(np.linalg.norm(step)) < 1e-9:
            break
    centre = origin + params[0] * basis_u + params[1] * basis_v
    return centre, float(params[2])


def fit_arm_arc(points, side="left"):
    """Fit one straight-arm sweep; never raises, returns an :class:`ArmFit`."""
    try:
        points = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        return ArmFit(side, False, "invalid_samples", 0)
    if points.ndim != 2 or points.shape[1] != 3:
        return ArmFit(side, False, "invalid_samples", 0)
    count = int(points.shape[0])
    if not np.all(np.isfinite(points)):
        return ArmFit(side, False, "non_finite_samples", count)
    if count < SWEEP_MIN_SAMPLES:
        return ArmFit(side, False, "too_few_samples", count)

    origin = points.mean(axis=0)
    centred = points - origin
    _, singular, _ = np.linalg.svd(centred, full_matrices=False)
    if singular[0] < 1e-6 or singular[1] / singular[0] < MIN_PLANE_SPREAD_RATIO:
        return ArmFit(side, False, "degenerate_geometry", count)
    # A down->forward straight-arm sweep lies in a VERTICAL plane (it
    # contains gravity).  Constrain the plane normal to the horizontal: a free
    # PCA plane is tilted by lateral wobble on short arcs, and every degree of
    # tilt moves the fitted centre ~1 cm sideways.  A genuinely non-vertical
    # sweep shows up as a large RMS and is rejected.
    _, _, horizontal_vt = np.linalg.svd(centred[:, :2], full_matrices=False)
    basis_u = np.array([horizontal_vt[0, 0], horizontal_vt[0, 1], 0.0])
    basis_v = UP.copy()
    normal = np.cross(basis_u, basis_v)
    initial = _circle_fit_2d(np.c_[centred @ basis_u, centred @ basis_v])
    if initial is None:
        return ArmFit(side, False, "degenerate_geometry", count)
    shoulder, length = _refine_3d_circle(points, origin, basis_u, basis_v, *initial)
    if not np.all(np.isfinite(shoulder)) or not math.isfinite(length) or length <= 0.0:
        return ArmFit(side, False, "degenerate_geometry", count)

    offsets = points - shoulder
    distances = np.linalg.norm(offsets, axis=1)
    rms = float(np.sqrt(np.mean((distances - length) ** 2)))
    directions = offsets / np.maximum(distances, 1e-9)[:, None]

    # Circular span of the in-plane angles: 2 pi minus the largest gap.
    angles = np.sort(np.arctan2(directions @ basis_v, directions @ basis_u))
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * math.pi])
    span = float(2.0 * math.pi - gaps.max())

    elevation = np.arcsin(np.clip(directions @ UP, -1.0, 1.0))
    down_angle = float(math.pi / 2.0 + elevation.min())
    top_index = int(np.argmax(elevation))
    top_elevation = float(elevation[top_index])
    forward = directions[top_index].copy()
    forward[2] = 0.0
    forward_norm = float(np.linalg.norm(forward))
    forward = forward / forward_norm if forward_norm > 1e-6 else None

    common = dict(
        sample_count=count,
        shoulder=shoulder,
        length_m=length,
        rms_m=rms,
        arc_span_rad=span,
        plane_normal=normal,
        forward_direction=forward,
        down_angle_rad=down_angle,
        top_elevation_rad=top_elevation,
    )
    if not MIN_HUMAN_ARM_LENGTH_M <= length <= MAX_HUMAN_ARM_LENGTH_M:
        return ArmFit(side, False, "implausible_arm_length", **common)
    if rms > MAX_FIT_RMS_M:
        return ArmFit(side, False, "fit_rms_too_high", **common)
    if span < MIN_ARC_SPAN_RAD:
        return ArmFit(side, False, "arc_too_short", **common)
    if down_angle > MAX_DOWN_ANGLE_RAD:
        return ArmFit(side, False, "arc_missing_down_pose", **common)
    if top_elevation < MIN_TOP_ELEVATION_RAD or forward is None:
        return ArmFit(side, False, "arc_missing_forward_pose", **common)
    scale = float(np.clip(G1_29_STRAIGHT_ARM_REACH_M / length, MIN_TRANSLATION_SCALE, MAX_TRANSLATION_SCALE))
    return ArmFit(side, True, "accepted", scale=scale, **common)


# --------------------------------------------------------------------------
# Two-arm calibration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HumanArmCalibration:
    accepted: bool
    reason: str
    fits: dict = field(default_factory=dict)
    yaw_rad: float | None = None
    shoulder_separation_m: float | None = None

    @property
    def rotation(self):
        return yaw_rotation(-self.yaw_rad) if self.yaw_rad is not None else None

    def scale(self, side):
        return self.fits[side].scale

    def shoulder(self, side):
        return self.fits[side].shoulder

    def telemetry(self):
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "robot_straight_arm_reach_m": G1_29_STRAIGHT_ARM_REACH_M,
            "body_yaw_deg": None if self.yaw_rad is None else math.degrees(self.yaw_rad),
            "shoulder_separation_m": self.shoulder_separation_m,
            "sides": {side: fit.telemetry() for side, fit in self.fits.items()},
        }


def calibrate_human_arms(left_points, right_points):
    fits = {"left": fit_arm_arc(left_points, "left"), "right": fit_arm_arc(right_points, "right")}
    for side in SIDES:
        if not fits[side].accepted:
            return HumanArmCalibration(False, f"{side}_{fits[side].reason}", fits)

    left, right = fits["left"], fits["right"]
    lateral = left.shoulder - right.shoulder
    lateral_horizontal = np.array([lateral[0], lateral[1], 0.0])
    separation = float(np.linalg.norm(lateral_horizontal))
    if not MIN_SHOULDER_SEPARATION_M <= separation <= MAX_SHOULDER_SEPARATION_M:
        return HumanArmCalibration(False, "implausible_shoulder_separation", fits, shoulder_separation_m=separation)
    if abs(float(lateral[2])) > MAX_SHOULDER_HEIGHT_DIFFERENCE_M:
        return HumanArmCalibration(False, "implausible_shoulder_height", fits, shoulder_separation_m=separation)

    normals = []
    for fit in (left, right):
        normal = fit.plane_normal if fit.plane_normal @ lateral >= 0.0 else -fit.plane_normal
        normals.append(normal)
    body_y = normals[0] + normals[1]
    body_y[2] = 0.0
    if np.linalg.norm(body_y) < 1e-6:
        return HumanArmCalibration(False, "degenerate_body_axis", fits, shoulder_separation_m=separation)
    body_y /= np.linalg.norm(body_y)
    if _angle_between(body_y, lateral_horizontal) > MAX_LATERAL_AXIS_DISAGREEMENT_RAD:
        return HumanArmCalibration(False, "body_axes_disagree", fits, shoulder_separation_m=separation)
    body_x = np.cross(body_y, UP)
    swept_forward = left.forward_direction + right.forward_direction
    if _angle_between(body_x, swept_forward) > MAX_FORWARD_AXIS_DISAGREEMENT_RAD:
        return HumanArmCalibration(False, "forward_axis_disagrees", fits, shoulder_separation_m=separation)
    yaw = math.atan2(float(body_x[1]), float(body_x[0]))
    return HumanArmCalibration(True, "accepted", fits, yaw_rad=yaw, shoulder_separation_m=separation)


def map_controller_position(calibration, side, controller_position):
    """``S_robot + k R_align (p_C - S_h)`` in the pelvis-root robot frame."""
    fit = calibration.fits[side]
    return G1_29_SHOULDER_ORIGINS_M[side] + fit.scale * (
        calibration.rotation @ (np.asarray(controller_position, dtype=float) - fit.shoulder)
    )


# --------------------------------------------------------------------------
# L-pose gate
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LPoseGateDecision:
    accepted: bool
    reason: str
    errors_m: dict = field(default_factory=dict)
    tolerance_m: float = L_POSE_TOLERANCE_M

    def telemetry(self):
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "tolerance_m": float(self.tolerance_m),
            "errors_m": {side: (None if value is None else float(value)) for side, value in self.errors_m.items()},
        }

    def message(self):
        errors = ", ".join(
            f"{side} {value * 100:.1f} cm" for side, value in self.errors_m.items() if value is not None
        )
        return f"{self.reason} ({errors}; limit {self.tolerance_m * 100:.0f} cm)" if errors else self.reason


def evaluate_l_pose_gate(calibration, controller_poses, measured_wrist_poses, tolerance_m=L_POSE_TOLERANCE_M):
    if calibration is None or not calibration.accepted:
        return LPoseGateDecision(True, "no_human_calibration", tolerance_m=tolerance_m)
    controllers = _rigid_pair(controller_poses)
    wrists = _rigid_pair(measured_wrist_poses)
    if controllers is None or wrists is None:
        return LPoseGateDecision(False, "invalid_pose_sample", tolerance_m=tolerance_m)
    errors = {}
    for index, side in enumerate(SIDES):
        mapped = map_controller_position(calibration, side, controllers[index][:3, 3])
        errors[side] = float(np.linalg.norm(mapped - wrists[index][:3, 3]))
    if all(error < tolerance_m for error in errors.values()):
        return LPoseGateDecision(True, "operator_in_l_pose", errors, tolerance_m)
    return LPoseGateDecision(False, "operator_not_in_l_pose", errors, tolerance_m)


# --------------------------------------------------------------------------
# Sweep session (pre-arm only)
# --------------------------------------------------------------------------

class HumanArmSweep:
    """Collect one pre-arm sweep; pure state machine driven by the caller.

    ``request`` returns False (and does nothing) while tracking is active or
    before preparation.  ``observe`` must be called with each pre-arm
    TeleData; it never commands anything.  ``abort`` is used on start/stop.
    """

    def __init__(self, duration_s=SWEEP_DURATION_S):
        self.duration_s = float(duration_s)
        self.active = False
        self._start = 0.0
        self._last_timestamp = 0.0
        self._points = {side: [] for side in SIDES}
        self.stale_or_invalid_count = 0

    def request(self, now, *, preparation_ready, tracking_active):
        if not preparation_ready or tracking_active or self.active:
            return False
        self.active = True
        self._start = float(now)
        self._last_timestamp = float(now)
        self._points = {side: [] for side in SIDES}
        self.stale_or_invalid_count = 0
        return True

    def abort(self):
        was_active = self.active
        self.active = False
        self._points = {side: [] for side in SIDES}
        return was_active

    def observe(self, tele_data, now, *, tracking_active):
        """Return a :class:`HumanArmCalibration` when the sweep ends, else None."""
        if not self.active:
            return None
        if tracking_active:
            self.abort()
            return HumanArmCalibration(False, "aborted_tracking_active")
        timestamp = getattr(tele_data, "controller_sample_timestamp", 0.0)
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            timestamp = 0.0
        if (
            math.isfinite(timestamp)
            and timestamp > self._last_timestamp
            and controller_sample_is_fresh(timestamp, now)
        ):
            pair = _rigid_pair((getattr(tele_data, "left_wrist_pose", None), getattr(tele_data, "right_wrist_pose", None)))
            if pair is not None:
                self._last_timestamp = timestamp
                for side, pose in zip(SIDES, pair):
                    self._points[side].append(pose[:3, 3].copy())
            else:
                self.stale_or_invalid_count += 1
        else:
            self.stale_or_invalid_count += 1
        if float(now) - self._start < self.duration_s:
            return None
        points = self._points
        self.active = False
        self._points = {side: [] for side in SIDES}
        return calibrate_human_arms(np.array(points["left"]).reshape(-1, 3), np.array(points["right"]).reshape(-1, 3))


# --------------------------------------------------------------------------
# Directional target provider wrapping the base calibrator
# --------------------------------------------------------------------------

def _orthonormalize_rotation(rotation):
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _rotation_log(rotation):
    """Return the axis/angle for a finite rotation, including angles near pi."""
    skew_axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    sine_twice = float(np.linalg.norm(skew_axis))
    angle = math.atan2(sine_twice, float(np.trace(rotation) - 1.0))
    if angle < 1e-9 or not math.isfinite(angle):
        return np.array([1.0, 0.0, 0.0]), 0.0
    if math.pi - angle < 1e-3:
        eigenvalues, eigenvectors = np.linalg.eigh((rotation + rotation.T) / 2.0)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if float(axis @ skew_axis) < 0.0:
            axis = -axis
        return axis / np.linalg.norm(axis), angle
    return skew_axis / sine_twice, angle


def _rotation_exp(axis, angle):
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _blend_orientation(current, anchor, blend):
    """Geodesically move ``current`` toward ``anchor`` by ``blend`` in SO(3)."""
    if blend <= 0.0:
        return current
    axis, angle = _rotation_log(current.T @ anchor)
    return _orthonormalize_rotation(current @ _rotation_exp(axis, angle * blend))


class HumanCalibratedWristCalibrator:
    """Duck-typed wrapper adding the human-calibrated directional mapping.

    Without an accepted human calibration every call is delegated verbatim.
    With one, the base receives a virtual controller pose (same rotation,
    virtual position) such that its own relative translation law yields
    ``map(p_C) - blend(t) * e0``.  Only the base's public
    ``calibrate/targets/reset_for_start_request/consume_first_target`` and the
    optional ``translation_scale``/``set_translation_scale`` API are used.
    """

    def __init__(self, base):
        object.__setattr__(self, "_base", base)
        # Fixed scale the base was constructed with (0.7 in the scale branch);
        # restored whenever no human calibration is installed.
        self._default_scale = getattr(base, "translation_scale", None)
        self._human = None
        self._active = None
        self.last_gate_decision = None

    def __getattr__(self, name):
        # Telemetry counters etc. of the base (only reached for missing attrs).
        base = self.__dict__.get("_base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    @property
    def base(self):
        return self._base

    @property
    def calibrated(self):
        return bool(getattr(self._base, "calibrated", False))

    @property
    def human_calibration(self):
        return self._human

    def set_human_calibration(self, calibration):
        """Install (or clear with None) a calibration; only before calibrate()."""
        if self.calibrated:
            raise RuntimeError("human calibration can only change before calibrate()")
        if calibration is not None and not calibration.accepted:
            calibration = None
        self._human = calibration
        self._active = None

    def reset_for_start_request(self, request_timestamp):
        self._active = None
        self.last_gate_decision = None
        self._base.reset_for_start_request(request_timestamp)

    def consume_first_target(self):
        return self._base.consume_first_target()

    def check_l_pose(self, controller_poses, measured_wrist_poses):
        decision = evaluate_l_pose_gate(self._human, controller_poses, measured_wrist_poses)
        self.last_gate_decision = decision
        return decision

    def calibrate(self, controller_poses, measured_wrist_poses, sample_timestamp, request_timestamp, now=None):
        self._active = None
        if self._human is None:
            self._set_base_scale(self._default_scale)
            return self._base.calibrate(controller_poses, measured_wrist_poses, sample_timestamp, request_timestamp, now=now)
        decision = self.check_l_pose(controller_poses, measured_wrist_poses)
        if not decision.accepted:
            self._base.reset_for_start_request(request_timestamp)
            return False
        controllers = _rigid_pair(controller_poses)
        wrists = _rigid_pair(measured_wrist_poses)
        base_scale = self._configure_base_scale()
        active = {
            "base_scale": base_scale,
            "calibration_timestamp": float(sample_timestamp),
            "offsets": [],
            "orientation_anchors": [],
        }
        for index, side in enumerate(SIDES):
            mapped = map_controller_position(self._human, side, controllers[index][:3, 3])
            active["offsets"].append(mapped - wrists[index][:3, 3])
            active["orientation_anchors"].append(wrists[index][:3, :3].copy())
        self._active = active
        virtual = self._virtual_poses(controllers, sample_timestamp)
        calibrated = self._base.calibrate(virtual, measured_wrist_poses, sample_timestamp, request_timestamp, now=now)
        if not calibrated:
            self._active = None
        return calibrated

    def targets(self, controller_poses, sample_timestamp, now=None):
        if self._active is None:
            return self._base.targets(controller_poses, sample_timestamp, now=now)
        controllers = _rigid_pair(controller_poses)
        if controllers is None:
            # Let the base record its own invalid-pose rejection.
            return self._base.targets(controller_poses, sample_timestamp, now=now)
        targets = self._base.targets(self._virtual_poses(controllers, sample_timestamp), sample_timestamp, now=now)
        return self._apply_extension_orientation_policy(targets)

    def mapped_positions(self, controller_poses, sample_timestamp):
        """Directional robot-frame wrist positions (after offset blend)."""
        if self._active is None:
            return None
        controllers = _rigid_pair(controller_poses)
        if controllers is None:
            return None
        blend = self._blend(sample_timestamp)
        return tuple(
            map_controller_position(self._human, side, controllers[index][:3, 3])
            - blend * self._active["offsets"][index]
            for index, side in enumerate(SIDES)
        )

    def _set_base_scale(self, scale):
        setter = getattr(self._base, "set_translation_scale", None)
        if scale is None or not callable(setter):
            return
        try:
            setter(float(np.clip(scale, MIN_TRANSLATION_SCALE, MAX_TRANSLATION_SCALE)))
        except (RuntimeError, ValueError, TypeError):
            pass

    def _configure_base_scale(self):
        # The base scale only sets the units of the virtual controller stream
        # (its jump limits then act on ~controller-sized deltas); the mapped
        # target itself is independent of it.
        self._set_base_scale(float(np.mean([self._human.scale(side) for side in SIDES])))
        scale = getattr(self._base, "translation_scale", 1.0)
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        return scale if math.isfinite(scale) and scale > 0.0 else 1.0

    def _blend(self, sample_timestamp):
        try:
            elapsed = float(sample_timestamp) - self._active["calibration_timestamp"]
        except (TypeError, ValueError):
            elapsed = 0.0
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            return 1.0
        return max(0.0, 1.0 - elapsed / OFFSET_BLEND_S)

    def _virtual_poses(self, controllers, sample_timestamp):
        # Base law: target = W0 + k_b (v - v0).  With
        #   v = (map(p) + (1 - blend) e0) / k_b   and   v0 = map(p0) / k_b
        # (blend = 1 at the calibration sample) it yields
        #   W0 + map(p) - map(p0) + (1 - blend) e0 = map(p) - blend * e0.
        # Rotation is passed through unchanged (orientation stays 1:1).
        base_scale = self._active["base_scale"]
        blend = self._blend(sample_timestamp)
        virtual = []
        for index, side in enumerate(SIDES):
            mapped = map_controller_position(self._human, side, controllers[index][:3, 3])
            pose = controllers[index].copy()
            pose[:3, 3] = (mapped + (1.0 - blend) * self._active["offsets"][index]) / base_scale
            virtual.append(pose)
        return tuple(virtual)

    def _apply_extension_orientation_policy(self, targets):
        """Blend only the extension boundary toward a measured feasible wrist.

        ``targets`` has already passed the base calibrator's SE(3), freshness,
        discontinuity, and workspace gates.  It is deliberately not repaired:
        any invalid result remains ``None`` and the normal fail-closed path
        applies.  The outer EE limiter subsequently limits angular steps.
        """
        if targets is None or self._active is None:
            return targets
        targets = _rigid_pair(targets)
        if targets is None:
            return None
        adjusted = []
        span = EXTENSION_ORIENTATION_BLEND_END_M - EXTENSION_ORIENTATION_BLEND_START_M
        for index, (side, target) in enumerate(zip(SIDES, targets)):
            reach = float(np.linalg.norm(target[:3, 3] - G1_29_SHOULDER_ORIGINS_M[side]))
            blend = float(np.clip((reach - EXTENSION_ORIENTATION_BLEND_START_M) / span, 0.0, 1.0))
            # Smoothstep has zero slope at both boundaries: no orientation kink
            # when entering/exiting the feasibility zone.
            blend = blend * blend * (3.0 - 2.0 * blend)
            pose = target.copy()
            pose[:3, :3] = _blend_orientation(
                pose[:3, :3], self._active["orientation_anchors"][index], blend
            )
            adjusted.append(pose)
        return tuple(adjusted) if _rigid_pair(adjusted) is not None else None
