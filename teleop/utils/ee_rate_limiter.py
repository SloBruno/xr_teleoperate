"""Three-band Cartesian rate limiter for the dual G1 wrist targets.

Ported (reimplemented, numpy only) from NVIDIA IsaacTeleop
``src/python/isaaccapture/retargeters/rate_limiter.py`` --
``RateLimiterConfig`` / ``EePoseRateLimiter._compute_fn`` /
``_clamp_position_step`` / ``_clamp_orientation_step`` / ``_clamped_dt``
(Apache-2.0, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES).  The
retargeting-engine plumbing and quaternion representation were replaced by
4x4 SE(3) matrices and axis-angle (log/exp) geodesic steps.

Bands, per side, relative to the last COMMITTED (= emitted and published)
target, never the last input:

a) PASS: step within ``max_velocity * dt`` -> the input target is returned
   unchanged (zero lag; this replaces the 4-tap WeightedMovingFilter whose
   group delay was ~1.5 control cycles);
b) CLAMP: larger step -> linear step clamped along the straight line and
   angular step clamped along the geodesic (axis-angle), so a persistent far
   target is approached at bounded speed;
c) REJECT_HOLD: the INPUT moved faster than ``reject_*`` relative to the last
   accepted input (IK/tracking discontinuity) -> hold the last committed
   target; after ``max_consecutive_rejections`` consecutive anomalous frames
   the input is REACCEPTED as a new regime and approached through band (b).

Coordination with ``ControllerWristCalibrator``: controller-sample
teleports (>0.15 m / >90 deg between consecutive samples) are already
rejected upstream and re-anchored, so they never reach this limiter.  The
reject tier here is deliberately set ABOVE that envelope so it only catches
discontinuities in the calibrated target itself; it does not re-anchor.

dt is the monotonic time since the last commit, clamped to [min_dt, max_dt];
a missing, non-finite, duplicate or backwards timestamp uses ``nominal_dt``.
Non-finite / non-rigid / malformed targets fail closed (``None``) without
touching any state.

Two-phase use: ``limit()`` proposes; the caller runs IK and the residual
gate and calls ``commit()`` only when the proposal was actually published,
so a rejected IK solution never becomes the rate-limit reference.
"""

from dataclasses import dataclass
import math

import numpy as np


BAND_FIRST = "first"
BAND_PASS = "pass"
BAND_CLAMP = "clamp"
BAND_REJECT_HOLD = "reject_hold"
BAND_REACCEPT = "reaccept"

_MIN_ANGLE_RAD = 1e-9


@dataclass(frozen=True)
class EeRateLimiterConfig:
    """Limits; defaults are the NVIDIA ``RateLimiterConfig`` defaults."""

    max_linear_velocity: float = 0.25
    max_angular_velocity: float = 1.5
    nominal_dt: float = 1.0 / 60.0
    max_dt: float = 0.1
    min_dt: float = 1e-4
    reject_linear_velocity: float | None = None
    reject_angular_velocity: float | None = None
    max_consecutive_rejections: int | None = 30

    def __post_init__(self):
        for name in ("max_linear_velocity", "max_angular_velocity", "nominal_dt", "max_dt", "min_dt"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not self.min_dt <= self.nominal_dt <= self.max_dt:
            raise ValueError("require 0 < min_dt <= nominal_dt <= max_dt")
        for reject_name, clamp_name in (
            ("reject_linear_velocity", "max_linear_velocity"),
            ("reject_angular_velocity", "max_angular_velocity"),
        ):
            reject = getattr(self, reject_name)
            if reject is None:
                continue
            if not math.isfinite(reject) or reject <= 0.0:
                raise ValueError(f"{reject_name} must be finite and > 0")
            if reject < getattr(self, clamp_name):
                raise ValueError(f"require {reject_name} >= {clamp_name}")
        if self.max_consecutive_rejections is not None and self.max_consecutive_rejections < 1:
            raise ValueError("max_consecutive_rejections must be >= 1 or None")


# Reviewed G1_29 parameters (tuned by inert replay of real sessions, see
# commit message): NVIDIA's 0.25 m/s / 1.5 rad/s were sized for the SO-101
# and would themselves add tens of mm of lag to human-speed wrist motion.
G1_29_EE_RATE_LIMITER_CONFIG = EeRateLimiterConfig(
    max_linear_velocity=1.0,
    max_angular_velocity=6.0,
    nominal_dt=1.0 / 30.0,
    max_dt=0.12,
    min_dt=0.02,
    reject_linear_velocity=3.0,
    reject_angular_velocity=25.0,
    max_consecutive_rejections=5,
)


@dataclass(frozen=True)
class EeRateLimitResult:
    targets: tuple
    bands: tuple
    dt: float


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


def _orthonormalize(rotation):
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def rotation_log(rotation):
    """(unit axis, angle in [0, pi]) of a rotation matrix; robust near 0 and pi."""
    skew_axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    sine_twice = float(np.linalg.norm(skew_axis))
    angle = math.atan2(sine_twice, float(np.trace(rotation) - 1.0))
    if not math.isfinite(angle) or angle < _MIN_ANGLE_RAD:
        return np.array([1.0, 0.0, 0.0]), 0.0
    if math.pi - angle < 1e-3:
        eigenvalues, eigenvectors = np.linalg.eigh((rotation + rotation.T) / 2.0)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if float(axis @ skew_axis) < 0.0:
            axis = -axis
        return axis / np.linalg.norm(axis), angle
    return skew_axis / sine_twice, angle


def rotation_exp(axis, angle):
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _clamp_position_step(target, previous, max_step):
    delta = target - previous
    distance = float(np.linalg.norm(delta))
    if distance <= max_step or distance == 0.0:
        return target.copy(), False
    return previous + delta * (max_step / distance), True


def _clamp_rotation_step(target, previous, max_step):
    """Geodesic (slerp-equivalent) step from ``previous`` toward ``target``."""
    axis, angle = rotation_log(previous.T @ target)
    if angle <= max_step:
        return target.copy(), False
    return _orthonormalize(previous @ rotation_exp(axis, max_step)), True


def _finite_time(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class DualEePoseRateLimiter:
    """Three-band governor for the (left, right) wrist SE(3) target pair."""

    def __init__(self, config=None):
        self._cfg = config if config is not None else EeRateLimiterConfig()
        self.reset()

    @property
    def config(self):
        return self._cfg

    @property
    def last_committed_targets(self):
        """Copy of the last emitted (committed) target pair, or ``None``."""
        if self._last_targets is None:
            return None
        return tuple(t.copy() for t in self._last_targets)

    def reset(self):
        """Forget the baseline: the next valid frame passes through and latches."""
        self._last_targets = None
        self._last_time = None
        self._last_inputs = None
        self._rejections = [0, 0]
        self._pending = None
        self._hold_runs = [0, 0]
        self.max_hold_run = 0
        self.band_counts = {}

    def _dt(self, now):
        now = _finite_time(now)
        if now is None or self._last_time is None:
            return self._cfg.nominal_dt, now
        delta = now - self._last_time
        if delta <= 0.0:
            return self._cfg.nominal_dt, now
        return min(max(delta, self._cfg.min_dt), self._cfg.max_dt), now

    def _is_anomalous(self, side, target, dt):
        previous = self._last_inputs[side]
        cfg = self._cfg
        if cfg.reject_linear_velocity is not None and (
            float(np.linalg.norm(target[:3, 3] - previous[:3, 3])) > cfg.reject_linear_velocity * dt
        ):
            return True
        if cfg.reject_angular_velocity is not None:
            _, angle = rotation_log(previous[:3, :3].T @ target[:3, :3])
            if angle > cfg.reject_angular_velocity * dt:
                return True
        return False

    def limit(self, targets, now):
        """Propose the limited target pair, or ``None`` for invalid input."""
        self._pending = None
        try:
            if targets is None or len(targets) != 2 or not all(_is_rigid_se3(t) for t in targets):
                return None
        except TypeError:
            return None
        targets = tuple(np.asarray(t, dtype=float).copy() for t in targets)

        if self._last_targets is None:
            self._pending = (targets, _finite_time(now), targets, [0, 0], (BAND_FIRST, BAND_FIRST))
            return EeRateLimitResult(tuple(t.copy() for t in targets), (BAND_FIRST, BAND_FIRST), self._cfg.nominal_dt)

        dt, now_value = self._dt(now)
        limited, bands = [], []
        inputs = list(self._last_inputs)
        rejections = list(self._rejections)
        for side in (0, 1):
            target = targets[side]
            last = self._last_targets[side]
            band = None
            if self._is_anomalous(side, target, dt):
                rejections[side] += 1
                cap = self._cfg.max_consecutive_rejections
                if cap is None or rejections[side] <= cap:
                    limited.append(last.copy())
                    bands.append(BAND_REJECT_HOLD)
                    continue
                band = BAND_REACCEPT
            rejections[side] = 0
            position, position_clamped = _clamp_position_step(
                target[:3, 3], last[:3, 3], self._cfg.max_linear_velocity * dt
            )
            rotation, rotation_clamped = _clamp_rotation_step(
                target[:3, :3], last[:3, :3], self._cfg.max_angular_velocity * dt
            )
            out = np.eye(4)
            out[:3, :3] = rotation
            out[:3, 3] = position
            if not np.all(np.isfinite(out)):
                return None
            limited.append(out)
            inputs[side] = target
            if band is None:
                band = BAND_CLAMP if (position_clamped or rotation_clamped) else BAND_PASS
            bands.append(band)
        limited = tuple(limited)
        bands = tuple(bands)
        self._pending = (limited, now_value, tuple(inputs), rejections, bands)
        return EeRateLimitResult(tuple(t.copy() for t in limited), bands, dt)

    def commit(self):
        """Adopt the last proposal as the emitted reference (after publication)."""
        if self._pending is None:
            return False
        limited, now_value, inputs, rejections, bands = self._pending
        self._pending = None
        self._last_targets = tuple(t.copy() for t in limited)
        if now_value is not None:
            self._last_time = now_value
        self._last_inputs = tuple(t.copy() for t in inputs)
        self._rejections = list(rejections)
        for side, band in enumerate(bands):
            self.band_counts[band] = self.band_counts.get(band, 0) + 1
            self._hold_runs[side] = self._hold_runs[side] + 1 if band == BAND_REJECT_HOLD else 0
            self.max_hold_run = max(self.max_hold_run, self._hold_runs[side])
        return True
