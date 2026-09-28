"""Arm enable ramp: ``blend_ratio`` 0 -> 1 between the held pose and IK.

Port of the Isaac ROS G1 teleop ``safety_controller`` ``blend_ratio`` enable
ramp (NVIDIA GR00T real-teleop workflow; see
/home/bruno/nvidia_teleop_study/RESUMO.md refs [4][5] -- the ROS node source
is not vendored, so this is a behavioural reimplementation).

At arming (first published tracking command after ``r``/A) the joint target
is blended ``q = (1 - a) * q_hold + a * q_ik`` with a smoothstep ``a(t)``
over ``duration_s``; feed-forward torque is scaled by ``a`` so the first
command equals the frozen measured hold (zero feed-forward) exactly.

Contract:
* ``begin`` is idempotent: a repeated start while the ramp runs or after it
  completed returns ``False`` and never restarts the ramp;
* ``interrupt`` (STOP/q) ends the ramp immediately and permanently for this
  session: ``apply`` then returns ``None`` so the caller falls back to the
  lifecycle stop hold -- the ramp never continues after STOP;
* invalid time never advances ``a`` and time going backwards never lowers it;
* non-finite / wrongly shaped vectors fail closed (``None``).

This only blends the COMMAND TARGET.  The arm_sdk authority weight
(kNotUsedJoint0.q) and the disarm return-to-home belong to
feat/graceful-shutdown-release.
"""

import math

import numpy as np


DEFAULT_ENABLE_RAMP_S = 0.8
MAX_ENABLE_RAMP_S = 3.0


def _finite_time(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _vector(value, size=None):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        return None
    if size is not None and array.size != size:
        return None
    return array.copy()


class ArmEnableRamp:
    def __init__(self, duration_s=DEFAULT_ENABLE_RAMP_S, joint_count=14):
        if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
            raise ValueError("duration_s must be a finite number")
        if not math.isfinite(duration_s) or not 0.0 < duration_s <= MAX_ENABLE_RAMP_S:
            raise ValueError(f"duration_s must be within (0, {MAX_ENABLE_RAMP_S}]")
        if isinstance(joint_count, bool) or not isinstance(joint_count, int) or joint_count <= 0:
            raise ValueError("joint_count must be a positive integer")
        self.duration_s = float(duration_s)
        self.joint_count = joint_count
        self._start = None
        self._hold = None
        self._alpha = 0.0
        self.interrupted = False

    @property
    def started(self):
        return self._start is not None

    @property
    def complete(self):
        return self.started and not self.interrupted and self._alpha >= 1.0

    @property
    def alpha(self):
        return self._alpha

    def begin(self, now, hold_q):
        """Start the ramp once from ``hold_q``; repeated calls are no-ops."""
        if self.interrupted or self.started:
            return False
        now = _finite_time(now)
        hold = _vector(hold_q, self.joint_count)
        if now is None or hold is None:
            return False
        self._start = now
        self._hold = hold
        self._alpha = 0.0
        return True

    def interrupt(self):
        self.interrupted = True

    def _advance(self, now):
        now = _finite_time(now)
        if now is None:
            return self._alpha
        s = min(max((now - self._start) / self.duration_s, 0.0), 1.0)
        alpha = s * s * (3.0 - 2.0 * s)  # smoothstep: zero slope at both ends
        self._alpha = max(self._alpha, alpha)
        return self._alpha

    def apply(self, q_target, tauff_target, now):
        """Return ``(q, tauff, alpha)`` or ``None`` (fail closed)."""
        if self.interrupted or not self.started:
            return None
        q = _vector(q_target, self._hold.size)
        tau = _vector(tauff_target, self._hold.size)
        if q is None or tau is None:
            return None
        alpha = self._advance(now)
        if alpha >= 1.0:
            return q, tau, 1.0
        return (1.0 - alpha) * self._hold + alpha * q, alpha * tau, alpha
