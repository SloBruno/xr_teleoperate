"""Lifecycle-serialized arm command publication."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArmCommandDecision:
    """The exact command selected under the lifecycle lock."""

    published: bool
    hold: bool
    selected_q: np.ndarray
    selected_tauff: np.ndarray
    publication: object = None


_HOLD_ATTRIBUTE = "_xr_teleop_frozen_hold_q"
_LAST_PUBLISHED_ATTRIBUTE = "_xr_teleop_last_published_q"

# Gravity feed-forward on a hold is computed only from a lowstate sample at
# most this old (same bound as the graceful-shutdown state check).
HOLD_GRAVITY_STATE_MAX_AGE = 0.25


def _clear_frozen_hold(arm_ctrl):
    setattr(arm_ctrl, _HOLD_ATTRIBUTE, None)


def _clear_tracking_reference(arm_ctrl):
    _clear_frozen_hold(arm_ctrl)
    setattr(arm_ctrl, _LAST_PUBLISHED_ATTRIBUTE, None)


def _frozen_hold(arm_ctrl, prefer_last_published=False):
    """Freeze ONE hold reference per hold run; never resampled inside a run.

    Legacy (no gravity model): the measured q at hold entry.  Gravity hold
    (C1): the last q actually handed to the writer by an accepted tracking
    command, so the arm does not step back by its tracking lag; measured q is
    only the fallback before the first accepted publication of this epoch.
    """
    hold_q = getattr(arm_ctrl, _HOLD_ATTRIBUTE, None)
    if hold_q is None:
        last_published = getattr(arm_ctrl, _LAST_PUBLISHED_ATTRIBUTE, None) if prefer_last_published else None
        measured_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=float).copy()
        if measured_q.ndim != 1 or measured_q.size == 0 or not np.all(np.isfinite(measured_q)):
            raise ValueError("measured arm state is not a finite joint vector")
        if (
            last_published is not None
            and np.shape(last_published) == measured_q.shape
            and np.all(np.isfinite(last_published))
        ):
            hold_q = np.asarray(last_published, dtype=float).copy()
        else:
            hold_q = measured_q
        setattr(arm_ctrl, _HOLD_ATTRIBUTE, hold_q.copy())
    return np.asarray(hold_q, dtype=float).copy()


def _fresh_measured_q_for_gravity(arm_ctrl, measured_q):
    """Measured q for the gravity model, or None when stale/invalid."""
    snapshot = getattr(arm_ctrl, "get_dual_arm_q_snapshot", None)
    if snapshot is None:
        return measured_q
    try:
        q, age = snapshot()
        q = np.asarray(q, dtype=float).copy()
        age = float(age)
    except Exception:
        return None
    if not np.isfinite(age) or age < 0.0 or age > HOLD_GRAVITY_STATE_MAX_AGE:
        return None
    if q.shape != measured_q.shape or not np.all(np.isfinite(q)):
        return None
    return q


def _hold_tauff(arm_ctrl, measured_q, gravity_tauff, tau_limit, scale):
    """Static gravity feed-forward at MEASURED q; zero whenever unavailable.

    Bounded per joint by ``tau_limit`` (the model effort limits) and scaled
    by ``scale`` (the enable-ramp feed-forward factor while it is running).
    Any model failure, stale/non-finite state or shape mismatch gives zero
    torque -- the previous hold behaviour -- never a guessed torque.
    """
    zeros = np.zeros_like(measured_q)
    if gravity_tauff is None:
        return zeros
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        return zeros
    if not np.isfinite(scale):
        return zeros
    scale = min(max(scale, 0.0), 1.0)
    q = _fresh_measured_q_for_gravity(arm_ctrl, measured_q)
    if q is None:
        return zeros
    try:
        tau = np.asarray(gravity_tauff(q), dtype=float).copy()
    except Exception:
        return zeros
    if tau.shape != measured_q.shape or not np.all(np.isfinite(tau)):
        return zeros
    if tau_limit is not None:
        try:
            limit = np.abs(np.asarray(tau_limit, dtype=float))
        except (TypeError, ValueError):
            return zeros
        if limit.shape != tau.shape or not np.all(np.isfinite(limit)):
            return zeros
        tau = np.clip(tau, -limit, limit)
    return scale * tau


def _command_is_finite(q_target, tauff_target, expected_shape=None):
    try:
        q = np.asarray(q_target, dtype=float)
        tau = np.asarray(tauff_target, dtype=float)
    except (TypeError, ValueError):
        return False
    return (
        q.ndim == 1
        and q.size > 0
        and (expected_shape is None or q.shape == expected_shape)
        and tau.shape == q.shape
        and np.all(np.isfinite(q))
        and np.all(np.isfinite(tau))
    )


def publish_arm_command(
    arm_ctrl,
    q_target,
    tauff_target,
    *,
    target_accepted,
    sample_fresh,
    lifecycle_lock,
    is_started,
    is_stopped,
    hold_gravity_tauff=None,
    hold_tau_limit=None,
    hold_gravity_scale=1.0,
):
    """Select and publish one command while retaining the exact hold decision.

    ``hold_gravity_tauff`` (q -> tau, e.g. ``G1_29_ArmIK.gravity_tauff``)
    enables the gravity hold: every tracking hold commands the last published
    q (frozen once per hold run) with gravity feed-forward at measured q.
    Without it the legacy measured-q / zero-torque hold is unchanged.  STOP,
    not-started and invalid-measured-state branches are identical in both.
    """
    with lifecycle_lock:
        try:
            measured_q = np.asarray(
                arm_ctrl.get_current_dual_arm_q(), dtype=float
            ).copy()
        except (AttributeError, TypeError, ValueError):
            measured_q = np.array([], dtype=float)
        measured_is_valid = measured_q.ndim == 1 and measured_q.size > 0 and np.all(np.isfinite(measured_q))
        if is_stopped():
            # STOP is terminal for tracking: enqueue nothing (no IK target, no
            # re-sampled hold). The writer keeps its last command until the
            # graceful shutdown sequence takes ownership, returns the arms
            # home, releases arm_sdk authority, and deactivates the writer.
            _clear_tracking_reference(arm_ctrl)
            if not measured_is_valid:
                return ArmCommandDecision(
                    False, True, measured_q, np.array([], dtype=float), None
                )
            return ArmCommandDecision(False, True, measured_q, np.zeros_like(measured_q), None)
        if not is_started():
            if not measured_is_valid:
                return ArmCommandDecision(
                    False, True, measured_q, np.array([], dtype=float), None
                )
            selected_tauff = np.zeros_like(measured_q)
            _clear_tracking_reference(arm_ctrl)
            publication = arm_ctrl.ctrl_dual_arm(measured_q, selected_tauff)
            return ArmCommandDecision(False, True, measured_q, selected_tauff, publication)
        gravity_hold = hold_gravity_tauff is not None

        def _publish_hold():
            hold_q = _frozen_hold(arm_ctrl, prefer_last_published=gravity_hold)
            selected_tauff = _hold_tauff(
                arm_ctrl, measured_q, hold_gravity_tauff, hold_tau_limit, hold_gravity_scale
            )
            publication = arm_ctrl.ctrl_dual_arm(hold_q, selected_tauff)
            return ArmCommandDecision(False, True, hold_q, selected_tauff, publication)

        if not target_accepted or not sample_fresh or not measured_is_valid:
            if not measured_is_valid:
                return ArmCommandDecision(
                    False, True, measured_q, np.array([], dtype=float), None
                )
            return _publish_hold()
        if not _command_is_finite(q_target, tauff_target, measured_q.shape):
            return _publish_hold()
        selected_q = np.asarray(q_target).copy()
        selected_tauff = np.asarray(tauff_target).copy()
        publication = arm_ctrl.ctrl_dual_arm(selected_q, selected_tauff)
        _clear_frozen_hold(arm_ctrl)
        setattr(arm_ctrl, _LAST_PUBLISHED_ATTRIBUTE, np.asarray(selected_q, dtype=float).copy())
        return ArmCommandDecision(True, False, selected_q, selected_tauff, publication)


def publish_if_authorized(
    arm_ctrl,
    q_target,
    tauff_target,
    *,
    target_accepted,
    sample_fresh,
    lifecycle_lock,
    is_started,
    is_stopped,
):
    """Publish an IK result only while lifecycle authority remains valid.

    The caller must perform IK before calling this function; this lock only
    covers the final authority check and the output mutation.
    """
    return publish_arm_command(
        arm_ctrl,
        q_target,
        tauff_target,
        target_accepted=target_accepted,
        sample_fresh=sample_fresh,
        lifecycle_lock=lifecycle_lock,
        is_started=is_started,
        is_stopped=is_stopped,
    ).published
