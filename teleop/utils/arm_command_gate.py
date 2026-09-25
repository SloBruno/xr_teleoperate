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


def _clear_frozen_hold(arm_ctrl):
    setattr(arm_ctrl, _HOLD_ATTRIBUTE, None)


def _frozen_hold(arm_ctrl):
    hold_q = getattr(arm_ctrl, _HOLD_ATTRIBUTE, None)
    if hold_q is None:
        measured_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=float).copy()
        if measured_q.ndim != 1 or measured_q.size == 0 or not np.all(np.isfinite(measured_q)):
            raise ValueError("measured arm state is not a finite joint vector")
        hold_q = measured_q
        setattr(arm_ctrl, _HOLD_ATTRIBUTE, hold_q.copy())
    return np.asarray(hold_q, dtype=float).copy()


def _command_is_finite(q_target, tauff_target):
    try:
        q = np.asarray(q_target, dtype=float)
        tau = np.asarray(tauff_target, dtype=float)
    except (TypeError, ValueError):
        return False
    return (
        q.ndim == 1
        and q.size > 0
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
):
    """Select and publish one command while retaining the exact hold decision."""
    with lifecycle_lock:
        if is_stopped() or not is_started():
            measured_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=float).copy()
            selected_tauff = np.zeros_like(measured_q)
            _clear_frozen_hold(arm_ctrl)
            if not (
                measured_q.ndim == 1
                and measured_q.size > 0
                and np.all(np.isfinite(measured_q))
            ):
                if is_stopped():
                    arm_ctrl.deactivate()
                return ArmCommandDecision(False, True, measured_q, selected_tauff, None)
            if is_stopped():
                # STOP is terminal for this output path: invalidate the writer
                # before returning so no pending/new target is enqueued.
                arm_ctrl.deactivate()
                return ArmCommandDecision(False, True, measured_q, selected_tauff, None)
            publication = arm_ctrl.ctrl_dual_arm(measured_q, selected_tauff)
            return ArmCommandDecision(False, True, measured_q, selected_tauff, publication)
        if not target_accepted or not sample_fresh or not _command_is_finite(q_target, tauff_target):
            hold_q = _frozen_hold(arm_ctrl)
            selected_tauff = np.zeros_like(hold_q)
            publication = arm_ctrl.ctrl_dual_arm(hold_q, selected_tauff)
            return ArmCommandDecision(False, True, hold_q, selected_tauff, publication)
        selected_q = np.asarray(q_target).copy()
        selected_tauff = np.asarray(tauff_target).copy()
        publication = arm_ctrl.ctrl_dual_arm(selected_q, selected_tauff)
        _clear_frozen_hold(arm_ctrl)
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
