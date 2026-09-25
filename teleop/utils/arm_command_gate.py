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
            measured_q = arm_ctrl.get_current_dual_arm_q().copy()
            selected_tauff = np.zeros_like(measured_q)
            publication = arm_ctrl.ctrl_dual_arm(measured_q, selected_tauff)
            if is_stopped():
                arm_ctrl.deactivate()
            return ArmCommandDecision(False, True, measured_q, selected_tauff, publication)
        if not target_accepted or not sample_fresh:
            measured_q = arm_ctrl.get_current_dual_arm_q().copy()
            selected_tauff = np.zeros_like(measured_q)
            publication = arm_ctrl.ctrl_dual_arm(measured_q, selected_tauff)
            return ArmCommandDecision(False, True, measured_q, selected_tauff, publication)
        selected_q = np.asarray(q_target).copy()
        selected_tauff = np.asarray(tauff_target).copy()
        publication = arm_ctrl.ctrl_dual_arm(selected_q, selected_tauff)
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
