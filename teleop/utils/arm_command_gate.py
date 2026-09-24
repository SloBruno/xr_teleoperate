"""Lifecycle-serialized arm command publication."""

import numpy as np


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
    with lifecycle_lock:
        if is_stopped() or not is_started():
            measured_q = arm_ctrl.get_current_dual_arm_q().copy()
            arm_ctrl.ctrl_dual_arm(measured_q, np.zeros_like(measured_q))
            if is_stopped():
                arm_ctrl.deactivate()
            return False
        if not target_accepted or not sample_fresh:
            measured_q = arm_ctrl.get_current_dual_arm_q().copy()
            arm_ctrl.ctrl_dual_arm(measured_q, np.zeros_like(measured_q))
            return False
        arm_ctrl.ctrl_dual_arm(q_target, tauff_target)
        return True
