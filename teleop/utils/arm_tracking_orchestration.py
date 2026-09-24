"""Pure, dependency-injected arm tracking cycle orchestration."""

from dataclasses import dataclass

import numpy as np

from teleop.utils.arm_command_gate import publish_if_authorized
from teleop.utils.quest_safety import controller_sample_is_fresh


@dataclass(frozen=True)
class ArmTrackingCycleResult:
    target_accepted: bool
    published: bool
    target: object = None


def run_arm_tracking_cycle(
    *,
    arm_ctrl,
    arm_ik,
    calibrator,
    controller_poses,
    sample_timestamp,
    now=None,
    current_q,
    current_dq,
    first_target=None,
    candidate_targets=None,
    lifecycle_lock,
    is_started,
    is_stopped,
):
    """Resolve one target, solve only accepted fresh targets, then gate output.

    The final lifecycle/freshness gate remains in ``publish_if_authorized``;
    this helper only makes the pre-gate flow deterministic and injectable.
    """
    sample_fresh = controller_sample_is_fresh(sample_timestamp, now)
    target = None
    if not is_stopped() and first_target is not None and sample_fresh:
        target = first_target
    elif not is_stopped() and candidate_targets is not None and sample_fresh:
        target = candidate_targets
    elif (
        not is_stopped()
        and calibrator is not None
        and getattr(calibrator, "calibrated", False)
        and sample_fresh
    ):
        target = calibrator.targets(controller_poses, sample_timestamp, now=now)

    if target is None:
        sol_q = np.asarray(current_q).copy()
        sol_tauff = np.zeros_like(sol_q)
    else:
        sol_q, sol_tauff = arm_ik.solve_ik(
            target[0], target[1], current_q, current_dq
        )

    final_sample_fresh = sample_fresh and controller_sample_is_fresh(sample_timestamp, now)
    published = publish_if_authorized(
        arm_ctrl,
        sol_q,
        sol_tauff,
        target_accepted=target is not None,
        sample_fresh=final_sample_fresh,
        lifecycle_lock=lifecycle_lock,
        is_started=is_started,
        is_stopped=is_stopped,
    )
    return ArmTrackingCycleResult(target is not None, published, target)
