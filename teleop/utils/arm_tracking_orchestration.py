"""Pure, dependency-injected arm tracking cycle orchestration."""

from dataclasses import dataclass

import numpy as np

from teleop.utils.arm_command_gate import publish_arm_command
from teleop.utils.quest_safety import controller_sample_is_fresh


@dataclass(frozen=True)
class ArmTrackingCycleResult:
    target_accepted: bool
    published: bool
    hold: bool
    requested_q: object
    requested_tauff: object
    selected_q: object
    selected_tauff: object
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

    The final lifecycle/freshness gate remains in ``publish_arm_command``;
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
    command = publish_arm_command(
        arm_ctrl,
        sol_q,
        sol_tauff,
        target_accepted=target is not None,
        sample_fresh=final_sample_fresh,
        lifecycle_lock=lifecycle_lock,
        is_started=is_started,
        is_stopped=is_stopped,
    )
    return ArmTrackingCycleResult(
        target_accepted=target is not None,
        published=command.published,
        hold=command.hold,
        requested_q=np.asarray(sol_q).copy(),
        requested_tauff=np.asarray(sol_tauff).copy(),
        selected_q=command.selected_q,
        selected_tauff=command.selected_tauff,
        target=target,
    )


def arm_recording_actions(cycle: ArmTrackingCycleResult):
    """Build arm actions from this cycle's selected command.

    ``selected_*`` is the command decision passed to the actuator, including
    a measured-q/zero-torque hold. It is pre-controller-limit telemetry;
    post-limit snapshots can be added separately without changing this record.
    """
    left_q = cycle.selected_q[:7].tolist()
    right_q = cycle.selected_q[-7:].tolist()
    return {
        "left_arm": {"qpos": left_q, "qvel": [], "torque": []},
        "right_arm": {"qpos": right_q, "qvel": [], "torque": []},
    }
