"""Pure, dependency-injected arm tracking cycle orchestration."""

from dataclasses import dataclass

import numpy as np

from teleop.utils.arm_command_gate import publish_arm_command
from teleop.utils.quest_safety import controller_sample_is_fresh


def _is_rigid_se3(transform):
    try:
        matrix = np.asarray(transform, dtype=float)
    except (TypeError, ValueError):
        return False
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return False
    rotation = matrix[:3, :3]
    return (
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    )


def _valid_target_pair(target):
    try:
        return len(target) == 2 and all(_is_rigid_se3(pose) for pose in target)
    except (TypeError, ValueError):
        return False


def _rotation_distance(first, second):
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _fk_matches_target(arm_ik, q, target):
    forward_kinematics = getattr(arm_ik, "forward_kinematics", None)
    if forward_kinematics is None:
        return True
    try:
        fk = forward_kinematics(q)
    except Exception:
        return False
    if not _valid_target_pair(fk):
        return False
    for actual, expected in zip(fk, target):
        if np.linalg.norm(actual[:3, 3] - expected[:3, 3]) > 0.10:
            return False
        if _rotation_distance(actual, expected) > 0.30:
            return False
    return True


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
    publication: object = None
    sample_fresh: bool = False
    decision_reason: str = ""
    arm_joint_split: tuple[int, int] = (7, 7)


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

    if target is not None and not _valid_target_pair(target):
        target = None

    if target is None:
        sol_q = np.asarray(current_q).copy()
        sol_tauff = np.zeros_like(sol_q)
    else:
        sol_q, sol_tauff = arm_ik.solve_ik(
            target[0], target[1], current_q, current_dq
        )
        if not _fk_matches_target(arm_ik, sol_q, target):
            target = None

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
    if command.hold:
        if is_stopped():
            decision_reason = "lifecycle_stop_hold"
        elif not is_started():
            decision_reason = "lifecycle_not_started_hold"
        elif target is None:
            decision_reason = "no_accepted_target_hold"
        else:
            decision_reason = "invalid_or_stale_ik_hold"
    else:
        decision_reason = "ik_command_selected"
    return ArmTrackingCycleResult(
        target_accepted=target is not None,
        published=command.published,
        hold=command.hold,
        requested_q=np.asarray(sol_q).copy(),
        requested_tauff=np.asarray(sol_tauff).copy(),
        selected_q=command.selected_q,
        selected_tauff=command.selected_tauff,
        target=target,
        publication=command.publication,
        sample_fresh=final_sample_fresh,
        decision_reason=decision_reason,
        arm_joint_split=tuple(getattr(arm_ctrl, "arm_joint_split", (7, 7))),
    )


def build_arm_recording_actions(cycle: ArmTrackingCycleResult):
    """Build JSON-ready arm actions from this cycle's selected command.

    ``selected_*`` is the command decision passed to the actuator, including
    a measured-q/zero-torque hold. It is pre-controller-limit telemetry;
    post-limit snapshots can be added separately without changing this record.

    Serialization contract: every qpos field returned here is a plain Python
    list, so production record builders must not call ``.tolist()`` on it.
    """
    left_count, right_count = cycle.arm_joint_split
    selected_q = np.asarray(cycle.selected_q)
    if selected_q.ndim != 1 or selected_q.size != left_count + right_count:
        raise ValueError("selected_q does not match arm_joint_split")
    left_q = selected_q[:left_count].tolist()
    right_q = selected_q[left_count:].tolist()
    return {
        "left_arm": {"qpos": left_q, "qvel": [], "torque": []},
        "right_arm": {"qpos": right_q, "qvel": [], "torque": []},
    }


def arm_recording_actions(cycle: ArmTrackingCycleResult):
    """Backward-compatible name for the production arm-action builder."""
    return build_arm_recording_actions(cycle)
