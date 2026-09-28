"""Pure, dependency-injected arm tracking cycle orchestration."""

from dataclasses import dataclass
import time

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
    """Residual gate on the RAW IK solution (0.10 m / 0.30 rad per side).

    ``q`` must be the unfiltered solver output: G1_29 no longer smooths
    inside ``solve_ik``, so the gate judges exactly what IK produced and a
    lagging filtered vector can no longer trip it (review 2026-09-28).
    """
    try:
        q_array = np.asarray(q, dtype=float)
    except (TypeError, ValueError):
        return False
    if q_array.ndim != 1 or q_array.size == 0 or not np.all(np.isfinite(q_array)):
        return False
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
    requested_target: object = None
    limiter_bands: tuple | None = None
    ramp_alpha: float | None = None


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
    rate_limiter=None,
    enable_ramp=None,
):
    """Resolve one target, solve only accepted fresh targets, then gate output.

    Order per cycle (G1_29 passes ``rate_limiter`` and ``enable_ramp``):

    1. requested target (first calibrated target / calibrator / candidate);
    2. three-band Cartesian ``rate_limiter.limit`` against the last
       PUBLISHED target (``None`` fails closed without IK);
    3. IK, then the residual gate on the raw IK ``q``;
    4. ``enable_ramp`` blends the arming hold pose into the IK command;
    5. ``publish_arm_command`` (final lifecycle/freshness authority);
    6. ``rate_limiter.commit`` only if the command was actually published.

    STOP interrupts the ramp permanently.  The joint-space velocity clip in
    ``robot_arm`` remains the last barrier downstream.
    """
    clock = now if now is not None else time.monotonic()
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
    requested_target = target

    if enable_ramp is not None and is_stopped():
        enable_ramp.interrupt()
    if enable_ramp is not None and enable_ramp.interrupted:
        target = None

    # Cartesian hold: a FRESH sample that the calibrator rejected (e.g. a
    # re-anchoring jump) re-solves the last PUBLISHED target instead of
    # freezing measured q.  Replay showed the measured-q hold steps the arm
    # backwards by its tracking lag (up to 82 deg in one cycle).  Never used
    # for stale samples, STOP, an interrupted ramp, or before a first publish;
    # the residual gate below still applies.
    cartesian_hold = False
    if (
        target is None
        and rate_limiter is not None
        and sample_fresh
        and not is_stopped()
        and calibrator is not None
        and getattr(calibrator, "calibrated", False)
        and not (enable_ramp is not None and enable_ramp.interrupted)
    ):
        held = getattr(rate_limiter, "last_committed_targets", None)
        if held is not None and _valid_target_pair(held):
            target = held
            cartesian_hold = True

    limiter_bands = None
    if target is not None and rate_limiter is not None and not cartesian_hold:
        limited = rate_limiter.limit(target, clock)
        if limited is None or not _valid_target_pair(limited.targets):
            target = None
        else:
            target = limited.targets
            limiter_bands = tuple(limited.bands)

    if target is None:
        sol_q = np.asarray(current_q).copy()
        sol_tauff = np.zeros_like(sol_q)
    else:
        sol_q, sol_tauff = arm_ik.solve_ik(
            target[0], target[1], current_q, current_dq
        )
        if not _fk_matches_target(arm_ik, sol_q, target):
            target = None

    command_q, command_tauff = sol_q, sol_tauff
    ramp_alpha = None
    if enable_ramp is not None and target is not None:
        if not enable_ramp.started and is_started() and not is_stopped():
            enable_ramp.begin(clock, current_q)
        blended = enable_ramp.apply(sol_q, sol_tauff, clock)
        if blended is None:
            target = None
        else:
            command_q, command_tauff, ramp_alpha = blended

    final_sample_fresh = sample_fresh and controller_sample_is_fresh(sample_timestamp, now)
    command = publish_arm_command(
        arm_ctrl,
        command_q,
        command_tauff,
        target_accepted=target is not None,
        sample_fresh=final_sample_fresh,
        lifecycle_lock=lifecycle_lock,
        is_started=is_started,
        is_stopped=is_stopped,
    )
    if command.published and rate_limiter is not None and limiter_bands is not None:
        rate_limiter.commit()
    if enable_ramp is not None and is_stopped():
        enable_ramp.interrupt()
    if command.hold:
        if is_stopped():
            decision_reason = "lifecycle_stop_hold"
        elif not is_started():
            decision_reason = "lifecycle_not_started_hold"
        elif target is None:
            decision_reason = "no_accepted_target_hold"
        else:
            decision_reason = "invalid_or_stale_ik_hold"
    elif cartesian_hold:
        decision_reason = "cartesian_hold_last_target"
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
        requested_target=requested_target,
        limiter_bands=limiter_bands,
        ramp_alpha=ramp_alpha,
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
