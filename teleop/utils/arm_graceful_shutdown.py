"""Graceful G1 arm shutdown: bounded return home, Dex3 open, authority release.

Contract (terminal ``q`` / right-controller ``B`` / Ctrl+C / exception):

1. Tracking has already stopped: the caller no longer produces IK targets.
2. Return both arms to the all-zero preparation pose along a smoothstep
   joint-space trajectory whose peak joint velocity never exceeds
   ``max_joint_velocity`` (duration is proportional to the largest joint
   distance). Waypoints go through ``arm_ctrl.ctrl_dual_arm`` so the existing
   writer gates (finite/shape) and publication receipts still apply.
3. Wait (bounded) for measured arrival within ``arrival_tolerance``.
4. Open Dex3 and stop its writer (via the ``open_hands`` callback).
4b. Torso lean (only if the controller's waist command was configured at
   ``r``, see teleop/utils/torso_lean.py): the waist target is set back to the
   neutral captured at ``r`` at the START of the arm return (the writer slews
   it at the configured rate limit, concurrently with the arms), and the
   procedure waits (bounded) until the WRITTEN waist command equals the
   neutral BEFORE the authority weight ramp. With the feature off nothing in
   this step runs and the sequence is byte-for-byte the previous one.
4d. Waist gravity feed-forward (only if ``configure_waist_gravity_ff`` ran,
   teleop/utils/waist_gravity_ff.py): after the waist is back at neutral the
   feed-forward tau is ramped to 0 (``RAMP_S``) and the procedure waits
   (bounded) for the writer to report it finished BEFORE the weight ramp, so
   the vendor controller never receives the waist with a residual tau.
4c. Dex3 close/hold at shutdown (``DEX3_SHUTDOWN_HAND=close|hold``, see
   teleop/utils/dex3_shutdown_hand.py): when ``close_hands`` is given it runs
   FIRST (before step 2): triggers lose authority and the hand ramps closed
   (bounded); it stays held by its writer while the arms return and during the
   weight ramp; ``release_hands`` (after step 5, before the arm writer is
   deactivated) publishes the final frame and stops the Dex3 writer. Step 4 is
   then skipped. If ``close_hands`` raises, step 4 (``open_hands``) runs as
   before (fail-safe). Without ``close_hands`` the sequence is unchanged.
5. Ramp the ``rt/arm_sdk`` authority weight (kNotUsedJoint0.q) linearly 1 -> 0
   so the Unitree motion controller takes the arms back smoothly, confirm the
   writer actually published weight 0, then deactivate the writer.

Fail-safe: invalid/non-finite/stale measured state or a trajectory that would
be too long skips the return motion (no invented trajectory) and only performs
the weight release. A writer that is not publishing cannot release anything;
then the writer is simply deactivated. Every phase is time-bounded, and the
procedure runs at most once per controller.

The module is dependency-injected (clock/sleep/emit) so it can be tested with
fakes; it performs no I/O itself. ``emit`` must be nonblocking.
"""

from dataclasses import dataclass, field

import numpy as np

DEFAULT_MAX_JOINT_VELOCITY = 0.5        # rad/s, per joint, peak
DEFAULT_WAYPOINT_DT = 0.02              # 50 Hz waypoint updates (writer runs at 250 Hz)
DEFAULT_MIN_RETURN_DURATION = 0.5       # s
DEFAULT_MAX_RETURN_DURATION = 12.0      # s; longer plans are refused (fail-safe)
DEFAULT_ARRIVAL_TOLERANCE = 0.05        # rad, same as launcher preparation
DEFAULT_SETTLE_TIMEOUT = 2.0            # s
DEFAULT_RELEASE_DURATION = 2.0          # s, weight 1 -> 0 (upstream: 101 steps x 20 ms)
DEFAULT_RELEASE_DT = 0.02               # s
DEFAULT_RELEASE_CONFIRM_TIMEOUT = 0.5   # s to observe a published weight-0 frame
DEFAULT_STATE_MAX_AGE = 0.25            # s
DEFAULT_WRITER_MAX_AGE = 0.25           # s since the last successful arm write
DEFAULT_WAIST_NEUTRAL_TOL = 1e-4        # rad, written waist command == neutral
DEFAULT_WAIST_MEASURED_TOL = 0.05       # rad, measured waist near neutral (logged)
DEFAULT_WAIST_EXTRA_TIMEOUT = 1.5       # s on top of distance / rate
DEFAULT_WAIST_FF_RAMP_TIMEOUT = 1.0     # s for the waist feed-forward ramp-out (ramp 0.5 s)
SMOOTHSTEP_PEAK_VELOCITY_FACTOR = 1.5   # max d/ds of 3s^2 - 2s^3


def smoothstep(s):
    s = min(1.0, max(0.0, float(s)))
    return s * s * (3.0 - 2.0 * s)


def plan_return_duration(start_q, goal_q, max_joint_velocity=DEFAULT_MAX_JOINT_VELOCITY,
                         min_duration=DEFAULT_MIN_RETURN_DURATION):
    """Duration whose smoothstep peak velocity is <= max_joint_velocity."""
    start = np.asarray(start_q, dtype=float)
    goal = np.asarray(goal_q, dtype=float)
    distance = float(np.max(np.abs(goal - start))) if start.size else 0.0
    return max(float(min_duration),
               SMOOTHSTEP_PEAK_VELOCITY_FACTOR * distance / float(max_joint_velocity))


def interpolate_return(start_q, goal_q, elapsed, duration):
    start = np.asarray(start_q, dtype=float)
    goal = np.asarray(goal_q, dtype=float)
    if duration <= 0.0:
        return goal.copy()
    return start + (goal - start) * smoothstep(elapsed / duration)


def _finite_vector(value, size):
    try:
        vector = np.asarray(value, dtype=float).reshape(-1).copy()
    except (TypeError, ValueError):
        return None
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        return None
    return vector


@dataclass
class GracefulShutdownResult:
    returned_home: bool = False
    return_skipped_reason: str | None = None
    arrival_confirmed: bool = False
    hands_opened: bool = False
    hands_closed: bool = False
    hands_released: bool = False
    hand_summary: dict | None = None
    waist_return_requested: bool = False
    waist_neutral_reached: bool = False
    waist_ff_zeroed: bool = False
    weight_released: bool = False
    release_confirmed: bool = False
    deactivated: bool = False
    cancelled: bool = False
    events: list = field(default_factory=list)


def run_graceful_arm_shutdown(
    arm_ctrl,
    *,
    clock,
    sleep,
    emit=None,
    open_hands=None,
    close_hands=None,
    release_hands=None,
    gravity_tauff=None,
    goal_q=None,
    attempt_return=True,
    start_command_tolerance=0.3,
    max_joint_velocity=DEFAULT_MAX_JOINT_VELOCITY,
    waypoint_dt=DEFAULT_WAYPOINT_DT,
    min_return_duration=DEFAULT_MIN_RETURN_DURATION,
    max_return_duration=DEFAULT_MAX_RETURN_DURATION,
    arrival_tolerance=DEFAULT_ARRIVAL_TOLERANCE,
    settle_timeout=DEFAULT_SETTLE_TIMEOUT,
    release_duration=DEFAULT_RELEASE_DURATION,
    release_dt=DEFAULT_RELEASE_DT,
    release_confirm_timeout=DEFAULT_RELEASE_CONFIRM_TIMEOUT,
    state_max_age=DEFAULT_STATE_MAX_AGE,
    writer_max_age=DEFAULT_WRITER_MAX_AGE,
    waist_neutral_tol=DEFAULT_WAIST_NEUTRAL_TOL,
    waist_extra_timeout=DEFAULT_WAIST_EXTRA_TIMEOUT,
    waist_ff_ramp_timeout=DEFAULT_WAIST_FF_RAMP_TIMEOUT,
):
    """Run the graceful shutdown once; never raises, always ends deactivated.

    ``arm_ctrl`` must provide the G1_29 shutdown API: ``arm_joint_split``,
    ``get_dual_arm_q_snapshot() -> (q, age_s)``, ``get_arm_command() -> (q, tau)``,
    ``ctrl_dual_arm(q, tau)``, ``set_motion_authority_weight(w)``,
    ``get_publication_status() -> dict``, ``motion_mode`` and ``deactivate()``.
    A second call returns the first result without commanding anything.
    """
    previous = getattr(arm_ctrl, "_xr_teleop_graceful_shutdown_result", None)
    if previous is not None:
        return previous
    result = GracefulShutdownResult()
    arm_ctrl._xr_teleop_graceful_shutdown_result = result

    def event(name, **detail):
        record = {"event": name, "t": clock(), **detail}
        result.events.append(record)
        if emit is not None:
            try:
                emit(name, detail)
            except BaseException:
                pass

    size = int(sum(int(count) for count in arm_ctrl.arm_joint_split))
    goal = np.zeros(size) if goal_q is None else _finite_vector(goal_q, size)

    def measured():
        try:
            q, age = arm_ctrl.get_dual_arm_q_snapshot()
            age = float(age)
        except BaseException:
            return None, "state_read_failed"
        q = _finite_vector(q, size)
        if q is None:
            return None, "state_invalid"
        if not np.isfinite(age) or age > state_max_age:
            return None, "state_stale"
        return q, None

    def writer_alive():
        try:
            status = arm_ctrl.get_publication_status()
        except BaseException:
            return False
        last = status.get("last_publish_monotonic")
        return (
            bool(status.get("active"))
            and last is not None
            and np.isfinite(last)
            and clock() - float(last) <= writer_max_age
        )

    def tau_for(q, fallback):
        if gravity_tauff is not None:
            try:
                tau = _finite_vector(gravity_tauff(q), size)
            except BaseException:
                tau = None
            if tau is not None:
                return tau
        return fallback

    def waist_status():
        getter = getattr(arm_ctrl, "get_waist_command", None)
        if getter is None:
            return None
        try:
            status = getter()
        except BaseException:
            return None
        return status if isinstance(status, dict) and status.get("enabled") else None

    # ---- Phase -1: Dex3 close/hold (opt-in callback; default mode close) ----
    hands_close_failed = False
    if close_hands is not None:
        try:
            summary = close_hands()
            result.hands_closed = True
            result.hand_summary = summary if isinstance(summary, dict) else None
            event("shutdown_dex3_closed", summary=result.hand_summary)
        except BaseException as error:
            hands_close_failed = True
            event("shutdown_error", phase="dex3_close", error=type(error).__name__)

    # ---- Phase 0: torso lean -> neutral (only when the feature took the waist)
    waist0 = waist_status()
    if waist0 is not None:
        try:
            if arm_ctrl.waist_return_to_neutral():
                result.waist_return_requested = True
                event("shutdown_waist_return_started",
                      max_distance_rad=float(np.max(np.abs(np.asarray(waist0["written"]) - np.asarray(waist0["neutral"])))))
        except BaseException as error:
            event("shutdown_error", phase="waist_return", error=type(error).__name__)

    last_tau = np.zeros(size)
    try:
        # ---- Phase 1: bounded return to the preparation pose -------------
        start_q, reason = measured()
        if not attempt_return:
            reason = "return_not_requested"
        elif goal is None:
            reason = "goal_invalid"
        if reason is None and not writer_alive():
            reason = "writer_not_publishing"
        commanded_q, tau0 = None, None
        if reason is None:
            try:
                raw_q, raw_tau = arm_ctrl.get_arm_command()
                commanded_q = _finite_vector(raw_q, size)
                tau0 = _finite_vector(raw_tau, size)
            except BaseException:
                commanded_q, tau0 = None, None
            # Start from the command the servos are already tracking so the
            # first waypoint is continuous; fall back to measured q if the
            # command is unusable or far from the measurement.
            if commanded_q is not None and np.all(np.abs(commanded_q - start_q) <= start_command_tolerance):
                start_q = commanded_q
            if tau0 is None:
                tau0 = np.zeros(size)
            duration = plan_return_duration(start_q, goal, max_joint_velocity, min_return_duration)
            if duration > max_return_duration:
                reason = "return_too_far"
        if reason is not None:
            result.return_skipped_reason = reason
            event("shutdown_return_skipped", reason=reason)
        else:
            event("shutdown_return_started", duration_s=duration,
                  max_joint_velocity=float(max_joint_velocity),
                  max_distance_rad=float(np.max(np.abs(goal - start_q))))
            t0 = clock()
            try:
                while True:
                    elapsed = clock() - t0
                    s = min(1.0, elapsed / duration)
                    q = interpolate_return(start_q, goal, elapsed, duration)
                    # Feed-forward is continuous: blend from the last
                    # commanded tau to the gravity model at q (zero if no
                    # model is available) with the same smoothstep profile.
                    blend = smoothstep(s)
                    last_tau = (1.0 - blend) * tau0 + blend * tau_for(q, np.zeros(size))
                    arm_ctrl.ctrl_dual_arm(q, last_tau)
                    if s >= 1.0:
                        break
                    sleep(waypoint_dt)
                result.returned_home = True
                deadline = clock() + settle_timeout
                while True:
                    current, _ = measured()
                    if current is not None and np.all(np.abs(current - goal) <= arrival_tolerance):
                        result.arrival_confirmed = True
                        break
                    if clock() >= deadline:
                        break
                    sleep(waypoint_dt)
                event("shutdown_return_finished", arrival_confirmed=result.arrival_confirmed,
                      duration_s=clock() - t0)
            except BaseException as error:  # e.g. a second Ctrl+C: cancel motion, still release
                result.cancelled = True
                event("shutdown_return_cancelled", error=type(error).__name__)
    except BaseException as error:
        event("shutdown_error", phase="return", error=type(error).__name__)

    # ---- Phase 1b: waist command back at neutral before releasing authority
    if result.waist_return_requested:
        try:
            rate = float(waist0.get("max_rate") or 0.0)
            dist = float(np.max(np.abs(np.asarray(waist0["written"]) - np.asarray(waist0["neutral"]))))
            timeout = (dist / rate if rate > 0 else 0.0) + float(waist_extra_timeout)
            deadline = clock() + timeout
            while True:
                status = waist_status()
                if status is not None and np.all(np.abs(np.asarray(status["written"]) - np.asarray(status["neutral"]))
                                                 <= waist_neutral_tol):
                    result.waist_neutral_reached = True
                    break
                if clock() >= deadline or not writer_alive():
                    break
                sleep(waypoint_dt)
            measured_ok = None
            snap = getattr(arm_ctrl, "get_waist_q_snapshot", None)
            if snap is not None and status is not None:
                try:
                    wq, wage = snap()
                    if wq is not None and np.isfinite(wage) and wage <= state_max_age:
                        measured_ok = bool(np.all(np.abs(np.asarray(wq, float) - np.asarray(status["neutral"]))
                                                  <= DEFAULT_WAIST_MEASURED_TOL))
                except BaseException:
                    measured_ok = None
            event("shutdown_waist_return_finished", neutral_reached=result.waist_neutral_reached,
                  measured_near_neutral=measured_ok)
        except BaseException as error:
            event("shutdown_error", phase="waist_wait", error=type(error).__name__)

    # ---- Phase 1c: waist gravity feed-forward -> 0 before releasing authority
    ff_getter = getattr(arm_ctrl, "get_waist_gravity_ff", None)
    ff_status = None
    if ff_getter is not None:
        try:
            ff_status = ff_getter()
        except BaseException:
            ff_status = None
    if isinstance(ff_status, dict) and ff_status.get("configured"):
        try:
            if arm_ctrl.waist_gravity_ff_ramp_out():
                event("shutdown_waist_ff_ramp_out_started", gain=float(ff_status.get("gain", 0.0)))
                deadline = clock() + float(waist_ff_ramp_timeout)
                while True:
                    status = ff_getter()
                    if isinstance(status, dict) and status.get("finished"):
                        result.waist_ff_zeroed = True
                        break
                    if clock() >= deadline or not writer_alive():
                        break
                    sleep(waypoint_dt)
                event("shutdown_waist_ff_ramp_out_finished", zeroed=result.waist_ff_zeroed)
        except BaseException as error:
            event("shutdown_error", phase="waist_ff_ramp_out", error=type(error).__name__)

    # ---- Phase 2: open Dex3 and stop its writer -------------------------
    if open_hands is not None and (close_hands is None or hands_close_failed):
        try:
            open_hands()
            result.hands_opened = True
        except BaseException as error:
            event("shutdown_error", phase="dex3_open", error=type(error).__name__)

    # ---- Phase 3: authority weight ramp-down ----------------------------
    try:
        if not getattr(arm_ctrl, "motion_mode", False):
            event("weight_release_skipped", reason="not_motion_mode")
        elif not writer_alive():
            event("weight_release_skipped", reason="writer_not_publishing")
        else:
            event("weight_release_started", duration_s=float(release_duration))
            steps = max(1, int(round(release_duration / release_dt)))
            for index in range(steps + 1):
                arm_ctrl.set_motion_authority_weight(1.0 - index / steps)
                if index < steps:
                    sleep(release_dt)
            result.weight_released = True
            deadline = clock() + release_confirm_timeout
            while True:
                status = arm_ctrl.get_publication_status()
                if status.get("last_published_weight") == 0.0:
                    result.release_confirmed = True
                    break
                if clock() >= deadline:
                    break
                sleep(release_dt)
            event("weight_release_finished", confirmed=result.release_confirmed)
    except BaseException as error:
        event("shutdown_error", phase="weight_release", error=type(error).__name__)

    # ---- Phase 3b: Dex3 final frame + writer stop (close/hold modes) -----
    if close_hands is not None and not hands_close_failed and release_hands is not None:
        try:
            release_hands()
            result.hands_released = True
            event("shutdown_dex3_released")
        except BaseException as error:
            event("shutdown_error", phase="dex3_release", error=type(error).__name__)

    # ---- Phase 4: stop the writer (always) ------------------------------
    try:
        arm_ctrl.deactivate()
        result.deactivated = True
    except BaseException as error:
        event("shutdown_error", phase="deactivate", error=type(error).__name__)
    event("shutdown_arm_released", deactivated=result.deactivated,
          release_confirmed=result.release_confirmed)
    return result
