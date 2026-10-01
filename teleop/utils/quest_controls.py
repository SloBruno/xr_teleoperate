import math
import time


STICK_DEADZONE = 0.12
STICK_PRECISION_EXPONENT = 3
# LocoClient.Move accepts vx/vy in m/s and vyaw in rad/s.  The SDK only
# serializes floats and declares no positive minimum, so these are the
# smallest reviewed non-zero operator caps; validate them on hardware before
# reducing them further.
# Defaults = BotBrain frontend G1 profile (g1-r1.ts: 0.5 m/s, 0.3 rad/s); hard limits below = BotBrain G1 driver saturation.
MIN_OPERATOR_WALK_SPEED_MPS = 0.5
MIN_OPERATOR_TURN_RATE_RADPS = 0.3
# BotBrain saturation: 0.6 m/s, 1.0 rad/s.  Caps are raisable (CLI/env) up to these.
MAX_WALK_SPEED_CAP_MPS = 0.6
MAX_TURN_RATE_CAP_RADPS = 1.0
# BotBrain twist_mux/dead-man: zero when no fresh stick for 0.2 s.
LOCO_STICK_TIMEOUT_S = 0.2


def _checked_cap(name, value, limit):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0 or value > limit:
        raise ValueError(f"{name}={value} must be in (0, {limit}]")
    return value


def resolve_speed_caps(walk_cap=None, turn_cap=None, environ=None):
    """(walk m/s, yaw rad/s): arg > env G1_WALK_SPEED_CAP/G1_TURN_RATE_CAP > default."""
    env = environ if environ is not None else {}
    if walk_cap is None and env.get("G1_WALK_SPEED_CAP") not in (None, ""):
        walk_cap = env["G1_WALK_SPEED_CAP"]
    if turn_cap is None and env.get("G1_TURN_RATE_CAP") not in (None, ""):
        turn_cap = env["G1_TURN_RATE_CAP"]
    walk = MIN_OPERATOR_WALK_SPEED_MPS if walk_cap is None else _checked_cap("walk cap", walk_cap, MAX_WALK_SPEED_CAP_MPS)
    yaw = MIN_OPERATOR_TURN_RATE_RADPS if turn_cap is None else _checked_cap("turn cap", turn_cap, MAX_TURN_RATE_CAP_RADPS)
    return walk, yaw


def speed_cap_banner(walk_cap, turn_cap):
    return (f"=== LOCOMOTION CAP ACTIVE: walk {walk_cap:.2f} m/s (max {MAX_WALK_SPEED_CAP_MPS}), "
            f"turn {turn_cap:.2f} rad/s (max {MAX_TURN_RATE_CAP_RADPS}) ===")


def loco_stick_is_fresh(sample_timestamp, now=None):
    if now is None:
        now = time.monotonic()
    if not math.isfinite(sample_timestamp) or sample_timestamp <= 0.0 or not math.isfinite(now):
        return False
    return 0.0 <= now - sample_timestamp <= LOCO_STICK_TIMEOUT_S


def _clamp_stick_value(value):
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def _shape_stick_value(value):
    """Add a center deadzone and cubic precision without capping full scale."""
    value = _clamp_stick_value(value)
    magnitude = abs(value)
    if magnitude <= STICK_DEADZONE:
        return 0.0
    normalized = (magnitude - STICK_DEADZONE) / (1.0 - STICK_DEADZONE)
    return math.copysign(normalized ** STICK_PRECISION_EXPONENT, value)


def joystick_to_locomotion(left_xy, right_xy, walk_cap=MIN_OPERATOR_WALK_SPEED_MPS, turn_cap=MIN_OPERATOR_TURN_RATE_RADPS):
    left_x, left_y = (_shape_stick_value(value) for value in left_xy)
    right_x, _ = (_shape_stick_value(value) for value in right_xy)
    return (
        -left_y * walk_cap,
        -left_x * walk_cap,
        -right_x * turn_cap,
    )


def _raw_axis(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _raw_xy(xy):
    """Raw stick pair as JSON-safe floats (before clamp/deadzone/curve)."""
    try:
        x, y = xy
    except Exception:
        return None
    return [_raw_axis(x), _raw_axis(y)]


def _shaped_xy(xy):
    try:
        x, y = xy
        return [_shape_stick_value(x), _shape_stick_value(y)]
    except Exception:
        return None


def stick_snapshot(left_xy, right_xy, loco_wrapper=None):
    """In-memory diagnostic record: raw sticks, shaped sticks, command, Move code.

    Pure (no I/O) and never raises; the command is the unchanged
    joystick_to_locomotion result.  Sign conventions are NOT altered here.
    """
    try:
        command = list(joystick_to_locomotion(left_xy, right_xy))
    except Exception:
        command = None
    return {
        "raw_left_xy": _raw_xy(left_xy),
        "raw_right_xy": _raw_xy(right_xy),
        "shaped_left_xy": _shaped_xy(left_xy),
        "shaped_right_xy": _shaped_xy(right_xy),
        "command": command,
        "last_move_code": getattr(loco_wrapper, "last_move_code", None),
        "nonzero_move_codes": getattr(loco_wrapper, "nonzero_move_codes", None),
    }


def _note_moving(loco_wrapper, moving):
    try:
        loco_wrapper._was_moving = moving
    except Exception:
        pass


def _explicit_stop_on_release(loco_wrapper, reason):
    """One non-blocking StopMove per non-zero -> zero transition; never raises."""
    _note_moving(loco_wrapper, False)
    stop = getattr(loco_wrapper, "StopMove", None)
    if stop is None:
        return
    try:
        stop(reason)
    except Exception:
        pass


def dispatch_joystick_locomotion(loco_wrapper, motion_enabled, controller_is_fresh, left_xy, right_xy,
                                 walk_cap=MIN_OPERATOR_WALK_SPEED_MPS, turn_cap=MIN_OPERATOR_TURN_RATE_RADPS):
    """Send only enabled, fresh joystick locomotion; stale input releases to zero.

    Transition non-zero -> zero additionally issues one explicit StopMove
    (reason release / stale / motion_disabled).
    """
    locomotion = (0.0, 0.0, 0.0)
    was_moving = bool(getattr(loco_wrapper, "_was_moving", False))
    if not motion_enabled:
        if was_moving and loco_wrapper is not None:
            _explicit_stop_on_release(loco_wrapper, "motion_disabled")
        return locomotion
    if controller_is_fresh:
        locomotion = joystick_to_locomotion(left_xy, right_xy, walk_cap, turn_cap)
    loco_wrapper.Move(*locomotion)
    if any(locomotion):
        _note_moving(loco_wrapper, True)
    elif was_moving:
        _explicit_stop_on_release(loco_wrapper, "release" if controller_is_fresh else "stale")
    return locomotion
