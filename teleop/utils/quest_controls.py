import math


STICK_DEADZONE = 0.12
STICK_PRECISION_EXPONENT = 3
# LocoClient.Move accepts vx/vy in m/s and vyaw in rad/s.  The SDK only
# serializes floats and declares no positive minimum, so these are the
# smallest reviewed non-zero operator caps; validate them on hardware before
# reducing them further.
MIN_OPERATOR_WALK_SPEED_MPS = 0.15
MIN_OPERATOR_TURN_RATE_RADPS = 0.10


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


def joystick_to_locomotion(left_xy, right_xy):
    left_x, left_y = (_shape_stick_value(value) for value in left_xy)
    right_x, _ = (_shape_stick_value(value) for value in right_xy)
    return (
        -left_y * MIN_OPERATOR_WALK_SPEED_MPS,
        -left_x * MIN_OPERATOR_WALK_SPEED_MPS,
        -right_x * MIN_OPERATOR_TURN_RATE_RADPS,
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


def dispatch_joystick_locomotion(loco_wrapper, motion_enabled, controller_is_fresh, left_xy, right_xy):
    """Send only enabled, fresh joystick locomotion; stale input releases to zero."""
    locomotion = (0.0, 0.0, 0.0)
    if not motion_enabled:
        return locomotion
    if controller_is_fresh:
        locomotion = joystick_to_locomotion(left_xy, right_xy)
    loco_wrapper.Move(*locomotion)
    return locomotion
