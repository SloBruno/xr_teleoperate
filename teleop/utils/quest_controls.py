import math


STICK_DEADZONE = 0.12
STICK_PRECISION_EXPONENT = 3
# LocoClient.Move accepts vx/vy in m/s and vyaw in rad/s.  The SDK only
# serializes floats and declares no positive minimum, so these are the
# smallest reviewed non-zero operator caps; validate them on hardware before
# reducing them further.
MIN_OPERATOR_WALK_SPEED_MPS = 0.10
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


def dispatch_joystick_locomotion(loco_wrapper, motion_enabled, controller_is_fresh, left_xy, right_xy):
    """Send only enabled, fresh joystick locomotion; stale input releases to zero."""
    locomotion = (0.0, 0.0, 0.0)
    if not motion_enabled:
        return locomotion
    if controller_is_fresh:
        locomotion = joystick_to_locomotion(left_xy, right_xy)
    loco_wrapper.Move(*locomotion)
    return locomotion
