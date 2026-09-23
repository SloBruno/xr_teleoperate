import math


STICK_DEADZONE = 0.12
STICK_PRECISION_EXPONENT = 3


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
    return (-left_y, -left_x, -right_x)
