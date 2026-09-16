import math


def _clamp_stick_value(value):
    value = float(value)
    if math.isnan(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def joystick_to_locomotion(left_xy, right_xy):
    left_x, left_y = (_clamp_stick_value(value) for value in left_xy)
    right_x, _ = (_clamp_stick_value(value) for value in right_xy)
    return (-left_y, -left_x, -right_x)
