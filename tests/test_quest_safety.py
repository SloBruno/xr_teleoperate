import math
import importlib

import pytest


def safety_api():
    try:
        return importlib.import_module("teleop.utils.quest_safety")
    except ImportError as error:
        pytest.fail(f"quest safety API is missing: {error}")


def test_controller_sample_freshness_limit_is_documented_and_monotonic():
    api = safety_api()
    CONTROLLER_SAMPLE_FRESHNESS_LIMIT_S = api.CONTROLLER_SAMPLE_FRESHNESS_LIMIT_S
    controller_sample_is_fresh = api.controller_sample_is_fresh
    assert CONTROLLER_SAMPLE_FRESHNESS_LIMIT_S == 0.25
    assert controller_sample_is_fresh(10.0, now=10.25)
    assert not controller_sample_is_fresh(10.0, now=10.250001)
    assert not controller_sample_is_fresh(0.0, now=0.1)


def test_hand_pose_freshness_uses_the_same_monotonic_limit():
    hand_sample_is_fresh = safety_api().hand_sample_is_fresh
    assert hand_sample_is_fresh(10.0, now=10.25)
    assert not hand_sample_is_fresh(10.0, now=10.250001)
    assert not hand_sample_is_fresh(0.0, now=0.1)


def test_stale_joystick_is_replaced_with_exact_zero_velocity_input():
    fresh_controller_value = safety_api().fresh_controller_value
    assert fresh_controller_value((0.8, -0.4), sample_timestamp=1.0, now=1.251) == (0.0, 0.0)


def test_fresh_hand_mode_sticks_remain_available_for_move_mapping():
    fresh_controller_value = safety_api().fresh_controller_value
    from teleop.utils.quest_controls import joystick_to_locomotion

    left = fresh_controller_value((0.8, -0.4), sample_timestamp=1.0, now=1.1)
    right = fresh_controller_value((0.0, 0.0), sample_timestamp=1.0, now=1.1)
    expected_forward = 0.15 * ((0.4 - 0.12) / (1.0 - 0.12)) ** 3
    expected_lateral = -0.15 * ((0.8 - 0.12) / (1.0 - 0.12)) ** 3
    assert joystick_to_locomotion(left, right) == pytest.approx(
        (expected_forward, expected_lateral, 0.0)
    )


def test_stale_left_and_right_triggers_neutralize_independently():
    fresh_controller_value = safety_api().fresh_controller_value
    assert fresh_controller_value(0.9, sample_timestamp=1.0, now=1.3) == 0.0
    assert fresh_controller_value(0.2, sample_timestamp=2.0, now=2.1) == 0.2


def test_nonfinite_controller_timestamp_is_not_fresh():
    controller_sample_is_fresh = safety_api().controller_sample_is_fresh
    assert not controller_sample_is_fresh(math.nan, now=1.0)


def test_status_locomotion_carries_stick_snapshot_and_survives_bad_snapshot():
    from teleop.utils.teleop_status import TeleopStatusMonitor
    out = []
    mon = TeleopStatusMonitor(out.append)
    snap = {"raw_left_xy": [0.0, -1.0], "command": [0.15, 0.0, 0.0], "last_move_code": 0}
    s = mon.observe(now=10.0, lifecycle="tracking", controller_sample_timestamp=10.0,
                    motion_enabled=True, locomotion=(0.15, 0.0, 0.0), stick=snap)
    assert s["locomotion"]["command"] == [0.15, 0.0, 0.0]
    assert s["locomotion"]["raw_left_xy"] == [0.0, -1.0]
    assert s["locomotion"]["last_move_code"] == 0
    s2 = TeleopStatusMonitor(out.append).observe(now=1.0, lifecycle="tracking",
        controller_sample_timestamp=1.0, stick=5)
    assert s2 is None or "command" in s2["locomotion"]
