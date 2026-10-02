from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def command(v=0.6):
    return np.array([0.0, 0.0, 0.0, v, v, v, v])


def test_active_grip_holds_last_protected_command_across_long_state_gaps():
    from teleop.utils.dex3_state_grace import Dex3StateGrace

    grace = Dex3StateGrace(grace_s=1.5)
    protected = command()
    fresh = grace.update(10.0, fresh=True, grip_active=True, safe=True,
                         q_cmd=protected, enable=[True] * 7)
    assert fresh["state"] == "fresh"

    for now in (10.2, 10.7, 11.25):
        held = grace.update(now, fresh=False, grip_active=True, safe=False)
        assert held["state"] == "holding_no_feedback"
        np.testing.assert_allclose(held["q_cmd"], protected)
        assert held["enable"] == [True] * 7
        assert held["held_command"] == protected.tolist()

    # A quiet state topic is not a command to open a latched grip.
    held = grace.update(12.01, fresh=False, grip_active=True, safe=False)
    assert held["state"] == "holding_no_feedback"
    np.testing.assert_allclose(held["q_cmd"], protected)
    assert held["enable"] == [True] * 7
    assert held["reason"] == "cached_protected_command"
    assert held["gap_count"] == 1
    assert held["gap_max_s"] >= 1.8


def test_grace_never_holds_without_safe_fresh_feedback_or_active_grip():
    from teleop.utils.dex3_state_grace import Dex3StateGrace

    grace = Dex3StateGrace(grace_s=1.5)
    fallback = command(0.4)
    no_cache = grace.update(1.0, fresh=False, grip_active=True, safe=False,
                            fallback_q=fallback)
    assert no_cache["state"] == "holding_no_feedback"
    assert no_cache["reason"] == "no_cached_protected_command"
    np.testing.assert_allclose(no_cache["q_cmd"], fallback)
    grace.update(2.0, fresh=True, grip_active=True, safe=False,
                 q_cmd=command(), enable=[True] * 7)
    assert grace.update(2.2, fresh=False, grip_active=True, safe=False)["state"] == "expired"

    grace.update(3.0, fresh=True, grip_active=True, safe=True,
                 q_cmd=command(), enable=[True] * 7)
    released = grace.update(3.2, fresh=False, grip_active=False, safe=False)
    assert released["state"] == "expired"
    assert released["reason"] == "grip_not_active"


def test_fresh_recovery_recomputes_and_emits_only_gap_transitions():
    from teleop.utils.dex3_state_grace import Dex3StateGrace

    grace = Dex3StateGrace(grace_s=1.5)
    grace.update(1.0, fresh=True, grip_active=True, safe=True,
                 q_cmd=command(), enable=[True] * 7)
    assert grace.update(1.2, fresh=False, grip_active=True, safe=False)["warning"] == "state_gap_started"
    assert grace.update(1.3, fresh=False, grip_active=True, safe=False)["warning"] is None
    recovered = grace.update(1.4, fresh=True, grip_active=True, safe=True,
                             q_cmd=command(0.4), enable=[True] * 7)
    assert recovered["state"] == "fresh"
    assert recovered["warning"] == "state_gap_recovered"
    np.testing.assert_allclose(recovered["q_cmd"], command(0.4))


def test_short_gap_holds_only_joints_with_safe_fresh_feedback():
    from teleop.utils.dex3_state_grace import Dex3StateGrace

    grace = Dex3StateGrace(grace_s=1.5)
    protected = np.array([0.0, -0.25, -0.35, -0.41, -0.42, -0.43, -0.44])
    # Thumb1 faulted/disabled; long fingers have an active protected grip.
    safe = [True, False, True, True, True, True, True]
    enable = [True, False, True, True, True, True, True]
    grace.update(10.0, fresh=True, grip_active=True, safe=safe,
                 q_cmd=protected, enable=enable)

    held = grace.update(10.2, fresh=False, grip_active=True, safe=False)

    assert held["state"] == "holding_no_feedback"
    assert held["q_cmd"][1] == 0.0
    assert held["enable"][1] is False
    np.testing.assert_allclose(held["q_cmd"][3:7], protected[3:7])
    assert held["enable"][3:7] == [True] * 4
    assert held["state_grace_joint_hold"] == [0, 2, 3, 4, 5, 6]
    assert held["state_grace_blocked_joints"] == [1]


def test_short_gap_opens_a_thermal_or_faulted_long_finger_instead_of_holding_it():
    from teleop.utils.dex3_state_grace import Dex3StateGrace

    grace = Dex3StateGrace(grace_s=1.5)
    protected = command()
    # Finger1 (joint 4) is thermally cut off; it must not inherit a hold.
    safe = [True, True, True, True, False, True, True]
    enable = [True, True, True, True, False, True, True]
    grace.update(20.0, fresh=True, grip_active=True, safe=safe,
                 q_cmd=protected, enable=enable)

    held = grace.update(20.2, fresh=False, grip_active=True, safe=False)

    assert held["q_cmd"][4] == 0.0
    assert held["enable"][4] is False
    np.testing.assert_allclose(held["q_cmd"][[3, 5, 6]], protected[[3, 5, 6]])
    assert held["state_grace_blocked_joints"] == [4]


def test_small_callback_race_future_timestamp_is_fresh_but_real_future_is_not():
    from teleop.utils.dex3_state_grace import state_is_fresh

    # The subscriber can stamp after the control loop captured `now`.
    assert state_is_fresh(10.0, {"timestamp": 10.035}) is True
    assert state_is_fresh(10.0, {"timestamp": 10.100}) is False


def test_future_state_timestamp_never_clears_an_active_protected_grip():
    from teleop.utils.dex3_state_grace import Dex3StateGrace

    protected = command(0.6)
    grace = Dex3StateGrace()
    grace.update(10.0, fresh=True, grip_active=True, safe=True,
                 q_cmd=protected, enable=[True] * 7)
    held = grace.update(10.1, fresh=False, grip_active=True, safe=False,
                        gap_eligible=False)
    assert held["state"] == "holding_no_feedback"
    np.testing.assert_allclose(held["q_cmd"], protected)
