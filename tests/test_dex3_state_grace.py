from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def command(v=0.6):
    return np.array([0.0, 0.0, 0.0, v, v, v, v])


def test_active_grip_holds_last_protected_command_indefinitely_without_state():
    from teleop.utils.dex3_state_grace import Dex3StateGrace

    grace = Dex3StateGrace(grace_s=1.5)
    protected = command()
    enable = [True, True, False, True, True, True, True]
    fresh = grace.update(10.0, fresh=True, grip_active=True, safe=True,
                         q_cmd=protected, enable=enable)
    assert fresh["state"] == "fresh"

    for now in (10.2, 11.25, 15.0):
        held = grace.update(now, fresh=False, grip_active=True, safe=False)
        assert held["state"] == "holding_no_feedback"
        assert held["reason"] == "cached_protected_command"
        np.testing.assert_allclose(held["q_cmd"], protected)
        assert held["enable"] == enable  # per-joint fault disable is retained
    assert held["hold_duration_s"] == 4.8
    assert held["warning"] is None


def test_active_grip_without_prior_feedback_uses_explicit_bounded_fallback():
    from teleop.utils.dex3_state_grace import Dex3StateGrace

    grace = Dex3StateGrace()
    fallback = command(0.4)
    held = grace.update(1.0, fresh=False, grip_active=True, safe=False,
                        fallback_q_cmd=fallback)
    assert held["state"] == "holding_no_feedback"
    assert held["reason"] == "no_cached_protected_command"
    np.testing.assert_allclose(held["q_cmd"], fallback)
    assert held["enable"] is None
    assert held["warning"] == "state_gap_started"


def test_trigger_release_stops_retention_and_fresh_recovery_is_deduplicated():
    from teleop.utils.dex3_state_grace import Dex3StateGrace

    grace = Dex3StateGrace()
    grace.update(1.0, fresh=True, grip_active=True, safe=True,
                 q_cmd=command(), enable=[True] * 7)
    assert grace.update(1.2, fresh=False, grip_active=True, safe=False)["warning"] == "state_gap_started"
    released = grace.update(1.3, fresh=False, grip_active=False, safe=False)
    assert released["state"] == "expired"
    assert released["reason"] == "grip_not_active"
    recovered = grace.update(1.4, fresh=True, grip_active=True, safe=True,
                             q_cmd=command(0.4), enable=[True] * 7)
    assert recovered["state"] == "fresh"
    assert recovered["warning"] == "state_gap_recovered"
