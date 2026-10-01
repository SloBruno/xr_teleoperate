from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def command(v=0.6):
    return np.array([0.0, 0.0, 0.0, v, v, v, v])


def test_active_grip_holds_last_protected_command_for_short_state_gaps_then_expires():
    from teleop.utils.dex3_state_grace import Dex3StateGrace

    grace = Dex3StateGrace(grace_s=1.5)
    protected = command()
    fresh = grace.update(10.0, fresh=True, grip_active=True, safe=True,
                         q_cmd=protected, enable=[True] * 7)
    assert fresh["state"] == "fresh"

    for now in (10.2, 10.7, 11.25):
        held = grace.update(now, fresh=False, grip_active=True, safe=False)
        assert held["state"] == "holding"
        np.testing.assert_allclose(held["q_cmd"], protected)
        assert held["enable"] == [True] * 7
        assert held["held_command"] == protected.tolist()

    expired = grace.update(12.01, fresh=False, grip_active=True, safe=False)
    assert expired["state"] == "expired"
    assert expired["q_cmd"] is None
    assert expired["reason"] == "state_grace_expired"
    assert expired["gap_count"] == 1
    assert expired["gap_max_s"] >= 1.8


def test_grace_never_holds_without_safe_fresh_feedback_or_active_grip():
    from teleop.utils.dex3_state_grace import Dex3StateGrace

    grace = Dex3StateGrace(grace_s=1.5)
    assert grace.update(1.0, fresh=False, grip_active=True, safe=False)["state"] == "expired"
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
