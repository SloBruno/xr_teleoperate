import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.utils.dex3_controls import (
    TriggerHold, TRIGGER_HOLD_MAX_AGE_S, trigger_sample_usable, open_reasons)


def test_fresh_passes_and_clamps():
    h = TriggerHold()
    i = h.update(1.0, 100.0, 100.1)
    assert i["trigger_effective"] == 1.0 and not i["stale"] and i["trigger_raw"] == 1.0
    assert h.update(7.0, 100.2, 100.25)["trigger_effective"] == 1.0


def test_stale_holds_last_valid_until_limit_then_opens():
    h = TriggerHold()
    h.update(1.0, 100.0, 100.1)
    i = h.update(0.0, 100.0, 100.4)  # age 0.4 s: stale but inside hold
    assert i["stale"] and i["trigger_state"] == "held" and i["trigger_effective"] == 1.0
    i = h.update(0.0, 100.0, 100.0 + TRIGGER_HOLD_MAX_AGE_S + 0.05)
    assert i["trigger_state"] == "expired" and i["trigger_effective"] == 0.0
    assert i["expired_count"] == 1 and i["dropouts"] == 1


def test_missing_or_invalid_timestamp_fails_open():
    h = TriggerHold()
    h.update(1.0, 100.0, 100.1)
    for ts in (0.0, float("nan")):
        assert h.update(1.0, ts, 100.2)["trigger_effective"] == 0.0
    h.update(1.0, 100.3, 100.31)
    assert h.update(1.0, 200.0, 100.4)["trigger_effective"] == 0.0  # future sample


def test_released_trigger_stays_zero_while_held():
    h = TriggerHold()
    h.update(0.0, 100.0, 100.1)
    assert h.update(0.0, 100.0, 100.4)["trigger_effective"] == 0.0


def test_gap_and_rate_stats():
    h = TriggerHold()
    for ts in (100.0, 100.1, 101.35, 101.45):
        i = h.update(1.0, ts, ts)
    assert i["gap_max_s"] == 1.25 and i["update_hz"] > 1


def test_usable_window():
    assert trigger_sample_usable(100.0, 100.45)
    assert not trigger_sample_usable(100.0, 100.55)
    assert not trigger_sample_usable(0.0, 1.0)


def test_open_reasons():
    assert open_reasons("expired", 0.0, None) == ["stale_expired"]
    assert open_reasons("fresh", 0.0, None) == ["trigger_low"]
    f = {"state_stale": False, "fault": [False] * 7, "stall": [False] * 3 + [True] * 4,
         "grip_hold": [False] * 7, "derate": [1.0] * 7}
    assert "protection_relax" in open_reasons("fresh", 1.0, f)
    f["state_stale"] = True
    assert "state_stale" in open_reasons("fresh", 1.0, f)


def test_grip_latch_holds_1_25_second_dropout_and_releases_only_after_fresh_low_debounce():
    from teleop.utils.dex3_controls import GripLatch
    h = GripLatch()
    assert h.update(1.0, 100.0, 100.0)["grip_latch_state"] == "active"
    held = h.update(0.0, 100.0, 101.25)
    assert held["trigger_effective"] == 1.0 and held["grip_latch_state"] == "held_stale"
    # Fresh low starts evidence; one frame cannot release a grip.
    low = h.update(0.0, 101.26, 101.26)
    assert low["trigger_effective"] == 1.0 and low["fresh_low_count"] == 1
    released = h.update(0.0, 101.87, 101.87)
    assert released["grip_latch_state"] == "released"
    assert released["trigger_effective"] == 0.0
    assert released["fresh_low_duration_s"] >= 0.6


def test_grip_latch_expiry_is_two_seconds_and_rejects_future_or_invalid_timestamps():
    from teleop.utils.dex3_controls import GripLatch
    h = GripLatch()
    h.update(1.0, 10.0, 10.0)
    assert h.update(0.0, 10.0, 11.25)["trigger_effective"] == 1.0
    expired = h.update(0.0, 10.0, 12.01)
    assert expired["grip_latch_state"] == "expired" and expired["trigger_effective"] == 0.0
    h.update(1.0, 20.0, 20.0)
    invalid = h.update(1.0, 21.0, 20.1)
    assert invalid["grip_latch_state"] == "invalid" and invalid["trigger_effective"] == 0.0


def test_grip_latches_are_side_independent_and_explicit_stop_wins():
    from teleop.utils.dex3_controls import GripLatch
    left, right = GripLatch(), GripLatch()
    left.update(1.0, 30.0, 30.0)
    right.update(1.0, 30.0, 30.0)
    assert left.update(0.0, 30.1, 30.1)["trigger_effective"] == 1.0
    assert right.update(1.0, 30.1, 30.1)["trigger_effective"] == 1.0
    stopped = right.update(1.0, 30.2, 30.2, stop=True)
    assert stopped["grip_latch_state"] == "stopped" and stopped["trigger_effective"] == 0.0
