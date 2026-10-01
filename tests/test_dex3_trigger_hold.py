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
