import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import analyze_dex3_trigger_path as a


def rec(t, q, reason=None):
    j = [{}] * 3 + [{"q": q}] * 4
    return {"event": "full_pose_telemetry", "timestamp_monotonic": t, "dex3": {"right": {"extended": {
        "state": {"joints": j, "hand": {"pressure_max": [5.0]}},
        "published_command": {"q": [0, 0, 0, 1.15, 1.15, 1.15, 1.15]},
        "trigger_path": {"trigger_raw": 0.0, "age_ms": 400.0, "stale": True, "trigger_state": "expired",
                         "open_reasons": reason or ["stale_expired"], "gap_max_s": 1.2}}}}}


def test_detects_open_event_with_reason_and_histograms():
    recs = [rec(t / 10, 1.1) for t in range(5)] + [rec(0.5 + t / 10, 0.6) for t in range(5)]
    out = a.analyze(recs)["right"]
    assert len(out["open_events"]) >= 1 and out["open_events"][0]["reasons"] == ["stale_expired"]
    assert out["age_hist_ms"] and out["max_pressure"] == 5.0


def test_detects_measured_open_even_when_q_cmd_changes_with_it():
    recs = [rec(0.0, 1.2), rec(0.2, 0.7, ["controller_loss_timeout"])]
    recs[1]["dex3"]["right"]["extended"]["published_command"]["q"][3:] = [0.0] * 4
    out = a.analyze(recs)["right"]
    assert out["open_events"]
    assert out["open_events"][0]["command_changed"] is True
