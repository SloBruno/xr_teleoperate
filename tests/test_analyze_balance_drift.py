"""Offline balance-drift analyzer on a synthetic JSONL."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import analyze_balance_drift as abd


def rec(t, speed, cmd=0.0, dx=0.0, tau=1.0, pitch=0.04, steps=()):
    return {
        "event": "full_pose_telemetry", "timestamp": 1000.0 + t, "timestamp_monotonic": 50.0 + t,
        "timestamp_utc": f"T{t:.2f}", "lifecycle": "tracking",
        "locomotion": {"stick": {"left": [0.0, 0.0]}},
        "balance": {
            "loco_command": [cmd, 0.0, 0.0],
            "imu_pelvis": {"rpy": [0.0, pitch, 0.0]}, "imu_torso": {"rpy": [0.0, pitch + 0.05, 0.0]},
            "arms": {"tau_est": [tau] * 14, "q": [0.0] * 14},
            "com": {"level": "ok", "dx_mm": dx, "dy_mm": 0.0},
            "derived": {"horizontal_speed": speed, "body_velocity": [speed, 0.0, 0.0],
                        "command_zero": cmd == 0.0, "uncommanded_motion": cmd == 0.0 and speed > 0.05,
                        "pelvis_minus_torso_rpy": [0.0, -0.05, 0.0],
                        "step_events": [{"foot": f, "contact": False} for f in steps]},
            "config_changes": [],
        },
    }


def synthetic(path):
    rows = [rec(i * 0.05, 0.0) for i in range(40)]                        # quiet 2 s
    rows += [rec(2 + i * 0.05, 0.15, dx=60.0, tau=9.0, pitch=0.09, steps=(0,) if i == 3 else ())
             for i in range(20)]                                          # 1 s drift, arm out
    rows += [rec(3 + i * 0.05, 0.3, cmd=0.3) for i in range(20)]           # commanded walk
    rows.insert(5, {"event": "tracking_started", "timestamp_monotonic": 50.2})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n{broken\n")
    return path


def test_finds_single_uncommanded_interval_and_correlates(tmp_path):
    rep = abd.analyze(synthetic(tmp_path / "t.jsonl"), min_duration_s=0.3)
    assert rep["samples"] == 80 and rep["bad_lines"] == 1
    assert len(rep["intervals"]) == 1
    iv = rep["intervals"][0]
    assert abs(iv["duration_s"] - 0.95) < 0.06
    assert iv["max_speed"] == 0.15 and iv["step_events"] == 1
    assert iv["com_dx_mm_max"] == 60.0 and iv["arm_tau_abs_max"] == 9.0
    assert iv["loco_command_abs_max"] == 0.0
    assert iv["pelvis_pitch_delta_vs_baseline"] > 0.04
    assert rep["correlation"]["speed_vs_com_dx"] > 0.8
    assert rep["commanded_motion_s"] > 0.9


def test_cli_prints_json(tmp_path, capsys):
    p = synthetic(tmp_path / "t.jsonl")
    assert abd.main([str(p), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["intervals"]) == 1


def test_no_balance_block_is_reported(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"event": "full_pose_telemetry", "timestamp_monotonic": 1.0}) + "\n")
    rep = abd.analyze(p)
    assert rep["samples"] == 0 and rep["intervals"] == []
