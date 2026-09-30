import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "analyze_dex3_telemetry.py"


def load():
    spec = importlib.util.spec_from_file_location("analyze_dex3_telemetry", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def joint(q, tau, temp=None):
    j = {"q": q, "dq": 0.0, "tau_est": tau}
    if temp is not None:
        j.update({"temperature": [temp, temp + 1], "mode": 17, "motorstate": 0})
    return j


def record(t, cmd_q, meas_q, tau, temp=None, err=None, lost=None, age=20, rate=100.0, side="left"):
    hand = {"power_v": 24.0, "power_a": 0.5} if temp is not None else {}
    if err is not None:
        hand["error"] = err
    if lost is not None:
        hand["pressure_lost"] = lost
    ext = {"state": {"joints": [joint(meas_q, tau, temp)] * 7, "hand": hand},
           "state_age_ms": age, "rate_hz": rate,
           "published_command": {"q": [cmd_q] * 7, "kp": [1.5] * 7}}
    return {"event": "full_pose_telemetry", "timestamp_monotonic": t,
            "dex3": {side: {"extended": ext}, "right": {"extended": None}}}


def write(tmp_path, records):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n{broken\n")
    return path


def test_summary_per_finger_tracking_saturation_tau_temp_lost_age(tmp_path):
    recs = [
        record(1.0, 1.05, 1.0, 0.4, temp=30),
        record(2.0, 1.05, 0.3, 2.0, temp=34, err=[0, 0], lost=[0]),
        record(3.0, 1.05, 0.2, 2.6, temp=40, err=[4, 0], lost=[2], age=150),
    ]
    mod = load()
    summary = mod.summarize(mod.load_records([write(tmp_path, recs)]))
    j1 = summary["left"]["joints"][1]
    assert j1["name"] == "Thumb1"
    assert j1["samples"] == 3
    assert j1["tracking_error_max"] == 0.85
    assert j1["tau_est_abs_max"] == 2.6
    assert j1["temperature_max"] == 41 and j1["temperature_delta"] == 10
    assert j1["saturated_fraction"] > 0  # error > threshold
    assert summary["left"]["hand"]["error_nonzero_samples"] == 1
    assert summary["left"]["hand"]["lost_nonzero_samples"] == 1
    assert summary["left"]["state_age_ms_max"] == 150
    assert summary["left"]["rate_hz_min"] == 100.0
    assert summary["left"]["hand"]["power_a_max"] == 0.5
    assert "right" not in summary or not summary["right"]["joints"]


def test_cli_prints_json_and_is_read_only(tmp_path):
    path = write(tmp_path, [record(1.0, 0.5, 0.5, 0.1, temp=30)])
    before = path.read_bytes()
    out = subprocess.run([sys.executable, str(SCRIPT), str(path), "--json"],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out)["left"]["joints"]["0"]["name"] == "Thumb0"
    assert path.read_bytes() == before


def test_handles_records_without_extended_and_nulls(tmp_path):
    recs = [{"event": "full_pose_telemetry", "dex3": {"left": {"measured_q": [0] * 7}}},
            {"event": "x"}, record(1.0, None, None, None)]
    mod = load()
    summary = mod.summarize(mod.load_records([write(tmp_path, recs)]))
    assert summary["left"]["joints"][0]["tracking_error_max"] is None
