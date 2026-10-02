#!/usr/bin/env python3
"""Find uncommanded body motion in pose-telemetry JSONL and correlate it.

Usage: python tools/analyze_balance_drift.py pose-telemetry-*.jsonl [--json]
       [--speed 0.05] [--min-duration 0.3]

Needs records written with G1_BALANCE_TELEMETRY=1 (``balance`` block).
For each interval where the locomotion command is zero but the measured body
speed exceeds the threshold, reports duration, speed, step events, modelled
CoM shift, arm tau_est, pelvis/torso rpy vs the pre-session baseline and the
command actually published.  Read-only; no robot access.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _get(d, *path):
    for p in path:
        if isinstance(d, dict):
            d = d.get(p)
        elif isinstance(d, list) and isinstance(p, int) and -len(d) <= p < len(d):
            d = d[p]
        else:
            return None
    return d


def pearson(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pairs) / math.sqrt(sxx * syy)


def load(path):
    samples, bad = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if rec.get("event") != "full_pose_telemetry" or not isinstance(rec.get("balance"), dict):
                continue
            t = _num(rec.get("timestamp_monotonic"))
            if t is None:
                continue
            samples.append(rec)
    samples.sort(key=lambda r: r["timestamp_monotonic"])
    return samples, bad


def _row(rec, speed_threshold):
    b = rec["balance"]
    d = b.get("derived") or {}
    speed = _num(d.get("horizontal_speed"))
    cmd = b.get("loco_command")
    cmd_abs = max((abs(_num(v) or 0.0) for v in cmd), default=0.0) if isinstance(cmd, list) else None
    command_zero = d.get("command_zero")
    if command_zero is None and cmd_abs is not None:
        command_zero = cmd_abs < 1e-3
    tau = _get(b, "arms", "tau_est")
    tau_abs = max((abs(_num(v) or 0.0) for v in tau), default=None) if isinstance(tau, list) else None
    return {
        "t": rec["timestamp_monotonic"],
        "utc": rec.get("timestamp_utc"),
        "speed": speed,
        "uncommanded": bool(command_zero) and speed is not None and speed > speed_threshold,
        "commanded": command_zero is False,
        "cmd_abs": cmd_abs,
        "com_dx": _num(_get(b, "com", "dx_mm")),
        "com_dy": _num(_get(b, "com", "dy_mm")),
        "tau_abs": tau_abs,
        "pelvis_roll": _num(_get(b, "imu_pelvis", "rpy", 0)),
        "pelvis_pitch": _num(_get(b, "imu_pelvis", "rpy", 1)),
        "torso_roll": _num(_get(b, "imu_torso", "rpy", 0)),
        "torso_pitch": _num(_get(b, "imu_torso", "rpy", 1)),
        "pt_pitch": _num(_get(d, "pelvis_minus_torso_rpy", 1)),
        "steps": len(d.get("step_events") or []),
        "config_changes": b.get("config_changes") or [],
    }


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _max(vals, key=lambda v: v):
    vals = [v for v in vals if v is not None]
    return max(vals, key=key) if vals else None


def analyze(path, speed_threshold=0.05, min_duration_s=0.3, gap_s=0.25):
    samples, bad = load(path)
    rows = [_row(r, speed_threshold) for r in samples]
    quiet = [r for r in rows if not r["uncommanded"] and not r["commanded"]]
    base_pitch = _mean([r["pelvis_pitch"] for r in quiet])
    base_roll = _mean([r["pelvis_roll"] for r in quiet])
    intervals, cur = [], []

    def close():
        if not cur:
            return
        dur = cur[-1]["t"] - cur[0]["t"]
        if dur >= min_duration_s:
            pp = _max([r["pelvis_pitch"] for r in cur], key=lambda v: abs(v - (base_pitch or 0.0)))
            pr = _max([r["pelvis_roll"] for r in cur], key=lambda v: abs(v - (base_roll or 0.0)))
            intervals.append({
                "start_monotonic": cur[0]["t"], "end_monotonic": cur[-1]["t"],
                "start_utc": cur[0]["utc"], "end_utc": cur[-1]["utc"],
                "duration_s": round(dur, 3),
                "samples": len(cur),
                "max_speed": _max([r["speed"] for r in cur]),
                "mean_speed": _mean([r["speed"] for r in cur]),
                "step_events": sum(r["steps"] for r in cur),
                "loco_command_abs_max": _max([r["cmd_abs"] for r in cur]),
                "com_dx_mm_max": _max([r["com_dx"] for r in cur], key=abs),
                "com_dy_mm_max": _max([r["com_dy"] for r in cur], key=abs),
                "arm_tau_abs_max": _max([r["tau_abs"] for r in cur]),
                "pelvis_pitch_delta_vs_baseline": (None if pp is None or base_pitch is None
                                                   else abs(pp - base_pitch)),
                "pelvis_roll_delta_vs_baseline": (None if pr is None or base_roll is None
                                                  else abs(pr - base_roll)),
                "pelvis_minus_torso_pitch_mean": _mean([r["pt_pitch"] for r in cur]),
                "config_changes": [c for r in cur for c in r["config_changes"]],
            })

    for r in rows:
        if r["uncommanded"] and (not cur or r["t"] - cur[-1]["t"] <= gap_s):
            cur.append(r)
            continue
        close()
        cur = [r] if r["uncommanded"] else []
    close()

    commanded_s = 0.0
    for a, b in zip(rows, rows[1:]):
        if a["commanded"] and b["t"] - a["t"] <= gap_s:
            commanded_s += b["t"] - a["t"]
    zero_rows = [r for r in rows if not r["commanded"]]
    speeds = [r["speed"] for r in zero_rows]
    return {
        "file": str(path),
        "samples": len(rows),
        "bad_lines": bad,
        "speed_threshold_mps": speed_threshold,
        "baseline": {"pelvis_pitch": base_pitch, "pelvis_roll": base_roll},
        "intervals": intervals,
        "uncommanded_s": round(sum(i["duration_s"] for i in intervals), 3),
        "commanded_motion_s": round(commanded_s, 3),
        "correlation": {  # over zero-command samples only
            "speed_vs_com_dx": pearson(speeds, [r["com_dx"] for r in zero_rows]),
            "speed_vs_arm_tau": pearson(speeds, [r["tau_abs"] for r in zero_rows]),
            "speed_vs_pelvis_pitch": pearson(speeds, [r["pelvis_pitch"] for r in zero_rows]),
            "speed_vs_pelvis_roll": pearson(speeds, [r["pelvis_roll"] for r in zero_rows]),
            "speed_vs_pelvis_minus_torso_pitch": pearson(speeds, [r["pt_pitch"] for r in zero_rows]),
        },
        "config_changes": [c for r in rows for c in r["config_changes"]],
    }


def _fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--speed", type=float, default=0.05)
    ap.add_argument("--min-duration", type=float, default=0.3)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = analyze(a.jsonl, a.speed, a.min_duration)
    if a.json:
        print(json.dumps(rep, indent=2))
        return 0
    print(f"{rep['file']}: {rep['samples']} amostras com balance, {rep['bad_lines']} linhas ruins")
    print(f"movimento nao comandado: {rep['uncommanded_s']} s em {len(rep['intervals'])} intervalos; "
          f"comandado: {rep['commanded_motion_s']} s")
    for i in rep["intervals"]:
        print(f"  {i['start_utc']} +{i['duration_s']}s v_max={_fmt(i['max_speed'])} passos={i['step_events']} "
              f"cmd={_fmt(i['loco_command_abs_max'])} CoMdx={_fmt(i['com_dx_mm_max'], 0)}mm "
              f"tau={_fmt(i['arm_tau_abs_max'], 1)} dpitch={_fmt(i['pelvis_pitch_delta_vs_baseline'])} "
              f"droll={_fmt(i['pelvis_roll_delta_vs_baseline'])}")
    print("correlacoes (cmd=0):", {k: _fmt(v, 2) for k, v in rep["correlation"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
