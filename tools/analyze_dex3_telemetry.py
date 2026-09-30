#!/usr/bin/env python3
"""Read-only summary of extended Dex3-1 telemetry from pose-telemetry JSONL.

Per finger (per side): tracking error (cmd - measured), saturation (fraction
of samples with |error| above --sat-threshold), tau_est, temperature max/delta,
hand error/lost, state age and receive rate. Never writes or modifies inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

LEFT_NAMES = ["Thumb0", "Thumb1", "Thumb2", "Middle0", "Middle1", "Index0", "Index1"]
RIGHT_NAMES = ["Thumb0", "Thumb1", "Thumb2", "Index0", "Index1", "Middle0", "Middle1"]
NAMES = {"left": LEFT_NAMES, "right": RIGHT_NAMES}


def _num(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_records(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    yield record


def _nonzero(values):
    if isinstance(values, (int, float)):
        values = [values]
    if not isinstance(values, list):
        return False
    return any((_num(v) or 0) != 0 for v in values)


def _max(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _min(values):
    values = [v for v in values if v is not None]
    return min(values) if values else None


def _r(value, digits=6):
    return None if value is None else round(value, digits)


def summarize(records, sat_threshold=0.3):
    acc = {}
    for record in records:
        dex3 = record.get("dex3") if isinstance(record, dict) else None
        if not isinstance(dex3, dict):
            continue
        for side in ("left", "right"):
            ext = (dex3.get(side) or {}).get("extended") if isinstance(dex3.get(side), dict) else None
            if not isinstance(ext, dict):
                continue
            s = acc.setdefault(side, {"joints": [dict(err=[], tau=[], temp=[], sat=0, n=0, dq=[], modes=set())
                                                   for _ in range(7)],
                                      "age": [], "rate": [], "err_nz": 0, "lost_nz": 0,
                                      "power_a": [], "power_v": [], "records": 0})
            s["records"] += 1
            s["age"].append(_num(ext.get("state_age_ms")))
            s["rate"].append(_num(ext.get("rate_hz")))
            state = ext.get("state") or {}
            cmd_q = ((ext.get("published_command") or {}).get("q")) or []
            for i, joint in enumerate((state.get("joints") or [])[:7]):
                if not isinstance(joint, dict):
                    continue
                j = s["joints"][i]
                j["n"] += 1
                q = _num(joint.get("q"))
                cq = _num(cmd_q[i]) if i < len(cmd_q) else None
                if q is not None and cq is not None:
                    err = abs(cq - q)
                    j["err"].append(err)
                    if err > sat_threshold:
                        j["sat"] += 1
                j["tau"].append(_num(joint.get("tau_est")))
                j["dq"].append(_num(joint.get("dq")))
                temps = joint.get("temperature")
                if isinstance(temps, list):
                    t = _max([_num(x) for x in temps])
                    if t is not None:
                        j["temp"].append(t)
                if joint.get("mode") is not None:
                    j["modes"].add(joint.get("mode"))
            hand = state.get("hand") or {}
            if _nonzero(hand.get("error")):
                s["err_nz"] += 1
            if _nonzero(hand.get("pressure_lost")):
                s["lost_nz"] += 1
            s["power_a"].append(_num(hand.get("power_a")))
            s["power_v"].append(_num(hand.get("power_v")))
    summary = {}
    for side, s in acc.items():
        joints = {}
        for i, j in enumerate(s["joints"]):
            tau_abs = [abs(v) for v in j["tau"] if v is not None]
            joints[i] = {
                "name": NAMES[side][i],
                "samples": j["n"],
                "tracking_error_max": _r(_max(j["err"])),
                "tracking_error_mean": _r(sum(j["err"]) / len(j["err"])) if j["err"] else None,
                "saturated_fraction": _r(j["sat"] / len(j["err"])) if j["err"] else None,
                "tau_est_abs_max": _r(_max(tau_abs)),
                "dq_abs_max": _r(_max([abs(v) for v in j["dq"] if v is not None])),
                "temperature_max": _r(_max(j["temp"])),
                "temperature_delta": _r(_max(j["temp"]) - j["temp"][0]) if j["temp"] else None,
                "modes_seen": sorted(j["modes"]),
            }
        summary[side] = {
            "records": s["records"],
            "joints": joints,
            "hand": {
                "error_nonzero_samples": s["err_nz"],
                "lost_nonzero_samples": s["lost_nz"],
                "power_a_max": _r(_max(s["power_a"])),
                "power_v_min": _r(_min(s["power_v"])),
            },
            "state_age_ms_max": _max(s["age"]),
            "rate_hz_min": _r(_min(s["rate"])),
        }
    return summary


def format_text(summary):
    lines = []
    for side, s in summary.items():
        lines.append(f"== {side}: {s['records']} records; age_max={s['state_age_ms_max']} ms; "
                     f"rate_min={s['rate_hz_min']} Hz; err_nz={s['hand']['error_nonzero_samples']}; "
                     f"lost_nz={s['hand']['lost_nonzero_samples']}; power_a_max={s['hand']['power_a_max']}")
        lines.append(f"{'joint':8} {'n':>5} {'err_max':>8} {'err_mean':>8} {'sat%':>6} {'|tau|max':>9} "
                     f"{'|dq|max':>8} {'T_max':>6} {'dT':>5}")
        for j in s["joints"].values():
            def f(v, w, p=3):
                return f"{'-':>{w}}" if v is None else f"{v:{w}.{p}f}"
            sat = None if j["saturated_fraction"] is None else 100 * j["saturated_fraction"]
            lines.append(f"{j['name']:8} {j['samples']:5d} {f(j['tracking_error_max'], 8)} "
                         f"{f(j['tracking_error_mean'], 8)} {f(sat, 6, 1)} {f(j['tau_est_abs_max'], 9)} "
                         f"{f(j['dq_abs_max'], 8)} {f(j['temperature_max'], 6, 1)} {f(j['temperature_delta'], 5, 1)}")
    return "\n".join(lines) if lines else "no extended dex3 records found"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--sat-threshold", type=float, default=0.3, help="|cmd-meas| rad counted as saturated/stalled")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    summary = summarize(load_records(args.files), args.sat_threshold)
    print(json.dumps(summary, indent=2) if args.json else format_text(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
