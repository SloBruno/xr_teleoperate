#!/usr/bin/env python3
"""Read-only: open events, trigger gap/age histograms, pressure vs closure from pose-telemetry.

Uses dex3.<side>.extended.trigger_path (raw trigger, age_ms, stale, open_reasons, q_cmd,
pressure_peak). An open event = all 4 long fingers (slots 3-6) retreat > OPEN_DROP_RAD
within OPEN_WINDOW_S of measured q while the commanded q stays constant (+-0.05).
Usage: analyze_dex3_trigger_path.py FILE.jsonl [...]
"""
from __future__ import annotations
import json, sys
from collections import Counter

OPEN_DROP_RAD = 0.2
OPEN_WINDOW_S = 1.0
CMD_CONST_RAD = 0.05
BUCKETS = (50, 100, 250, 500, 1000, 2000)


def load(paths):
    for p in paths:
        with open(p, encoding="utf-8") as h:
            for line in h:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("event") == "full_pose_telemetry":
                    yield r


def hist(values):
    c = Counter()
    for v in values:
        for b in BUCKETS:
            if v <= b:
                c[f"<={b}ms"] += 1
                break
        else:
            c[f">{BUCKETS[-1]}ms"] += 1
    return dict(c)


def analyze(records):
    rows = {"left": [], "right": []}
    for r in records:
        t = r.get("timestamp_monotonic")
        for side in rows:
            ext = ((r.get("dex3") or {}).get(side) or {}).get("extended") or {}
            st = (ext.get("state") or {}).get("joints") or []
            q = [j.get("q") for j in st[3:7]] if len(st) >= 7 else None
            cmd = (ext.get("published_command") or {}).get("q")
            cmd = cmd[3:7] if cmd and len(cmd) >= 7 else None
            tp = ext.get("trigger_path") or {}
            pr = ((st and (ext.get("state") or {}).get("hand") or {}).get("pressure_max")) or []
            pmax = max([p for p in pr if p is not None], default=None)
            if tp.get("pressure_peak") is not None:
                pmax = max(pmax or 0, tp["pressure_peak"])
            if t is not None and q and None not in q:
                rows[side].append((t, q, cmd, tp, pmax))
    out = {}
    for side, rs in rows.items():
        events, ages, gaps, state_gaps = [], [], [], []
        grace_counts = Counter()
        i = 0
        for k, (t, q, cmd, tp, pmax) in enumerate(rs):
            if tp.get("age_ms") is not None:
                ages.append(tp["age_ms"])
            if tp.get("gap_max_s") is not None:
                gaps.append(tp["gap_max_s"] * 1000)
            if tp.get("state_grace_state") is not None:
                grace_counts[tp["state_grace_state"]] += 1
            if tp.get("state_gap_max_s") is not None:
                state_gaps.append(tp["state_gap_max_s"] * 1000)
            while rs[i][0] < t - OPEN_WINDOW_S:
                i += 1
            t0, q0, cmd0 = rs[i][0], rs[i][1], rs[i][2]
            if cmd and cmd0 and all(abs(a) > 0.5 for a in q0):
                command_changed = not all(abs(a - b) <= CMD_CONST_RAD for a, b in zip(cmd, cmd0))
                drop = [abs(a) - abs(b) for a, b in zip(q0, q)]
                # A measured retreat is an opening event even if q_cmd moved at
                # the same time: the old const-command guard hid the exact
                # controller-caused openings this tool must report.
                if all(d > OPEN_DROP_RAD for d in drop):
                    if not events or t - events[-1]["t"] > 2.0:
                        events.append({"t": round(t, 3), "drops": [round(d, 2) for d in drop],
                                       "command_changed": command_changed,
                                       "trigger_raw": tp.get("trigger_raw"), "trigger_effective": tp.get("trigger_effective"),
                                       "age_ms": tp.get("age_ms"), "grip_latch_state": tp.get("grip_latch_state"),
                                       "trigger_state": tp.get("trigger_state"),
                                       "state_grace_state": tp.get("state_grace_state"),
                                       "state_grace_reason": tp.get("state_grace_reason"),
                                       "state_age_ms": tp.get("state_age_ms"),
                                       "state_gap_max_s": tp.get("state_gap_max_s"),
                                       "reasons": tp.get("open_reasons") or [tp.get("exact_open_reason") or "unknown (no trigger-path telemetry)"],
                                       "pressure_peak": pmax})
        contact = [r for r in rs if (r[4] or 0) > 0]
        out[side] = {"records": len(rs), "open_events": events, "age_hist_ms": hist(ages),
                     "trigger_gap_max_hist_ms": hist(gaps),
                     "state_grace_counts": dict(grace_counts),
                     "state_gap_max_ms": None if not state_gaps else round(max(state_gaps), 3),
                     "stale_records": sum(1 for r in rs if r[3].get("stale")),
                     "records_with_pressure": len(contact),
                     "max_pressure": max([r[4] for r in rs if r[4] is not None], default=None)}
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(json.dumps(analyze(load(sys.argv[1:])), indent=2))
