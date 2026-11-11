#!/usr/bin/env python3
"""Offline back-drive replay of a recorded session through the current Dex3 protector.

Feeds the recorded trigger target (``trigger_path.q_target``, pre-protection) and
the recorded measured hand state into ``Dex3HandProtector`` and compares, for
every stall episode, the command the live (old) controller actually published
with the command the current code would publish while the joint was pushed back.

ESTIMATE ONLY: the measured q was produced while the old controller let the
joint be back-driven with zero resistance; a resisting command would change the
physical motion, so this shows the command/torque the new code would have
applied to those observed states, not the motion that would have resulted.
Pure offline: reads JSONL, never touches DDS/robot.

Usage: replay_dex3_backdrive.py SESSION.jsonl [--side right] [--json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.utils.dex3_protection import (  # noqa: E402
    CLOSE_TORQUE_CEILING_NM, DEX3_KP, Dex3HandProtector, JOINT_NAMES)


def load(path, side):
    """Yield (t_mono, utc, state_joints, q_target, published_q, recorded_stall)."""
    with open(path) as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("event") != "full_pose_telemetry" or rec.get("lifecycle") != "tracking":
                continue
            ext = ((rec.get("dex3") or {}).get(side) or {}).get("extended") or {}
            joints = (ext.get("state") or {}).get("joints")
            tp = ext.get("trigger_path") or {}
            pub = (ext.get("published_command") or {}).get("q")
            if not joints or not tp.get("q_target") or not pub:
                continue
            stall = (ext.get("protection") or {}).get("stall") or [False] * 7
            yield (rec["timestamp_monotonic"], rec.get("timestamp_utc"), joints,
                   tp["q_target"], pub, stall)


def replay(path, side="right", open_pose=None):
    open_pose = np.zeros(7) if open_pose is None else np.asarray(open_pose, float)
    prot = Dex3HandProtector(list(open_pose))
    carry = [dict() for _ in range(7)]
    episodes = {i: [] for i in range(7)}
    cur = {}
    new_close_max = [0.0] * 7
    for t, utc, joints, q_target, pub, rec_stall in load(path, side):
        for i, j in enumerate(joints):
            for k, v in j.items():
                if v is not None:
                    carry[i][k] = v  # slow fields only emitted on change
        state = {
            "timestamp": t,
            "q": [x.get("q") for x in carry], "dq": [x.get("dq") for x in carry],
            "tau": [x.get("tau_est") for x in carry],
            "temp": [max(x["temperature"]) if x.get("temperature") else None for x in carry],
            "mode": [x.get("mode") for x in carry], "motorstate": [x.get("motorstate") for x in carry],
        }
        res = prot.update(t, q_target, state)
        for i in range(7):
            q = state["q"][i]
            if q is None:
                continue
            closing = np.sign(q_target[i] - open_pose[i])
            if closing != 0 and res.enable[i]:
                new_close_max[i] = max(new_close_max[i], DEX3_KP * max(0.0, closing * (res.q_cmd[i] - q)))
            if res.stall[i] and i not in cur:
                hs = prot._stall_sign[i]
                cur[i] = dict(joint=JOINT_NAMES[i], start_utc=utc, anchor=prot._stall_anchor[i],
                              sign=hs, worst_q=q, worst_utc=utc, old_cmd_at_worst=pub[i],
                              new_cmd_at_worst=float(res.q_cmd[i]), max_backdrive_rad=0.0,
                              new_restoring_nm_at_worst=0.0, old_restoring_nm_at_worst=0.0,
                              max_new_restoring_nm=0.0, recorded_stall=bool(rec_stall[i]))
            if i in cur:
                e = cur[i]
                if not res.stall[i]:
                    e["end_utc"] = utc
                    episodes[i].append(cur.pop(i))
                    continue
                back = e["sign"] * (e["anchor"] - q)
                restoring = DEX3_KP * e["sign"] * (res.q_cmd[i] - q) if res.enable[i] else 0.0
                e["max_new_restoring_nm"] = max(e["max_new_restoring_nm"], restoring)
                if back > e["max_backdrive_rad"]:
                    e.update(max_backdrive_rad=back, worst_q=q, worst_utc=utc,
                             old_cmd_at_worst=pub[i], new_cmd_at_worst=float(res.q_cmd[i]),
                             new_restoring_nm_at_worst=restoring,
                             old_restoring_nm_at_worst=DEX3_KP * e["sign"] * (pub[i] - q))
    for i, e in cur.items():
        e["end_utc"] = None
        episodes[i].append(e)
    return {"episodes": episodes, "new_close_max_nm": new_close_max}


def main(argv):
    side = argv[argv.index("--side") + 1] if "--side" in argv else "right"
    out = replay(argv[1], side)
    if "--json" in argv:
        print(json.dumps(out, indent=1))
        return 0
    for i, eps in out["episodes"].items():
        for e in eps:
            print(f"{side} {e['joint']}: stall {e['start_utc']}..{e['end_utc']} anchor {e['anchor']:+.3f} | "
                  f"worst q {e['worst_q']:+.3f} ({e['max_backdrive_rad']:.2f} rad back) @ {e['worst_utc']} | "
                  f"old cmd {e['old_cmd_at_worst']:+.3f} ({e['old_restoring_nm_at_worst']:+.2f} N*m) -> "
                  f"new cmd {e['new_cmd_at_worst']:+.3f} ({e['new_restoring_nm_at_worst']:+.2f} N*m)")
    for i in range(7):
        print(f"{side} {JOINT_NAMES[i]}: new max closing implicit {out['new_close_max_nm'][i]:.2f} N*m "
              f"(ceiling {CLOSE_TORQUE_CEILING_NM[i]:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
