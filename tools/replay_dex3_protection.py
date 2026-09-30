#!/usr/bin/env python3
"""Offline replay of a recorded pose-telemetry session through Dex3 protection.

ESTIMATE ONLY: the recorded measured state (q, dq, tau_est, temperature) was
produced by the *old* unprotected controller. A real motor would react
differently to the protected command (lower torque -> less heating, less
penetration), so this only shows what the protector would have *commanded*
given the states that were actually observed. Pure offline: reads a JSONL file,
never touches DDS/robot.

Usage: replay_dex3_protection.py SESSION.jsonl [--json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.utils.dex3_protection import (  # noqa: E402
    DEX3_KP, DERATE_START_C, Dex3HandProtector, JOINT_NAMES)


def load_rows(path):
    """Yield (t_rel, side, joints, published_q) for tracking records."""
    t0 = None
    with open(path) as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("event") != "full_pose_telemetry":
                continue
            ts = rec.get("timestamp_monotonic")
            if t0 is None:
                t0 = ts
            if rec.get("lifecycle") != "tracking":
                continue
            for side in ("left", "right"):
                ext = ((rec.get("dex3") or {}).get(side) or {}).get("extended") or {}
                state = ext.get("state") or {}
                cmd = ext.get("published_command")
                if state.get("joints") and cmd and cmd.get("q"):
                    yield ts - t0, side, state["joints"], cmd["q"]


def replay(path, open_pose=None):
    open_pose = np.zeros(7) if open_pose is None else open_pose
    prot = {s: Dex3HandProtector(open_pose) for s in ("left", "right")}
    carry = {s: [dict() for _ in range(7)] for s in prot}
    stats = {s: {i: dict(old_max_nm=0.0, new_max_nm=0.0, old_close_max_nm=0.0, new_close_max_nm=0.0, n=0, limited=0,
                         first_limited=None, first_stall=None, first_derate=None,
                         first_hot_latch=None, first_fault=None, stall_samples=0,
                         first_temp65=None, first_temp70=None, first_temp80=None)
             for i in range(7)} for s in prot}
    for t, side, joints, cmd_q in load_rows(path):
        c = carry[side]
        for i, j in enumerate(joints):
            for k, v in j.items():
                if v is not None:
                    c[i][k] = v  # slow fields are only emitted on change
        temps = [max(x["temperature"]) if x.get("temperature") else None for x in c]
        state = {
            "timestamp": t,
            "q": [x.get("q") for x in c], "dq": [x.get("dq") for x in c],
            "tau": [x.get("tau_est") for x in c], "temp": temps,
            "mode": [x.get("mode") for x in c], "motorstate": [x.get("motorstate") for x in c],
        }
        res = prot[side].update(t, cmd_q, state)
        for i in range(7):
            q = state["q"][i]
            if q is None:
                continue
            s = stats[side][i]
            s["n"] += 1
            old = DEX3_KP * abs(cmd_q[i] - q)
            new = DEX3_KP * abs(res.q_cmd[i] - q) if res.enable[i] else 0.0
            s["old_max_nm"] = max(s["old_max_nm"], old)
            direction = np.sign(cmd_q[i] - open_pose[i])
            if direction != 0:  # closing-direction implicit torque only
                s["old_close_max_nm"] = max(s["old_close_max_nm"], max(0.0, direction * (cmd_q[i] - q)) * DEX3_KP)
                if res.enable[i]:
                    s["new_close_max_nm"] = max(s["new_close_max_nm"], max(0.0, direction * (res.q_cmd[i] - q)) * DEX3_KP)
            s["new_max_nm"] = max(s["new_max_nm"], new)
            for key, flag in (("first_limited", res.torque_limited[i]), ("first_stall", res.stall[i]),
                              ("first_derate", res.derate[i] < 1.0), ("first_fault", res.fault[i])):
                if flag and s[key] is None:
                    s[key] = t
            s["limited"] += bool(res.torque_limited[i])
            s["stall_samples"] += bool(res.stall[i])
            T = temps[i]
            if T is not None:
                for key, th in (("first_temp65", DERATE_START_C), ("first_temp70", 70), ("first_temp80", 80)):
                    if T >= th and s[key] is None:
                        s[key] = t
    return stats


def main(argv):
    stats = replay(argv[1])
    if "--json" in argv:
        print(json.dumps(stats, indent=1))
        return 0
    for side, joints in stats.items():
        for i in (1, 2):
            s = joints[i]
            print(f"{side:5s} {JOINT_NAMES[i]}: closing implicit torque max old {s['old_close_max_nm']:.2f} -> new {s['new_close_max_nm']:.2f} N*m (any dir: {s['old_max_nm']:.2f}->{s['new_max_nm']:.2f})"
                  f" | limited {s['limited']}/{s['n']} | first stall {s['first_stall']} | first derate {s['first_derate']}"
                  f" | first fault {s['first_fault']} | recorded T>=65 {s['first_temp65']} T>=70 {s['first_temp70']} T>=80 {s['first_temp80']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
