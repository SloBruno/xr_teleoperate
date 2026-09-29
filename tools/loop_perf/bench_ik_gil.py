#!/usr/bin/env python3
"""INERT: does G1_29_ArmIK.solve_ik release the GIL?  (no DDS, no robot I/O)

A Python counter thread spins while the main thread solves IK on logged
samples. If the counter advances ~as fast as when idle, solve_ik releases the
GIL; if it barely advances, IK holds it (then other Python threads cannot run
during IK but also cannot slow it down except at entry/exit).
"""
import json
import sys
import threading
import time

import numpy as np

sys.dont_write_bytecode = True
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK  # noqa: E402

rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if r.get("event") == "full_pose_telemetry" and r.get("lifecycle") == "tracking"
        and (r["arm"]["calibrated_cartesian_target"] or {}).get("left") is not None]
ik = G1_29_ArmIK()
cnt = [0]
stop = [False]


def spin():
    while not stop[0]:
        cnt[0] += 1


def run(n, label, spinner):
    th = None
    if spinner:
        th = threading.Thread(target=spin, daemon=True)
        th.start()
        time.sleep(0.05)
    times = []
    c0 = cnt[0]
    t_all = time.perf_counter()
    for r in rows[:n]:
        tgt = r["arm"]["calibrated_cartesian_target"]
        q = np.array(r["arm"]["left"]["measured_q"] + r["arm"]["right"]["measured_q"])
        t = time.perf_counter()
        ik.solve_ik(np.array(tgt["left"]), np.array(tgt["right"]), q, np.zeros(14))
        times.append((time.perf_counter() - t) * 1000)
    wall = time.perf_counter() - t_all
    stop[0] = True
    if th:
        th.join()
    stop[0] = False
    return {"label": label, "ik_p50_ms": round(float(np.median(times)), 2),
            "ik_p95_ms": round(float(np.percentile(times, 95)), 2),
            "spin_per_s": round((cnt[0] - c0) / wall)}


def idle_spin_rate():
    c0 = cnt[0]
    th = threading.Thread(target=spin, daemon=True)
    th.start()
    time.sleep(1.0)
    stop[0] = True
    th.join()
    stop[0] = False
    return cnt[0] - c0


print(json.dumps({"idle_spin_per_s": idle_spin_rate(),
                  "alone": run(150, "alone", False),
                  "with_spinner": run(150, "with_spinner", True)}, indent=1))
