#!/usr/bin/env python3
"""Offline ESTIMATE: finger (slots 3-6) closing torque/error, old vs new protection+pose.
Measured state comes from the old controller, so reach/heating are NOT simulated."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_dex3_protection as rp
from teleop.utils import dex3_protection as dp

OLD_CEIL = (0.75, 0.75, 0.75, 1.0, 1.0, 1.0, 1.0)
SCALE = np.array([1, 1, 1, 1.15 / 0.78539816, 1.30 / 0.87266463, 1.15 / 0.78539816, 1.30 / 0.87266463])

def run(path, new):
    kw = {} if new else dict(close_ceiling_nm=OLD_CEIL)
    prot = {s: dp.Dex3HandProtector(np.zeros(7), **kw) for s in ("left", "right")}
    carry = {s: [dict() for _ in range(7)] for s in prot}
    acc = {(s, i): dict(tq=[], err=[], hold=0, stall=0, n=0) for s in prot for i in range(3, 7)}
    for t, side, joints, cmd_q in rp.load_rows(path):
        c = carry[side]
        for i, j in enumerate(joints):
            for k, v in j.items():
                if v is not None: c[i][k] = v
        st = {"timestamp": t, "q": [x.get("q") for x in c], "dq": [x.get("dq") for x in c], "tau": [x.get("tau_est") for x in c],
              "temp": [max(x["temperature"]) if x.get("temperature") else None for x in c],
              "mode": [x.get("mode") for x in c], "motorstate": [x.get("motorstate") for x in c]}
        tgt = np.asarray(cmd_q) * (SCALE if new else 1.0)
        # old log is already protected by 8371cde; recover the trigger target from the full-trigger pose
        res = prot[side].update(t, tgt, st)
        for i in range(3, 7):
            q = st["q"][i]
            if q is None or abs(cmd_q[i]) < 0.9 * abs(0.8 if i in (3,5) else 0.87): continue
            a = acc[(side, i)]; a["n"] += 1
            a["tq"].append(dp.DEX3_KP * abs(res.q_cmd[i] - q)); a["err"].append(abs(tgt[i] - q))
            a["hold"] += bool(res.grip_hold[i]); a["stall"] += bool(res.stall[i])
    return acc

if __name__ == "__main__":
    o, n = run(sys.argv[1], False), run(sys.argv[1], True)
    print("side joint | OLD: tq p90/max N*m, stall%  | NEW: tq p90/max, stall%, hold%, target-meas err max")
    for k in sorted(o):
        a, b = o[k], n[k]
        if not a["tq"]: continue
        f = lambda x: (np.percentile(x["tq"], 90), max(x["tq"]))
        print(k, "| %.2f/%.2f %.0f%% | %.2f/%.2f %.0f%% %.0f%% %.2f" % (*f(a), 100*a["stall"]/a["n"], *f(b), 100*b["stall"]/b["n"], 100*b["hold"]/b["n"], max(b["err"])))
