#!/usr/bin/env python3
"""Offline closed-loop replay: finger (slot 3, right sign) blocked by a compliant box.

Plant (heat constant fitted to the 2026-10-01 box session: ~0.4 C/s at 0.65 N*m): contact at q0=0.72 rad, box stiffness K N*m/rad; heat = c*tau^2 - cool*(T-25).
Trigger 1.0 with 0.12 s stale-sample dropouts (trigger forced 0, as controller_sample_is_fresh does >0.25 s) every 1.5 s. Compares the
322af1c state machine (module loaded from a file) with the current one.
Usage: replay_dex3_grip_cycle.py OLD_MODULE.py   (pure offline; no DDS/robot)
"""
import importlib.util, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.utils import dex3_protection as NEW


def load(path):
    spec = importlib.util.spec_from_file_location("old_prot", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def sim(mod, dur=45.0, dip_every=1.5, dip_to=0.0, K=8.0, q0=0.72, heat=0.95, cool=0.01, dt=0.02):
    prot = mod.Dex3HandProtector(np.zeros(7)); kp = 1.5
    closed = np.array([0, 0, 0, 1.15, 1.30, 1.15, 1.30]); q = 0.0; T = 30.0
    trans = 0; prev = False; rows = []; k = 0
    t = 0.0
    while t < dur:
        trig = dip_to if (dip_every and (t % dip_every) < 0.12 and t > 2) else 1.0
        tgt = trig * closed
        blocked = q >= q0 - 1e-3
        st = {"timestamp": t, "q": [0.0] * 3 + [q] * 4, "dq": [0.0 if blocked else 3.0] * 7,
              "tau": [8e5 if blocked else 0.0] * 7, "temp": [T] * 7, "mode": [1] * 7, "motorstate": [0] * 7}
        r = prot.update(t, tgt, st)
        qc = float(r.q_cmd[3]); tau = kp * (qc - q)
        # quasi-static plant
        q = min(qc, q0 + max(0.0, (kp * (qc - q0)) / (kp + K))) if qc > q0 else min(qc, q + 0.15)
        tau_n = kp * (qc - q)
        T += (heat * tau_n ** 2 - cool * (T - 25.0)) * dt
        s = bool(r.stall[3])
        if prev and not s: trans += 1
        prev = s
        rows.append((t, tau_n, T, s, bool(r.grip_hold[3]), r.derate[3]))
        t += dt
    a = np.array([(x[1]) for x in rows if x[0] > 4.0])
    return dict(stall_off_transitions=trans, torque_min=a.min(), torque_mean=a.mean(), torque_max=a.max(),
                Tmax=max(x[2] for x in rows), min_derate=min(x[5] for x in rows),
                hold_frac=np.mean([x[4] for x in rows]),
                torque_series=[(round(x[0], 1), round(x[1], 2), round(x[2], 1), x[3], x[4]) for x in rows[::int(1 / dt) ]])


if __name__ == "__main__":
    old = load(sys.argv[1]) if len(sys.argv) > 1 else None
    for name, mod in (("322af1c", old), ("NEW", NEW)):
        if mod is None: continue
        for jit in (0.0, 1.5):
            r = sim(mod, dip_every=jit)
            ts = r.pop("torque_series")
            print(name, "jitter" if jit else "clean ", {k: round(float(v), 2) for k, v in r.items()})
            if jit and name == "NEW": print("  t,torque,T,stall,hold:", ts[:6], "...", ts[-3:])
