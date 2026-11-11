#!/usr/bin/env python3
"""Offline replay of RECORDED Quest stick telemetry through the waist-pitch path.

No DDS, no robot. Input: an extract of the robot's teleop-status.jsonl
(``locomotion.agv.raw_right_xy`` + ``controller.fresh/age_ms``, 1 Hz). Each
tracking segment of the recording becomes one armed session of the real
WaistPitchSupervisor + vendored ComputedTorqueController, closed-loop against
the vendored whole-body pitch plant (ct_sim/body_plant.py) at 50 Hz:

  * a status row with a fresh controller -> that stick sample is posted every
    cycle for 1 s (zero-order hold of the 1 Hz record, timestamp = now);
  * a row with a stale/absent controller -> nothing is posted for 1 s (the
    last Quest sample ages past 0.25 s exactly as on the robot);
  * after the segment a synthetic LEFT Y press (the button is not in the
    recording) -> home to q=0.0 -> release; publications after release are
    counted (must be 0).

Prints one JSON line per segment plus a summary; exit 1 if any invariant
fails (cap, rate, hold-on-stale drift, release only at neutral).

Usage: python tools/g1d_waist_pitch_replay.py STATUS.jsonl[.gz] [--arms rest|extended|worst]
"""
import argparse
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from teleop.utils import g1d_waist_pitch as wp  # noqa: E402
from g1d_waist_pitch_sim import POSES  # noqa: E402

ARMS = {"rest": POSES["rest (elbows 1.4)"], "extended": POSES["extended forward"],
        "worst": POSES["worst-case reach"], "zero": POSES["zero"]}
STEPS_PER_ROW = int(round(1.0 / wp.CTRL_DT))


def load_segments(path, min_rows=5):
    opener = gzip.open if str(path).endswith(".gz") else open
    segs, cur = [], []
    with opener(path, "rt") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("lifecycle") != "tracking":
                if len(cur) >= min_rows:
                    segs.append(cur)
                cur = []
                continue
            c = d.get("controller") or {}
            xy = d.get("raw_right_xy")
            cur.append((bool(c.get("fresh")) and xy is not None, tuple(xy) if xy else None))
    if len(cur) >= min_rows:
        segs.append(cur)
    return segs


def replay_segment(stack, body, rows, arms, tau_limit=85.0):
    from body_plant import BodyPlant
    clock = SimpleNamespace(t=1000.0)
    now = lambda: clock.t  # noqa: E731
    plant = BodyPlant(body, q0=0.0, tau_limit=tau_limit, arm_traj=lambda t: (dict(arms), {}, {}))
    pub = []

    class Bus:
        def read_state(self):
            st = plant.state()
            st.temperature = (35, 36)
            return st, now()

        def foreign_writers(self):
            return []

        def create_cmd_writer(self):
            return True

        def publish_cmd(self, writer_id, cmd, t=None):
            pub.append((now(), cmd))
            plant.cmd = cmd

        def close(self):
            pass

    def arm_snapshot():
        ms = [SimpleNamespace(q=0.0, dq=0.0) for _ in range(29)]
        for n, i in stack.body_state.ARM_SLOTS.items():
            ms[i].q = arms.get(n, 0.0)
        return ms, now()

    ctrl = wp.build_teleop_controller(stack, body)
    sup = wp.WaistPitchSupervisor(ctrl, Bus(), wp.make_arm_filter(stack, body), arm_snapshot,
                                  clock=now, log=lambda lvl, m: None, main_alive=lambda: True)
    m = {"rows": len(rows), "fresh_rows": 0, "stale_rows": 0, "rows_beyond_deadzone": 0,
         "q_max": 0.0, "q_min": 0.0, "max_cmd_step": 0.0, "max_ref_rate": 0.0, "max_plant_dq": 0.0,
         "stale_max_drift": 0.0, "stale_moves_cmd": 0, "rate_err_max": 0.0,
         "hold_snaps": 0, "max_hold_snap": 0.0}

    def tick(stick, y=False):
        clock.t += wp.CTRL_DT
        plant.step(wp.CTRL_DT)
        if stick is not None:
            sup.submit(stick, y, now())
        n0 = len(pub)
        q_meas = plant.q
        sup.step()
        if len(pub) > n0 and n0 > 0:
            step = pub[-1][1].q - pub[-2][1].q
            snap = abs(pub[-1][1].q - q_meas) <= 0.002 and abs(pub[-2][1].q - q_meas) > abs(step) * 0.5
            if snap and abs(step) > wp.PITCH_RATE_MAX * wp.CTRL_DT * 1.5:
                # vendored stop()/hold: reference re-anchored onto the MEASURED q
                # (removes PD lead, never a new target): counted separately
                m["hold_snaps"] += 1
                m["max_hold_snap"] = max(m["max_hold_snap"], abs(step))
            else:
                m["max_cmd_step"] = max(m["max_cmd_step"], abs(step))
        m["q_max"] = max(m["q_max"], plant.q)
        m["q_min"] = min(m["q_min"], plant.q)
        m["max_plant_dq"] = max(m["max_plant_dq"], abs(plant.dq))
        if ctrl.v_ref is not None:
            m["max_ref_rate"] = max(m["max_ref_rate"], abs(ctrl.v_ref))

    sup.request_arm(start_thread=False)
    for _ in range(5):
        tick((0.0, 0.0))
    if sup.state != wp.ACTIVE:
        return dict(m, error=f"not armed: {sup.state} {sup.last_reason}")
    for fresh, xy in rows:
        if fresh:
            m["fresh_rows"] += 1
            rate = wp.pitch_rate_from_stick(xy, 1.0, 1.0)
            if rate != 0.0:
                m["rows_beyond_deadzone"] += 1
            q_ref0 = ctrl.q_ref
            for _ in range(STEPS_PER_ROW):
                tick(xy)
            # requested rate vs achieved reference rate over the row; only
            # meaningful when not blocked by the cap / neutral boundary
            achieved = (ctrl.q_ref - q_ref0) / 1.0
            inside = 0.02 < ctrl.q_ref < wp.PITCH_Q_MAX - 0.02 and 0.02 < q_ref0 < wp.PITCH_Q_MAX - 0.02
            if inside:
                m["rate_err_max"] = max(m["rate_err_max"], abs(achieved) - abs(rate) if abs(achieved) > abs(rate)
                                        else 0.0)
        else:
            m["stale_rows"] += 1
            q0, n_pub0 = plant.q, len(pub)
            cmd_q0 = pub[-1][1].q if pub else None
            for _ in range(STEPS_PER_ROW):
                tick(None)
            m["stale_max_drift"] = max(m["stale_max_drift"], abs(plant.q - q0))
            # after the first 0.25 s the stale sample must not advance the command
            if cmd_q0 is not None and any(abs(c.q - cmd_q0) > 0.01 for _, c in pub[n_pub0 + 20:]):
                m["stale_moves_cmd"] += 1
    q_before_y = plant.q
    t_y = clock.t
    tick((0.0, 0.0), True)                       # synthetic LEFT Y (fresh sample)
    for _ in range(int(30 / wp.CTRL_DT)):
        tick(None)
        if sup.finished:
            break
    t_home = clock.t - t_y
    n_after = len(pub)
    for _ in range(STEPS_PER_ROW):
        tick((1.0, 0.0))                         # stick after release must do nothing
    m.update({k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    m.update(q_before_failsafe=round(q_before_y, 4), failsafe_return_s=round(t_home, 2),
             final_state=sup.state, q_release=round(plant.q, 4),
             published_after_release=len(pub) - n_after, admission_refusals=sup.admission_refusals)
    return m


def check(m):
    return (m.get("error") is None and m["q_max"] <= wp.PITCH_Q_MAX + 0.01 and m["q_min"] >= -0.012
            and m["max_ref_rate"] <= wp.PITCH_RATE_MAX + 1e-6 and m["max_cmd_step"] <= wp.PITCH_RATE_MAX * wp.CTRL_DT * 1.5
            and m["stale_moves_cmd"] == 0 and m["stale_max_drift"] < 0.01
            and m["final_state"] == wp.RELEASED and abs(m["q_release"]) <= 0.012
            and m["published_after_release"] == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("status")
    ap.add_argument("--arms", default="rest", choices=sorted(ARMS))
    ap.add_argument("--tau-limit", type=float, default=85.0)
    ap.add_argument("--max-segments", type=int, default=0)
    args = ap.parse_args()
    stack = wp.load_pitch_stack()
    sys.path.insert(0, str(wp.VENDOR_DIR / "ct_sim"))
    body = wp.build_body_model(stack)
    segs = load_segments(args.status)
    if args.max_segments:
        segs = segs[:args.max_segments]
    ok = True
    tot = {"segments": len(segs), "rows": 0, "fresh_rows": 0, "stale_rows": 0, "rows_beyond_deadzone": 0}
    worst = {"q_max": 0.0, "max_cmd_step": 0.0, "max_ref_rate": 0.0, "stale_max_drift": 0.0,
             "failsafe_return_s": 0.0, "abs_q_release": 0.0, "max_hold_snap": 0.0, "max_plant_dq": 0.0}
    print(f"arms={args.arms} cap={wp.PITCH_Q_MAX} rate<={wp.PITCH_RATE_MAX} motor<={args.tau_limit} N.m")
    for i, rows in enumerate(segs):
        m = replay_segment(stack, body, rows, ARMS[args.arms], args.tau_limit)
        good = check(m)
        ok &= good
        print(("OK   " if good else "FAIL ") + json.dumps(dict(segment=i, **m)))
        for k in ("rows", "fresh_rows", "stale_rows", "rows_beyond_deadzone"):
            tot[k] += m.get(k, 0)
        tot["hold_snaps"] = tot.get("hold_snaps", 0) + m.get("hold_snaps", 0)
        for k in ("q_max", "max_cmd_step", "max_ref_rate", "stale_max_drift", "failsafe_return_s",
                  "max_hold_snap", "max_plant_dq"):
            worst[k] = max(worst[k], m.get(k, 0.0))
        worst["abs_q_release"] = max(worst["abs_q_release"], abs(m.get("q_release", 0.0)))
    print("SUMMARY " + json.dumps(dict(ok=ok, **tot, **{"worst_" + k: round(v, 4) for k, v in worst.items()})))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
