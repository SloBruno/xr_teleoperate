#!/usr/bin/env python3
"""Offline simulation of the Quest waist-pitch path (no DDS, no robot).

Runs the real WaistPitchSupervisor + vendored ComputedTorqueController
against the vendored whole-body pitch plant (ct_sim/body_plant.py, Pinocchio
RNEA, lifting breakaway ~2.1|g|, motor clipped at --tau-limit) with a
deterministic clock, for several arm poses, and prints one line per scenario:

  arm -> full stick forward (to the cap / admission) -> release (hold)
  -> stale Quest input (hold) -> LEFT Y fail-safe -> home to q=0.0 -> release

Usage: python tools/g1d_waist_pitch_sim.py [--tau-limit 85] [--jsonl out.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teleop.utils import g1d_waist_pitch as wp  # noqa: E402

JOINTS = ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint",
          "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint")
MIRROR = (1, -1, -1, 1, -1, 1, -1)


def pose(left):
    d = {}
    for j, v, m in zip(JOINTS, left, MIRROR):
        d["left_" + j] = v
        d["right_" + j] = v * m
    return d


POSES = {
    "zero": pose((0, 0, 0, 0, 0, 0, 0)),
    "rest (elbows 1.4)": pose((0, 0, 0, 1.4, 0, 0, 0)),
    "extended forward": pose((-1.57, 0, 0, 0, 0, 0, 0)),
    "worst-case reach": pose((-1.86, 0.25, 1.51, 1.37, 0.21, 0.08, 0.03)),
}


def scenario(stack, body, name, arms, tau_limit, jsonl):
    sys.path.insert(0, str(wp.VENDOR_DIR / "ct_sim"))
    from body_plant import BodyPlant
    clock = SimpleNamespace(t=1000.0)
    now = lambda: clock.t  # noqa: E731
    live_arms = dict(arms)
    plant = BodyPlant(body, q0=0.0, tau_limit=tau_limit, arm_traj=lambda t: (dict(live_arms), {}, {}))
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

    def arm_snapshot():
        from types import SimpleNamespace as NS
        ms = [NS(q=0.0, dq=0.0) for _ in range(29)]
        for n, i in stack.body_state.ARM_SLOTS.items():
            ms[i].q = live_arms.get(n, 0.0)
        return ms, now()

    ctrl = wp.build_teleop_controller(stack, body)
    events = []
    sup = wp.WaistPitchSupervisor(ctrl, Bus(), wp.make_arm_filter(stack, body), arm_snapshot,
                                  clock=now, log=lambda lvl, m: events.append(m), main_alive=lambda: True)
    peak = {"tau": 0.0, "dq": 0.0, "q": 0.0}

    def run(seconds, stick=None, y=False):
        for _ in range(int(round(seconds / wp.CTRL_DT))):
            clock.t += wp.CTRL_DT
            plant.step(wp.CTRL_DT)
            if stick is not None:
                sup.submit(stick, y, now())
            sup.step()
            peak["tau"] = max(peak["tau"], abs(plant.tau_m))
            peak["dq"] = max(peak["dq"], abs(plant.dq))
            peak["q"] = max(peak["q"], plant.q)
            if jsonl:
                jsonl.write(json.dumps({"scenario": name, "t": clock.t, "q": plant.q, "dq": plant.dq,
                                        "tau_motor": plant.tau_m, "state": sup.state,
                                        "target": ctrl.target}) + "\n")

    sup.request_arm(start_thread=False)
    run(0.2, (0.0, 0.0))
    run(6.0, (1.0, 0.0))
    q_fwd = plant.q
    run(2.0, (0.0, 0.0))
    q_hold = plant.q
    run(1.5, None)                      # Quest stale
    q_stale = plant.q
    t_fs = clock.t
    peak_return = {"tau": 0.0}
    run(0.02, (0.0, 0.0), True)         # left Y on a fresh sample
    for _ in range(int(20 / wp.CTRL_DT)):
        run(wp.CTRL_DT, None)
        peak_return["tau"] = max(peak_return["tau"], abs(plant.tau_m))
        if sup.finished:
            break
    n_after = len(pub)
    run(1.0, (1.0, 0.0))
    return {
        "pose": name, "q_forward": round(q_fwd, 4), "admission_refusals": sup.admission_refusals,
        "predicted_lift_at_q": round(ctrl.predicted_lift(q_fwd), 1),
        "hold_drift": round(abs(q_hold - q_fwd), 4), "stale_drift": round(abs(q_stale - q_hold), 4),
        "final_state": sup.state, "q_release": round(plant.q, 4),
        "return_s": round(clock.t - t_fs - 1.0, 2), "peak_return_tau": round(peak_return["tau"], 1),
        "peak_dq": round(peak["dq"], 3), "published_after_release": len(pub) - n_after,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-limit", type=float, default=85.0)
    ap.add_argument("--jsonl", default=None)
    args = ap.parse_args()
    stack = wp.load_pitch_stack()
    body = wp.build_body_model(stack)
    out = open(args.jsonl, "w") if args.jsonl else None
    ok = True
    print(f"cap={wp.PITCH_Q_MAX} rad, rate<={wp.PITCH_RATE_MAX} rad/s, motor limit {args.tau_limit} N.m")
    for name, arms in POSES.items():
        r = scenario(stack, body, name, arms, args.tau_limit, out)
        good = (r["final_state"] == wp.RELEASED and abs(r["q_release"]) <= 0.012
                and r["published_after_release"] == 0 and r["q_forward"] <= wp.PITCH_Q_MAX + 0.02
                and r["hold_drift"] < 0.01 and r["stale_drift"] < 0.01)
        ok &= good
        print(("OK  " if good else "FAIL") + " " + json.dumps(r))
    if out:
        out.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
