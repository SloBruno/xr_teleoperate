"""Inert replay: CoM shift from measured arm joints vs measured sport velocity.
Offline only; reads pose-telemetry-*.jsonl. Needs pinocchio (conda env tv).
Usage: python tools/com_replay.py FILE.jsonl [...]"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop.utils.com_monitor import ArmCoMModel, pearson  # noqa: E402


def load(path):
    t, q, v = [], [], []
    for line in open(path):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("event") != "full_pose_telemetry" or d.get("lifecycle") != "tracking":
            continue
        a = d.get("arm", {})
        ql, qr = a.get("left", {}).get("measured_q"), a.get("right", {}).get("measured_q")
        rs = d.get("locomotion", {}).get("robot_state", {})
        sv = rs.get("sport_velocity")
        if not ql or not qr or sv is None:
            continue
        t.append(d["timestamp_monotonic"]); q.append(list(ql) + list(qr)); v.append(sv)
        cmd = d.get("locomotion", {}).get("dispatched_command") or [0, 0, 0]
        v[-1] = list(sv) + list(cmd)
    return np.array(t), np.array(q), np.array(v)


def main(paths):
    model = ArmCoMModel.from_urdf(ROOT / "assets/g1/g1_body29_hand14.urdf")
    for p in paths:
        t, q, v = load(p)
        if len(t) < 50:
            print(p, "poucas amostras", len(t)); continue
        dx = np.array([model.delta_com(x)[0] for x in q]) * 1000  # mm
        dy = np.array([model.delta_com(x)[1] for x in q]) * 1000
        vx, vy = v[:, 0], v[:, 1]
        cmd_idle = (np.abs(v[:, 3:6]).sum(axis=1) == 0)
        sel = cmd_idle  # sem comando de marcha
        dvx = np.gradient(dx, t)
        print(f"\n{Path(p).name[:40]} n={len(t)} dur={t[-1]-t[0]:.0f}s idle={sel.sum()}")
        print(f" dCoM_x mm: min {dx.min():.1f} p50 {np.median(dx):.1f} max {dx.max():.1f} | dCoM_y mm: min {dy.min():.1f} max {dy.max():.1f}")
        print(f" |sport_vx| idle: p50 {np.median(np.abs(vx[sel])) if sel.any() else float('nan'):.3f} p95 {np.percentile(np.abs(vx[sel]),95) if sel.any() else float('nan'):.3f} max {np.abs(vx[sel]).max() if sel.any() else float('nan'):.3f} m/s")
        if sel.sum() > 30:
            print(f" corr(dCoM_x, vx) sem comando: {pearson(dx[sel], vx[sel]):.3f}   corr(dCoM_y, vy): {pearson(dy[sel], vy[sel]):.3f}")
            # janelas: vx medio quando dx alto vs baixo
            hi = dx[sel] > np.percentile(dx[sel], 75); lo = dx[sel] < np.percentile(dx[sel], 25)
            print(f" |vx| medio quartil dx alto {np.abs(vx[sel][hi]).mean():.4f} vs baixo {np.abs(vx[sel][lo]).mean():.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
