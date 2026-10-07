#!/usr/bin/env python3
"""Offline report for a task saved by tools/arm_path_web.py.

    python tools/arm_path_report.py <task_dir> [--window-s 0.05] [--min-step-m 0.001] [--no-png]

Re-computes the per-arm metrics from left.csv / right.csv (raw columns) with the
chosen filter, prints a table, writes ``report.json`` and, if matplotlib is
installed, ``left.png`` / ``right.png`` (3D + XY/XZ/YZ projections with equal
axes + X/Y/Z vs time + cumulative length). Needs only numpy (+ matplotlib).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
import arm_path_metrics as apm  # noqa: E402

SIDES = ("left", "right")
LABEL = {"left": "Esquerdo", "right": "Direito"}


def read_arm(path):
    import csv
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([float(r["t_rel_s"]) for r in rows])
    P = np.array([[float(r["x_raw"]), float(r["y_raw"]), float(r["z_raw"])] for r in rows]).reshape(-1, 3)
    return t, P


def compute(task_dir, window_s=apm.DEFAULT_WINDOW_S, min_step_m=apm.DEFAULT_MIN_STEP_M):
    out = {"task_dir": os.path.abspath(task_dir), "filter": apm.filter_desc(window_s, min_step_m), "arms": {}}
    try:
        with open(os.path.join(task_dir, "summary.json"), encoding="utf-8") as fh:
            s = json.load(fh)
        out.update(name=s.get("name"), start_utc=s.get("start_utc"), point=s.get("point"), frame=s.get("frame"))
    except FileNotFoundError:
        pass
    data = {}
    for sd in SIDES:
        t, P = read_arm(os.path.join(task_dir, f"{sd}.csv"))
        F = apm.smooth(t, P, window_s)
        out["arms"][sd] = apm.path_metrics(t, P, window_s, min_step_m, filtered=F)
        data[sd] = (t, P, F)
    return out, data


def _set_equal_3d(ax, P):
    lo, hi = P.min(axis=0), P.max(axis=0)
    c, r = (lo + hi) / 2, max((hi - lo).max() / 2, 0.01)
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def plot_arm(path_png, sd, t, P, F, m, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"{title} — {LABEL[sd]}: filtrado {m['length_filtered_m']:.3f} m · bruto {m['length_raw_m']:.3f} m · "
                 f"líquido {m['net_displacement_m']:.3f} m · {m['duration_s']:.1f} s · v máx {m['max_speed_m_s']:.2f} m/s")
    tn = (t - t[0]) / max(t[-1] - t[0], 1e-9) if len(t) else t
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], c=tn, cmap="turbo", s=2)
    ax.plot(*F.T, color="0.6", lw=0.5)
    ax.scatter(*F[0], color="blue", s=50, label="início"); ax.scatter(*F[-1], color="red", marker="s", s=50, label="fim")
    ax.set_xlabel("X frente (m)"); ax.set_ylabel("Y esquerda (m)"); ax.set_zlabel("Z cima (m)"); ax.legend(loc="upper left")
    _set_equal_3d(ax, F)
    for k, (i, j, nm) in enumerate(((0, 1, "Topo XY"), (0, 2, "Lateral XZ"), (1, 2, "Frontal YZ"))):
        a = fig.add_subplot(2, 3, 2 + k)
        seg = np.stack([F[:-1][:, [i, j]], F[1:][:, [i, j]]], axis=1) * 100
        lc = LineCollection(seg, cmap="turbo", linewidths=1.6); lc.set_array(tn[:-1]); a.add_collection(lc)
        a.plot(*(F[0, [i, j]] * 100), "o", color="blue"); a.plot(*(F[-1, [i, j]] * 100), "s", color="red")
        a.autoscale(); a.set_aspect("equal", adjustable="datalim"); a.grid(alpha=0.3)
        a.set_title(nm); a.set_xlabel("XYZ"[i] + " (cm)"); a.set_ylabel("XYZ"[j] + " (cm)")
        if (i, j) == (1, 2):
            a.invert_xaxis()
    a = fig.add_subplot(2, 3, 5)
    for k, c in enumerate(("tab:red", "tab:green", "tab:blue")):
        a.plot(t, P[:, k], color=c, alpha=0.3, lw=0.8); a.plot(t, F[:, k], color=c, lw=1.4, label="XYZ"[k])
    a.set_xlabel("t (s)"); a.set_ylabel("m"); a.set_title("X/Y/Z × tempo (claro = bruto)"); a.legend(); a.grid(alpha=0.3)
    a = fig.add_subplot(2, 3, 6)
    a.plot(t, apm.cumulative_length(P), "--", color="0.5", label="bruto")
    a.plot(t, apm.cumulative_length(F, m.get("min_step_m", 0.0)), color="k", label="filtrado")
    a.set_xlabel("t (s)"); a.set_ylabel("m"); a.set_title("Comprimento acumulado"); a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dir")
    ap.add_argument("--window-s", type=float, default=apm.DEFAULT_WINDOW_S)
    ap.add_argument("--min-step-m", type=float, default=apm.DEFAULT_MIN_STEP_M)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args(argv)
    rep, data = compute(args.task_dir, args.window_s, args.min_step_m)
    print(f"Tarefa: {rep.get('name')}  ({rep.get('start_utc')})  ponto: {(rep.get('point') or {}).get('label')}")
    print(f"Filtro: média móvel {args.window_s * 1000:.0f} ms, passo mínimo {args.min_step_m * 1000:.1f} mm")
    print(f"{'braço':10s} {'filtrado':>9s} {'bruto':>9s} {'líquido':>9s} {'|dx|':>7s} {'|dy|':>7s} {'|dz|':>7s} {'dur s':>7s} {'v méd':>7s} {'v máx':>7s}")
    for sd in SIDES:
        m = rep["arms"][sd]
        ax = m["length_axis_filtered_m"]
        print(f"{LABEL[sd]:10s} {m['length_filtered_m']:9.4f} {m['length_raw_m']:9.4f} {m['net_displacement_m']:9.4f} "
              f"{ax[0]:7.3f} {ax[1]:7.3f} {ax[2]:7.3f} {m['duration_s']:7.2f} {m['mean_speed_m_s']:7.3f} {m['max_speed_m_s']:7.3f}")
    with open(os.path.join(args.task_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
    if not args.no_png:
        try:
            import matplotlib  # noqa: F401
        except Exception:
            print("matplotlib ausente: PNGs não gerados (pip install matplotlib)")
            return 0
        for sd in SIDES:
            t, P, F = data[sd]
            if len(t) < 2:
                continue
            m = dict(rep["arms"][sd], min_step_m=args.min_step_m)
            png = os.path.join(args.task_dir, f"{sd}.png")
            plot_arm(png, sd, t, P, F, m, rep.get("name") or os.path.basename(args.task_dir.rstrip("/")))
            print("PNG:", png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
