"""Path-length metrics for a sampled 3D point trajectory (numpy only).

Used by ``tools/arm_path_web.py`` (live + on save) and
``tools/arm_path_report.py`` (offline). Pure functions, no pinocchio/DDS.

Why two lengths: the sum of |p[i+1]-p[i]| over a noisy signal grows with the
noise (each step adds ~sqrt(2)*sigma per axis even when the arm is still), so the
*raw* length over-estimates. The *filtered* length first smooths the positions
with a centred moving average (default 50 ms window, zero phase: no lag, no
shrink of slow motion) and optionally ignores sub-threshold jitter with a
hysteresis step (``min_step_m``: a new point only counts once it is at least
that far from the last counted point).
"""
from __future__ import annotations

import numpy as np

DEFAULT_WINDOW_S = 0.05
DEFAULT_MIN_STEP_M = 0.001


def smooth(t, P, window_s=DEFAULT_WINDOW_S):
    """Centred moving average in time (window ``window_s``, edges shrink).

    ``t`` (N,), ``P`` (N,3). Returns (N,3). window_s <= 0 -> copy of P.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    n = len(t)
    if n < 3 or window_s is None or window_s <= 0:
        return P.copy()
    h = 0.5 * float(window_s)
    lo = np.searchsorted(t, t - h, side="left")
    hi = np.searchsorted(t, t + h, side="right")
    # symmetric window around each sample (keep zero phase at the edges)
    k = np.minimum(np.arange(n) - lo, hi - 1 - np.arange(n))
    lo = np.arange(n) - k
    hi = np.arange(n) + k + 1
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(P, axis=0)])
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


def step_lengths(P):
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    if len(P) < 2:
        return np.zeros(0)
    return np.linalg.norm(np.diff(P, axis=0), axis=1)


def cumulative_length(P, min_step_m=0.0):
    """Cumulative path length per sample (N,), starts at 0.

    With ``min_step_m`` > 0 a hysteresis is used: the anchor only moves (and
    the length only grows) when the point is >= min_step_m from the anchor.
    """
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    n = len(P)
    out = np.zeros(n)
    if n < 2:
        return out
    if not min_step_m or min_step_m <= 0:
        out[1:] = np.cumsum(step_lengths(P))
        return out
    anchor, acc = P[0], 0.0
    for i in range(1, n):
        d = float(np.linalg.norm(P[i] - anchor))
        if d >= min_step_m:
            acc += d
            anchor = P[i]
        out[i] = acc
    return out


def path_metrics(t, P, window_s=DEFAULT_WINDOW_S, min_step_m=DEFAULT_MIN_STEP_M, filtered=None):
    """Metrics dict for one arm. Lengths in m, speeds in m/s, times in s."""
    t = np.asarray(t, dtype=float).reshape(-1)
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    n = len(t)
    F = smooth(t, P, window_s) if filtered is None else np.asarray(filtered, float).reshape(-1, 3)
    m: dict = {"n_samples": int(n)}
    if n < 2:
        m.update(duration_s=0.0, length_raw_m=0.0, length_filtered_m=0.0, net_displacement_m=0.0,
                 net_vector_m=[0.0, 0.0, 0.0], length_axis_raw_m=[0.0] * 3, length_axis_filtered_m=[0.0] * 3,
                 mean_speed_m_s=0.0, max_speed_m_s=0.0, raw_over_filtered=None,
                 start_xyz_m=None if n == 0 else [float(v) for v in F[0]],
                 end_xyz_m=None if n == 0 else [float(v) for v in F[-1]], bbox_min_m=None, bbox_max_m=None)
        return m
    dur = float(t[-1] - t[0])
    raw = float(step_lengths(P).sum())
    filt = float(cumulative_length(F, min_step_m)[-1])
    net = F[-1] - F[0]
    dt = np.diff(t)
    ok = dt > 1e-6
    sp = step_lengths(F)[ok] / dt[ok] if ok.any() else np.zeros(1)
    m.update(
        duration_s=round(dur, 4),
        length_raw_m=raw,
        length_filtered_m=filt,
        net_displacement_m=float(np.linalg.norm(net)),
        net_vector_m=[float(v) for v in net],
        length_axis_raw_m=[float(v) for v in np.abs(np.diff(P, axis=0)).sum(axis=0)],
        length_axis_filtered_m=[float(v) for v in np.abs(np.diff(F, axis=0)).sum(axis=0)],
        mean_speed_m_s=filt / dur if dur > 0 else 0.0,
        max_speed_m_s=float(sp.max()) if sp.size else 0.0,
        raw_over_filtered=(raw / filt) if filt > 1e-9 else None,
        start_xyz_m=[float(v) for v in F[0]],
        end_xyz_m=[float(v) for v in F[-1]],
        bbox_min_m=[float(v) for v in F.min(axis=0)],
        bbox_max_m=[float(v) for v in F.max(axis=0)],
    )
    return m


def filter_desc(window_s=DEFAULT_WINDOW_S, min_step_m=DEFAULT_MIN_STEP_M):
    return {
        "method": "centred moving average in time (zero phase), then path length = sum of 3D steps"
                  + (" with hysteresis min step" if min_step_m and min_step_m > 0 else ""),
        "window_s": float(window_s or 0.0),
        "min_step_m": float(min_step_m or 0.0),
        "max_speed_from": "filtered signal, per-sample finite difference",
        "net_displacement_from": "filtered signal, first -> last sample",
    }
