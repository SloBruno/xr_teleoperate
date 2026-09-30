#!/usr/bin/env python3
"""Read-only analyzer for loop_diagnostics status records (+ optional pose-telemetry).

Usage: analyze_loop_feed.py teleop-status.jsonl [pose-telemetry.jsonl ...]
The VEREDITO lines are rule-based CORRELATIONS, not proof of causation.
"""
from __future__ import annotations

import json
import math
import sys


def load(paths):
    diag, pose = [], []
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("event") == "loop_diagnostics":
                    diag.append(r)
                elif "loop_timing" in r:
                    pose.append(r)
    return diag, pose


def pearson(x, y):
    n = len(x)
    if n < 5:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _med(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else None


def analyze(diag, pose=()):
    out = {"windows": len(diag), "verdicts": []}
    if not diag:
        out["verdicts"].append("sem registros loop_diagnostics")
        return out
    hz = [d.get("loop_hz") for d in diag]
    out["loop_hz_median"] = _med(hz)
    out["unique_feed_hz_median"] = _med([_g(d, "feed", "unique_hz") for d in diag])
    out["total_ms_p95_median"] = _med([_g(d, "total_ms", "p95") for d in diag])
    # stage p95 (median over windows), excluding sleep
    stages = {}
    for d in diag:
        for k, v in (d.get("stages_ms") or {}).items():
            if v.get("p95") is not None:
                stages.setdefault(k, []).append(v["p95"])
    stage_p95 = {k: _med(v) for k, v in stages.items()}
    out["stage_p95_ms"] = stage_p95
    work = {k: v for k, v in stage_p95.items() if k != "sleep"}
    V = out["verdicts"]
    if work:
        k = max(work, key=work.get)
        V.append(f"etapa dominante {k} p95 {work[k]:.1f} ms (ciclo alvo ~{1000/30:.0f} ms)")
    # gap histogram totals from last window (cumulative)
    hist = {}
    for d in diag:
        for k, v in (_g(d, "feed", "gap_hist") or {}).items():
            hist[k] = hist.get(k, 0) + v
    out["gap_hist"] = hist
    bursts = [b for d in diag for b in (_g(d, "feed", "stale_bursts") or [])]
    out["stale_bursts"] = len(bursts)
    dom = {}
    for b in bursts:
        dom[b.get("dominant_stage")] = dom.get(b.get("dominant_stage"), 0) + 1
    out["stale_burst_dominant_stage"] = dom
    if bursts:
        top = max(dom, key=dom.get)
        V.append(f"{len(bursts)} rajadas stale; etapa dominante nelas: {top} ({dom[top]}x)")
    # correlation: gap p95 vs render p95 per window
    gp = [(_g(d, "feed", "gap_ms", "p95"), _g(d, "stages_ms", "render", "p95")) for d in diag]
    gp = [(a, b) for a, b in gp if a is not None and b is not None]
    r = pearson([a for a, _ in gp], [b for _, b in gp]) if gp else None
    out["corr_gap_render"] = r
    if r is not None and r >= 0.6:
        V.append(f"gaps do controller coincidem com picos de render (r={r:.2f}) [correlação, não causa]")
    # controller-wait vs network: if 'controller' stage tiny but unique feed low -> feed limited upstream
    fh, uh = out["loop_hz_median"], out["unique_feed_hz_median"]
    ctl = stage_p95.get("controller")
    if fh and uh and uh < 0.8 * fh and (ctl or 0) < 5:
        V.append(f"feed único ({uh} Hz) < loop ({fh} Hz) sem espera no controller: amostras chegam lentas do Quest (rede/Quest/event loop do Vuer)")
    # route
    routes = {}
    for d in diag:
        rt = _g(d, "system", "net", "route")
        if rt:
            routes[rt] = routes.get(rt, 0) + 1
    out["quest_routes"] = routes
    if routes:
        rt = max(routes, key=routes.get)
        V.append(f"rota do Quest: {rt}")
        if rt == "tailscale" and uh and uh < 20:
            V.append("feed Quest limitado por rede? (Tailscale + feed baixo) [correlação; repetir em Wi-Fi direto]")
    else:
        V.append("rota do Quest: não observada")
    # CPU / GC
    busy = [_g(d, "system", "cpu", "busy_pct") for d in diag]
    busy = [b for b in busy if b is not None]
    own = [_g(d, "system", "threads", "own_cpu_pct") for d in diag]
    own = [b for b in own if b is not None]
    out["cpu_busy_pct_median"] = _med(busy)
    out["own_cpu_pct_median"] = _med(own)
    if own and _med(own) >= 95:
        V.append(f"processo de teleop perto de 1 núcleo ({_med(own):.0f}% mediana): provável contenção GIL [correlação]")
    if busy and _med(busy) >= 85:
        V.append(f"CPU saturada (busy mediana {_med(busy):.0f}%)")
    gcw = [_g(d, "gc", "window_max_ms") for d in diag]
    gcw = [g for g in gcw if g is not None]
    out["gc_max_ms"] = max(gcw) if gcw else None
    if gcw and max(gcw) >= 30:
        V.append(f"pausa de GC até {max(gcw):.0f} ms")
    temps = [_g(d, "system", "thermal", "max_c") for d in diag]
    temps = [t for t in temps if t is not None]
    out["temp_max_c"] = max(temps) if temps else None
    if temps and max(temps) >= 85:
        V.append(f"temperatura alta ({max(temps):.0f} °C), possível throttling")
    vid = [_g(d, "video", "fps") for d in diag]
    out["video_fps_median"] = _med(vid)
    out["video_raw_MBps_median"] = (_med([_g(d, "video", "raw_bytes_per_s") for d in diag]) or 0) / 1e6
    last = diag[-1]
    out["limiter_bands_final"] = last.get("limiter_bands")
    out["calibration_final"] = last.get("calibration")
    out["diag_errors"] = last.get("diag_errors")
    V.append("VEREDITO = correlações por regras; confirmar com experimento A/B (Wi-Fi vs Tailscale, vídeo on/off)")
    return out


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    diag, pose = load(argv[1:])
    res = analyze(diag, pose)
    print(json.dumps({k: v for k, v in res.items() if k != "verdicts"}, indent=2, default=str))
    print("\nVEREDITO (correlação, não causalidade):")
    for v in res["verdicts"]:
        print(" -", v)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
