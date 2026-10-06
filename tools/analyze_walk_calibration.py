#!/usr/bin/env python3
"""Analisa trechos marcados de calibracao de caminhada (trim) do G1.

Uso:
  python tools/analyze_walk_calibration.py pose-telemetry-*.jsonl \
      --markers calib-markers-*.jsonl [--json] [--csv saida.csv]
      [--trim-start 1.0] [--trim-end 0.5]

Alinha por relogio de parede (campo ``timestamp`` = time.time(), ou
``timestamp_utc``) a telemetria (``full_pose_telemetry`` com bloco ``balance``,
G1_BALANCE_TELEMETRY=1) com os marcadores de tools/mark_calibration_segments.py.
Agrupa por condicao e por offset da IMU da pelve (marcador ``o`` do terminal,
offset automatico gravado pela UI web em cada marcador [imu_offset_source =
dds_get/dds_passive/manual], eventos ``offset_set`` bem-sucedidos da UI, ou
config_change_status imu_offset_json; conflito gera aviso, marcador vence).
``--fit-offset``: regressao linear roll->trim vy e pitch->deriva frente na
parada / erro vx, com offset que zera (-b/a).  Yaw so reportado.

Trechos manuais: reta, giro_esquerda, giro_direita.  A PARADA e automatica:
apos o inicio de cada reta (ate o proximo marcador de inicio), o instante em
que o loco_command efetivo cai para ~0 de forma sustentada inicia a parada,
que termina quando a velocidade real fica ~0 por --still-hold s, no proximo
marcador ou apos --stop-max s.  Somente leitura; sem acesso ao robo.

Unidades: loco_command esta em unidades NORMALIZADAS do joystick
(rt/wirelesscontroller; robo converte por limites internos, ver
teleop/utils/loco_wireless.py).  O trim sugerido e nessa mesma escala.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.mark_calibration_segments import segments_from_markers  # noqa: E402

MANUAL_KINDS = ("reta", "giro_esquerda", "giro_direita")
KINDS = ("reta", "parada", "giro_esquerda", "giro_direita")
SHOULDER_LEFT, SHOULDER_RIGHT = (0, 1, 2), (7, 8, 9)  # arm tau_est order: L7 + R7


# ------------------------------------------------------------------ helpers
def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _get(d, *path):
    for p in path:
        if isinstance(d, dict):
            d = d.get(p)
        elif isinstance(d, list) and isinstance(p, int) and -len(d) <= p < len(d):
            d = d[p]
        else:
            return None
    return d


def _vec(v, n):
    if not isinstance(v, (list, tuple)) or len(v) < n:
        return None
    out = [_num(x) for x in v[:n]]
    return None if None in out else out


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def _mean(vals):
    return _stats(vals)["mean"]


def _wall(rec):
    t = _num(rec.get("timestamp"))
    if t is not None:
        return t
    s = rec.get("timestamp_utc")
    if isinstance(s, str):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _yaw_from_quat(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _unwrap(seq):
    out, prev, acc = [], None, 0.0
    for a in seq:
        if prev is not None:
            acc += math.remainder(a - prev, 2 * math.pi)
        else:
            acc = a
        out.append(acc)
        prev = a
    return out


# ------------------------------------------------------------------ loading
def parse_imu_offset(item):
    """config_change_status item {name, content} -> [roll, pitch, yaw] deg or None.
    Accepts content as JSON list, or JSON object with imu/imu_offset_json/imu_offset
    (value list or JSON string of a list).  secondaryimu_* is ignored."""
    if not isinstance(item, dict):
        return None
    name, content = str(item.get("name") or ""), item.get("content")
    if "secondary" in name:          # torso IMU: never the pelvis offset
        return None
    val = content
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(val, dict):
        # real G1 format (Unitree Explorer / config service): {"imu":[r,p,y]}
        val = next((val[k] for k in ("imu_offset_json", "imu_offset", "imu") if k in val), None)
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                return None
    elif "imu_offset" not in name:
        return None
    return _vec(val, 3) if isinstance(val, list) and len(val) == 3 else None


def _sample(rec):
    t = _wall(rec)
    if t is None:
        return None
    b = rec.get("balance")
    s = {"t": t, "balance": isinstance(b, dict) and "error" not in b}
    if not s["balance"]:
        return s
    offs = [parse_imu_offset(c) for c in (b.get("config_changes") or [])]
    offs = [o for o in offs if o is not None]
    s["offset_cfg"] = offs[-1] if offs else None
    s["cmd"] = _vec(b.get("loco_command"), 3)
    raw = b.get("loco_raw")
    if isinstance(raw, dict):
        s["raw"] = {k: _vec(raw.get(k), 2) for k in ("left", "right")}
    else:
        s["raw"] = None
    # odometry: odommodestate -> sportmodestate -> odom fusion/pelvis
    pos = yaw = vel = yaw_rate = None
    src = None
    for name in ("odommodestate", "sportmodestate"):
        sp = _get(b, "sport", name)
        if isinstance(sp, dict) and _vec(sp.get("position"), 2):
            pos = _vec(sp.get("position"), 2)
            yaw = _num(_get(sp, "imu", "rpy", 2))
            vel = _vec(sp.get("velocity"), 2)
            yaw_rate = _num(sp.get("yaw_speed"))
            src = name
            break
    if pos is None:
        for name in ("fusion", "pelvis", "torso"):
            od = _get(b, "odom", name)
            if isinstance(od, dict) and _vec(od.get("position"), 2):
                pos = _vec(od.get("position"), 2)
                q = _vec(od.get("orientation_wxyz"), 4)
                yaw = _yaw_from_quat(q) if q else None
                vel = _vec(od.get("linear"), 2)
                yaw_rate = _num(_get(od, "angular", 2))
                src = f"odom.{name}"
                break
    if vel is None:
        vel = _vec(_get(b, "derived", "body_velocity"), 2)
    speed = _num(_get(b, "derived", "horizontal_speed"))
    if speed is None and vel is not None:
        speed = math.hypot(*vel)
    tau = _get(b, "arms", "tau_est")
    tau = [_num(v) for v in tau] if isinstance(tau, list) and len(tau) >= 10 else None
    s.update({
        "pos": pos, "yaw": yaw, "vel": vel, "speed": speed, "yaw_rate": yaw_rate, "odom_src": src,
        "pelvis_rpy": _vec(_get(b, "imu_pelvis", "rpy"), 3),
        "torso_rpy": _vec(_get(b, "imu_torso", "rpy"), 3),
        "tau_sh_l": _mean([abs(tau[i]) for i in SHOULDER_LEFT if tau[i] is not None]) if tau else None,
        "tau_sh_r": _mean([abs(tau[i]) for i in SHOULDER_RIGHT if tau[i] is not None]) if tau else None,
        "com_dx": _num(_get(b, "com", "dx_mm")), "com_dy": _num(_get(b, "com", "dy_mm")),
    })
    return s


def load_telemetry(paths):
    samples, bad, files = [], 0, []
    for path in paths:
        n = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if not isinstance(rec, dict) or rec.get("event") != "full_pose_telemetry":
                    continue
                s = _sample(rec)
                if s is not None:
                    samples.append(s)
                    n += 1
        files.append({"file": str(path), "samples": n})
    samples.sort(key=lambda s: s["t"])
    return samples, bad, files


def load_markers(path):
    events, bad = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(ev, dict) and ev.get("event") == "calib_marker" and _num(ev.get("timestamp")) is not None:
                events.append(ev)
    return events, bad


# ---------------------------------------------------------------- detection
def _cmd_zero(s, eps):
    c = s.get("cmd")
    return None if c is None else all(abs(v) < eps for v in c)


def detect_release(samples, eps=0.02, hold_s=0.2):
    """Last nonzero->zero transition whose zero run lasts >= hold_s (or until
    the data ends with at least hold_s).  Returns time or None."""
    found, prev_nonzero, run_start = None, False, None
    for s in samples:
        z = _cmd_zero(s, eps)
        if z is None:
            continue
        if z:
            if prev_nonzero and run_start is None:
                run_start = s["t"]
            if run_start is not None and s["t"] - run_start >= hold_s:
                found = run_start
        else:
            prev_nonzero, run_start = True, None
    return found


def detect_still(samples, t0, speed_eps, hold_s):
    """(time robot became still, time stillness confirmed) after t0, else (None, None)."""
    run = None
    for s in samples:
        if s["t"] < t0 or s.get("speed") is None:
            continue
        if s["speed"] < speed_eps:
            run = s["t"] if run is None else run
            if s["t"] - run >= hold_s - 1e-9:
                return run, s["t"]
        else:
            run = None
    return None, None


# ---------------------------------------------------------------- metrics
def _window(samples, a, b):
    return [s for s in samples if a <= s["t"] <= b]


def _trimmed(samples, a, b, ts, te):
    w = _window(samples, a + ts, b - te) if b - a > ts + te else []
    return w or _window(samples, a, b)


def _missing(win):
    miss = []
    if not win:
        return ["amostras"]
    bal = [s for s in win if s["balance"]]
    if not bal:
        return ["balance"]
    for key, label in (("cmd", "loco_command"), ("raw", "loco_raw"), ("pos", "odom"),
                       ("yaw", "yaw"), ("pelvis_rpy", "imu_pelvis"), ("torso_rpy", "imu_torso"),
                       ("tau_sh_l", "arms.tau_est"), ("com_dx", "com")):
        if all(s.get(key) is None for s in bal):
            miss.append(label)
    return miss


def _common(win):
    bal = [s for s in win if s["balance"]]
    cmd = [s["cmd"] for s in bal if s.get("cmd")]
    raws = [s["raw"] for s in bal if s.get("raw")]

    def raw(side, i):
        return _stats([r[side][i] for r in raws if r.get(side)])

    def rpy(key, i):
        return _mean([s[key][i] for s in bal if s.get(key)])
    return {
        "amostras": len(win),
        "cmd": {k: _stats([c[i] for c in cmd]) for i, k in enumerate(("vx", "vy", "omega"))},
        "stick": {"left_x": raw("left", 0), "left_y": raw("left", 1),
                  "right_x": raw("right", 0), "right_y": raw("right", 1)},
        "cmd_zero_frac": (sum(1 for c in cmd if max(map(abs, c)) < 1e-3) / len(cmd)) if cmd else None,
        "pelve_roll": rpy("pelvis_rpy", 0), "pelve_pitch": rpy("pelvis_rpy", 1),
        "torso_roll": rpy("torso_rpy", 0), "torso_pitch": rpy("torso_rpy", 1),
        "tau_ombro_esq": _mean([s.get("tau_sh_l") for s in bal]),
        "tau_ombro_dir": _mean([s.get("tau_sh_r") for s in bal]),
        "com_dx_mm": _mean([s.get("com_dx") for s in bal]),
        "com_dy_mm": _mean([s.get("com_dy") for s in bal]),
        "odom_fonte": next((s["odom_src"] for s in bal if s.get("odom_src")), None),
    }


def _odom_pair(win):
    pts = [s for s in win if s.get("pos") is not None]
    return (pts[0], pts[-1]) if len(pts) >= 2 else (None, None)


def _yaw_unwrapped(win):
    ys = [(s["t"], s["yaw"]) for s in win if s.get("yaw") is not None]
    if len(ys) < 2:
        return None
    u = _unwrap([y for _, y in ys])
    return u[-1] - u[0]


def _yaw_integrated(win):
    ys = [(s["t"], s["yaw_rate"]) for s in win if s.get("yaw_rate") is not None]
    if len(ys) < 2:
        return None
    return sum(0.5 * (a[1] + b[1]) * (b[0] - a[0]) for a, b in zip(ys, ys[1:]))


def _frame_disp(p0, p1, heading):
    dx, dy = p1["pos"][0] - p0["pos"][0], p1["pos"][1] - p0["pos"][1]
    if heading is None:
        return math.hypot(dx, dy), None, None
    fwd = dx * math.cos(heading) + dy * math.sin(heading)
    left = -dx * math.sin(heading) + dy * math.cos(heading)
    return math.hypot(dx, dy), fwd, left


def analyze_reta(samples, a, b, ts, te):
    full = _window(samples, a, b)
    win = _trimmed(samples, a, b, ts, te)
    out = _common(win)
    out["missing"] = _missing(full)
    out["trim_sugerido"] = {"vy": out["cmd"]["vy"]["mean"], "omega": out["cmd"]["omega"]["mean"]}
    out["velocidade_media_mps"] = _mean([s.get("speed") for s in win])
    out["yaw_rate_medio"] = _mean([s.get("yaw_rate") for s in win])
    p0, p1 = _odom_pair(full)
    if p0 is None:
        out.update(distancia_m=None, avanco_m=None, deriva_lateral_m=None, deriva_heading_deg=None)
    else:
        dist, fwd, left = _frame_disp(p0, p1, p0.get("yaw"))
        dyaw = _yaw_unwrapped(full)
        out.update(distancia_m=dist, avanco_m=fwd, deriva_lateral_m=left,
                   deriva_heading_deg=None if dyaw is None else math.degrees(dyaw))
    dt = p1["t"] - p0["t"] if p0 is not None else 0.0
    out["velocidade_frente_mps"] = (out["avanco_m"] / dt) if p0 is not None and dt > 0 and out["avanco_m"] is not None else None
    return out


def analyze_parada(samples, a, b, still_eps, still_hold):
    win = _window(samples, a, b)
    out = _common(win)
    out["missing"] = _missing(win)
    still_t, _ = detect_still(win, a, still_eps, still_hold)
    out["tempo_ate_parar_s"] = None if still_t is None else still_t - a
    out["parou"] = still_t is not None
    out["velocidade_max_mps"] = max((s["speed"] for s in win if s.get("speed") is not None), default=None)
    p0, p1 = _odom_pair(win)
    if p0 is None:
        out.update(deslocamento_m=None, deriva_frente_m=None, deriva_lateral_m=None, giro_deg=None)
    else:
        dist, fwd, left = _frame_disp(p0, p1, p0.get("yaw"))
        dyaw = _yaw_unwrapped(win)
        out.update(deslocamento_m=dist, deriva_frente_m=fwd, deriva_lateral_m=left,
                   giro_deg=None if dyaw is None else math.degrees(dyaw))
    return out


def analyze_giro(samples, a, b, ts, te, kind):
    full = _window(samples, a, b)
    out = _common(_trimmed(samples, a, b, ts, te))
    out["missing"] = _missing(full)
    ang, method = _yaw_unwrapped(full), "imu_yaw_desembrulhado"
    if ang is None:
        ang, method = _yaw_integrated(full), "integral_yaw_speed"
    out["angulo_deg"] = None if ang is None else math.degrees(ang)
    out["angulo_metodo"] = method if ang is not None else None
    want = 1 if kind == "giro_esquerda" else -1
    out["sentido_ok"] = None if ang is None else (ang * want > 0)
    p0, p1 = _odom_pair(full)
    out["translacao_m"] = None if p0 is None else _frame_disp(p0, p1, None)[0]
    out["yaw_rate_medio"] = _mean([s.get("yaw_rate") for s in full])
    return out


# --------------------------------------------------------------- offset fit
def linear_fit(xs, ys):
    """Least squares y = a*x + b; offset_zero = -b/a with warnings."""
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n, levels = len(pts), len({round(x, 6) for x, _ in pts})
    out = {"n": n, "niveis": levels, "inclinacao": None, "intercepto": None, "r2": None,
           "offset_zero": None, "faixa_medida": None, "avisos": []}
    if levels < 3:
        out["avisos"].append(f"apenas {levels} niveis distintos de offset (<3): ajuste pouco confiavel")
    if n < 2 or levels < 2:
        out["avisos"].append("pontos insuficientes para regressao")
        return out
    mx = statistics.fmean(x for x, _ in pts)
    my = statistics.fmean(y for _, y in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    syy = sum((y - my) ** 2 for _, y in pts)
    a = sxy / sxx
    b = my - a * mx
    out.update(inclinacao=a, intercepto=b,
               r2=(sxy * sxy / (sxx * syy)) if syy > 0 else 1.0,
               faixa_medida=[min(x for x, _ in pts), max(x for x, _ in pts)])
    if abs(a) < 1e-12:
        out["avisos"].append("inclinacao ~0: offset nao afeta a metrica")
        return out
    z = -b / a
    out["offset_zero"] = z
    lo, hi = out["faixa_medida"]
    if not lo - 1e-9 <= z <= hi + 1e-9:
        out["avisos"].append(f"offset_zero {z:+.2f} fora da faixa medida [{lo:+.2f}, {hi:+.2f}]: extrapolacao")
    return out


def fit_offsets(segs):
    by_cond = {}
    for s in segs:
        if s.get("imu_offset") is not None:
            by_cond.setdefault(s["condition"], []).append(s)
    out = {}
    for cond, items in by_cond.items():
        retas = [s for s in items if s["kind"] == "reta"]
        paradas = [s for s in items if s["kind"] == "parada"]
        erro_vx = []
        for s in retas:
            v, c = s.get("velocidade_frente_mps"), _get(s, "cmd", "vx", "mean")
            erro_vx.append(None if v is None or c is None else v - c)
        out[cond] = {
            "roll_vs_trim_vy": linear_fit([s["imu_offset"][0] for s in retas],
                                          [_metric(s, "trim_vy") for s in retas]),
            "pitch_vs_deriva_frente_parada": linear_fit([s["imu_offset"][1] for s in paradas],
                                                        [s.get("deriva_frente_m") for s in paradas]),
            "pitch_vs_erro_vx": linear_fit([s["imu_offset"][1] for s in retas], erro_vx),
            "yaw": {"niveis": sorted({s["imu_offset"][2] for s in items}),
                    "trim_omega_por_yaw": {f"{y:+.2f}": _mean([_metric(s, "trim_omega") for s in retas
                                                                if s["imu_offset"][2] == y])
                                           for y in sorted({s["imu_offset"][2] for s in retas})},
                    "recomendacao": None,
                    "nota": "yaw apenas reportado; nenhum ajuste recomendado"},
            "nota_erro_vx": "erro_vx = velocidade real a frente (m/s) - vx comandado (NORMALIZADO): "
                            "compare apenas a variacao entre offsets",
        }
    return out


def _offset_label(cond, off):
    if off is None:
        return f"{cond} @ offset=desconhecido"
    return f"{cond} @ roll={off[0]:+.2f} pitch={off[1]:+.2f} yaw={off[2]:+.2f}"


def _aggregate(segs, keyfn):
    groups = {}
    for s in segs:
        groups.setdefault(keyfn(s), {}).setdefault(s["kind"], []).append(s)
    out = {}
    for key, kinds in groups.items():
        out[key] = {}
        for kind, items in kinds.items():
            row = {"n": len(items), "tentativas": [i["attempt"] for i in items]}
            for name in REPORT_METRICS.get(kind, ()) + COMMON_METRICS:
                vals = [v for v in (_metric(i, name) for i in items) if v is not None]
                row[name] = {"mean": statistics.fmean(vals) if vals else None,
                             "std": statistics.stdev(vals) if len(vals) > 1 else None,
                             "n": len(vals)}
            out[key][kind] = row
    return out


# ------------------------------------------------------------------ main
REPORT_METRICS = {
    "reta": ("trim_vy", "trim_omega", "distancia_m", "deriva_lateral_m", "deriva_heading_deg",
             "velocidade_media_mps", "yaw_rate_medio", "cmd_vx"),
    "parada": ("deslocamento_m", "deriva_frente_m", "deriva_lateral_m", "tempo_ate_parar_s", "giro_deg"),
    "giro_esquerda": ("angulo_deg", "translacao_m", "cmd_omega", "cmd_vx", "cmd_vy"),
    "giro_direita": ("angulo_deg", "translacao_m", "cmd_omega", "cmd_vx", "cmd_vy"),
}
COMMON_METRICS = ("pelve_pitch", "pelve_roll", "torso_pitch", "tau_ombro_esq", "tau_ombro_dir",
                  "com_dx_mm", "com_dy_mm")


def _metric(seg, name):
    if name == "trim_vy":
        return _get(seg, "trim_sugerido", "vy")
    if name == "trim_omega":
        return _get(seg, "trim_sugerido", "omega")
    if name.startswith("cmd_"):
        return _get(seg, "cmd", name[4:], "mean")
    return seg.get(name)


def analyze(telemetry_paths, markers_path, trim_start=1.0, trim_end=0.5, release_eps=0.02,
            release_hold=0.2, still_speed=0.03, still_hold=0.5, stop_max=5.0, fit_offset=False):
    samples, bad, files = load_telemetry(telemetry_paths)
    events, bad_m = load_markers(markers_path)
    marked = segments_from_markers(events)
    warnings = []
    data_end = samples[-1]["t"] if samples else None
    if not samples:
        warnings.append("telemetria sem amostras full_pose_telemetry")
    elif not any(s["balance"] for s in samples):
        warnings.append("telemetria sem bloco balance (rode com G1_BALANCE_TELEMETRY=1)")
    segs = []
    for m in marked:
        if m["kind"] not in MANUAL_KINDS:
            continue
        a = m["start"]
        b = m["end"] if m["end"] is not None else data_end
        if b is None:
            continue
        cfg = [x["offset_cfg"] for x in samples if x["t"] <= a and x.get("offset_cfg") is not None]
        tel_off = cfg[-1] if cfg else None
        off = m.get("imu_offset")
        off_src = None
        if off is not None:
            src = m.get("imu_offset_source")
            off_src = f"marcador:{src}" if src else "marcador"
        if off is None and tel_off is not None:
            off, off_src = tel_off, "config_change_status"
        elif off is not None and tel_off is not None and any(abs(u - v) > 1e-6 for u, v in zip(off, tel_off)):
            warnings.append(f"{m['condition']} t{m['attempt']} {m['kind']}: conflito de offset IMU "
                            f"marcador={off} vs telemetria={tel_off} (usado o marcador)")
        base = {"condition": m["condition"] or "?", "attempt": m["attempt"], "kind": m["kind"],
                "imu_offset": off, "imu_offset_fonte": off_src, "imu_offset_telemetria": tel_off,
                "start": a, "end": b, "auto": False, "notes": m["notes"],
                "fechado_por_marcador": m["end"] is not None}
        if m["kind"] == "reta":
            limit = m["next_boundary"] if m["next_boundary"] is not None else data_end
            rel = detect_release(_window(samples, a, limit), release_eps, release_hold)
            reta_end = min(b, rel) if rel is not None else b
            seg = dict(base, end=reta_end, soltou_joystick_s=None if rel is None else rel - a)
            seg.update(analyze_reta(samples, a, reta_end, trim_start, trim_end))
            segs.append(seg)
            if rel is None:
                warnings.append(f"{base['condition']} t{base['attempt']} reta: parada nao detectada "
                                f"(loco_command ~0 sustentado ausente ou sem balance)")
                continue
            hard = min(limit, rel + stop_max)
            _, confirmed = detect_still(_window(samples, rel, hard), rel, still_speed, still_hold)
            p_end = confirmed if confirmed is not None else hard
            pseg = dict(base, kind="parada", start=rel, end=p_end, auto=True, notes=[],
                        fechado_por_marcador=False)
            pseg.update(analyze_parada(samples, rel, p_end, still_speed, still_hold))
            segs.append(pseg)
        else:
            seg = dict(base)
            seg.update(analyze_giro(samples, a, b, trim_start, trim_end, m["kind"]))
            segs.append(seg)
    for s in segs:
        s["duration_s"] = s["end"] - s["start"]
        if s["missing"]:
            warnings.append(f"{s['condition']} t{s['attempt']} {s['kind']}: ausente {', '.join(s['missing'])}")

    agg_out = _aggregate(segs, lambda s: s["condition"])
    agg_offset = _aggregate(segs, lambda s: _offset_label(s["condition"], s.get("imu_offset")))
    diff = {}
    if "com_caixa" in agg_out and "sem_caixa" in agg_out:
        for kind in KINDS:
            c, s_ = agg_out["com_caixa"].get(kind), agg_out["sem_caixa"].get(kind)
            if not c or not s_:
                continue
            diff[kind] = {}
            for name in REPORT_METRICS[kind] + COMMON_METRICS:
                cm, sm = c[name]["mean"], s_[name]["mean"]
                diff[kind][name] = None if cm is None or sm is None else cm - sm
    return {
        "telemetry_files": files, "markers_file": str(markers_path),
        "samples": len(samples), "bad_lines": bad, "bad_marker_lines": bad_m,
        "params": {"trim_start_s": trim_start, "trim_end_s": trim_end, "release_eps": release_eps,
                   "release_hold_s": release_hold, "still_speed_mps": still_speed,
                   "still_hold_s": still_hold, "stop_max_s": stop_max},
        "unidades": {"loco_command": "normalizado (joystick rt/wirelesscontroller), nao m/s",
                     "odom": "m, rad (odometria do robo; precisao nao validada)"},
        "segments": segs, "aggregate": agg_out, "aggregate_by_offset": agg_offset,
        "fit_offset": fit_offsets(segs) if fit_offset else None,
        "diferenca_com_menos_sem": diff,
        "warnings": warnings,
    }


# ------------------------------------------------------------------ output
def _f(v, nd=3):
    return "-" if v is None else (f"{v:+.{nd}f}" if isinstance(v, float) else str(v))


def _ms(d, nd=3):
    if not d or d.get("mean") is None:
        return "-"
    return f"{d['mean']:+.{nd}f}" + (f"±{d['std']:.{nd}f}" if d.get("std") is not None else "")


def render_text(rep):
    L = [f"telemetria: {rep['samples']} amostras ({rep['bad_lines']} linhas ruins); "
         f"marcadores: {rep['markers_file']}",
         "obs.: loco_command em unidades NORMALIZADAS do joystick (nao m/s)", ""]
    for s in rep["segments"]:
        head = (f"[{s['condition']} t{s['attempt']}] {s['kind']}{' (auto)' if s['auto'] else ''} "
                f"{s['end'] - s['start']:.1f}s")
        if s["kind"] == "reta":
            L.append(f"{head}  trim sugerido vy={_f(s['trim_sugerido']['vy'])} "
                     f"omega={_f(s['trim_sugerido']['omega'])}  vx={_ms(s['cmd']['vx'])} "
                     f"dist={_f(s['distancia_m'], 2)}m lateral={_f(s['deriva_lateral_m'], 3)}m "
                     f"heading={_f(s['deriva_heading_deg'], 1)}° v={_f(s['velocidade_media_mps'], 2)}m/s")
        elif s["kind"] == "parada":
            L.append(f"{head}  deslocamento={_f(s['deslocamento_m'], 3)}m "
                     f"(frente={_f(s['deriva_frente_m'], 3)} lateral={_f(s['deriva_lateral_m'], 3)}) "
                     f"tempo ate parar={_f(s['tempo_ate_parar_s'], 2)}s giro={_f(s['giro_deg'], 1)}°")
        else:
            L.append(f"{head}  angulo={_f(s['angulo_deg'], 1)}° ({s['angulo_metodo'] or '-'}) "
                     f"omega_cmd={_ms(s['cmd']['omega'])} vx={_ms(s['cmd']['vx'])} "
                     f"vy={_ms(s['cmd']['vy'])} translacao={_f(s['translacao_m'], 3)}m")
        L.append(f"    pelve pitch/roll={_f(s['pelve_pitch'])}/{_f(s['pelve_roll'])} "
                 f"torso pitch={_f(s['torso_pitch'])} tau ombro E/D={_f(s['tau_ombro_esq'], 1)}/"
                 f"{_f(s['tau_ombro_dir'], 1)} CoM dx/dy={_f(s['com_dx_mm'], 0)}/{_f(s['com_dy_mm'], 0)}mm"
                 + (f"  AUSENTE: {', '.join(s['missing'])}" if s["missing"] else "")
                 + (f"  notas: {s['notes']}" if s["notes"] else ""))
    L.append("\n== agregado por condicao (media±desvio entre tentativas) ==")
    for cond, kinds in rep["aggregate"].items():
        for kind in KINDS:
            row = kinds.get(kind)
            if not row:
                continue
            vals = " ".join(f"{n}={_ms(row[n])}" for n in REPORT_METRICS[kind] if row[n]["n"])
            L.append(f"{cond} {kind} n={row['n']}: {vals}")
    if len(rep["aggregate_by_offset"]) > 1 or any("desconhecido" not in k for k in rep["aggregate_by_offset"]):
        L.append("\n== agregado por condicao e offset IMU (graus) ==")
        for key, kinds in sorted(rep["aggregate_by_offset"].items()):
            for kind in ("reta", "parada"):
                row = kinds.get(kind)
                if row:
                    vals = " ".join(f"{n}={_ms(row[n])}" for n in REPORT_METRICS[kind][:4] if row[n]["n"])
                    L.append(f"{key} {kind} n={row['n']}: {vals}")
    if rep.get("fit_offset"):
        L.append("\n== ajuste de offset (--fit-offset) ==")
        for cond, fits in rep["fit_offset"].items():
            for name in ("roll_vs_trim_vy", "pitch_vs_deriva_frente_parada", "pitch_vs_erro_vx"):
                f = fits[name]
                L.append(f"{cond} {name}: a={_f(f['inclinacao'], 4)} b={_f(f['intercepto'], 4)} "
                         f"R2={_f(f['r2'], 2)} n={f['n']} niveis={f['niveis']} "
                         f"offset que zera={_f(f['offset_zero'], 2)}°"
                         + (f"  AVISO: {'; '.join(f['avisos'])}" if f["avisos"] else ""))
            L.append(f"{cond} yaw: niveis={fits['yaw']['niveis']} (so reportado, sem recomendacao)")
    if rep["diferenca_com_menos_sem"]:
        L.append("\n== diferenca com_caixa - sem_caixa ==")
        for kind, row in rep["diferenca_com_menos_sem"].items():
            L.append(f"{kind}: " + " ".join(f"{n}={_f(v)}" for n, v in row.items()
                                            if n in REPORT_METRICS[kind] and v is not None))
    if rep["warnings"]:
        L.append("\navisos:")
        L += [f"  - {w}" for w in rep["warnings"]]
    return "\n".join(L)


CSV_FIELDS = ("condition", "attempt", "kind", "auto", "start", "end", "duracao_s", "trim_vy",
              "trim_omega", "cmd_vx", "cmd_vy", "cmd_omega", "distancia_m", "deriva_lateral_m",
              "deriva_heading_deg", "velocidade_media_mps", "deslocamento_m", "tempo_ate_parar_s",
              "angulo_deg", "translacao_m") + COMMON_METRICS + ("imu_offset", "imu_offset_fonte", "missing",)


def write_csv(rep, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for s in rep["segments"]:
            row = {k: s.get(k) for k in CSV_FIELDS}
            row.update(duracao_s=s["end"] - s["start"], trim_vy=_metric(s, "trim_vy"),
                       trim_omega=_metric(s, "trim_omega"), cmd_vx=_metric(s, "cmd_vx"),
                       cmd_vy=_metric(s, "cmd_vy"), cmd_omega=_metric(s, "cmd_omega"),
                       missing=";".join(s["missing"]),
                       imu_offset=None if s.get("imu_offset") is None else " ".join(f"{v:g}" for v in s["imu_offset"]))
            w.writerow(row)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Analisa calibracao de caminhada (trim) por trechos marcados.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("telemetry", nargs="+", type=Path, help="pose-telemetry-*.jsonl")
    ap.add_argument("--markers", required=True, type=Path, help="calib-markers-*.jsonl")
    ap.add_argument("--trim-start", type=float, default=1.0, help="descarta s no inicio do trecho")
    ap.add_argument("--trim-end", type=float, default=0.5, help="descarta s no fim do trecho")
    ap.add_argument("--release-eps", type=float, default=0.02, help="|cmd| abaixo = joystick solto")
    ap.add_argument("--release-hold", type=float, default=0.2, help="s sustentado para detectar soltura")
    ap.add_argument("--still-speed", type=float, default=0.03, help="m/s considerado parado")
    ap.add_argument("--still-hold", type=float, default=0.5, help="s parado para fechar a parada")
    ap.add_argument("--stop-max", type=float, default=5.0, help="duracao maxima da parada (s)")
    ap.add_argument("--fit-offset", action="store_true",
                    help="regressao offset IMU (roll->trim vy; pitch->deriva frente na parada e erro vx)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args(argv)
    rep = analyze(a.telemetry, a.markers, a.trim_start, a.trim_end, a.release_eps, a.release_hold,
                  a.still_speed, a.still_hold, a.stop_max, a.fit_offset)
    if a.csv:
        write_csv(rep, a.csv)
    print(json.dumps(rep, indent=2, ensure_ascii=False) if a.json else render_text(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
