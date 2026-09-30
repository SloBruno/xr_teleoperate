#!/usr/bin/env python3
"""Read-only correlation of RAW left-stick y against the sign of vx.

Input: teleop-status.jsonl (locomotion.raw_left_xy / command) and/or
pose-telemetry JSONL (record["locomotion"]).  Consecutive records with
|raw y| >= --push-threshold form a "push window".  Per window we report the
raw-y sign, the vx sign and a verdict label.  This only correlates; it does
not decide whether the device or the code is wrong -- the user must say which
physical direction they pushed.

Expected chain (code): vx = -shaped(raw_y) * cap, so forward (raw y<0) -> vx>0.
  raw y<0 and vx<0  -> CODIGO (contradicts the code chain)
  raw y>0 and vx<0  -> EIXO_INVERTIDO_NO_DISPOSITIVO, *if* the user pushed
                       forward (use --user-pushed forward); code is consistent
  otherwise         -> CONSISTENTE / SEM_DADOS
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def _num(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_records(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    yield record


def extract_samples(records):
    """Yield (timestamp, raw_y, vx) from status or pose records; skip unusable."""
    for index, record in enumerate(records):
        loco = record.get("locomotion")
        if not isinstance(loco, dict):
            continue
        raw = loco.get("raw_left_xy")
        command = loco.get("command")
        if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
            continue
        if not (isinstance(command, (list, tuple)) and command):
            continue
        raw_y, vx = _num(raw[1]), _num(command[0])
        if raw_y is None or vx is None:
            continue
        timestamp = _num(record.get("timestamp")) or _num(record.get("timestamp_monotonic"))
        yield (timestamp if timestamp is not None else float(index)), raw_y, vx


def _sign(value, eps=1e-9):
    return 1 if value > eps else -1 if value < -eps else 0


def windows(samples, push_threshold=0.5, gap_s=2.0):
    current = []
    last_t = None
    for t, raw_y, vx in samples:
        active = abs(raw_y) >= push_threshold
        if active and current and last_t is not None and t - last_t > gap_s:
            yield current
            current = []
        if active:
            current.append((t, raw_y, vx))
            last_t = t
        elif current:
            yield current
            current = []
            last_t = None
    if current:
        yield current


def classify(raw_sign, vx_sign, user_pushed=None):
    if raw_sign < 0 and vx_sign < 0:
        return "CODIGO"
    if raw_sign > 0 and vx_sign < 0:
        if user_pushed == "forward":
            return "EIXO_INVERTIDO_NO_DISPOSITIVO"
        return "EIXO_INVERTIDO_NO_DISPOSITIVO?(confirme que empurrou para frente)"
    if vx_sign == 0:
        return "SEM_COMANDO"
    return "CONSISTENTE"


def summarize(samples, push_threshold=0.5, gap_s=2.0, user_pushed=None):
    samples = list(samples)
    result = []
    for window in windows(samples, push_threshold, gap_s):
        mean_y = sum(s[1] for s in window) / len(window)
        vx_neg = sum(1 for s in window if s[2] < 0)
        vx_pos = sum(1 for s in window if s[2] > 0)
        vx_sign = -1 if vx_neg > vx_pos else 1 if vx_pos > vx_neg else 0
        result.append({
            "start": window[0][0], "end": window[-1][0], "n": len(window),
            "mean_raw_y": mean_y, "raw_y_sign": _sign(mean_y),
            "vx_negative": vx_neg, "vx_positive": vx_pos, "vx_sign": vx_sign,
            "verdict": classify(_sign(mean_y), vx_sign, user_pushed),
        })
    return {"samples": len(samples), "windows": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="teleop-status.jsonl and/or pose-telemetry JSONL")
    parser.add_argument("--push-threshold", type=float, default=0.5)
    parser.add_argument("--gap-s", type=float, default=2.0)
    parser.add_argument("--user-pushed", choices=["forward", "backward"], default=None,
                        help="what the operator says they physically did")
    args = parser.parse_args(argv)
    report = summarize(extract_samples(load_records(args.paths)),
                       args.push_threshold, args.gap_s, args.user_pushed)
    print(f"samples with raw stick + vx: {report['samples']}")
    if not report["windows"]:
        print("SEM_DADOS: nenhuma janela de empurrão (logs antigos não trazem raw_left_xy).")
        return 0
    for w in report["windows"]:
        print(f"[{w['start']:.2f}..{w['end']:.2f}] n={w['n']} raw_y={w['mean_raw_y']:+.2f} "
              f"vx(-:{w['vx_negative']} +:{w['vx_positive']}) -> {w['verdict']}")
    print("Obs: correlação apenas; quem decide é o operador (qual direção empurrou).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
