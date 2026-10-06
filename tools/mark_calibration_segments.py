#!/usr/bin/env python3
"""Terminal de marcadores manuais para calibracao de caminhada (trim) do G1.

Ferramenta INDEPENDENTE da teleop: nao cria DDS, nao publica nada, nao toca no
robo.  Apenas le linhas do teclado (input(), funciona via SSH comum) e grava
JSONL append-only (flush + fsync a cada evento) com relogio de parede UTC e
monotonico, para alinhar depois com pose-telemetry-*.jsonl.

A PARADA (soltar o joystick) NAO e marcada aqui: o analisador a detecta
automaticamente pelo loco_command efetivo.  Pode ser operado por outra pessoa
ou entre tentativas.

Uso: python tools/mark_calibration_segments.py [--out ARQ.jsonl]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DIR = Path("/home/unitree/.local/state/xr_teleoperate")

CONDITIONS = {"s": "sem_caixa", "c": "com_caixa"}
SEGMENT_KEYS = {"r": "reta", "e": "giro_esquerda", "d": "giro_direita"}

HELP = """teclas (letra + Enter):
  s        condicao atual = sem caixa
  c        condicao atual = com caixa
  t N      tentativa N (ex.: t 2)
  r        inicio da RETA (andando, corrigindo com joystick)
  e        inicio do giro 360 para a ESQUERDA
  d        inicio do giro 360 para a DIREITA
  f        fim do trecho aberto
  o R P Y  offset da IMU da pelve em uso (graus roll pitch yaw, como no app
           Unitree Explorer; vale ate mudar). Ex.: o 0.5 0 0
  x texto  nota livre (ex.: x pe escorregou)
  u        desfazer ultimo marcador
  ?        esta ajuda
  q        sair
(a parada apos soltar o joystick e detectada automaticamente pelo analisador;
 abrir um trecho fecha o anterior automaticamente)"""


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def default_out_path(now: float | None = None) -> Path:
    stamp = datetime.fromtimestamp(time.time() if now is None else now,
                                   timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_DIR / f"calib-markers-{stamp}.jsonl"


def effective_markers(events):
    """Apply 'undo' events: return markers that survive, in order."""
    alive = []
    for ev in events:
        if ev.get("event") != "calib_marker":
            continue
        if ev.get("type") == "undo":
            target = ev.get("undo_of")
            idx = next((i for i in range(len(alive) - 1, -1, -1)
                        if alive[i].get("seq") == target), None)
            if idx is None and target is None:
                idx = next((i for i in range(len(alive) - 1, -1, -1)
                            if alive[i].get("type") not in ("session_start", "session_end")), None)
            if idx is not None:
                alive.pop(idx)
            continue
        alive.append(ev)
    return alive


def _is_pelvis_offset_set(ev):
    """Successful pelvis-IMU offset write recorded by tools/calib_web.py."""
    return (ev.get("type") == "offset_set" and ev.get("ok") is True and not ev.get("dry_run")
            and ev.get("offset_key", "imu") == "imu" and ev.get("imu_offset") is not None)


def segments_from_markers(events):
    """Markers -> [{kind, condition, attempt, start, end, next_boundary, notes}].

    Times are wall clock (time.time()).  ``next_boundary`` is the next
    segment_start/session_end after the segment (used to search the automatic
    stop after a reta)."""
    alive = sorted(effective_markers(events), key=lambda e: (float(e["timestamp"]), e.get("seq", 0)))
    segs, cur, offset = [], None, None
    for ev in alive:
        typ, t = ev.get("type"), float(ev["timestamp"])
        if typ == "imu_offset" or _is_pelvis_offset_set(ev):
            offset = ev.get("imu_offset")
        if typ in ("segment_start", "segment_end", "session_end") and cur is not None:
            cur["end"] = t
            segs.append(cur)
            cur = None
        if typ == "segment_start":
            auto = ev.get("imu_offset_source") is not None and ev.get("imu_offset") is not None
            cur = {"kind": ev.get("kind"), "condition": ev.get("condition"),
                   "attempt": ev.get("attempt"), "start": t, "end": None,
                   "start_seq": ev.get("seq"), "notes": [],
                   # web UI writes the live offset (+ its source) on every marker
                   "imu_offset": ev.get("imu_offset") if auto else (
                       offset if offset is not None else ev.get("imu_offset")),
                   "imu_offset_source": ev.get("imu_offset_source") if auto else None}
        elif typ == "note" and cur is not None:
            cur["notes"].append(ev.get("note"))
    if cur is not None:
        segs.append(cur)  # left open: analyzer uses end of data
    boundaries = sorted(float(e["timestamp"]) for e in alive
                        if e.get("type") in ("segment_start", "session_end"))
    for s in segs:
        s["next_boundary"] = next((b for b in boundaries if b > s["start"]), None)
    return segs


class MarkerSession:
    def __init__(self, path, wall_clock=time.time, monotonic_clock=time.monotonic,
                 offset_provider=None):
        """``offset_provider`` (optional, web UI): callable -> (pelvis offset
        [r,p,y] deg or None, source str) recorded automatically on every event."""
        self._wall, self._mono = wall_clock, monotonic_clock
        self._offset_provider = offset_provider
        self._fh = None
        self.path = None if path is None else Path(path)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        self._seq = 0
        self._events = []
        self.closed = False
        self._write("session_start", note=f"pid={os.getpid()}")

    # ---------------------------------------------------------------- state
    def state(self):
        cond = att = offset = None
        open_seg = None
        for ev in effective_markers(self._events):
            typ = ev["type"]
            if typ == "condition":
                cond = ev["condition"]
            elif typ == "attempt":
                att = ev["attempt"]
            elif typ == "imu_offset" or _is_pelvis_offset_set(ev):
                offset = ev["imu_offset"]
            elif typ == "segment_start":
                open_seg = {"kind": ev["kind"], "since_utc": ev["timestamp_utc"], "seq": ev["seq"]}
            elif typ in ("segment_end", "session_end"):
                open_seg = None
        return {"condition": cond, "attempt": att, "open_segment": open_seg, "imu_offset": offset}

    def status_line(self):
        st = self.state()
        seg = st["open_segment"]
        seg_s = f"{seg['kind']} (desde {seg['since_utc'][11:23]})" if seg else "nenhum"
        off = st["imu_offset"]
        off_s = "desconhecido" if off is None else "/".join(f"{v:+.2f}" for v in off)
        return (f"[condicao={st['condition'] or '?'} tentativa={st['attempt'] or '?'} "
                f"offset_imu(r/p/y)={off_s} trecho aberto={seg_s}]")

    # ---------------------------------------------------------------- write
    def _write(self, typ, **fields):
        st = self.state() if self._events else {"condition": None, "attempt": None, "imu_offset": None}
        wall, mono = float(self._wall()), float(self._mono())
        self._seq += 1
        ev = {"event": "calib_marker", "schema_version": 1, "seq": self._seq, "type": typ,
              "timestamp": wall, "timestamp_utc": utc_iso(wall), "timestamp_monotonic": mono,
              "clock_domain": {"timestamp": "wall_clock_utc", "timestamp_monotonic": "monotonic"},
              "condition": st["condition"], "attempt": st["attempt"], "imu_offset": st["imu_offset"],
              "kind": None, "note": None}
        if self._offset_provider is not None:
            try:
                off, src = self._offset_provider()
            except Exception:
                off, src = None, "erro"
            ev["imu_offset"] = None if off is None else [float(v) for v in off]
            ev["imu_offset_source"] = src or "desconhecido"
        ev.update(fields)
        self._events.append(ev)
        if self._fh is not None:
            self._fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass
        return ev

    def record(self, typ, **fields):
        """Public append (web UI): same schema, fsync'd like key presses."""
        if self.closed:
            raise RuntimeError("sessao encerrada")
        return self._write(typ, **fields)

    def events(self):
        return list(self._events)

    def handle(self, line):
        line = (line or "").strip()
        if not line:
            return self.status_line()
        key, _, arg = line.partition(" ")
        key, arg = key.lower(), arg.strip()
        if key in ("?", "h", "help"):
            return HELP
        if key in CONDITIONS:
            ev = self._write("condition", condition=CONDITIONS[key])
            return f"OK #{ev['seq']} condicao = {CONDITIONS[key].replace('_', ' ')}  {self.status_line()}"
        if key == "t":
            try:
                n = int(arg)
            except ValueError:
                return "ERRO: use 't N' (ex.: t 2); nada gravado"
            ev = self._write("attempt", attempt=n)
            return f"OK #{ev['seq']} tentativa = {n}  {self.status_line()}"
        if key in SEGMENT_KEYS:
            prev = self.state()["open_segment"]
            ev = self._write("segment_start", kind=SEGMENT_KEYS[key])
            closed = f" (fechou {prev['kind']})" if prev else ""
            return f"OK #{ev['seq']} INICIO {SEGMENT_KEYS[key]} {ev['timestamp_utc']}{closed}  {self.status_line()}"
        if key == "f":
            prev = self.state()["open_segment"]
            if prev is None:
                return f"nenhum trecho aberto; nada gravado  {self.status_line()}"
            ev = self._write("segment_end", kind=prev["kind"])
            return f"OK #{ev['seq']} FIM {prev['kind']} {ev['timestamp_utc']}  {self.status_line()}"
        if key == "o":
            try:
                vals = [float(v.replace(",", ".")) for v in arg.split()]
            except ValueError:
                vals = []
            if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
                return "ERRO: use 'o ROLL PITCH YAW' em graus (ex.: o 0.5 0 0); nada gravado"
            ev = self._write("imu_offset", imu_offset=vals)
            return f"OK #{ev['seq']} offset IMU = {vals}  {self.status_line()}"
        if key == "x":
            if not arg:
                return "ERRO: use 'x texto'; nada gravado"
            ev = self._write("note", note=arg)
            return f"OK #{ev['seq']} nota gravada  {self.status_line()}"
        if key == "u":
            alive = [e for e in effective_markers(self._events) if e["type"] != "session_start"]
            if not alive:
                return "nada para desfazer"
            target = alive[-1]
            self._write("undo", undo_of=target["seq"])
            desc = target.get("kind") or target.get("condition") or target.get("attempt") or target.get("note")
            return f"desfeito #{target['seq']} ({target['type']} {desc})  {self.status_line()}"
        if key == "q":
            self.close()
            return "sessao encerrada"
        if key == "p":
            return ("tecla 'p' removida: a parada e detectada automaticamente pelo "
                    "analisador (loco_command ~0). Nada gravado.")
        return f"tecla desconhecida '{key}'; '?' para ajuda. Nada gravado."

    def close(self):
        if self.closed:
            return
        self._write("session_end")
        self.closed = True
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Terminal de marcadores manuais (calibracao de caminhada). "
                                             "Nao usa DDS; so grava JSONL.",
                                 epilog=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"arquivo JSONL (padrao {DEFAULT_DIR}/calib-markers-<UTC>.jsonl)")
    a = ap.parse_args(argv)
    out = a.out or default_out_path()
    session = MarkerSession(out)
    print(f"gravando em {out}\n{HELP}\n{session.status_line()}", flush=True)
    try:
        while not session.closed:
            try:
                line = input("> ")
            except EOFError:
                break
            print(session.handle(line), flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        session.close()
        print(f"arquivo: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
