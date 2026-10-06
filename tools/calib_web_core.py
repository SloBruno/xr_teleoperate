"""Logica pura da UI web de calibracao de caminhada do G1 (sem DDS, sem HTTP).

Tudo aqui e testavel com um backend fake:
  * payload EXATO do servico ``config`` (igual ao app Unitree Explorer);
  * parsing de respostas GET / config_change_status / requests passivos;
  * politica de seguranca para escrita do offset da IMU;
  * controlador da sessao (marcadores + offset + historico + logs JSONL).

O backend DDS real fica em tools/calib_web.py (import tardio do SDK).
Unidades do offset: GRAUS [roll, pitch, yaw] (medido: +0,1 -> +0,1 deg no rpy).
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from tools.mark_calibration_segments import (MarkerSession, effective_markers, utc_iso)

# ------------------------------------------------------------- config service
CONFIG_SERVICE = "config"                 # rt/api/config/{request,response}
API_SET, API_GET = 1001, 1002             # 1001 confirmado (captura); 1002 [INF]
CONFIG_NAMES = {"imu": "imu_offset_json", "secondaryimu": "secondaryimu_offset_json"}
NAME_TO_KEY = {v: k for k, v in CONFIG_NAMES.items()}
KEY_LABEL = {"imu": "IMU pelve (imu_offset_json)", "secondaryimu": "IMU torso (secondaryimu_offset_json)"}
AXES = ("roll", "pitch", "yaw")
RPC_CODES = {0: "ok", 3001: "erro desconhecido", 3102: "falha ao enviar", 3103: "api nao registrada",
             3104: "timeout", 3105: "api_id da resposta nao confere", 3203: "api nao implementada no servidor",
             3204: "lease invalido", -1: "simulado/sem backend"}
CODE_TIMEOUT = 3104


def _clean(v):
    v = round(float(v), 3)
    return 0.0 if v == 0 else v          # -0.0 -> 0.0


def offset_json(key, vals):
    """content string, e.g. '{"imu":[-0.1,2.5,0.0]}' (compact, floats)."""
    if key not in CONFIG_NAMES:
        raise ValueError(f"chave desconhecida {key!r}")
    vals = [_clean(v) for v in vals]
    if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
        raise ValueError("offset precisa de 3 valores finitos")
    return json.dumps({key: vals}, separators=(",", ":"))


def build_set_parameter(key, vals):
    """Request_.parameter do SET (api 1001) exatamente como o app envia:
    {"name":"imu_offset_json","content":"{\\"imu\\":[-0.1,2.5,0.0]}"}"""
    return json.dumps({"name": CONFIG_NAMES[key], "content": offset_json(key, vals)},
                      separators=(",", ":"))


def build_get_parameter(key):
    return json.dumps({"name": CONFIG_NAMES[key]}, separators=(",", ":"))


def _loads(v):
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "replace")
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return None
    return v


def _vec3(v):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return None
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(x) for x in out) else None


def parse_offset_content(key, content):
    """content (str JSON ou dict) -> [r,p,y] ou None.  Aceita {"imu":[..]},
    lista pura, ou {"content": ...} aninhado."""
    val = _loads(content)
    for _ in range(3):
        if isinstance(val, dict):
            if key in val:
                val = _loads(val[key])
            elif "content" in val:
                val = _loads(val["content"])
            elif CONFIG_NAMES.get(key) in val:
                val = _loads(val[CONFIG_NAMES[key]])
            else:
                return None
        else:
            break
    return _vec3(val)


def parse_get_response(key, data):
    """Response_.data do GET.  Formato esperado (config_api.hpp go2):
    {"content":"{\\"imu\\":[...]}"}; tolera content direto."""
    return parse_offset_content(key, data)


def parse_set_request(parameter):
    """Request_.parameter (SET) -> (key, vals) ou None (led_json etc. ignorados)."""
    p = _loads(parameter)
    if not isinstance(p, dict):
        return None
    key = NAME_TO_KEY.get(str(p.get("name") or ""))
    if key is None:
        return None
    vals = parse_offset_content(key, p.get("content"))
    return None if vals is None else (key, vals)


def parse_config_change(name, content):
    """rt/config_change_status {name, content} -> (key, vals) ou None."""
    key = NAME_TO_KEY.get(str(name or ""))
    if key is None:
        return None
    vals = parse_offset_content(key, content)
    return None if vals is None else (key, vals)


# ------------------------------------------------------------------- policy
class OffsetPolicy:
    def __init__(self, window_deg=3.0, hard_abs_deg=10.0, max_step_deg=0.5, min_interval_s=1.0,
                 allow_yaw=False, loco_eps=0.05, speed_eps=0.05, yaw_rate_eps=0.1, max_age_s=1.0):
        self.window_deg = float(window_deg)
        self.hard_abs_deg = min(float(hard_abs_deg), 10.0)   # teto rigido
        self.max_step_deg = min(float(max_step_deg), 0.5)
        self.min_interval_s = max(float(min_interval_s), 1.0)
        self.allow_yaw = bool(allow_yaw)
        self.loco_eps, self.speed_eps = float(loco_eps), float(speed_eps)
        self.yaw_rate_eps, self.max_age_s = float(yaw_rate_eps), float(max_age_s)

    def as_dict(self):
        return {k: getattr(self, k) for k in ("window_deg", "hard_abs_deg", "max_step_deg",
                                              "min_interval_s", "allow_yaw", "loco_eps", "speed_eps")}

    def validate(self, *, key, current, target, base, uncertain=False, last_write_mono=None,
                 now=None, in_progress=False):
        """-> lista de erros (vazia = permitido).  Nada aqui faz I/O."""
        errs = []
        if key not in CONFIG_NAMES:
            return [f"chave desconhecida {key!r}"]
        tgt = None
        try:
            tgt = [float(v) for v in target]
        except (TypeError, ValueError):
            errs.append("alvo invalido")
        if tgt is not None and (len(tgt) != 3 or not all(math.isfinite(v) for v in tgt)):
            errs.append("alvo precisa de 3 numeros finitos (NaN/inf recusado)")
            tgt = None
        if base is None:
            errs.append("base desconhecida: leia do robo (GET) ou confirme um valor base manual")
        if current is None:
            errs.append("offset atual desconhecido")
        if uncertain:
            errs.append("offset atual incerto (escrita anterior sem confirmacao): releia (GET) ou confirme manualmente")
        if in_progress:
            errs.append("outra escrita em andamento")
        if last_write_mono is not None and now is not None and now - last_write_mono < self.min_interval_s:
            errs.append(f"intervalo minimo entre escritas {self.min_interval_s:.1f}s "
                        f"(faltam {self.min_interval_s - (now - last_write_mono):.1f}s)")
        if tgt is None or current is None or base is None:
            return errs
        for i, ax in enumerate(AXES):
            if abs(tgt[i]) > self.hard_abs_deg + 1e-9:
                errs.append(f"{ax}: |{tgt[i]:+.2f}| > limite rigido {self.hard_abs_deg:.1f} deg")
            if abs(tgt[i] - base[i]) > self.window_deg + 1e-9:
                errs.append(f"{ax}: {tgt[i]:+.2f} fora da janela base {base[i]:+.2f} +/- {self.window_deg:.1f} deg")
            if abs(tgt[i] - current[i]) > self.max_step_deg + 1e-9:
                errs.append(f"{ax}: passo {tgt[i] - current[i]:+.2f} > maximo {self.max_step_deg:.2f} deg por comando")
        if not self.allow_yaw and abs(tgt[2] - current[2]) > 1e-9:
            errs.append("yaw e somente leitura (use --allow-yaw para liberar)")
        if all(abs(a - b) < 1e-9 for a, b in zip(tgt, current)):
            errs.append("alvo igual ao valor atual")
        return errs

    def motion_warnings(self, live):
        """Avisos (exigem 2a confirmacao, nao bloqueiam)."""
        w = []
        wc = live.get("wireless") or {}
        if wc.get("age") is None or wc["age"] > self.max_age_s:
            w.append("comando de locomocao (rt/wirelesscontroller) sem dado recente: estado desconhecido")
        elif not wc.get("zero"):
            w.append(f"joystick de locomocao NAO esta ~0 (lx={wc.get('lx'):+.2f} ly={wc.get('ly'):+.2f} "
                     f"rx={wc.get('rx'):+.2f})")
        od = live.get("odom") or {}
        if od.get("age") is None or od["age"] > self.max_age_s:
            w.append("velocidade do corpo (rt/odommodestate) sem dado recente: estado desconhecido")
        else:
            if od.get("speed") is not None and od["speed"] > self.speed_eps:
                w.append(f"robo em movimento: velocidade {od['speed']:.3f} m/s > {self.speed_eps:.2f}")
            if od.get("yaw_rate") is not None and abs(od["yaw_rate"]) > self.yaw_rate_eps:
                w.append(f"robo girando: yaw rate {od['yaw_rate']:+.2f} rad/s")
        return w


def step_toward(current, goal, max_step):
    return [_clean(c + max(-max_step, min(max_step, g - c))) for c, g in zip(current, goal)]


def verify_by_imu(history, t_set, before, after, *, pre_s=0.6, post_from=0.4, post_to=1.0):
    """Compara media do rpy (rad) antes/depois do SET com o delta esperado (deg).
    history: [(t_mono, [r,p,y] rad)].  So roll/pitch.  -> dict."""
    exp = [after[i] - before[i] for i in range(2)]
    pre = [r for t, r in history if t_set - pre_s <= t <= t_set and r]
    post = [r for t, r in history if t_set + post_from <= t <= t_set + post_to and r]
    if len(pre) < 2 or len(post) < 2:
        return {"verified": None, "reason": "sem amostras de IMU suficientes",
                "expected_deg": exp, "measured_deg": None}
    meas = [math.degrees(sum(r[i] for r in post) / len(post) - sum(r[i] for r in pre) / len(pre))
            for i in range(2)]
    ok = all(abs(m - e) <= max(0.04, 0.4 * abs(e)) for m, e in zip(meas, exp))
    return {"verified": bool(ok), "reason": "imu_rpy", "expected_deg": exp,
            "measured_deg": [round(m, 4) for m in meas], "n_pre": len(pre), "n_post": len(post)}


# ------------------------------------------------------------ jsonl writer
class JsonlAppender:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, rec):
        with self._lock:
            if self._fh is None:
                return
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass

    def close(self):
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


# -------------------------------------------------------------- fake backend
class FakeBackend:
    """Backend sem DDS: testes, --sim e --dry-run (SET simulado)."""

    def __init__(self, *, can_write=True, can_get=True, values=None, get_code=0, set_code=0,
                 live=None, imu_follow=True, clock=time.monotonic):
        self.can_write, self.can_get = can_write, can_get
        self.values = {k: list(v) for k, v in (values or {}).items()}
        self.get_code, self.set_code = get_code, set_code
        self._live = live
        self.imu_follow = imu_follow
        self._clock = clock
        self.calls = []                 # (api_id, parameter)
        self.passive = deque()          # (t, key, vals, source)
        self.base_rpy = {"imu": [0.0, math.radians(2.5), 0.0], "secondaryimu": [0.0, 0.0, 0.0]}
        self.history = {"imu": deque(maxlen=400), "secondaryimu": deque(maxlen=400)}

    def start(self):
        return self

    def close(self):
        pass

    def live(self):
        if self._live is not None:
            return self._live() if callable(self._live) else self._live
        out = {"sources": {}, "wireless": {"age": 0.0, "lx": 0.0, "ly": 0.0, "rx": 0.0, "zero": True},
               "odom": {"age": 0.0, "speed": 0.0, "yaw_rate": 0.0}, "mode": "fake"}
        now = self._clock()
        for k, name in (("imu", "pelvis"), ("secondaryimu", "torso")):
            h = self.history[k]
            out[f"{name}_rpy_deg"] = [math.degrees(x) for x in h[-1][1]] if h else None
            out[f"{name}_age"] = (now - h[-1][0]) if h else None
        return out

    def tick_imu(self, t=None):
        t = self._clock() if t is None else t
        for k, h in self.history.items():
            off = self.values.get(k) or [0.0, 0.0, 0.0]
            h.append((t, [self.base_rpy[k][i] + (math.radians(off[i]) if self.imu_follow else 0.0)
                          for i in range(3)]))

    def imu_history(self, key):
        return list(self.history.get(key, ()))

    def drain_passive(self):
        out = list(self.passive)
        self.passive.clear()
        return out

    def config_get(self, key, timeout=1.0):
        param = build_get_parameter(key)
        self.calls.append((API_GET, param))
        if self.get_code != 0:
            return self.get_code, None
        if key not in self.values:
            return 3001, None
        return 0, json.dumps({"content": offset_json(key, self.values[key])}, separators=(",", ":"))

    def config_set(self, key, vals, timeout=2.0):
        param = build_set_parameter(key, vals)
        self.calls.append((API_SET, param))
        if self.set_code == 0:
            self.values[key] = [_clean(v) for v in vals]
        return self.set_code, ""


# --------------------------------------------------------------- controller
CONDITIONS = ("sem_caixa", "com_caixa")
KINDS = ("reta", "giro_esquerda", "giro_direita")
KIND_LABEL = {"reta": "Reta", "giro_esquerda": "Giro Esq.", "giro_direita": "Giro Dir."}


def default_stamp(now=None):
    return datetime.fromtimestamp(time.time() if now is None else now, timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class CalibController:
    def __init__(self, backend, out_dir, *, policy=None, mode="normal", wall_clock=time.time,
                 monotonic_clock=time.monotonic, sleep=time.sleep, stamp=None, markers_path=None,
                 log=print):
        if mode not in ("normal", "dry-run", "read-only"):
            raise ValueError(mode)
        self.backend, self.mode, self.policy = backend, mode, policy or OffsetPolicy()
        self._wall, self._mono, self._sleep, self._log = wall_clock, monotonic_clock, sleep, log
        stamp = stamp or default_stamp(wall_clock())
        out_dir = Path(out_dir)
        self.markers_path = Path(markers_path) if markers_path else out_dir / f"calib-markers-{stamp}.jsonl"
        self.offset_log_path = out_dir / f"offset-changes-{stamp}.jsonl"
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        # per key: value/source/t/uncertain + base
        self.offsets = {k: {"value": None, "source": None, "t": None, "t_wall": None, "uncertain": False,
                            "base": None, "base_source": None} for k in CONFIG_NAMES}
        self.get_available = None       # None = nao tentado
        self.last_write_mono = None
        self.history = []               # offset change records
        self.session = MarkerSession(self.markers_path, wall_clock=wall_clock,
                                     monotonic_clock=monotonic_clock, offset_provider=self._pelvis_offset)
        self._offset_log = None

    # ---------------------------------------------------------- helpers
    @property
    def writes_enabled(self):
        return self.mode != "read-only" and getattr(self.backend, "can_write", False)

    @property
    def get_enabled(self):
        return self.mode != "read-only" and getattr(self.backend, "can_get", False)

    def _pelvis_offset(self):
        o = self.offsets["imu"]
        if o["value"] is None:
            return None, "desconhecido"
        return list(o["value"]), (o["source"] + ("_incerto" if o["uncertain"] else ""))

    def _set_value(self, key, vals, source, *, uncertain=False):
        o = self.offsets[key]
        o.update(value=[_clean(v) for v in vals], source=source, t=self._mono(), t_wall=self._wall(),
                 uncertain=uncertain)
        if o["base"] is None and not uncertain:
            o["base"], o["base_source"] = list(o["value"]), source

    def _offset_logger(self):
        if self._offset_log is None:
            self._offset_log = JsonlAppender(self.offset_log_path)
        return self._offset_log

    # ----------------------------------------------------------- reading
    def initial_read(self, timeout=1.0):
        """Unica acao automatica permitida: GET de leitura (nunca em read-only)."""
        if not self.get_enabled:
            self._log("[calib_web] GET inicial pulado (read-only ou backend sem GET)")
            return
        for key in CONFIG_NAMES:
            self.read_get(key, timeout=timeout)

    def read_get(self, key, timeout=1.0):
        if not self.get_enabled:
            return {"ok": False, "error": "GET desabilitado (read-only)"}
        try:
            code, data = self.backend.config_get(key, timeout=timeout)
        except Exception as e:  # backend failure is never fatal
            code, data = 3001, None
            self._log(f"[calib_web] GET {key} falhou: {e!r}")
        vals = parse_get_response(key, data) if code == 0 else None
        with self._lock:
            if vals is not None:
                self.get_available = True
                self._set_value(key, vals, "dds_get")
            elif self.get_available is None:
                self.get_available = False
        self._log(f"[calib_web] GET {CONFIG_NAMES[key]} code={code} ({RPC_CODES.get(code, '?')}) -> {vals}")
        return {"ok": vals is not None, "code": code, "code_text": RPC_CODES.get(code, "?"), "value": vals}

    def poll_passive(self):
        try:
            items = self.backend.drain_passive()
        except Exception:
            items = []
        with self._lock:
            for t, key, vals, src in items:
                o = self.offsets.get(key)
                if o is None or vals is None:
                    continue
                changed = o["value"] is None or any(abs(a - b) > 1e-6 for a, b in zip(o["value"], vals))
                self._set_value(key, vals, src)
                if changed:
                    self._log(f"[calib_web] offset {key} visto passivamente ({src}): {vals}")
                    if key == "imu" and not self.session.closed:
                        self.session.record("note", note=f"offset {key} observado ({src}): {vals}")

    def set_manual(self, key, vals, set_base=False):
        v = _vec3(vals)
        if key not in CONFIG_NAMES or v is None:
            return {"ok": False, "error": "valor manual invalido (3 numeros finitos)"}
        if any(abs(x) > self.policy.hard_abs_deg for x in v):
            return {"ok": False, "error": f"|valor| > {self.policy.hard_abs_deg} deg"}
        with self._lock:
            self._set_value(key, v, "manual")
            if set_base or self.offsets[key]["base"] is None:
                self.offsets[key]["base"], self.offsets[key]["base_source"] = list(v), "manual"
        if not self.session.closed:
            self.session.record("note", note=f"offset {key} confirmado manualmente: {v}"
                                + (" (base)" if set_base else ""))
        return {"ok": True, "value": v}

    # ---------------------------------------------------------- writing
    def preview(self, key, target, *, advanced=False, _inside_apply=False):
        with self._lock:
            o = self.offsets.get(key)
            if o is None:
                return {"ok": False, "errors": [f"chave desconhecida {key!r}"], "warnings": []}
            errs = []
            if not self.writes_enabled:
                errs.append("escrita desabilitada (modo read-only)")
            if key == "secondaryimu" and not advanced:
                errs.append("IMU do torso so no modo avancado")
            errs += self.policy.validate(key=key, current=o["value"], target=target, base=o["base"],
                                         uncertain=o["uncertain"], last_write_mono=self.last_write_mono,
                                         now=self._mono(),
                                         in_progress=(not _inside_apply) and self._write_lock.locked())
            warns = self.policy.motion_warnings(self._safe_live())
            tgt = _vec3(target)
            return {"ok": not errs, "errors": errs, "warnings": warns, "key": key,
                    "before": o["value"], "after": None if tgt is None else [_clean(v) for v in tgt],
                    "base": o["base"],
                    "parameter": build_set_parameter(key, tgt) if (not errs and tgt) else None}

    def apply(self, key, target, *, expect_current=None, confirm=False, ack_motion=False, advanced=False):
        if not confirm:
            return {"ok": False, "errors": ["confirmacao ausente"]}
        if not self._write_lock.acquire(blocking=False):
            return {"ok": False, "errors": ["outra escrita em andamento"]}
        try:
            pv = self.preview(key, target, advanced=advanced, _inside_apply=True)
            if not pv["ok"]:
                return pv
            before = list(pv["before"])
            if expect_current is not None and (_vec3(expect_current) is None or any(
                    abs(a - b) > 1e-6 for a, b in zip(_vec3(expect_current), before))):
                return dict(pv, ok=False, errors=["valor atual mudou desde a confirmacao; revise"])
            if pv["warnings"] and not ack_motion:
                return dict(pv, ok=False, needs_motion_ack=True,
                            errors=["robo pode estar em movimento: confirme novamente"])
            after = pv["after"]
            t_mono, t_wall = self._mono(), self._wall()
            self.last_write_mono = t_mono
            dry = self.mode == "dry-run"
            try:
                code, _data = self.backend.config_set(key, after, timeout=2.0)
            except Exception as e:
                code = 3001
                self._log(f"[calib_web] SET falhou: {e!r}")
            ok = code == 0
            with self._lock:
                if ok:
                    self._set_value(key, after, "ui_set_simulado" if dry else "ui_set")
                else:
                    # resultado desconhecido: o robo pode ter aplicado. Bloqueia ate reconfirmar.
                    self.offsets[key]["uncertain"] = True
            verification = {"imu": None, "get": None}
            if ok:
                self._sleep(1.05)
                verification["imu"] = verify_by_imu(self.backend.imu_history(key), t_mono, before, after)
                if self.get_available and self.get_enabled:
                    g = self.read_get(key, timeout=1.0)
                    match = g["value"] is not None and all(abs(a - b) < 1e-3 for a, b in zip(g["value"], after))
                    verification["get"] = {"verified": match, "value": g["value"], "code": g.get("code")}
            verified = ok and (bool(verification["get"] and verification["get"]["verified"])
                               or bool(verification["imu"] and verification["imu"]["verified"]))
            rec = {"event": "imu_offset_change", "schema_version": 1, "key": key,
                   "name": CONFIG_NAMES[key], "before": before, "after": after, "base": pv["base"],
                   "code": code, "code_text": RPC_CODES.get(code, "?"), "ok": ok, "dry_run": dry,
                   "verified": verified, "verification": verification,
                   "parameter": build_set_parameter(key, after), "warnings_acked": pv["warnings"],
                   "timestamp": t_wall, "timestamp_utc": utc_iso(t_wall), "timestamp_monotonic": t_mono,
                   "clock_domain": {"timestamp": "wall_clock_utc", "timestamp_monotonic": "monotonic"}}
            self.history.append(rec)
            self._offset_logger().write(rec)
            if not self.session.closed:
                fields = {"offset_key": key, "offset_before": before, "offset_after": after, "ok": ok,
                          "code": code, "verified": verified, "dry_run": dry}
                if key == "imu":
                    fields["imu_offset"] = after if ok else before
                    fields["imu_offset_source"] = self.offsets["imu"]["source"]
                self.session.record("offset_set", **fields)
            self._log(f"[calib_web] SET {CONFIG_NAMES[key]} {before} -> {after} code={code} "
                      f"verificado={verified}{' (DRY-RUN)' if dry else ''}")
            return {"ok": ok, "code": code, "code_text": RPC_CODES.get(code, "?"), "verified": verified,
                    "verification": verification, "before": before, "after": after, "errors": [] if ok else
                    [f"SET retornou code={code} ({RPC_CODES.get(code, '?')}); offset marcado como incerto"]}
        finally:
            self._write_lock.release()

    def restore_step(self, key):
        """Proximo passo (<= max_step) rumo a base; o cliente ainda confirma."""
        with self._lock:
            o = self.offsets[key]
            if o["value"] is None or o["base"] is None:
                return {"ok": False, "errors": ["base ou valor atual desconhecido"]}
            tgt = step_toward(o["value"], o["base"], self.policy.max_step_deg)
            remaining = max(abs(b - c) for b, c in zip(o["base"], o["value"]))
            steps = math.ceil(remaining / self.policy.max_step_deg - 1e-9) if remaining > 1e-9 else 0
        return {"ok": steps > 0, "target": tgt, "steps_remaining": steps,
                "errors": [] if steps else ["ja esta na base"]}

    # ---------------------------------------------------------- markers
    def marker(self, action, **kw):
        with self._lock:
            s = self.session
            if s.closed:
                return {"ok": False, "error": "sessao encerrada"}
            st = s.state()
            if action == "condition":
                cond = kw.get("condition")
                if cond not in CONDITIONS:
                    return {"ok": False, "error": "condicao invalida"}
                s.record("condition", condition=cond)
                s.record("attempt", attempt=self.next_attempt(cond))
            elif action == "attempt":
                try:
                    n = int(kw.get("attempt"))
                except (TypeError, ValueError):
                    return {"ok": False, "error": "tentativa invalida"}
                if n < 1 or n > 99:
                    return {"ok": False, "error": "tentativa fora de 1..99"}
                s.record("attempt", attempt=n)
            elif action == "segment":
                kind = kw.get("kind")
                if kind not in KINDS:
                    return {"ok": False, "error": "trecho invalido"}
                if st["condition"] is None:
                    return {"ok": False, "error": "escolha Sem caixa / Com caixa primeiro"}
                done = self.done_kinds(st["condition"], st["attempt"])
                if st["attempt"] is None or (kind == "reta" and "reta" in done):
                    s.record("attempt", attempt=self.next_attempt(st["condition"], after=st["attempt"]))
                s.record("segment_start", kind=kind)
            elif action == "end":
                if st["open_segment"] is None:
                    return {"ok": False, "error": "nenhum trecho aberto"}
                s.record("segment_end", kind=st["open_segment"]["kind"])
            elif action == "undo":
                alive = [e for e in effective_markers(s.events())
                         if e["type"] not in ("session_start", "offset_set")]
                if not alive:
                    return {"ok": False, "error": "nada para desfazer"}
                s.record("undo", undo_of=alive[-1]["seq"])
            elif action == "note":
                note = str(kw.get("note") or "").strip()[:500]
                if not note:
                    return {"ok": False, "error": "nota vazia"}
                s.record("note", note=note)
            else:
                return {"ok": False, "error": f"acao desconhecida {action!r}"}
            return {"ok": True}

    def done_kinds(self, cond, att):
        return {e["kind"] for e in effective_markers(self.session.events())
                if e["type"] == "segment_start" and e.get("condition") == cond and e.get("attempt") == att}

    def next_attempt(self, cond, after=None):
        n = 1 if after is None else after + 1
        if after is None:
            while self.done_kinds(cond, n) >= set(KINDS):
                n += 1
        return n

    def checklist(self):
        out = {}
        evs = effective_markers(self.session.events())
        for cond in CONDITIONS:
            atts = {}
            for e in evs:
                if e["type"] == "segment_start" and e.get("condition") == cond and e.get("attempt"):
                    atts.setdefault(e["attempt"], set()).add(e["kind"])
            n = max([3] + list(atts))
            out[cond] = [{"attempt": i, **{k: k in atts.get(i, ()) for k in KINDS}} for i in range(1, n + 1)]
        return out

    # ------------------------------------------------------------ state
    def _safe_live(self):
        try:
            return self.backend.live() or {}
        except Exception:
            return {}

    def state(self):
        self.poll_passive()
        now_m = self._mono()
        with self._lock:
            st = self.session.state()
            seg = st["open_segment"]
            if seg:
                ev = next((e for e in self.session.events() if e["seq"] == seg["seq"]), None)
                seg = dict(seg, elapsed_s=None if ev is None else now_m - ev["timestamp_monotonic"])
            recent = []
            for e in effective_markers(self.session.events())[-12:][::-1]:
                recent.append({k: e.get(k) for k in ("seq", "type", "kind", "condition", "attempt", "note",
                                                     "timestamp_utc", "imu_offset", "imu_offset_source",
                                                     "offset_after", "ok")})
            offs = {}
            for k, o in self.offsets.items():
                offs[k] = dict(o, age_s=None if o["t"] is None else now_m - o["t"], label=KEY_LABEL[k],
                               name=CONFIG_NAMES[k],
                               differs_from_base=(o["value"] is not None and o["base"] is not None and any(
                                   abs(a - b) > 1e-6 for a, b in zip(o["value"], o["base"]))))
            return {"mode": self.mode, "writes_enabled": self.writes_enabled, "get_enabled": self.get_enabled,
                    "get_available": self.get_available, "policy": self.policy.as_dict(),
                    "markers_file": str(self.markers_path), "offset_log_file": str(self.offset_log_path),
                    "session": {"condition": st["condition"], "attempt": st["attempt"], "open_segment": seg,
                                "closed": self.session.closed},
                    "recent": recent, "checklist": self.checklist(), "offsets": offs,
                    "history": self.history[-20:][::-1],
                    "write_interval_left_s": None if self.last_write_mono is None else max(
                        0.0, self.policy.min_interval_s - (now_m - self.last_write_mono)),
                    "live": self._safe_live(),
                    "motion_warnings": self.policy.motion_warnings(self._safe_live())}

    def close(self):
        lines = []
        with self._lock:
            for k, o in self.offsets.items():
                if o["value"] is not None and o["base"] is not None and any(
                        abs(a - b) > 1e-6 for a, b in zip(o["value"], o["base"])):
                    lines.append(f"ATENCAO: {CONFIG_NAMES[k]} atual {o['value']} difere da base {o['base']} "
                                 f"(nada foi restaurado automaticamente)")
            if not self.session.closed:
                self.session.close()
            if self._offset_log is not None:
                self._offset_log.close()
        for line in lines:
            self._log(f"[calib_web] {line}")
        return lines
