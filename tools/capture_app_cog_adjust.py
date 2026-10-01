#!/usr/bin/env python3
"""Captura 100% PASSIVA do que o app Unitree Explorer faz ao ajustar o centro de gravidade.

Roda NO ROBO. Somente leitura: subscribers DDS (best-effort), leitura de arquivos,
`ss` e (opcional) tcpdump com sudo sem senha. NUNCA cria escritores DDS, nunca envia
mensagens, nunca chama RPC/Set*. Falhas viram contadores; nada bloqueia.

Uso:
  python tools/capture_app_cog_adjust.py [--iface enP8p1s0] [--duration 0]
Enter marca as fases: 1o=ANTES, 2o=MEXENDO, 3o=DEPOIS. Ctrl+C encerra e imprime o resumo.
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

PHASES = ("inicio", "antes", "mexendo", "depois")
MARKER_NAMES = ("antes", "mexendo", "depois")

# ---- Conhecimento estatico -------------------------------------------------
KNOWN_API_TOPICS = (
    "rt/api/sport/request", "rt/api/sport/response",
    "rt/api/arm/request", "rt/api/arm/response",
    "rt/api/motion_switcher/request", "rt/api/motion_switcher/response",
    "rt/api/robot_state/request", "rt/api/robot_state/response",
    "rt/api/config/request", "rt/api/config/response",
    "rt/api/voice/request", "rt/api/voice/response",
)
# (topic, type) de estado conhecidos -> projecao de campos (None = completo)
STATE_TOPICS = {
    "rt/sportmodestate": ("unitree_go.msg.dds_.SportModeState_", None, 0.0),
    "rt/odommodestate": ("unitree_go.msg.dds_.SportModeState_", None, 0.0),
    "rt/lf/sportmodestate": ("unitree_go.msg.dds_.SportModeState_", None, 0.0),
    "rt/lf/odommodestate": ("unitree_go.msg.dds_.SportModeState_", None, 0.0),
    "rt/lowstate": ("unitree_hg.msg.dds_.LowState_", ("imu_state", "wireless_remote"), 0.5),
    "rt/lf/lowstate": ("unitree_hg.msg.dds_.LowState_", ("imu_state", "wireless_remote"), 0.5),
    "rt/wirelesscontroller": ("unitree_go.msg.dds_.WirelessController_", None, 0.2),
    "rt/dex3/left/state": ("unitree_hg.msg.dds_.HandState_", ("imu_state", "motor_state.q"), 0.5),
    "rt/dex3/right/state": ("unitree_hg.msg.dds_.HandState_", ("imu_state", "motor_state.q"), 0.5),
}
STATE_PATTERNS = (re.compile(r"^rt/lf/.+"), re.compile(r"^rt/dex3/.+/state$"))

KNOWN_API_IDS = {
    "rt/api/sport": {
        7001: "GET_FSM_ID", 7002: "GET_FSM_MODE", 7003: "GET_BALANCE_MODE(7301 no G1)",
        7004: "GET_SWING_HEIGHT", 7005: "GET_STAND_HEIGHT", 7006: "GET_PHASE",
        7007: "GET_?", 7101: "SET_FSM_ID", 7102: "SET_BALANCE_MODE", 7103: "SET_SWING_HEIGHT",
        7104: "SET_STAND_HEIGHT", 7105: "SET_VELOCITY", 7106: "SET_ARM_TASK",
        7107: "SET_SPEED_MODE", 7108: "SET_MOTION(param ARRAY desconhecido)", 7109: "ARM_SDK_STATUS",
    },
    "rt/api/arm": {7106: "EXECUTE_ACTION", 7107: "GET_ACTION_LIST"},
    "rt/api/motion_switcher": {1001: "CHECK_MODE", 1002: "SELECT_MODE", 1003: "RELEASE_MODE",
                               1004: "SET_SILENT", 1005: "GET_SILENT"},
    "rt/api/robot_state": {1001: "SERVICE_SWITCH", 1002: "REPORT_FREQ", 1003: "SERVICE_LIST"},
    "rt/api/voice": {1001: "TTS", 1002: "ASR", 1003: "START_PLAY", 1004: "STOP_PLAY",
                     1005: "GET_VOLUME", 1006: "SET_VOLUME", 1010: "SET_RGB_LED"},
}
LEASE_IDS = {101: "LEASE_APPLY", 102: "LEASE_RENEWAL", 1: "API_VERSION"}


def api_family(topic):
    m = re.match(r"^(rt/api/[^/]+)/(request|response)$", topic or "")
    return m.group(1) if m else None


def api_name(topic, api_id):
    """Nome conhecido ou None (None => desconhecido)."""
    if api_id in LEASE_IDS:
        return LEASE_IDS[api_id]
    return KNOWN_API_IDS.get(api_family(topic) or "", {}).get(api_id)


# ---- Utilidades -------------------------------------------------------------
_SECRET = re.compile(r"(password|passwd|token|secret|credential|cookie|auth|private)", re.I)


def redact_text(s):
    """Mascara valores de chaves sensiveis em texto JSON/chave=valor."""
    if not isinstance(s, str):
        return s
    s = re.sub(r'("[^"]*(?:password|passwd|token|secret|credential|cookie|auth|private)[^"]*"\s*:\s*)"[^"]*"',
               r'\1"<redacted>"', s, flags=re.I)
    s = re.sub(r"((?:password|passwd|token|secret|credential)\w*\s*[=:]\s*)\S+", r"\1<redacted>", s, flags=re.I)
    return s


def to_jsonable(obj, depth=6, max_seq=64, max_str=2048):
    """Converte dataclass IDL/np/bytes em JSON seguro e limitado."""
    if depth < 0:
        return "<depth>"
    if obj is None or isinstance(obj, (bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and abs(obj) != float("inf") else str(obj)
    if isinstance(obj, str):
        return obj[:max_str]
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj[:64]).hex()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, depth - 1, max_seq, max_str) for k, v in list(obj.items())[:max_seq]}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(getattr(obj, k, None), depth - 1, max_seq, max_str)
                for k in obj.__dataclass_fields__}
    if hasattr(obj, "tolist"):
        try:
            return to_jsonable(obj.tolist(), depth - 1, max_seq, max_str)
        except Exception:
            pass
    if isinstance(obj, (list, tuple)) or hasattr(obj, "__iter__"):
        try:
            items = list(obj)
        except Exception:
            return str(obj)[:max_str]
        out = [to_jsonable(v, depth - 1, max_seq, max_str) for v in items[:max_seq]]
        if len(items) > max_seq:
            out.append("<+%d>" % (len(items) - max_seq))
        return out
    return str(obj)[:max_str]


def project(d, fields):
    """Projeta dict aninhado por caminhos 'a.b'. Em listas, aplica o resto do caminho a cada item."""
    if not fields:
        return d

    def pick(node, parts):
        if not parts:
            return node
        if isinstance(node, list):
            return [pick(x, parts) for x in node]
        if isinstance(node, dict) and parts[0] in node:
            return pick(node[parts[0]], parts[1:])
        return None

    return {f: pick(d, f.split(".")) for f in fields}


class SizedJsonl:
    """Arquivo JSONL com teto de bytes; excedente vira contador `dropped`."""

    def __init__(self, path, max_bytes):
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self.size = 0
        self.dropped = 0
        self.errors = 0
        self._lock = threading.Lock()
        self._fh = None

    def append(self, rec):
        try:
            line = json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
            with self._lock:
                if self.size + len(line) > self.max_bytes:
                    self.dropped += 1
                    return False
                if self._fh is None:
                    self._fh = open(self.path, "a", encoding="utf-8")
                self._fh.write(line)
                self._fh.flush()
                self.size += len(line)
            return True
        except Exception:
            self.errors += 1
            return False

    def close(self):
        with self._lock:
            if self._fh:
                try:
                    self._fh.close()
                finally:
                    self._fh = None


# ---- Nucleo (sem DDS) -------------------------------------------------------
class Capture:
    """Registra eventos, contadores por fase, snapshots e marcadores. Thread-safe e nao-bloqueante."""

    def __init__(self, outdir, clock=time.time, api_max_bytes=20_000_000, state_max_bytes=20_000_000,
                 snap_max_bytes=20_000_000, param_max=4096, binary_max=512):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.t0 = clock()
        self.phase = "inicio"
        self.lock = threading.Lock()
        self.api = SizedJsonl(self.dir / "api_events.jsonl", api_max_bytes)
        self.state = SizedJsonl(self.dir / "state_decimated.jsonl", state_max_bytes)
        self.snaps = SizedJsonl(self.dir / "snapshots.jsonl", snap_max_bytes)
        self.markers_log = SizedJsonl(self.dir / "markers.jsonl", 1_000_000)
        self.param_max = param_max
        self.binary_max = binary_max
        self.counts = {}   # topic -> {type,kind,total,phase:{},first_t,last_t}
        self.topics = {}   # topic -> {type, pubs, subs, first_seen_phase, first_seen_t}
        self.markers = []
        self.errors = {}
        self._last_state_write = {}
        self.latest = {}   # topic -> (t, jsonable-lazy obj)
        self.latest_raw = {}
        self.extra = {}

    # -- util
    def err(self, key):
        self.errors[key] = self.errors.get(key, 0) + 1

    def _count(self, topic, typ, kind):
        c = self.counts.setdefault(topic, {"type": typ, "kind": kind, "total": 0,
                                           "phase": {}, "first_t": None, "last_t": None})
        t = self.clock()
        c["total"] += 1
        c["phase"][self.phase] = c["phase"].get(self.phase, 0) + 1
        if c["first_t"] is None:
            c["first_t"] = t
        c["last_t"] = t
        return t

    # -- marcadores
    def mark(self, name=None):
        with self.lock:
            idx = sum(1 for m in self.markers)
            if name is None:
                name = MARKER_NAMES[idx] if idx < len(MARKER_NAMES) else "extra%d" % (idx - len(MARKER_NAMES) + 1)
            t = self.clock()
            self.markers.append({"name": name, "t": t, "rel": round(t - self.t0, 3)})
            if name in PHASES:
                self.phase = name
            self.markers_log.append({"t": t, "name": name, "phase_after": self.phase})
            return name

    # -- descoberta
    def note_topic(self, topic, type_name, kind, participant=None):
        with self.lock:
            e = self.topics.setdefault(topic, {"type": type_name, "pubs": 0, "subs": 0, "participants": [],
                                               "first_seen_phase": self.phase, "first_seen_t": self.clock()})
            if kind == "pub":
                e["pubs"] += 1
            elif kind == "sub":
                e["subs"] += 1
            if participant and participant not in e["participants"] and len(e["participants"]) < 8:
                e["participants"].append(participant)

    # -- eventos API
    def on_api(self, topic, kind, msg):
        """kind: 'request'|'response'. msg: Request_/Response_ (ou fake com mesmos atributos)."""
        try:
            h = getattr(msg, "header", None)
            ident = getattr(h, "identity", None)
            rec = {"t": self.clock(), "phase": self.phase, "topic": topic, "kind": kind,
                   "id": getattr(ident, "id", None), "api_id": getattr(ident, "api_id", None)}
            if kind == "request":
                lease = getattr(h, "lease", None)
                pol = getattr(h, "policy", None)
                p = getattr(msg, "parameter", "") or ""
                rec.update(lease_id=getattr(lease, "id", None),
                           priority=getattr(pol, "priority", None), noreply=getattr(pol, "noreply", None),
                           parameter=redact_text(p[: self.param_max]), parameter_len=len(p),
                           parameter_sha1=hashlib.sha1(p.encode("utf-8", "replace")).hexdigest()[:12])
            else:
                st = getattr(h, "status", None)
                d = getattr(msg, "data", "") or ""
                rec.update(code=getattr(st, "code", None), data=redact_text(d[: self.param_max]), data_len=len(d))
            b = getattr(msg, "binary", None) or []
            rec["binary_len"] = len(b)
            if len(b):
                rec["binary_hex"] = bytes(list(b)[: self.binary_max]).hex()
            fam = api_family(topic)
            rec["api_name"] = api_name(topic, rec["api_id"]) if kind == "request" or fam else None
            rec["known"] = rec["api_name"] is not None
            with self.lock:
                self._count(topic, "api." + kind, "api")
            self.api.append(rec)
        except Exception:
            self.err("on_api")

    # -- eventos de estado
    def on_state(self, topic, msg, type_name=None, fields=None, min_interval=0.5):
        try:
            with self.lock:
                t = self._count(topic, type_name, "state")
                self.latest_raw[topic] = (t, msg)
                last = self._last_state_write.get(topic, -1e9)
                if min_interval and t - last < min_interval:
                    return
                self._last_state_write[topic] = t
            d = project(to_jsonable(msg), fields)
            self.state.append({"t": t, "phase": self.phase, "topic": topic, "data": d})
        except Exception:
            self.err("on_state")

    # -- snapshots 1 Hz
    def snapshot(self):
        try:
            snap = {"t": self.clock(), "phase": self.phase}
            with self.lock:
                latest = dict(self.latest_raw)
            for topic, (t, msg) in latest.items():
                if topic in ("rt/sportmodestate", "rt/odommodestate", "rt/lf/sportmodestate"):
                    snap[topic] = to_jsonable(msg)
                elif topic in ("rt/lowstate", "rt/lf/lowstate"):
                    imu = to_jsonable(getattr(msg, "imu_state", None))
                    if isinstance(imu, dict):
                        snap["imu_state"] = {k: imu.get(k) for k in ("quaternion", "rpy", "gyroscope",
                                                                     "accelerometer") if k in imu}
                    snap["imu_state"] = snap.get("imu_state", imu)
                else:
                    continue
                snap.setdefault("_age", {})[topic] = round(snap["t"] - t, 3)
            self.snaps.append(snap)
        except Exception:
            self.err("snapshot")

    # -- persistencia
    def flush(self):
        def dump(name, obj):
            try:
                tmp = self.dir / (name + ".tmp")
                tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
                tmp.replace(self.dir / name)
            except Exception:
                self.err("flush_" + name)
        with self.lock:
            counts = json.loads(json.dumps(self.counts, default=str))
            topics = json.loads(json.dumps(self.topics, default=str))
            markers = list(self.markers)
        dump("counts.json", counts)
        dump("topics.json", topics)
        dump("markers.json", markers)
        meta = {"t0": self.t0, "t_end": self.clock(), "host": socket.gethostname(), "errors": self.errors,
                "dropped": {"api": self.api.dropped, "state": self.state.dropped, "snapshots": self.snaps.dropped},
                "write_errors": {"api": self.api.errors, "state": self.state.errors, "snapshots": self.snaps.errors},
                "passive": True}
        meta.update(self.extra)
        dump("meta.json", meta)

    def close(self):
        self.flush()
        for f in (self.api, self.state, self.snaps, self.markers_log):
            f.close()


# ---- Arquivos de configuracao (leitura) -------------------------------------
NAME_RE = re.compile(r"(?<![a-z])(balance|cog|com|center|centre|imu|calib\w*|offset|trim|param\w*|config\w*)(?![a-z])",
                     re.I)
SKIP_DIRS = {".git", ".cache", "miniconda3", "anaconda3", "node_modules", "site-packages", "__pycache__",
             "proc", "cyclonedds", "unitree_sdk2_python", "envs", "pkgs", "ssl", "ssh", "certs",
             "xr_teleoperate", "xr_teleoperate_slo", "docker", "snap", ".vscode-server", ".conda"}
SKIP_EXT = {".so", ".pyc", ".png", ".jpg", ".jpeg", ".a", ".o", ".h", ".hpp", ".pem", ".key", ".crt", ".bin",
            ".mp4", ".wav", ".mp3", ".gz", ".zip", ".log", ".whl", ".pt", ".onnx", ".engine"}
SECRET_NAME = re.compile(r"(passw|shadow|token|secret|credential|id_rsa|\.key$|\.pem$)", re.I)
TEXT_MAX = 64 * 1024
HASH_MAX = 4 * 1024 * 1024
DEFAULT_ROOTS = ("/unitree", "/home/unitree", "/etc", "/opt")


def find_candidates(roots=DEFAULT_ROOTS, budget_s=25.0, max_files=4000, now=time.time, walker=os.walk):
    """Lista arquivos com nome plausivel. Somente leitura, com orcamento de tempo."""
    out, deadline, truncated = [], now() + budget_s, False
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, dns, fns in walker(root, followlinks=False):
            if now() > deadline or len(out) >= max_files:
                truncated = True
                break
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".cache")]
            for fn in fns:
                if SECRET_NAME.search(fn) or os.path.splitext(fn)[1].lower() in SKIP_EXT:
                    continue
                if NAME_RE.search(fn) or NAME_RE.search(os.path.basename(dp)):
                    out.append(os.path.join(dp, fn))
                    if len(out) >= max_files:
                        break
    return out, truncated


def stat_file(path, keep_text=False):
    """{mtime,size,sha256?,text?}; erros viram {'error':...}. Somente leitura."""
    try:
        st = os.stat(path)
        rec = {"mtime": st.st_mtime, "size": st.st_size}
        if st.st_size <= HASH_MAX and os.path.isfile(path):
            with open(path, "rb") as fh:
                data = fh.read(HASH_MAX)
            rec["sha256"] = hashlib.sha256(data).hexdigest()
            if keep_text and st.st_size <= TEXT_MAX and b"\0" not in data:
                try:
                    rec["text"] = data.decode("utf-8")
                except UnicodeDecodeError:
                    pass
        return rec
    except Exception as e:
        return {"error": type(e).__name__}


def snapshot_files(paths, previous=None):
    """Re-stat; so le/hasheia de novo quando mtime/size mudou (previous ajuda a economizar)."""
    out = {}
    for p in paths:
        try:
            st = os.stat(p)
        except Exception as e:
            out[p] = {"error": type(e).__name__}
            continue
        prev = (previous or {}).get(p)
        if prev and prev.get("mtime") == st.st_mtime and prev.get("size") == st.st_size and "error" not in prev:
            out[p] = prev
        else:
            out[p] = stat_file(p, keep_text=True)
    return out


def diff_files(before, after):
    """Lista de mudancas entre dois snapshots + diffs unificados de textos pequenos."""
    changes, diffs = [], {}
    for p in sorted(set(before) | set(after)):
        b, a = before.get(p), after.get(p)
        if b is None:
            changes.append({"path": p, "change": "new"})
        elif a is None:
            changes.append({"path": p, "change": "removed"})
        elif b.get("sha256") != a.get("sha256") or b.get("mtime") != a.get("mtime") or b.get("size") != a.get("size"):
            if b.get("sha256") == a.get("sha256") and "sha256" in b:
                changes.append({"path": p, "change": "touched", "mtime_before": b.get("mtime"),
                                "mtime_after": a.get("mtime")})
                continue
            changes.append({"path": p, "change": "modified", "mtime_before": b.get("mtime"),
                            "mtime_after": a.get("mtime"), "size_before": b.get("size"),
                            "size_after": a.get("size")})
            if "text" in b and "text" in a:
                d = "".join(difflib.unified_diff(b["text"].splitlines(True), a["text"].splitlines(True),
                                                 "antes", "depois", n=2))
                diffs[p] = redact_text(d)[:20000]
    return changes, diffs


def strip_text(snap):
    return {p: {k: v for k, v in r.items() if k != "text"} for p, r in snap.items()}


def recent_files(roots, since, budget_s=15.0, cap=300, now=time.time):
    """Nomes (sem conteudo) modificados desde `since`, para pegar configs com nome inesperado."""
    out, deadline = [], now() + budget_s
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, dns, fns in os.walk(root, followlinks=False):
            if now() > deadline or len(out) >= cap:
                return out
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in fns:
                if SECRET_NAME.search(fn) or fn.endswith((".log", ".pyc", ".tmp")):
                    continue
                p = os.path.join(dp, fn)
                try:
                    if os.stat(p).st_mtime >= since:
                        out.append(p)
                except OSError:
                    pass
    return out


class FileWatcher:
    def __init__(self, cap, roots=DEFAULT_ROOTS, finder=find_candidates):
        self.cap, self.roots, self.finder = cap, roots, finder
        self.paths, self.base, self.truncated = [], {}, False

    def start(self):
        try:
            self.paths, self.truncated = self.finder(self.roots)
            self.base = snapshot_files(self.paths)
            json.dump(strip_text(self.base), open(self.cap.dir / "files_before.json", "w"), indent=1)
        except Exception:
            self.cap.err("files_start")

    def finish(self):
        try:
            after = snapshot_files(self.paths, self.base)
            changes, diffs = diff_files(self.base, after)
            rec = recent_files(self.roots, self.cap.t0)
            json.dump(strip_text(after), open(self.cap.dir / "files_after.json", "w"), indent=1)
            json.dump({"changes": changes, "recent_any_name": rec, "candidates": len(self.paths),
                       "truncated": self.truncated}, open(self.cap.dir / "file_changes.json", "w"), indent=1)
            if diffs:
                (self.cap.dir / "diffs").mkdir(exist_ok=True)
                for i, (p, d) in enumerate(diffs.items()):
                    (self.cap.dir / "diffs" / ("%02d_%s.diff" % (i, re.sub(r"\W+", "_", p)[-80:]))).write_text(
                        "# " + p + "\n" + d)
        except Exception:
            self.cap.err("files_finish")


# ---- Rede (opcional, passiva) -----------------------------------------------
def parse_ss(text):
    """Extrai (proto,state,local,peer) de `ss -tun` (sem processos)."""
    rows = []
    for ln in text.splitlines()[1:]:
        f = ln.split()
        if len(f) >= 6:
            rows.append((f[0], f[1], f[4], f[5]))
    return rows


class NetWatcher:
    def __init__(self, cap, run=subprocess.run, which=None, period=3.0):
        self.cap, self.run, self.period = cap, run, period
        import shutil
        self.which = which or shutil.which
        self.proc = None
        self.seen = set()
        self.status = "nao iniciado"
        self.stop = threading.Event()

    def listening_once(self, tag):
        try:
            r = self.run(["ss", "-tulnH"], capture_output=True, text=True, timeout=5)
            (self.cap.dir / ("listening_%s.txt" % tag)).write_text(r.stdout)
        except Exception:
            self.cap.err("ss_listen")

    def poll_connections(self):
        try:
            r = self.run(["ss", "-tunH"], capture_output=True, text=True, timeout=5)
            for ln in r.stdout.splitlines():
                f = ln.split()
                if len(f) >= 5:
                    key = (f[0], f[1], f[3], f[4])
                    if key not in self.seen and len(self.seen) < 5000:
                        self.seen.add(key)
                        with open(self.cap.dir / "connections.jsonl", "a") as fh:
                            fh.write(json.dumps({"t": self.cap.clock(), "phase": self.cap.phase,
                                                 "proto": f[0], "state": f[1], "local": f[3],
                                                 "peer": f[4]}) + "\n")
        except Exception:
            self.cap.err("ss_poll")

    def start_tcpdump(self):
        tcpdump = self.which("tcpdump")
        if not tcpdump:
            self.status = "tcpdump ausente"
            return
        try:
            ok = self.run(["sudo", "-n", "-l", tcpdump], capture_output=True, text=True, timeout=5)
            if ok.returncode != 0:
                self.status = "sudo sem senha indisponivel para tcpdump"
                return
        except Exception:
            self.status = "sudo indisponivel"
            return
        try:
            # somente cabecalhos de linha (-q), sem payload (-s 0 nao e usado; texto nao imprime dados)
            cmd = ["sudo", "-n", tcpdump, "-i", "any", "-n", "-q", "-tt", "-l",
                   "not port 22 and not portrange 7400-7600 and not port 41641 and not port 53"]
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.status = "ativo"
            threading.Thread(target=self._pump, daemon=True).start()
        except Exception:
            self.status = "falha ao iniciar tcpdump"
            self.cap.err("tcpdump_start")

    def _pump(self, max_lines=200_000):
        n = 0
        with open(self.cap.dir / "net_headers.txt", "w") as fh:
            for ln in self.proc.stdout:
                n += 1
                if n > max_lines:
                    break
                fh.write("%s %s" % (self.cap.phase, ln))
                if n % 200 == 0:
                    fh.flush()

    def start(self):
        self.listening_once("inicio")
        self.start_tcpdump()

    def finish(self):
        self.listening_once("fim")
        self.poll_connections()
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.cap.extra["net_status"] = self.status
        self.cap.extra["net_connections_seen"] = len(self.seen)


# ---- Descoberta e subscribers DDS ------------------------------------------
def norm_type(t):
    return (t or "").replace("::", ".")


def wanted_subscription(topic, type_name):
    """Decide (kind, fields, min_interval) ou None. Nunca assina tipos desconhecidos."""
    tn = norm_type(type_name)
    if tn.endswith("unitree_api.msg.dds_.Request_") or topic.endswith("/request") and "Request_" in tn:
        return ("api_request", None, 0.0)
    if tn.endswith("unitree_api.msg.dds_.Response_") or topic.endswith("/response") and "Response_" in tn:
        return ("api_response", None, 0.0)
    if topic in STATE_TOPICS:
        return ("state",) + STATE_TOPICS[topic][1:]
    if any(p.match(topic) for p in STATE_PATTERNS) and tn in KNOWN_STATE_TYPES:
        return ("state", None, 0.5)
    return None


KNOWN_STATE_TYPES = {"unitree_go.msg.dds_.SportModeState_", "unitree_hg.msg.dds_.LowState_",
                     "unitree_go.msg.dds_.LowState_", "unitree_hg.msg.dds_.HandState_",
                     "unitree_go.msg.dds_.WirelessController_"}


class Subscriptions:
    """Mantem subscribers; `open_reader(topic, type_name)` e injetavel (fake nos testes)."""

    def __init__(self, cap, open_reader):
        self.cap, self.open_reader = cap, open_reader
        self.readers = {}  # topic -> (kind, fields, interval, reader, type_name)

    def ensure(self, topic, type_name):
        if topic in self.readers:
            return False
        want = wanted_subscription(topic, type_name)
        if want is None:
            return False
        try:
            reader = self.open_reader(topic, norm_type(type_name))
        except Exception:
            self.cap.err("subscribe_fail")
            return False
        if reader is None:
            self.cap.err("subscribe_fail")
            return False
        self.readers[topic] = (want[0], want[1], want[2], reader, norm_type(type_name))
        return True

    def poll_once(self, max_n=64):
        n = 0
        for topic, (kind, fields, interval, reader, tn) in list(self.readers.items()):
            try:
                msgs = reader(max_n)
            except Exception:
                self.cap.err("read_fail")
                continue
            for m in msgs or []:
                n += 1
                if kind == "api_request":
                    self.cap.on_api(topic, "request", m)
                elif kind == "api_response":
                    self.cap.on_api(topic, "response", m)
                else:
                    self.cap.on_state(topic, m, tn, fields, interval)
        return n


def discover_once(cap, fetch, subs):
    """fetch() -> [{topic,type,kind,participant}]. Registra tudo e assina o que for permitido."""
    try:
        items = fetch()
    except Exception:
        cap.err("discover_fail")
        return 0
    for it in items:
        cap.note_topic(it["topic"], it.get("type"), it.get("kind"), it.get("participant"))
        subs.ensure(it["topic"], it.get("type"))
    return len(items)


def _dds_open_reader_factory(participant):
    from cyclonedds.core import Qos, Policy
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from importlib import import_module
    qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(32))

    def open_reader(topic, type_name):
        mod, _, cls = type_name.rpartition(".")
        klass = getattr(import_module("unitree_sdk2py.idl." + mod), cls)
        rd = DataReader(participant, Topic(participant, topic, klass), qos=qos)

        def read(n):
            return list(rd.take(N=n))
        return read
    return open_reader


def _dds_fetch_factory(participant):
    from cyclonedds.builtin import BuiltinTopicDcpsPublication, BuiltinTopicDcpsSubscription
    from cyclonedds.sub import DataReader
    readers = {"pub": DataReader(participant, BuiltinTopicDcpsPublication),
               "sub": DataReader(participant, BuiltinTopicDcpsSubscription)}

    def fetch():
        out = []
        for kind, rd in readers.items():
            for s in rd.read(N=512):
                out.append({"topic": s.topic_name, "type": s.type_name, "kind": kind,
                            "participant": str(getattr(s, "participant_key", ""))})
        return out
    return fetch


# ---- Principal --------------------------------------------------------------
def stdin_marker_thread(cap, stop, stream=None):
    stream = stream or sys.stdin

    def run():
        while not stop.is_set():
            try:
                line = stream.readline()
            except Exception:
                return
            if line == "":
                return
            name = cap.mark()
            print("[marcador] %s (t=+%.1fs)" % (name.upper(), cap.markers[-1]["rel"]), flush=True)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def run_capture(args, fetch=None, open_reader=None, stdin=None, clock=time.time):
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out_base).expanduser() / ("cog_capture_" + ts)
    cap = Capture(outdir, clock=clock)
    cap.extra.update(argv=sys.argv, iface=args.iface, domain=args.domain)
    subs = Subscriptions(cap, open_reader or (lambda t, ty: None))
    if fetch is None or open_reader is None:
        from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize
        ChannelFactoryInitialize(args.domain, args.iface)
        participant = ChannelFactory()._ChannelFactory__participant
        fetch = fetch or _dds_fetch_factory(participant)
        subs.open_reader = open_reader or _dds_open_reader_factory(participant)
    fw = None if args.no_files else FileWatcher(cap)
    nw = None if args.no_net else NetWatcher(cap)
    stop = threading.Event()
    if fw:
        threading.Thread(target=fw.start, daemon=True).start()
    if nw:
        nw.start()
    for t in KNOWN_API_TOPICS:
        subs.ensure(t, "unitree_api.msg.dds_." + ("Request_" if t.endswith("request") else "Response_"))
    for t, (ty, _f, _i) in STATE_TOPICS.items():
        subs.ensure(t, ty)
    print("Captura PASSIVA em %s" % outdir, flush=True)
    print("Enter = marcador (1o ANTES, 2o MEXENDO, 3o DEPOIS). Ctrl+C encerra.", flush=True)
    if not args.no_stdin:
        stdin_marker_thread(cap, stop, stdin)
    last_disc = last_snap = last_flush = last_net = 0.0
    end = clock() + args.duration if args.duration else None
    try:
        while not stop.is_set():
            now = clock()
            if end and now >= end:
                break
            if now - last_disc >= args.discover_period:
                last_disc = now
                discover_once(cap, fetch, subs)
            if now - last_snap >= args.snap_period:
                last_snap = now
                cap.snapshot()
            if nw and now - last_net >= 3:
                last_net = now
                nw.poll_connections()
            if now - last_flush >= 5:
                last_flush = now
                cap.flush()
            if not subs.poll_once():
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    stop.set()
    discover_once(cap, fetch, subs)
    cap.snapshot()
    if nw:
        nw.finish()
    if fw:
        fw.finish()
    cap.extra["subscribed_topics"] = sorted(subs.readers)
    cap.close()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import analyze_cog_capture as an
        report = an.build_report(outdir)
        (outdir / "report.md").write_text(report)
        print(report)
    except Exception as e:
        print("analise falhou: %r (dados em %s)" % (e, outdir))
    print("\nDados: %s" % outdir)
    return outdir


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="enP8p1s0")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--duration", type=float, default=0.0, help="segundos (0 = ate Ctrl+C)")
    ap.add_argument("--out-base", default="/home/unitree/.local/state/xr_teleoperate")
    ap.add_argument("--snap-period", type=float, default=1.0)
    ap.add_argument("--discover-period", type=float, default=2.0)
    ap.add_argument("--no-files", action="store_true")
    ap.add_argument("--no-net", action="store_true")
    ap.add_argument("--no-stdin", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run_capture(parse_args())
