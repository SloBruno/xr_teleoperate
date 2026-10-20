"""Optional offline web control panel for the G1D teach/replay session.

Stdlib only (``http.server.ThreadingHTTPServer`` in daemon threads). The panel
is an *input*, exactly like the terminal keys: HTTP handlers never touch IO or
DDS and never wait for the control tick.

* ``/api/stop`` calls the lock-free ``session.request_quit()`` directly and
  discards every queued web action (stop wins).
* teach/save/replay are put on a small bounded queue (``put_nowait``; full =>
  HTTP 503 "busy") and a dispatcher thread calls the SAME ``request_*``
  lifecycle methods the keys use; if the session lock is held the dispatcher,
  not the HTTP handler, waits.
* status is read through the lock-free ``session.status_view()``.

Safety: random token per process (``hmac.compare_digest``), required by every
``/api`` call and by the page itself; one controlling browser at a time
(others read-only until it has been silent for ``controller_timeout_s``).
Heartbeat is authority/liveness input only; it never gates an active replay.
"""

from __future__ import annotations

import hmac
import json
import os
import queue
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from teleop.utils.teach_replay import DEX3_LIMITS, RecordingError, load_recording

DEFAULT_PORT = 8095
DEADMAN_S = 1.0
CONTROLLER_TIMEOUT_S = 3.0
MAX_BODY = 4096
MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # conservative documented request/file limit
TRASH_DIR = "trash"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}(\.aborted)?\.jsonl$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_UTC_PREFIX_RE = re.compile(r"^(\d{8}T\d{6}_\d{6}Z)")


class NameError_(ValueError):
    pass


def recording_base(filename: str) -> str:
    for suffix in (".aborted.jsonl", ".jsonl"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def safe_recording_path(teach_dir, name) -> Path:
    """Path of a recording file directly inside ``teach_dir`` or raise."""
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise NameError_("nome de gravação inválido")
    root = Path(teach_dir).resolve()
    path = (root / name).resolve()
    if path.parent != root:
        raise NameError_("fora do diretório de gravações")
    return path


def renamed_name(name: str, label) -> str:
    if not isinstance(label, str) or not _LABEL_RE.fullmatch(label):
        raise NameError_("rótulo inválido (use letras, números, _ ou -, até 64)")
    aborted = name.endswith(".aborted.jsonl")
    base = recording_base(name)
    match = _UTC_PREFIX_RE.match(base)
    prefix = match.group(1) if match else base
    new = f"{prefix}_{label}" + (".aborted.jsonl" if aborted else ".jsonl")
    if not _NAME_RE.fullmatch(new):
        raise NameError_("nome resultante inválido")
    return new


def _summarize(path: Path) -> dict:
    """Duration/date of a recording file (handler thread only, never the tick)."""
    info = {"duration_s": None, "date": None, "samples": 0, "status": None}
    first_t = last_t = None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                kind = rec.get("type")
                if i == 0 and kind == "header":
                    info["date"] = rec.get("created_utc")
                elif kind == "sample":
                    try:
                        t = float(rec["t"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    first_t = t if first_t is None else first_t
                    last_t = t
                    info["samples"] += 1
                elif kind == "footer":
                    info["status"] = rec.get("status")
    except OSError:
        pass
    if first_t is not None and last_t is not None:
        info["duration_s"] = round(last_t - first_t, 3)
    if info["date"] is None:
        try:
            info["date"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
        except OSError:
            pass
    return info


class WebPanel:
    def __init__(self, session, teach_dir, *, host="0.0.0.0", port=DEFAULT_PORT, token=None,
                 clock=time.monotonic, freshness_fn=None, deadman_s=DEADMAN_S,
                 controller_timeout_s=CONTROLLER_TIMEOUT_S, queue_size=4,
                 arm_lower=None, arm_upper=None):
        self.session = session
        self.teach_dir = Path(teach_dir)
        self.host = host
        self._port = int(port)
        self.token = token or secrets.token_urlsafe(16)
        self.clock = clock
        self.freshness_fn = freshness_fn
        self.deadman_s = float(deadman_s)
        self.controller_timeout_s = float(controller_timeout_s)
        self.arm_lower = arm_lower
        self.arm_upper = arm_upper
        self._actions: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._ctl_lock = threading.Lock()
        self._controller = None          # (client_id, last_seen)
        self._file_lock = threading.Lock()
        self._summary_cache = {}
        self._server = None
        self._threads = []

    # ------------------------------------------------------------ lifecycle
    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server is not None else self._port

    def url(self, ip: str) -> str:
        return f"http://{ip}:{self.port}/?t={self.token}"

    def start(self):
        panel = self

        class Handler(_Handler):
            pass

        Handler.panel = panel
        self._server = ThreadingHTTPServer((self.host, self._port), Handler)
        self._server.daemon_threads = True
        serve = threading.Thread(target=self._server.serve_forever, name="teach-web", daemon=True)
        dispatch = threading.Thread(target=self._dispatch, name="teach-web-dispatch", daemon=True)
        self._threads = [serve, dispatch]
        serve.start()
        dispatch.start()
        return self

    def stop(self, timeout=1.0):
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        self._clear_actions()
        try:
            self._actions.put_nowait(None)
        except queue.Full:
            pass
        for thread in self._threads:
            thread.join(timeout)

    # ------------------------------------------------------------- actions
    def _clear_actions(self):
        with self._actions.mutex:
            self._actions.queue.clear()

    def _dispatch(self):
        while True:
            item = self._actions.get()
            if item is None:
                return
            if self.session.quit_requested:
                continue                    # stop wins over everything still queued
            kind, arg = item
            try:
                if kind == "teach":
                    self.session.request_teach()
                elif kind == "save":
                    self.session.request_save()
                elif kind == "replay":
                    name, repeat, continuous = arg
                    self.session.request_replay(name, origin="web", repeat=repeat, continuous=continuous)
            except Exception:
                pass

    def enqueue(self, kind, arg=None) -> bool:
        try:
            self._actions.put_nowait((kind, arg))
            return True
        except queue.Full:
            return False

    def stop_now(self):
        self.session.request_quit()          # lock-free, before anything else
        self._clear_actions()

    def queued_replay_bases(self):
        with self._actions.mutex:
            items = list(self._actions.queue)
        out = set()
        for kind, arg in items:
            if kind != "replay" or not arg:
                continue
            name = arg[0] if isinstance(arg, tuple) else arg
            if name:
                out.add(recording_base(Path(name).name))
        return out

    # ---------------------------------------------------------- controller
    def touch(self, client_id) -> bool:
        """Mark a poll from ``client_id``; True if it is (now) the controller."""
        now = self.clock()
        with self._ctl_lock:
            ctl = self._controller
            if ctl is None or ctl[0] == client_id or now - ctl[1] > self.controller_timeout_s:
                self._controller = (client_id, now)
                is_controller = True
            else:
                is_controller = False
        if is_controller:
            try:
                self.session.feed_web_deadman(self.deadman_s)
            except Exception:
                pass
        return is_controller

    # ---------------------------------------------------------- recordings
    def busy_bases(self):
        try:
            busy = set(self.session.status_view().get("busy_recordings") or [])
        except Exception:
            busy = set()
        return {recording_base(b) for b in busy} | self.queued_replay_bases()

    def list_recordings(self, include_aborted=False):
        out = []
        try:
            entries = list(self.teach_dir.iterdir()) if self.teach_dir.is_dir() else []
        except OSError:
            entries = []
        for path in entries:
            name = path.name
            if not _NAME_RE.fullmatch(name) or not path.is_file():
                continue
            aborted = name.endswith(".aborted.jsonl")
            if aborted and not include_aborted:
                continue
            try:
                st = path.stat()
                key = (name, st.st_mtime_ns, st.st_size)
            except OSError:
                continue
            summary = self._summary_cache.get(key)
            if summary is None:
                summary = _summarize(path)
                self._summary_cache[key] = summary
            out.append({"name": name, "aborted": aborted, **summary})
        out.sort(key=lambda r: r["name"], reverse=True)
        return out

    def rename(self, name, label):
        with self._file_lock:
            src = safe_recording_path(self.teach_dir, name)
            dst = safe_recording_path(self.teach_dir, renamed_name(name, label))
            if not src.is_file():
                return 404, {"ok": False, "error": "gravação não encontrada"}
            if recording_base(name) in self.busy_bases():
                return 409, {"ok": False, "error": "gravação em uso (gravando/reproduzindo)"}
            try:
                os.link(src, dst)            # atomic, never overwrites
            except FileExistsError:
                return 409, {"ok": False, "error": "já existe uma gravação com esse nome"}
            except OSError as exc:
                return 500, {"ok": False, "error": str(exc)}
            os.unlink(src)
            return 200, {"ok": True, "name": dst.name}

    def delete(self, name):
        with self._file_lock:
            src = safe_recording_path(self.teach_dir, name)
            if not src.is_file():
                return 404, {"ok": False, "error": "gravação não encontrada"}
            if recording_base(name) in self.busy_bases():
                return 409, {"ok": False, "error": "gravação em uso (gravando/reproduzindo)"}
            trash = src.parent / TRASH_DIR
            try:
                trash.mkdir(exist_ok=True)
                target = trash / name
                if target.exists():
                    target = trash / f"{recording_base(name)}.{int(time.time())}.jsonl"
                os.replace(src, target)
            except OSError as exc:
                return 500, {"ok": False, "error": str(exc)}
            return 200, {"ok": True}

    def upload(self, filename, reader, content_length):
        """Stream, validate, and atomically install one recording off the tick."""
        if not isinstance(filename, str) or filename != Path(filename).name or not _NAME_RE.fullmatch(filename):
            return 400, {"ok": False, "error": "nome de gravação inválido"}
        try:
            length = int(content_length)
        except (TypeError, ValueError):
            return 400, {"ok": False, "error": "Content-Length obrigatório"}
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            return 413, {"ok": False, "error": "upload excede o limite de 16 MiB"}
        self.teach_dir.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            import tempfile
            fd, temp_name = tempfile.mkstemp(prefix=".g1d-upload-", suffix=".part", dir=self.teach_dir)
            temp = Path(temp_name)
            remaining = length
            with os.fdopen(fd, "wb") as handle:
                while remaining:
                    chunk = reader.read(min(64 * 1024, remaining))
                    if not chunk:
                        raise ValueError("corpo truncado")
                    handle.write(chunk)
                    remaining -= len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            load_recording(temp, arm_lower=self.arm_lower, arm_upper=self.arm_upper,
                           hand_limits=DEX3_LIMITS)
            with self._file_lock:
                target = self.teach_dir / filename
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                index = 0
                while target.exists():
                    index += 1
                    target = self.teach_dir / f"{stem}_{index}{suffix}"
                os.replace(temp, target)
            temp = None
            self._summary_cache.clear()
            return 201, {"ok": True, "name": target.name}
        except RecordingError as exc:
            return 400, {"ok": False, "error": str(exc)}
        except (OSError, ValueError) as exc:
            return 400, {"ok": False, "error": str(exc)}
        finally:
            if temp is not None:
                try:
                    temp.unlink()
                except OSError:
                    pass

    # -------------------------------------------------------------- status
    def status(self, client_id):
        controller = self.touch(client_id)
        try:
            view = dict(self.session.status_view())
        except Exception as exc:
            view = {"mode": "?", "message": f"status indisponível: {exc}"}
        fresh = {}
        if self.freshness_fn is not None:
            try:
                fresh = dict(self.freshness_fn())
            except Exception:
                fresh = {}
        view["fresh"] = fresh
        view["controller"] = controller
        return view

    def check_token(self, provided) -> bool:
        provided = provided if isinstance(provided, str) else ""
        return hmac.compare_digest(provided.encode("utf-8", "replace"), self.token.encode("utf-8"))


class _Handler(BaseHTTPRequestHandler):
    panel: WebPanel = None
    server_version = "G1DTeach/1"
    timeout = 10

    def log_message(self, *args):        # keep the operator terminal clean
        pass

    # ------------------------------------------------------------- helpers
    def _parse(self):
        parts = urlsplit(self.path)
        return parts.path, parse_qs(parts.query)

    def _token_ok(self, query):
        token = self.headers.get("X-Token")
        if token is None:
            token = (query.get("t") or [""])[0]
        return self.panel.check_token(token)

    def _client(self):
        cid = self.headers.get("X-Client-Id") or f"ip:{self.client_address[0]}"
        return cid[:64]

    def _send(self, code, payload, content_type="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        if length > MAX_BODY:
            raise ValueError("body too large")
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("body must be an object")
        return data

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        path, query = self._parse()
        if path in ("/", "/index.html"):
            if not self._token_ok(query):
                return self._send(401, TOKEN_REQUIRED_HTML.encode(), "text/html; charset=utf-8")
            return self._send(200, PAGE_HTML.encode(), "text/html; charset=utf-8")
        if not self._token_ok(query):
            return self._send(401, {"ok": False, "error": "token required"})
        if path == "/api/status":
            return self._send(200, self.panel.status(self._client()))
        if path == "/api/recordings":
            include = (query.get("all") or ["0"])[0] == "1"
            return self._send(200, {"ok": True, "recordings": self.panel.list_recordings(include)})
        return self._send(404, {"ok": False, "error": "not found"})

    # --------------------------------------------------------------- POST
    def do_POST(self):
        path, query = self._parse()
        if not self._token_ok(query):
            return self._send(401, {"ok": False, "error": "token required"})
        panel = self.panel
        if path == "/api/stop":                # always accepted, from any client
            panel.stop_now()
            return self._send(200, {"ok": True, "queued": False})
        if path == "/api/upload":
            if not panel.touch(self._client()):
                return self._send(409, {"ok": False, "error": "somente leitura: outro navegador controla"})
            filename = self.headers.get("X-Filename") or (query.get("name") or [""])[0]
            code, payload = panel.upload(filename, self.rfile, self.headers.get("Content-Length"))
            return self._send(code, payload)
        try:
            body = self._body()
        except ValueError as exc:
            return self._send(400, {"ok": False, "error": str(exc)})
        if path not in ("/api/heartbeat", "/api/teach", "/api/save", "/api/replay",
                        "/api/rename", "/api/delete"):
            return self._send(404, {"ok": False, "error": "not found"})
        if not panel.touch(self._client()):
            return self._send(409, {"ok": False, "error": "somente leitura: outro navegador controla"})
        if path == "/api/heartbeat":
            return self._send(200, {"ok": True})
        if path in ("/api/teach", "/api/save"):
            kind = path.rsplit("/", 1)[1]
            if not panel.enqueue(kind):
                return self._send(503, {"ok": False, "error": "busy"})
            return self._send(202, {"ok": True, "queued": True})
        try:
            if path == "/api/replay":
                name = body.get("name")
                arg = None
                if name is not None:
                    target = safe_recording_path(panel.teach_dir, name)
                    if name.endswith(".aborted.jsonl"):
                        raise NameError_("gravação abortada não pode ser reproduzida")
                    if not target.is_file():
                        return self._send(404, {"ok": False, "error": "gravação não encontrada"})
                    arg = str(target)
                try:
                    repeat = int(body.get("repeat", 1))
                except (TypeError, ValueError):
                    raise NameError_("repetições inválidas")
                continuous = body.get("continuous", False)
                if not isinstance(continuous, bool) or repeat < 1 or (continuous and repeat != 1):
                    raise NameError_("modo de repetição inválido")
                if not panel.enqueue("replay", (arg, repeat, continuous)):
                    return self._send(503, {"ok": False, "error": "busy"})
                return self._send(202, {"ok": True, "queued": True})
            if path == "/api/rename":
                code, payload = panel.rename(body.get("name"), body.get("label"))
                return self._send(code, payload)
            code, payload = panel.delete(body.get("name"))
            return self._send(code, payload)
        except NameError_ as exc:
            return self._send(400, {"ok": False, "error": str(exc)})


def local_ipv4_addresses():
    """(interface, ipv4) pairs for non-loopback interfaces (Linux ioctl, stdlib)."""
    import fcntl
    import socket
    import struct

    out = []
    try:
        names = [n for _, n in socket.if_nameindex()]
    except OSError:
        return out
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for name in names:
            if name == "lo":
                continue
            try:
                packed = fcntl.ioctl(sock.fileno(), 0x8915,  # SIOCGIFADDR
                                     struct.pack("256s", name[:15].encode()))
                out.append((name, socket.inet_ntoa(packed[20:24])))
            except OSError:
                continue
    finally:
        sock.close()
    return out


TOKEN_REQUIRED_HTML = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>G1D teach</title></head>
<body style="font-family:sans-serif;background:#111;color:#eee;padding:2em">
<h2>Token necessário (token required)</h2>
<p>Abra a URL completa impressa no terminal do robô (com <code>?t=...</code>).</p>
</body></html>"""


PAGE_HTML = r"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1D Ensinar/Reproduzir</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,sans-serif;background:#111;color:#eee;padding:10px 10px 120px}
h1{font-size:1.2em;margin:4px 0 10px}
.card{background:#1d1d1d;border:1px solid #333;border-radius:10px;padding:12px;margin-bottom:10px}
.row{display:flex;gap:8px;flex-wrap:wrap}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px}
.mode{font-size:1.6em;font-weight:bold}
.ok{color:#5d5}.bad{color:#f66}.warn{color:#fc4}
button{font-size:1.2em;padding:16px;border-radius:10px;border:0;flex:1 1 30%;min-width:120px;color:#fff;background:#356}
button:disabled{opacity:.35}
#teach{background:#2a6}#save{background:#36c}#replay{background:#a62}
#stop{position:fixed;left:10px;right:10px;bottom:10px;height:90px;font-size:2em;font-weight:bold;background:#d00;border:4px solid #fff}
#ro{display:none;background:#642;padding:8px;border-radius:8px;margin-bottom:10px}
#conn{display:none;background:#800;padding:8px;border-radius:8px;margin-bottom:10px}
table{width:100%;border-collapse:collapse}td{padding:6px 4px;border-bottom:1px solid #333;font-size:.95em}
td button{font-size:.9em;padding:8px;min-width:0;flex:none;background:#444}
.bar{height:14px;background:#333;border-radius:7px;overflow:hidden}.bar div{height:100%;background:#a62;width:0}
small{color:#999}
</style></head><body>
<h1>G1D Ensinar / Reproduzir</h1>
<div id="conn">Sem conexão com o robô</div>
<div id="ro">Somente leitura: outro navegador está controlando. PARAR continua funcionando.</div>
<div class="card">
 <div class="mode" id="mode">...</div>
 <div id="msg"></div>
 <div class="kv" style="margin-top:8px">
  <span>Braços</span><span id="f_arm">-</span>
  <span>Mão esq.</span><span id="f_left">-</span>
  <span>Mão dir.</span><span id="f_right">-</span>
  <span>Gravação</span><span id="rec">-</span>
 <span>Reprodução</span><span id="rep">-</span>
 </div>
 <div class="bar" style="margin-top:8px"><div id="prog"></div></div>
</div>
<div class="card row">
 <button id="teach">Ensinar (r)</button>
 <button id="save">Salvar (s)</button>
 <button id="replay">Reproduzir (p)</button>
 <label>Repetições <input id="repeat" type="number" min="1" value="1" style="width:5em"></label>
 <label><input id="continuous" type="checkbox"> Repetir continuamente</label>
</div>
<div class="card"><b>Enviar gravação</b><br><input id="uploadfile" type="file" accept=".jsonl">
 <button id="upload">Enviar</button><small>Limite: 16 MiB; a gravação é validada antes de aparecer na lista.</small></div>
<div class="card">
 <div class="row" style="justify-content:space-between;align-items:center">
  <b>Gravações</b>
  <label><input type="checkbox" id="showall"> mostrar abortadas</label>
 </div>
 <small>Selecionada para reproduzir: <span id="sel">mais recente</span></small>
 <table id="list"></table>
</div>
<button id="stop">PARAR (q)</button>
<script>
(function(){
var q=new URLSearchParams(location.search),T=q.get('t')||'';
var CID=sessionStorage.getItem('g1d_cid');
if(!CID){CID=Math.random().toString(36).slice(2)+Date.now().toString(36);sessionStorage.setItem('g1d_cid',CID);}
var selected=null,controller=false,busy=[];
function $(i){return document.getElementById(i);}
function api(method,path,body){
 var o={method:method,headers:{'X-Token':T,'X-Client-Id':CID},cache:'no-store'};
 if(body!==undefined){o.headers['Content-Type']='application/json';o.body=JSON.stringify(body);}
 return fetch(path,o).then(function(r){return r.json().catch(function(){return {};}).then(function(j){j._code=r.status;return j;});});
}
function fresh(id,v){var e=$(id);
 if(v===null||v===undefined){e.textContent='sem dados';e.className='bad';return;}
 e.textContent=(v*1000).toFixed(0)+' ms';e.className=v<0.2?'ok':(v<0.5?'warn':'bad');}
function poll(){
 api('GET','/api/status').then(function(s){
  $('conn').style.display='none';
  if(s._code!==200){$('msg').textContent='Erro '+s._code+' '+(s.error||'');return;}
  controller=!!s.controller;$('ro').style.display=controller?'none':'block';
  $('mode').textContent=(s.mode||'?').toUpperCase()+(s.quit_requested?' (parando)':'');
  $('msg').textContent=s.message||'';
  var f=s.fresh||{};fresh('f_arm',f.arm);fresh('f_left',f.left);fresh('f_right',f.right);
  $('rec').textContent=s.recording?(s.recording.name+' - '+s.recording.elapsed_s.toFixed(1)+' s'):'-';
  var p=s.replay?Math.round(100*s.replay.progress):0;
  $('rep').textContent=s.replay?(s.replay.name+' ('+(s.replay.phase==='ramp'?'rampa':'reproduzindo')+') '+p+'% ciclo '+s.replay.iteration+'/'+(s.replay.total===null?'contínuo':s.replay.total)):'-';
  $('prog').style.width=p+'%';
  var m=s.mode;
  $('teach').disabled=!controller||!(m==='idle'||m==='hold');
  $('replay').disabled=!controller||!(m==='idle'||m==='hold');
  $('save').disabled=!controller||m!=='teach';
  var nb=JSON.stringify(s.busy_recordings||[]);
  if(nb!==JSON.stringify(busy)){busy=s.busy_recordings||[];loadList();}
 }).catch(function(){$('conn').style.display='block';});
}
function base(n){return n.replace(/\.aborted\.jsonl$/,'').replace(/\.jsonl$/,'');}
function loadList(){
 api('GET','/api/recordings'+($('showall').checked?'?all=1':'')).then(function(r){
  var t=$('list');t.innerHTML='';
  (r.recordings||[]).forEach(function(rec){
   var tr=document.createElement('tr'),inuse=busy.indexOf(base(rec.name))>=0;
   var td=document.createElement('td');
   td.innerHTML='<b></b><br><small></small>';
   td.firstChild.textContent=rec.name+(rec.aborted?' (abortada)':'')+(inuse?' (em uso)':'');
   td.lastChild.textContent=(rec.duration_s!==null?rec.duration_s.toFixed(1)+' s':'?')+' - '+(rec.date||'');
   tr.appendChild(td);
   var ta=document.createElement('td');ta.style.whiteSpace='nowrap';
   function b(label,fn,dis){var x=document.createElement('button');x.textContent=label;x.disabled=dis;x.onclick=fn;ta.appendChild(x);}
   b(selected===rec.name?'Selecionada':'Selecionar',function(){selected=rec.name;$('sel').textContent=rec.name;loadList();},rec.aborted);
   b('Renomear',function(){var l=prompt('Novo rótulo (letras, números, _ ou -):','');if(!l)return;
     api('POST','/api/rename',{name:rec.name,label:l}).then(function(x){if(!x.ok)alert(x.error||('erro '+x._code));else if(selected===rec.name){selected=x.name;$('sel').textContent=x.name;}loadList();});},!controller||inuse);
   b('Apagar',function(){if(!confirm('Apagar '+rec.name+'? (vai para trash/)'))return;
     api('POST','/api/delete',{name:rec.name}).then(function(x){if(!x.ok)alert(x.error||('erro '+x._code));if(selected===rec.name){selected=null;$('sel').textContent='mais recente';}loadList();});},!controller||inuse);
   tr.appendChild(ta);t.appendChild(tr);
  });
 }).catch(function(){});
}
function act(path,body){api('POST',path,body).then(function(x){if(!x.ok)$('msg').textContent='Recusado: '+(x.error||x._code);poll();});}
$('teach').onclick=function(){act('/api/teach',{});};
$('save').onclick=function(){act('/api/save',{});setTimeout(loadList,1500);};
 $('replay').onclick=function(){act('/api/replay',{name:selected,repeat:parseInt($('repeat').value||'1',10),continuous:$('continuous').checked});};
 $('upload').onclick=function(){var f=$('uploadfile').files[0];if(!f)return;
   fetch('/api/upload',{method:'POST',headers:{'X-Token':T,'X-Client-Id':CID,'X-Filename':f.name},body:f}).then(function(r){return r.json();}).then(function(x){if(!x.ok)alert(x.error||'upload recusado');else loadList();});
 };
$('stop').onclick=function(){api('POST','/api/stop',{}).then(poll).catch(function(){$('conn').style.display='block';});};
$('showall').onchange=loadList;
setInterval(poll,200);setInterval(loadList,5000);poll();loadList();
})();
</script></body></html>
"""
