#!/usr/bin/env python3
"""Interface web de calibracao de caminhada do G1 (substitui o terminal de marcadores).

Processo INDEPENDENTE da teleop (nao importa teleop_hand_and_arm).  Servidor
stdlib (ThreadingHTTPServer) + HTML/JS puro com polling.

DDS:
  * leitores passivos: rt/lf/lowstate (IMU pelve), rt/secondary_imu (torso),
    rt/odommodestate, rt/wirelesscontroller, rt/config_change_status,
    rt/api/config/request + response (observa SETs do app Unitree Explorer);
  * escrita (so modo normal, so por acao confirmada na UI): servico ``config``
    api 1001 SET {"name":"imu_offset_json","content":"{\\"imu\\":[r,p,y]}"};
  * leitura ativa: api 1002 GET (nao confirmado no G1) somente na inicializacao
    e no botao "Reler" -- nunca em --read-only/--dry-run.
Nada e enviado automaticamente no encerramento.

Modos: normal | --dry-run (SET simulado, sem GET, sem writer DDS) |
--read-only (so passivo) | --sim (sem DDS, backend falso, para teste local).

Uso (robo):
  export LD_LIBRARY_PATH=/home/unitree/cyclonedds/build/lib
  export PYTHONPATH=/home/unitree/unitree_sdk2_python:$PYTHONPATH
  /home/unitree/miniconda3/envs/tv/bin/python tools/calib_web.py --iface enP8p1s0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.calib_web_core import (API_GET, API_SET, CONFIG_NAMES, CODE_TIMEOUT,  # noqa: E402
                                  CalibController, FakeBackend, OffsetPolicy, build_get_parameter,
                                  build_set_parameter, parse_config_change, parse_get_response,
                                  parse_set_request)

DEFAULT_DIR = Path("/home/unitree/.local/state/xr_teleoperate")
STATIC_DIR = Path(__file__).resolve().parent / "calib_web"
LOCO_ZERO_EPS = 0.05


# ------------------------------------------------------------ teleop probe
def teleop_active(proc="/proc", own_pid=None):
    """True se algum processo tem argv com .../teleop_hand_and_arm.py exato."""
    own_pid = os.getpid() if own_pid is None else own_pid
    found = []
    try:
        pids = [p for p in os.listdir(proc) if p.isdigit()]
    except OSError:
        return None
    for pid in pids:
        if int(pid) == own_pid:
            continue
        try:
            with open(os.path.join(proc, pid, "cmdline"), "rb") as fh:
                argv = [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
        except OSError:
            continue
        if argv and "python" in os.path.basename(argv[0]) and any(
                os.path.basename(a) == "teleop_hand_and_arm.py" for a in argv[1:]):
            found.append(int(pid))
    return found


# --------------------------------------------------------------- DDS backend
class DdsBackend:
    """Leitores passivos + (opcional) writer do servico config criado so na 1a chamada."""

    def __init__(self, iface, domain=0, dry_run=False, read_only=False, clock=time.monotonic, log=print):
        self.iface, self.domain = iface, domain
        self.dry_run, self.read_only = dry_run, read_only
        self.can_write = not read_only
        self.can_get = not (read_only or dry_run)
        self._clock, self._log = clock, log
        self._lock = threading.Lock()
        self._latest = {}            # name -> (t, extracted dict)
        self._counts = {}
        self.history = {"imu": deque(maxlen=600), "secondaryimu": deque(maxlen=600)}
        self._passive = deque(maxlen=64)
        self._pending_req = {}       # identity.id -> (t, key, vals)
        self._futures = {}           # identity.id -> [event, response]
        self._own_ids = set()
        self._subs = []
        self._pub = None
        self._next_id = int(time.time() * 1000)
        self._teleop = {"t": None, "pids": None}

    # -- setup ----------------------------------------------------------------
    def start(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_, Response_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, WirelessController_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowState_
        from teleop.utils.config_change_status_idl import ConfigChangeStatus_
        ChannelFactoryInitialize(self.domain, self.iface)
        topics = [("rt/lf/lowstate", LowState_, self._on_lowstate),
                  ("rt/secondary_imu", IMUState_, self._on_torso),
                  ("rt/odommodestate", SportModeState_, self._on_odom),
                  ("rt/wirelesscontroller", WirelessController_, self._on_wireless),
                  ("rt/config_change_status", ConfigChangeStatus_, self._on_cfg_status),
                  ("rt/api/config/request", Request_, self._on_request),
                  ("rt/api/config/response", Response_, self._on_response)]
        for topic, cls, handler in topics:
            try:
                sub = ChannelSubscriber(topic, cls)
                sub.Init(self._guard(handler), 10)
                self._subs.append(sub)
            except Exception as e:  # pragma: no cover - hardware
                self._log(f"[calib_web] falha ao assinar {topic}: {e!r}")
        return self

    def close(self):
        for s in self._subs:
            try:
                s.Close()
            except Exception:
                pass
        self._subs = []

    @staticmethod
    def _guard(fn):
        def h(msg):
            try:
                fn(msg)
            except Exception:
                pass
        return h

    def _store(self, name, data, min_dt=0.05):
        t = self._clock()
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + 1
            last = self._latest.get(name)
            if last is None or t - last[0] >= min_dt:
                self._latest[name] = (t, data)
        return t

    # -- handlers (curtos) ------------------------------------------------------
    def _on_lowstate(self, msg):
        rpy = [float(v) for v in msg.imu_state.rpy]
        t = self._store("pelvis", {"rpy": rpy})
        with self._lock:
            self.history["imu"].append((t, rpy))

    def _on_torso(self, msg):
        t = self._clock()
        with self._lock:
            last = self._latest.get("torso")
            if last is not None and t - last[0] < 0.04:
                return
        rpy = [float(v) for v in msg.rpy]
        self._store("torso", {"rpy": rpy}, 0.0)
        with self._lock:
            self.history["secondaryimu"].append((t, rpy))

    def _on_odom(self, msg):
        t = self._clock()
        with self._lock:
            last = self._latest.get("odom")
            if last is not None and t - last[0] < 0.05:
                return
        v = [float(x) for x in msg.velocity]
        self._store("odom", {"velocity": v, "yaw_speed": float(msg.yaw_speed), "mode": int(msg.mode)}, 0.0)

    def _on_wireless(self, msg):
        self._store("wireless", {"lx": float(msg.lx), "ly": float(msg.ly), "rx": float(msg.rx),
                                 "ry": float(msg.ry)}, 0.02)

    def _on_cfg_status(self, msg):
        r = parse_config_change(str(msg.name), str(msg.content))
        self._store("config_change_status", {"name": str(msg.name)}, 0.0)
        if r:
            with self._lock:
                self._passive.append((self._clock(), r[0], r[1], "dds_passive_status"))

    def _on_request(self, msg):
        rid, api = int(msg.header.identity.id), int(msg.header.identity.api_id)
        if api != API_SET:
            return
        r = parse_set_request(str(msg.parameter))
        if r is None:
            return
        with self._lock:
            if rid in self._own_ids:
                return
            self._pending_req[rid] = (self._clock(), r[0], r[1])
            if len(self._pending_req) > 64:
                self._pending_req.pop(next(iter(self._pending_req)))

    def _on_response(self, msg):
        rid, code = int(msg.header.identity.id), int(msg.header.status.code)
        with self._lock:
            fut = self._futures.get(rid)
            if fut is not None:
                fut[1] = (code, str(msg.data))
                fut[0].set()
                return
            req = self._pending_req.pop(rid, None)
            if req is not None and code == 0:
                self._passive.append((self._clock(), req[1], req[2], "dds_passive_app"))

    # -- consumer API -------------------------------------------------------------
    def drain_passive(self):
        with self._lock:
            out = list(self._passive)
            self._passive.clear()
        return out

    def imu_history(self, key):
        with self._lock:
            return list(self.history.get(key, ()))

    def _teleop_status(self):
        now = self._clock()
        if self._teleop["t"] is None or now - self._teleop["t"] > 2.0:
            self._teleop = {"t": now, "pids": teleop_active()}
        return self._teleop["pids"]

    def live(self):
        now = self._clock()
        with self._lock:
            latest = dict(self._latest)
            counts = dict(self._counts)
        out = {"sources": {}, "mode": "dds"}
        for name, (t, _d) in latest.items():
            out["sources"][name] = {"age": now - t, "count": counts.get(name, 0)}
        w = latest.get("wireless")
        if w:
            d = w[1]
            out["wireless"] = dict(d, age=now - w[0],
                                   zero=all(abs(d[k]) < LOCO_ZERO_EPS for k in ("lx", "ly", "rx")))
        else:
            out["wireless"] = {"age": None}
        o = latest.get("odom")
        if o:
            v = o[1]["velocity"]
            out["odom"] = {"age": now - o[0], "speed": math.hypot(v[0], v[1]), "vx": v[0], "vy": v[1],
                           "yaw_rate": o[1]["yaw_speed"], "mode": o[1]["mode"]}
        else:
            out["odom"] = {"age": None}
        for key, name in (("pelvis_rpy_deg", "pelvis"), ("torso_rpy_deg", "torso")):
            s = latest.get(name)
            out[key] = None if not s else [math.degrees(x) for x in s[1]["rpy"]]
            out[key.replace("rpy_deg", "age")] = None if not s else now - s[0]
        out["teleop_pids"] = self._teleop_status()
        return out

    # -- active RPC (lazy writer) ------------------------------------------------
    def _publisher(self):
        if self._pub is None:
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_
            self._pub = ChannelPublisher("rt/api/config/request", Request_)
            self._pub.Init()
        return self._pub

    def _call(self, api_id, parameter, timeout):
        from unitree_sdk2py.idl.unitree_api.msg.dds_ import (Request_, RequestHeader_, RequestIdentity_,
                                                             RequestLease_, RequestPolicy_)
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            self._own_ids.add(rid)
            ev = threading.Event()
            self._futures[rid] = [ev, None]
        try:
            req = Request_(RequestHeader_(RequestIdentity_(rid, api_id), RequestLease_(0),
                                          RequestPolicy_(0, False)), parameter, [])
            if not self._publisher().Write(req, 1.0):
                return 3102, None
            if not ev.wait(timeout):
                return CODE_TIMEOUT, None
            with self._lock:
                return self._futures[rid][1]
        finally:
            with self._lock:
                self._futures.pop(rid, None)

    def config_get(self, key, timeout=1.0):
        if not self.can_get:
            return -1, None
        return self._call(API_GET, build_get_parameter(key), timeout)

    def config_set(self, key, vals, timeout=2.0):
        param = build_set_parameter(key, vals)
        if self.read_only:
            raise RuntimeError("read-only")
        if self.dry_run:
            self._log(f"[calib_web] DRY-RUN: NAO enviado api {API_SET} {param}")
            return 0, ""
        return self._call(API_SET, param, timeout)


# ------------------------------------------------------------------- HTTP
def make_handler(ctrl, token=None, static_dir=STATIC_DIR):
    class Handler(BaseHTTPRequestHandler):
        server_version = "calib_web/1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self):
            if not token:
                return True
            q = parse_qs(urlparse(self.path).query)
            return self.headers.get("X-Calib-Token") == token or q.get("token", [None])[0] == token

        def do_GET(self):
            if not self._authorized():
                return self._send(401, {"ok": False, "error": "token invalido"})
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                try:
                    return self._send(200, (static_dir / "index.html").read_bytes(), "text/html; charset=utf-8")
                except OSError:
                    return self._send(500, {"ok": False, "error": "index.html ausente"})
            if path == "/api/state":
                return self._send(200, ctrl.state())
            return self._send(404, {"ok": False, "error": "nao encontrado"})

        def do_POST(self):
            if not self._authorized():
                return self._send(401, {"ok": False, "error": "token invalido"})
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._send(415, {"ok": False, "error": "use application/json"})
            try:
                n = min(int(self.headers.get("Content-Length") or 0), 65536)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except (ValueError, json.JSONDecodeError):
                return self._send(400, {"ok": False, "error": "json invalido"})
            path = urlparse(self.path).path
            try:
                if path == "/api/marker":
                    res = ctrl.marker(body.pop("action", None), **body)
                elif path == "/api/offset/preview":
                    res = ctrl.preview(body.get("key"), body.get("target"), advanced=bool(body.get("advanced")))
                elif path == "/api/offset/apply":
                    res = ctrl.apply(body.get("key"), body.get("target"),
                                     expect_current=body.get("expect_current"),
                                     confirm=body.get("confirm") is True,
                                     ack_motion=body.get("ack_motion") is True,
                                     advanced=bool(body.get("advanced")))
                elif path == "/api/offset/read":
                    res = ctrl.read_get(body.get("key"), timeout=1.0) if body.get("key") in CONFIG_NAMES \
                        else {"ok": False, "error": "chave invalida"}
                elif path == "/api/offset/manual":
                    res = ctrl.set_manual(body.get("key"), body.get("value"), set_base=bool(body.get("set_base")))
                elif path == "/api/offset/restore_step":
                    res = ctrl.restore_step(body.get("key")) if body.get("key") in CONFIG_NAMES \
                        else {"ok": False, "error": "chave invalida"}
                else:
                    return self._send(404, {"ok": False, "error": "nao encontrado"})
            except Exception as e:
                return self._send(500, {"ok": False, "error": repr(e)})
            return self._send(200, res)
    return Handler


def serve(ctrl, host, port, token=None):
    httpd = ThreadingHTTPServer((host, port), make_handler(ctrl, token))
    httpd.daemon_threads = True
    th = threading.Thread(target=httpd.serve_forever, name="calib_web_http", daemon=True)
    th.start()
    return httpd


def local_ips():
    ips = []
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=2).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[2] == "inet":
                ip = parts[3].split("/")[0]
                if not ip.startswith("127."):
                    ips.append((parts[1], ip))
    except Exception:
        pass
    return ips


def build_arg_parser():
    ap = argparse.ArgumentParser(description="UI web de calibracao de caminhada do G1 (marcadores + offset IMU).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--iface", default="enP8p1s0", help="interface DDS (barramento interno)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--markers", type=Path, default=None, help="arquivo calib-markers (padrao: novo com UTC)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="sem nenhuma escrita DDS (SET simulado, sem GET)")
    g.add_argument("--read-only", action="store_true", help="somente leitura passiva; sem GET/SET")
    ap.add_argument("--sim", action="store_true", help="sem DDS (backend falso) para testar a UI")
    ap.add_argument("--window", type=float, default=3.0, help="janela +/- graus em torno da base (padrao 3)")
    ap.add_argument("--hard-abs", type=float, default=10.0, help="|offset| maximo absoluto (<=10)")
    ap.add_argument("--max-step", type=float, default=0.5, help="passo maximo por comando (<=0.5)")
    ap.add_argument("--min-interval", type=float, default=1.0, help="s minimos entre escritas (>=1)")
    ap.add_argument("--allow-yaw", action="store_true", help="permitir alterar yaw (padrao: so leitura)")
    ap.add_argument("--run-seconds", type=float, default=None, help="encerrar apos N s (teste)")
    return ap


def main(argv=None):
    a = build_arg_parser().parse_args(argv)
    mode = "read-only" if a.read_only else ("dry-run" if a.dry_run else "normal")
    if a.sim:
        backend = FakeBackend(values={"imu": [0.0, 2.5, 0.0], "secondaryimu": [-0.8, 0.0, 0.0]},
                              can_write=not a.read_only, can_get=not (a.read_only or a.dry_run))
    else:
        backend = DdsBackend(a.iface, a.domain, dry_run=a.dry_run, read_only=a.read_only)
    backend.start()
    policy = OffsetPolicy(window_deg=a.window, hard_abs_deg=a.hard_abs, max_step_deg=a.max_step,
                          min_interval_s=a.min_interval, allow_yaw=a.allow_yaw)
    ctrl = CalibController(backend, a.out_dir, policy=policy, mode=mode, markers_path=a.markers)
    token = os.environ.get("CALIB_WEB_TOKEN") or None
    httpd = serve(ctrl, a.host, a.port, token)
    q = f"/?token={token}" if token else "/"
    print(f"[calib_web] modo={mode}{' SIM' if a.sim else ''} marcadores={ctrl.markers_path}", flush=True)
    print(f"[calib_web] alteracoes de offset -> {ctrl.offset_log_path}", flush=True)
    for ifname, ip in local_ips():
        print(f"[calib_web] abrir: http://{ip}:{a.port}{q}   ({ifname})", flush=True)
    print(f"[calib_web] local: http://127.0.0.1:{a.port}{q}   Ctrl+C para sair", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    sim_thread = None
    if a.sim:
        def tick():
            while not stop.is_set():
                backend.tick_imu()
                time.sleep(0.05)
        sim_thread = threading.Thread(target=tick, daemon=True)
        sim_thread.start()
    try:
        ctrl.initial_read(timeout=1.0)   # unica acao automatica: GET (nunca em read-only/dry-run)
        t_end = None if a.run_seconds is None else time.monotonic() + a.run_seconds
        while not stop.is_set() and (t_end is None or time.monotonic() < t_end):
            ctrl.poll_passive()
            stop.wait(0.5)
    except KeyboardInterrupt:
        print()
    finally:
        httpd.shutdown()
        httpd.server_close()
        for line in ctrl.close():
            print(f"[calib_web] {line}", flush=True)
        backend.close()
        print(f"[calib_web] encerrado. marcadores: {ctrl.markers_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
