#!/usr/bin/env python3
"""Pose compare web (Inspire teleop): robot wrist (FK) vs operator wrist (IK target).

Read-only side tool. Receives the teleop's opt-in UDP side channel
(``XR_POSE_STREAM=1``, see ``teleop/utils/pose_stream.py``), computes the robot
wrist position by forward kinematics of the commanded (sol_q) and measured
(lowstate) arm q with the SAME model/frames as the teleop IK
(``teleop/utils/arm_fk.py``), keeps a bounded in-memory buffer and serves a
PT-BR page with live X/Y/Z-vs-time plots and a "Salvar tarefa" recorder.

It publishes nothing on DDS and never talks to the robot.

    python tools/pose_compare_web.py                # http://0.0.0.0:8093
    python tools/pose_compare_web.py --fake-source  # synthetic UDP data (test)

Env: POSE_WEB_PORT, POSE_WEB_HOST, POSE_WEB_TOKEN (optional ?token=...),
     XR_POSE_STREAM_PORT (UDP, default 47555), POSE_WEB_TASK_DIR.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from teleop.utils import pose_stream as ps  # noqa: E402

PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_compare_web.html")
DEFAULT_TASK_DIR = os.path.expanduser("~/.local/state/xr_teleoperate/tasks")  # Dex3 line state dir
DEFAULT_HTTP_PORT = 8093
NAN = float("nan")
from teleop.utils.arm_fk import FRAME_DESC  # noqa: E402  (pinocchio imported lazily)

CSV_COLUMNS = (
    ["t_mono_s", "t_rel_s", "t_utc", "seq", "tracking", "fresh"]
    + [f"hand_{s}_{a}" for s in "LR" for a in "xyz"]
    + [f"robot_cmd_{s}_{a}" for s in "LR" for a in "xyz"]
    + [f"robot_meas_{s}_{a}" for s in "LR" for a in "xyz"]
    + [f"q_cmd_{i}" for i in range(ps.N_ARM)]
    + [f"q_meas_{i}" for i in range(ps.N_ARM)]
    # torso lean (XPS2 packets; empty for XPS1 / feature off). rad.
    + ["lean_active", "lean_target_pitch", "lean_target_roll", "lean_cmd_pitch", "lean_cmd_roll"]
    + [f"waist_{k}_{j}" for k in ("cmd", "meas") for j in ("yaw", "roll", "pitch")]
)


def _git_version():
    try:
        out = subprocess.run(["git", "-C", REPO, "describe", "--always", "--dirty", "--abbrev=12"],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _utc_iso(t_unix):
    return _dt.datetime.fromtimestamp(t_unix, _dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_name(name):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())[:60].strip("._")
    return s or "tarefa"


def _f(v):
    """JSON-safe float (NaN/inf -> None)."""
    try:
        v = float(v)
    except Exception:
        return None
    return round(v, 5) if math.isfinite(v) else None


def _xyz(v):
    return [None, None, None] if v is None else [_f(x) for x in v]


class Recorder:
    """'Salvar tarefa': collects samples between start and stop, writes CSV + JSON."""

    def __init__(self, task_dir, version="unknown", clock=time.monotonic):
        self.task_dir = task_dir
        self.version = version
        self.clock = clock
        self.lock = threading.Lock()
        self.state = "idle"  # idle | countdown | recording
        self.name = ""
        self.mode = None
        self.duration = None
        self.start_at = None
        self.samples = []
        self.last_result = None

    def start(self, name, mode="manual", duration=None, countdown=0.0):
        with self.lock:
            if self.state != "idle":
                raise RuntimeError("gravação já em andamento")
            if mode not in ("manual", "duration"):
                raise ValueError("modo inválido")
            if mode == "duration":
                duration = float(duration)
                if not (0 < duration <= 3600):
                    raise ValueError("duração deve estar entre 0 e 3600 s")
            countdown = max(0.0, min(float(countdown or 0.0), 10.0))
            self.name = _safe_name(name)
            self.mode = mode
            self.duration = duration if mode == "duration" else None
            self.start_at = self.clock() + countdown
            self.samples = []
            self.state = "countdown" if countdown > 0 else "recording"
            self.start_unix = time.time() + countdown
            return self.status_locked()

    def add(self, rec):
        """Called by the receiver for every sample (rec['recv_mono'] = server clock)."""
        finish = False
        with self.lock:
            if self.state == "idle":
                return None
            now = rec["recv_mono"]
            if self.state == "countdown" and now >= self.start_at:
                self.state = "recording"
            if self.state == "recording" and now >= self.start_at:
                if self.duration is not None and now > self.start_at + self.duration:
                    finish = True
                else:
                    self.samples.append(rec)
        return self.stop() if finish else None

    def tick(self):
        """Advance countdown / end a fixed-duration task even with no samples."""
        finish = False
        with self.lock:
            now = self.clock()
            if self.state == "countdown" and now >= self.start_at:
                self.state = "recording"
            if self.state == "recording" and self.duration is not None and now > self.start_at + self.duration:
                finish = True
        return self.stop() if finish else None

    def stop(self):
        with self.lock:
            if self.state == "idle":
                return self.last_result
            samples, name, mode, duration = self.samples, self.name, self.mode, self.duration
            start_unix = self.start_unix
            end_mono = self.clock()
            start_mono = self.start_at
            self.state, self.samples = "idle", []
        result = self._write(samples, name, mode, duration, start_unix, start_mono, end_mono)
        with self.lock:
            self.last_result = result
        return result

    def _write(self, samples, name, mode, duration, start_unix, start_mono, end_mono):
        os.makedirs(self.task_dir, exist_ok=True)
        stamp = _dt.datetime.fromtimestamp(start_unix, _dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = os.path.join(self.task_dir, f"{stamp}_{name}")
        n = 1
        while os.path.exists(base + ".csv") or os.path.exists(base + ".json"):
            n += 1
            base = os.path.join(self.task_dir, f"{stamp}_{name}_{n}")
        t0 = samples[0]["t_mono"] if samples else None
        with open(base + ".csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(CSV_COLUMNS)
            for r in samples:
                row = [f"{r['t_mono']:.6f}", f"{r['t_mono'] - t0:.6f}", _utc_iso(r["t_unix"]), r["seq"],
                       int(r["tracking"]), int(r["fresh"])]
                for key in ("hand_l", "hand_r", "rob_cmd_l", "rob_cmd_r", "rob_meas_l", "rob_meas_r"):
                    v = r.get(key)
                    row += ["" if v is None or not math.isfinite(x) else f"{x:.6f}" for x in (v if v is not None else (NAN,) * 3)]
                for key in ("q_cmd", "q_meas"):
                    row += ["" if not math.isfinite(x) else f"{x:.6f}" for x in r[key]]
                row.append(int(bool(r.get("lean_active"))) if r.get("lean_target") is not None else "")
                for key, n in (("lean_target", 2), ("lean_cmd", 2), ("waist_cmd", 3), ("waist_meas", 3)):
                    v = r.get(key)
                    row += [""] * n if v is None else ["" if not math.isfinite(x) else f"{x:.6f}" for x in v]
                w.writerow(row)
        dur = (samples[-1]["t_mono"] - samples[0]["t_mono"]) if len(samples) > 1 else 0.0
        meta = {
            "name": name,
            "mode": mode,
            "requested_duration_s": duration,
            "start_utc": _utc_iso(start_unix),
            "end_utc": _utc_iso(time.time()),
            "wall_duration_s": round(end_mono - start_mono, 3),
            "n_samples": len(samples),
            "data_duration_s": round(dur, 4),
            "rate_hz": round((len(samples) - 1) / dur, 2) if dur > 0 else None,
            "tracking_fraction": round(sum(r["tracking"] for r in samples) / len(samples), 4) if samples else None,
            "frame": FRAME_DESC,
            "units": {"position": "m", "q": "rad", "time": "s"},
            "hand": "operator wrist = IK target (tele_data.*_wrist_pose translation, IK frame; retargeted to the torso frame when G1_TORSO_LEAN is active)",
            "robot_cmd": "FK(sol_q commanded by IK)",
            "robot_meas": "FK(lowstate q measured)",
            "q_order": "G1_29 arm: left 7 (shoulder pitch,roll,yaw, elbow, wrist roll,pitch,yaw) then right 7",
            "git_version": self.version,
            "csv": os.path.basename(base + ".csv"),
            "columns": CSV_COLUMNS,
        }
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        return {"csv": base + ".csv", "json": base + ".json", "n_samples": len(samples), "name": name}

    def status_locked(self):
        now = self.clock()
        st = {"state": self.state, "name": self.name, "mode": self.mode, "duration": self.duration,
              "n_samples": len(self.samples)}
        if self.state == "countdown":
            st["countdown_left"] = round(max(0.0, self.start_at - now), 2)
        elif self.state == "recording":
            el = max(0.0, now - self.start_at)
            st["elapsed"] = round(el, 2)
            if self.duration is not None:
                st["remaining"] = round(max(0.0, self.duration - el), 2)
        if self.last_result:
            st["last"] = {k: (os.path.basename(v) if k in ("csv", "json") else v) for k, v in self.last_result.items()}
        return st

    def status(self):
        with self.lock:
            return self.status_locked()


def _deg(v):
    try:
        return [None if not math.isfinite(float(x)) else round(math.degrees(float(x)), 2) for x in v]
    except Exception:
        return None


def _lean_status(s):
    """Torso lean block for /api/status (degrees); None for XPS1 packets."""
    if not s or s.get("lean_target") is None:
        return None
    wc, wm = s.get("waist_cmd"), s.get("waist_meas")
    meas_rel = None
    if wc is not None and wm is not None and s.get("lean_cmd") is not None:
        # measured lean relative to neutral = (meas - cmd) + commanded lean
        try:
            meas_rel = [float(wm[2]) - float(wc[2]) + float(s["lean_cmd"][0]),
                        float(wm[1]) - float(wc[1]) + float(s["lean_cmd"][1])]
        except Exception:
            meas_rel = None
    return {"active": bool(s.get("lean_active")),
            "target_deg": _deg(s["lean_target"]), "cmd_deg": _deg(s["lean_cmd"]),
            "meas_deg": _deg(meas_rel) if meas_rel is not None else None,
            "waist_cmd_deg": _deg(wc) if wc is not None else None,
            "waist_meas_deg": _deg(wm) if wm is not None else None}


class PoseHub:
    """UDP receiver + FK + bounded buffer + recorder."""

    def __init__(self, fk=None, buffer_s=120.0, rate_hint=50.0, recorder=None, clock=time.monotonic,
                 skeleton=None, skeleton_hz=20.0):
        self.fk = fk
        self.skeleton = skeleton  # callable(q_meas) -> {"l","r","b"} or None (3D panel)
        self.skeleton_dt = 1.0 / skeleton_hz if skeleton_hz and skeleton_hz > 0 else 0.0
        self._last_skel = None  # next due time (server clock) for a skeleton sample
        self.clock = clock
        self.buf = collections.deque(maxlen=int(buffer_s * rate_hint * 1.5) + 10)
        self.lock = threading.Lock()
        self.cursor = 0
        self.recorder = recorder
        self.n_rx = 0
        self.n_bad = 0
        self.fk_errors = 0
        self.n_skel = 0
        self.last_rx = None
        self.last_seq = None
        self.lost = 0
        self._rx_times = collections.deque(maxlen=200)
        self._stop = threading.Event()
        self.sock = None

    def ingest(self, data):
        s = ps.unpack_sample(data)
        now = self.clock()
        if s is None:
            self.n_bad += 1
            return None
        cl = cr = ml = mr = None
        if self.fk is not None:
            try:
                cl, cr = self.fk(s["q_cmd"])
                ml, mr = self.fk(s["q_meas"])
            except Exception:
                self.fk_errors += 1
        s.update(rob_cmd_l=None if cl is None else [float(x) for x in cl],
                 rob_cmd_r=None if cr is None else [float(x) for x in cr],
                 rob_meas_l=None if ml is None else [float(x) for x in ml],
                 rob_meas_r=None if mr is None else [float(x) for x in mr],
                 recv_mono=now)
        if self.skeleton is not None and (self._last_skel is None or now >= self._last_skel):
            try:
                sk = self.skeleton(s["q_meas"])
            except Exception:
                sk = None
                self.fk_errors += 1
            if sk is not None:
                s["skel"] = sk
                # next due time (fixed cadence, no drift; resync after gaps)
                nxt = (now if self._last_skel is None else self._last_skel) + self.skeleton_dt
                self._last_skel = nxt if nxt > now else now + self.skeleton_dt
                self.n_skel += 1
        with self.lock:
            if self.last_seq is not None and s["seq"] > self.last_seq + 1:
                self.lost += s["seq"] - self.last_seq - 1
            self.last_seq = s["seq"]
            self.cursor += 1
            s["id"] = self.cursor
            self.buf.append(s)
            self.n_rx += 1
            self.last_rx = now
            self._rx_times.append(now)
        if self.recorder is not None:
            self.recorder.add(s)
        return s

    def bind(self, host="127.0.0.1", port=ps.DEFAULT_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)
        return self.sock.getsockname()[1]

    def run_udp(self):
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                data = None
            except OSError:
                if self._stop.is_set():
                    break
                time.sleep(0.05)
                continue
            if data is not None:
                self.ingest(data)
            if self.recorder is not None:
                self.recorder.tick()

    def stop(self):
        self._stop.set()
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass

    def rate(self):
        with self.lock:
            ts = [t for t in self._rx_times if self.clock() - t <= 2.0]
        return round((len(ts) - 1) / (ts[-1] - ts[0]), 1) if len(ts) > 2 and ts[-1] > ts[0] else 0.0

    def samples_since(self, since, max_n=6000):
        with self.lock:
            out = [s for s in self.buf if s["id"] > since]
        out = out[-max_n:]
        res = []
        for s in out:
            d = {
                "id": s["id"], "t": round(s["t_mono"], 4), "u": round(s["t_unix"], 3),
                "tr": int(s["tracking"]), "fr": int(s["fresh"]),
                "hl": _xyz(s["hand_l"]), "hr": _xyz(s["hand_r"]),
                "cl": _xyz(s["rob_cmd_l"]), "cr": _xyz(s["rob_cmd_r"]),
                "ml": _xyz(s["rob_meas_l"]), "mr": _xyz(s["rob_meas_r"]),
            }
            sk = s.get("skel")
            if sk is not None:  # ~20 Hz: arm skeleton from q measured (3D panel)
                d["sk"] = {k: [_xyz(p) for p in v] for k, v in sk.items()}
            res.append(d)
        return res

    def status(self):
        with self.lock:
            last = self.buf[-1] if self.buf else None
            age = None if self.last_rx is None else round(self.clock() - self.last_rx, 3)
            st = {"n_rx": self.n_rx, "n_bad": self.n_bad, "lost": self.lost, "fk_errors": self.fk_errors,
                  "fk": self.fk is not None,
                  "skeleton": self.skeleton is not None, "n_skel": self.n_skel, "age_s": age, "cursor": self.cursor,
                  "tracking": bool(last["tracking"]) if last else False,
                  "fresh": bool(last["fresh"]) if last else False,
                  "t_teleop": round(last["t_mono"], 4) if last else None,
                  "lean": _lean_status(last)}
        st["rate_hz"] = self.rate()
        return st


def list_tasks(task_dir):
    out = []
    try:
        names = sorted(os.listdir(task_dir), reverse=True)
    except FileNotFoundError:
        return out
    for n in names:
        if not n.endswith(".json"):
            continue
        p = os.path.join(task_dir, n)
        try:
            meta = json.load(open(p, encoding="utf-8"))
        except Exception:
            meta = {}
        stem = n[:-5]
        out.append({"stem": stem, "json": n, "csv": stem + ".csv" if os.path.exists(os.path.join(task_dir, stem + ".csv")) else None,
                    "name": meta.get("name"), "start_utc": meta.get("start_utc"), "n_samples": meta.get("n_samples"),
                    "data_duration_s": meta.get("data_duration_s"), "rate_hz": meta.get("rate_hz")})
    return out[:200]


def make_handler(hub, recorder, task_dir, token=None, page_path=PAGE_PATH, info=None):
    info = info or {}

    class Handler(BaseHTTPRequestHandler):
        server_version = "PoseCompareWeb/1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _auth(self, qs):
            if not token:
                return True
            got = self.headers.get("X-Token") or (qs.get("token") or [None])[0]
            if got == token:
                return True
            self._send(403, {"error": "token inválido (use ?token=...)"})
            return False

        def _send(self, code, obj, ctype="application/json; charset=utf-8", extra=None):
            body = obj if isinstance(obj, (bytes, bytearray)) else json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(u.query)
            if not self._auth(qs):
                return
            if u.path in ("/", "/index.html"):
                try:
                    body = open(page_path, "rb").read()
                except OSError:
                    return self._send(500, {"error": "página ausente"})
                return self._send(200, body, "text/html; charset=utf-8")
            if u.path == "/api/status":
                return self._send(200, {"stream": hub.status(), "record": recorder.status(), **info})
            if u.path == "/api/samples":
                try:
                    since = int((qs.get("since") or ["0"])[0])
                except ValueError:
                    since = 0
                return self._send(200, {"samples": hub.samples_since(since), "stream": hub.status(),
                                        "record": recorder.status()})
            if u.path == "/api/tasks":
                return self._send(200, {"tasks": list_tasks(task_dir), "dir": task_dir})
            if u.path.startswith("/tasks/"):
                fn = os.path.basename(urllib.parse.unquote(u.path[len("/tasks/"):]))
                p = os.path.join(task_dir, fn)
                if not fn or not fn.endswith((".csv", ".json")) or not os.path.isfile(p):
                    return self._send(404, {"error": "não encontrado"})
                ctype = "text/csv; charset=utf-8" if fn.endswith(".csv") else "application/json; charset=utf-8"
                return self._send(200, open(p, "rb").read(), ctype,
                                  {"Content-Disposition": f'attachment; filename="{fn}"'})
            return self._send(404, {"error": "rota desconhecida"})

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(u.query)
            if not self._auth(qs):
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(min(n, 65536)) or b"{}") if n else {}
            except Exception:
                return self._send(400, {"error": "JSON inválido"})
            if u.path == "/api/record/start":
                try:
                    st = recorder.start(body.get("name", ""), body.get("mode", "manual"),
                                        body.get("duration"), body.get("countdown", 0))
                except (RuntimeError, ValueError, TypeError) as e:
                    return self._send(409, {"error": str(e)})
                return self._send(200, {"record": st})
            if u.path == "/api/record/stop":
                res = recorder.stop()
                return self._send(200, {"result": None if res is None else
                                        {k: (os.path.basename(v) if k in ("csv", "json") else v) for k, v in res.items()},
                                        "record": recorder.status()})
            return self._send(404, {"error": "rota desconhecida"})

    return Handler


def local_urls(port, token=None):
    urls = []
    q = f"/?token={urllib.parse.quote(token)}" if token else "/"
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=3).stdout
        for line in out.splitlines():
            parts = line.split()
            ifname, addr = parts[1], parts[3].split("/")[0]
            if addr.startswith("127."):
                continue
            label = "Tailscale" if ifname.startswith("tailscale") or addr.startswith("100.") else (
                "Wi-Fi" if ifname.startswith("wl") else ifname)
            urls.append((label, f"http://{addr}:{port}{q}"))
    except Exception:
        pass
    urls.append(("local", f"http://127.0.0.1:{port}{q}"))
    return urls


def fake_source(port, stop, rate=50.0, host="127.0.0.1"):
    """Synthetic teleop: a known q trajectory; hand target = FK-ish circle; for tests."""
    import numpy as np
    tx = ps.PoseStreamSender(host=host, port=port, rate_hz=rate)
    t0 = time.monotonic()
    while not stop.is_set():
        t = time.monotonic() - t0
        q = np.zeros(14)
        q[0] = -0.4 + 0.3 * math.sin(t)
        q[3] = 0.6 + 0.2 * math.sin(0.7 * t)
        q[7] = -0.4 + 0.3 * math.cos(t)
        q[10] = 0.6 + 0.2 * math.cos(0.7 * t)
        qm = q - 0.03
        L = np.eye(4); L[:3, 3] = (0.30 + 0.05 * math.sin(t), 0.20, 0.10 + 0.05 * math.cos(t))
        R = np.eye(4); R[:3, 3] = (0.30 + 0.05 * math.cos(t), -0.20, 0.10 + 0.05 * math.sin(t))
        lp, lr = math.radians(6.0) * math.sin(0.3 * t), math.radians(3.0) * math.cos(0.3 * t)
        wc = (0.0, lr, lp)
        lean = {"active": t > 1.0, "target": (lp, lr), "cmd": (lp, lr), "waist_cmd": wc,
                "waist_meas": (0.0, lr - 0.005, lp - 0.01)}
        tx.maybe_send(t > 1.0, L, R, q, qm, lean=lean)
        time.sleep(0.5 / rate)
    tx.close()


def pid_alive(pid):
    """True while ``pid`` exists (signal 0); EPERM still means alive."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def build_fk_model():
    from teleop.utils.arm_fk import G1_29_WristFK
    return G1_29_WristFK()


def build_fk():
    return build_fk_model().wrist_xyz


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("POSE_WEB_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("POSE_WEB_PORT", DEFAULT_HTTP_PORT)))
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=int(os.environ.get("XR_POSE_STREAM_PORT", ps.DEFAULT_PORT)))
    ap.add_argument("--buffer-s", type=float, default=120.0)
    ap.add_argument("--task-dir", default=os.environ.get("POSE_WEB_TASK_DIR", DEFAULT_TASK_DIR))
    ap.add_argument("--no-fk", action="store_true", help="não carregar pinocchio (só mão)")
    ap.add_argument("--skeleton-hz", type=float, default=20.0, help="taxa do esqueleto 3D (FK do q medido)")
    ap.add_argument("--fake-source", action="store_true", help="gerador UDP sintético (teste)")
    ap.add_argument("--run-seconds", type=float, default=0.0, help="sair após N s (teste)")
    ap.add_argument("--exit-with-pid", type=int, default=0,
                    help="sair (salvando gravação em andamento) quando este PID terminar (launcher)")
    args = ap.parse_args(argv)
    token = os.environ.get("POSE_WEB_TOKEN") or None

    fk = skel = None
    if not args.no_fk:
        try:
            model = build_fk_model()
            fk, skel = model.wrist_xyz, model.skeleton
            print("[pose_web] FK G1_29 carregada (mesmo modelo/frames L_ee/R_ee do IK).", flush=True)
        except Exception as e:  # keep serving the hand data
            print(f"[pose_web] AVISO: FK indisponível ({e!r}); só a mão será plotada.", flush=True)
    version = _git_version()
    recorder = Recorder(args.task_dir, version=version)
    hub = PoseHub(fk=fk, buffer_s=args.buffer_s, recorder=recorder, skeleton=skel,
                  skeleton_hz=args.skeleton_hz)
    udp_port = hub.bind(args.udp_host, args.udp_port)
    threading.Thread(target=hub.run_udp, name="pose-udp", daemon=True).start()
    stop = threading.Event()
    if args.fake_source:
        threading.Thread(target=fake_source, args=(udp_port, stop), daemon=True).start()
        print("[pose_web] fonte UDP FALSA ativa (dados sintéticos).", flush=True)
    info = {"version": version, "frame": FRAME_DESC, "udp_port": udp_port, "fake": bool(args.fake_source),
            "buffer_s": args.buffer_s}
    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(hub, recorder, args.task_dir, token=token, info=info))
    httpd.daemon_threads = True
    print(f"[pose_web] UDP 127.0.0.1:{udp_port} | tarefas em {args.task_dir} | versão {version}", flush=True)
    for label, url in local_urls(httpd.server_address[1], token):
        print(f"[pose_web] página ({label}): {url}", flush=True)
    srv = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
    srv.start()
    try:
        deadline = time.monotonic() + args.run_seconds if args.run_seconds > 0 else None
        while deadline is None or time.monotonic() < deadline:
            if args.exit_with_pid > 0 and not pid_alive(args.exit_with_pid):
                print(f"[pose_web] launcher PID {args.exit_with_pid} terminou; encerrando.", flush=True)
                break
            time.sleep(0.5 if deadline is None else max(0.0, min(0.5, deadline - time.monotonic())))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if recorder.status()["state"] != "idle":
            res = recorder.stop()
            print(f"[pose_web] gravação em andamento salva: {res}", flush=True)
        httpd.shutdown()
        hub.stop()
        print("[pose_web] encerrado.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
