#!/usr/bin/env python3
"""Arm path web (G1_29): record the 3D path of each wrist / hand centre and its length.

Independent, READ-ONLY tool. Reads the MEASURED arm q passively from DDS
``rt/lowstate`` (unitree_hg LowState_, motors 15..28 = G1_29 arm, same order as
the teleop) with a subscriber only -- it never publishes anything. Works with
any teleop (Inspire or Dex3) running, or with none (manual motion).

Pipeline: DDS callback (~500-1000 Hz) only copies the 14 q under a short lock;
a sampler thread at ``--rate`` (default 100 Hz) runs FK with the SAME model and
frames as the teleop IK (``teleop/utils/arm_fk.py``) and feeds a live buffer
and the task recorder. On stop the task is written as one folder per task with
``left.csv``, ``right.csv`` and ``summary.json`` (metrics per arm, see
``tools/arm_path_metrics.py``).

    python tools/arm_path_web.py                       # DDS, iface enP8p1s0, :8095
    python tools/arm_path_web.py --point hand          # centre of the hand
    python tools/arm_path_web.py --fake-source         # synthetic q (no robot)

Env: ARM_PATH_TOKEN (optional ?token=...), ARM_PATH_PORT, ARM_PATH_HOST, ARM_PATH_DIR.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
for _p in (REPO, TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arm_path_metrics as apm  # noqa: E402
from pose_compare_web import _git_version, _safe_name, _utc_iso, local_urls  # noqa: E402
from teleop.utils.arm_fk import EE_OFFSET, FRAME_DESC  # noqa: E402  (pinocchio lazily)

PAGE_PATH = os.path.join(TOOLS, "arm_path_web.html")
JS_PATH = os.path.join(TOOLS, "path_plot.js")
DEFAULT_DIR = os.path.expanduser("~/.local/state/xr_teleoperate/arm_paths")  # Dex3 line state dir
DEFAULT_PORT = 8095
DEFAULT_IFACE = "enP8p1s0"
TOPIC = "rt/lowstate"
ARM_MOTORS = tuple(range(15, 29))  # G1_29_JointArmIndex: left 15..21, right 22..28
JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
SIDES = (("left", 0), ("right", 1))
# Centre of the Inspire RH56DFQ hand, in {left,right}_wrist_yaw_link (origin at
# wrist_yaw_joint, x along the forearm/hand). Derived from Unitree's official
# g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf (unitree_ros): hand base mounted at
# +0.0415 m x; the 4 finger MCP joints sit at +0.178 m x (y~0, z~+0.003); the palm
# centre = midpoint between the mount and the MCP line ~ +0.110 m x. The CoM of
# the hand base link is at +0.108 m x. Not measured on the physical robot.
HAND_CENTER_OFFSET = (0.110, 0.0, 0.0)
HAND_CENTER_SOURCE = ("derivado do URDF oficial Unitree g1_29dof_rev_1_0_with_inspire_hand_DFQ "
                      "(montagem +0.0415 m x; articulações MCP dos dedos +0.178 m x; centro = ponto médio "
                      "~+0.110 m x; CoM da base da mão +0.108 m x). Não medido fisicamente.")
# Centre of the Unitree Dex3-1 (3 fingers) hand, per side, in
# {left,right}_wrist_yaw_link. Derived from this repo's assets/g1/g1_body29_hand14.urdf
# (the IK model, Dex3): {left,right}_hand_palm_joint at (+0.0415, +-0.003, 0);
# index_0/middle_0 finger base joints at palm + (0.0777, +-0.0016, +-0.0285)
# -> wrist (+0.1192, +-0.0046, +-0.0285); thumb_0 at palm + 0.0255 x. Palm
# centre ~ midpoint between the mount and the finger-base line, z midway
# between index and middle: (+0.080, +-0.004, 0). The palm-link CoM sits at
# +0.104 m x. ESTIMATED from the URDF; not measured on the physical robot.
DEX3_HAND_CENTER_LEFT = (0.080, 0.004, 0.0)
DEX3_HAND_CENTER_RIGHT = (0.080, -0.004, 0.0)
DEX3_HAND_CENTER_SOURCE = ("Dex3-1 (3 dedos): ESTIMADO do URDF assets/g1/g1_body29_hand14.urdf (palma montada "
                           "+0.0415 m x; base dos dedos indicador/médio +0.119 m x; centro = ponto médio ~+0.080 m x, "
                           "y ±0.004 m; CoM da palma +0.104 m x). Não medido fisicamente.")
# Hand profiles selectable with --hand (default dex3 on this Dex3 line).
HAND_PROFILES = {
    "dex3": (DEX3_HAND_CENTER_LEFT, DEX3_HAND_CENTER_RIGHT, DEX3_HAND_CENTER_SOURCE),
    "inspire": (HAND_CENTER_OFFSET, HAND_CENTER_OFFSET, "Inspire RH56DFQ: " + HAND_CENTER_SOURCE),
}
DEFAULT_HAND = "dex3"
CSV_COLUMNS = (["t_rel_s", "t_utc", "x_raw", "y_raw", "z_raw", "x_filt", "y_filt", "z_filt",
                "dist_cum_raw_m", "dist_cum_filt_m", "lowstate_age_s"] + [f"q_{j}" for j in JOINTS])
_TASK_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[A-Za-z0-9._-]+$")


def parse_xyz(s):
    v = [float(x) for x in str(s).replace(" ", "").split(",")]
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise ValueError(f"esperado x,y,z: {s!r}")
    return tuple(v)


def _r(v, n=6):
    return None if v is None or not math.isfinite(v) else round(float(v), n)


class PointConfig:
    """Which point of each arm is measured: 'wrist' (L_ee/R_ee) or 'hand' (centre)."""

    def __init__(self, kind="wrist", hand_left=HAND_CENTER_OFFSET, hand_right=HAND_CENTER_OFFSET,
                 hand_source=HAND_CENTER_SOURCE):
        if kind not in ("wrist", "hand"):
            raise ValueError("ponto deve ser 'wrist' ou 'hand'")
        self.kind = kind
        self.hand_left = tuple(float(x) for x in hand_left)
        self.hand_right = tuple(float(x) for x in hand_right)
        self.hand_source = hand_source

    def offsets(self):
        if self.kind == "wrist":
            return tuple(EE_OFFSET), tuple(EE_OFFSET)
        return self.hand_left, self.hand_right

    def describe(self):
        ol, orr = self.offsets()
        return {
            "kind": self.kind,
            "label": "punho (L_ee/R_ee do IK)" if self.kind == "wrist" else "centro da mão",
            "offset_left_m": list(ol), "offset_right_m": list(orr),
            "offset_frame": "{left,right}_wrist_yaw_link (origem na wrist_yaw_joint, gira com o punho)",
            "offset_source": ("frame L_ee/R_ee do IK = +0.05 m x da wrist_yaw_joint" if self.kind == "wrist"
                              else self.hand_source),
            "hand_center_offset_left_m": list(self.hand_left),
            "hand_center_offset_right_m": list(self.hand_right),
            "hand_center_source": self.hand_source,
        }


class QState:
    """Latest measured arm q (14) + receive time; written by the DDS callback."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.q = None
        self.t = None
        self.n = 0
        self._times = collections.deque(maxlen=2000)

    def set(self, q14):
        now = self.clock()
        with self.lock:
            self.q = q14
            self.t = now
            self.n += 1
            self._times.append(now)

    def get(self):
        with self.lock:
            return self.q, self.t, self.n

    def rate(self):
        now = self.clock()
        with self.lock:
            ts = [t for t in self._times if now - t <= 1.0]
        return round((len(ts) - 1) / (ts[-1] - ts[0]), 1) if len(ts) > 2 and ts[-1] > ts[0] else 0.0

    def age(self):
        with self.lock:
            return None if self.t is None else self.clock() - self.t


class LowStateSource:
    """Passive DDS subscriber of rt/lowstate. NEVER creates a publisher."""

    def __init__(self, qstate, iface=DEFAULT_IFACE, domain=0):
        self.qstate = qstate
        self.iface = iface
        self.domain = domain
        self.sub = None
        self.n_bad = 0

    def on_msg(self, msg):
        try:
            ms = msg.motor_state
            q = np.fromiter((ms[i].q for i in ARM_MOTORS), dtype=float, count=14)
        except Exception:
            self.n_bad += 1
            return
        self.qstate.set(q)

    def start(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        if self.iface:
            ChannelFactoryInitialize(self.domain, self.iface)
        else:
            ChannelFactoryInitialize(self.domain)
        self.sub = ChannelSubscriber(TOPIC, LowState_)
        self.sub.Init(self.on_msg, 10)
        return self

    def stop(self):
        try:
            if self.sub is not None:
                self.sub.Close()
        except Exception:
            pass

    def describe(self):
        return f"DDS {TOPIC} (unitree_hg LowState_, motores 15..28), domínio {self.domain}, iface {self.iface or 'padrão'} — só subscriber"


class FakeSource:
    """Synthetic measured q at ~500 Hz (tests / demo). Small noise like encoders."""

    def __init__(self, qstate, rate=500.0, noise=0.0005, seed=0):
        self.qstate = qstate
        self.rate = rate
        self.noise = noise
        self._stop = threading.Event()
        self.rng = np.random.default_rng(seed)

    def q_at(self, t):
        q = np.zeros(14)
        q[0] = -0.3 + 0.35 * math.sin(0.8 * t)
        q[1] = 0.25 + 0.1 * math.sin(0.5 * t)
        q[3] = 0.7 + 0.35 * math.sin(1.1 * t)
        q[5] = 0.3 * math.sin(0.9 * t)
        q[7] = -0.3 + 0.3 * math.cos(0.6 * t)
        q[8] = -0.25 - 0.1 * math.cos(0.4 * t)
        q[10] = 0.8 + 0.3 * math.cos(0.9 * t)
        q[13] = 0.4 * math.cos(0.7 * t)
        return q

    def run(self):
        t0 = time.monotonic()
        while not self._stop.is_set():
            q = self.q_at(time.monotonic() - t0)
            self.qstate.set(q + self.rng.normal(0, self.noise, 14) if self.noise else q)
            time.sleep(1.0 / self.rate)

    def start(self):
        threading.Thread(target=self.run, name="fake-lowstate", daemon=True).start()
        return self

    def stop(self):
        self._stop.set()

    def describe(self):
        return "FONTE FALSA (q sintético, teste)"


class Recorder:
    """Task recording: start/stop (or fixed duration with countdown) -> per-arm files."""

    def __init__(self, out_dir, version="unknown", clock=time.monotonic, window_s=apm.DEFAULT_WINDOW_S,
                 min_step_m=apm.DEFAULT_MIN_STEP_M):
        self.out_dir = out_dir
        self.version = version
        self.clock = clock
        self.window_s = window_s
        self.min_step_m = min_step_m
        self.lock = threading.Lock()
        self.state = "idle"
        self.name = ""
        self.duration = None
        self.start_at = None
        self.start_unix = None
        self.meta_extra = {}
        self.rows = []  # (t_mono, t_unix, q14, pl, pr, age)
        self.cum_raw = [0.0, 0.0]
        self._live = None  # (n, metrics) cache for the filtered live length
        self.last_result = None

    def start(self, name, duration=None, countdown=0.0, meta_extra=None):
        with self.lock:
            if self.state != "idle":
                raise RuntimeError("gravação já em andamento")
            if duration not in (None, "", 0):
                duration = float(duration)
                if not (0 < duration <= 7200):
                    raise ValueError("duração deve estar entre 0 e 7200 s")
            else:
                duration = None
            countdown = max(0.0, min(float(countdown or 0.0), 10.0))
            self.name = _safe_name(name)
            self.duration = duration
            self.start_at = self.clock() + countdown
            self.start_unix = time.time() + countdown
            self.meta_extra = dict(meta_extra or {})
            self.rows, self.cum_raw, self._live = [], [0.0, 0.0], None
            self.state = "countdown" if countdown > 0 else "recording"
            return self._status_locked()

    def add(self, t_mono, t_unix, q14, pl, pr, age):
        finish = False
        with self.lock:
            if self.state == "idle":
                return None
            if self.state == "countdown" and t_mono >= self.start_at:
                self.state = "recording"
            if self.state == "recording" and t_mono >= self.start_at:
                if self.duration is not None and t_mono > self.start_at + self.duration:
                    finish = True
                else:
                    if self.rows:
                        prev = self.rows[-1]
                        for k, (a, b) in enumerate(((prev[3], pl), (prev[4], pr))):
                            self.cum_raw[k] += float(np.linalg.norm(np.subtract(b, a)))
                    self.rows.append((t_mono, t_unix, np.asarray(q14, float).copy(),
                                      np.asarray(pl, float).copy(), np.asarray(pr, float).copy(), age))
        return self.stop() if finish else None

    def tick(self):
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
            rows, name, duration, start_unix = self.rows, self.name, self.duration, self.start_unix
            wall = self.clock() - self.start_at
            meta_extra = self.meta_extra
            self.state, self.rows = "idle", []
        result = write_task(self.out_dir, rows, name, start_unix, wall, duration, meta_extra,
                            self.version, self.window_s, self.min_step_m)
        with self.lock:
            self.last_result = result
        return result

    def path_since(self, start_idx=0, max_n=20000):
        with self.lock:
            rows = self.rows[start_idx:]
        step = max(1, int(math.ceil(len(rows) / max_n))) if rows else 1
        return [{"t": round(r[0], 4), "l": [_r(v) for v in r[3]], "r": [_r(v) for v in r[4]]} for r in rows[::step]]

    def _live_metrics_locked(self):
        n = len(self.rows)
        if n < 2:
            return None
        if self._live is not None and n - self._live[0] < 25 and self.clock() - self._live[2] < 0.5:
            return self._live[1]
        rows = self.rows[-60000:] if n > 60000 else self.rows
        t = np.fromiter((r[0] for r in rows), float, len(rows))
        out = {}
        for side, k in SIDES:
            P = np.array([r[3 + k] for r in rows])
            F = apm.smooth(t, P, self.window_s)
            out[side] = {"length_raw_m": round(self.cum_raw[k], 4),
                         "length_filtered_m": round(float(apm.cumulative_length(F, self.min_step_m)[-1]), 4),
                         "net_displacement_m": round(float(np.linalg.norm(F[-1] - F[0])), 4)}
        self._live = (n, out, self.clock())
        return out

    def _status_locked(self):
        now = self.clock()
        st = {"state": self.state, "name": self.name, "duration": self.duration, "n_samples": len(self.rows)}
        if self.state == "countdown":
            st["countdown_left"] = round(max(0.0, self.start_at - now), 2)
        elif self.state == "recording":
            el = max(0.0, now - self.start_at)
            st["elapsed"] = round(el, 2)
            st["t_start"] = round(self.start_at, 4)
            if self.duration is not None:
                st["remaining"] = round(max(0.0, self.duration - el), 2)
            st["live"] = self._live_metrics_locked()
        if self.last_result:
            st["last"] = {k: v for k, v in self.last_result.items() if k != "path"}
        return st

    def status(self):
        with self.lock:
            return self._status_locked()


def write_task(out_dir, rows, name, start_unix, wall_s, duration, meta_extra, version,
               window_s=apm.DEFAULT_WINDOW_S, min_step_m=apm.DEFAULT_MIN_STEP_M):
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(start_unix))
    task_id = f"{stamp}_{name}"
    path = os.path.join(out_dir, task_id)
    n = 1
    while os.path.exists(path):
        n += 1
        task_id = f"{stamp}_{name}_{n}"
        path = os.path.join(out_dir, task_id)
    os.makedirs(path)
    t = np.array([r[0] for r in rows], float)
    t_rel = t - t[0] if len(t) else t
    arms = {}
    for side, k in SIDES:
        P = np.array([r[3 + k] for r in rows], float).reshape(-1, 3)
        F = apm.smooth(t, P, window_s)
        cr = apm.cumulative_length(P)
        cf = apm.cumulative_length(F, min_step_m)
        with open(os.path.join(path, f"{side}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(CSV_COLUMNS)
            for i, r in enumerate(rows):
                q7 = r[2][7 * k:7 * k + 7]
                w.writerow([f"{t_rel[i]:.4f}", _utc_iso(r[1])] + [f"{v:.6f}" for v in P[i]] + [f"{v:.6f}" for v in F[i]]
                           + [f"{cr[i]:.6f}", f"{cf[i]:.6f}", "" if r[5] is None else f"{r[5]:.4f}"]
                           + [f"{v:.6f}" for v in q7])
        arms[side] = apm.path_metrics(t, P, window_s, min_step_m, filtered=F)
        arms[side]["csv"] = f"{side}.csv"
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    ages = [r[5] for r in rows if r[5] is not None]
    summary = {
        "task_id": task_id,
        "name": name,
        "start_utc": _utc_iso(start_unix),
        "end_utc": _utc_iso(time.time()),
        "requested_duration_s": duration,
        "wall_duration_s": round(wall_s, 3),
        "data_duration_s": round(dur, 4),
        "n_samples": len(rows),
        "rate_hz_measured": round((len(rows) - 1) / dur, 2) if dur > 0 else None,
        "lowstate_age_s_max": round(max(ages), 4) if ages else None,
        "frame": FRAME_DESC.split("; point")[0],
        "units": {"position": "m", "length": "m", "speed": "m/s", "q": "rad", "time": "s"},
        "filter": apm.filter_desc(window_s, min_step_m),
        "metrics_definitions": {
            "length_raw_m": "soma de |p[i+1]-p[i]| no sinal bruto (inflada por ruído/tremor)",
            "length_filtered_m": "mesma soma após média móvel centrada (e limiar opcional) — use esta",
            "net_displacement_m": "|p_fim - p_início| (filtrado), deslocamento líquido",
            "length_axis_*_m": "soma de |dx|, |dy|, |dz|",
            "mean_speed_m_s": "length_filtered_m / duração",
            "max_speed_m_s": "máximo de |dp/dt| no sinal filtrado",
        },
        "q_order": "por braço: " + ", ".join(JOINTS) + " (lowstate motores 15..21 esquerdo, 22..28 direito)",
        "columns": CSV_COLUMNS,
        "git_version": version,
        "arms": arms,
    }
    summary.update(meta_extra or {})
    with open(os.path.join(path, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return {"task_id": task_id, "dir": path, "n_samples": len(rows), "name": name,
            "left_length_filtered_m": round(arms["left"]["length_filtered_m"], 4),
            "right_length_filtered_m": round(arms["right"]["length_filtered_m"], 4)}


def read_csv_arm(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    f = lambda r, c: float(r[c]) if r.get(c) not in (None, "") else float("nan")  # noqa: E731
    t = np.array([f(r, "t_rel_s") for r in rows])
    P = np.array([[f(r, "x_raw"), f(r, "y_raw"), f(r, "z_raw")] for r in rows]).reshape(-1, 3)
    F = np.array([[f(r, "x_filt"), f(r, "y_filt"), f(r, "z_filt")] for r in rows]).reshape(-1, 3)
    Q = np.array([[f(r, f"q_{j}") for j in JOINTS] for r in rows]).reshape(-1, 7)
    return t, P, F, Q


def load_task(out_dir, task_id, max_n=20000):
    """Saved task -> JSON-able dict for re-drawing (decimated to <= max_n points)."""
    if not _TASK_RE.match(task_id or ""):
        raise FileNotFoundError(task_id)
    path = os.path.join(out_dir, task_id)
    with open(os.path.join(path, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    out = {"summary": summary}
    for side, _ in SIDES:
        t, P, F, _Q = read_csv_arm(os.path.join(path, f"{side}.csv"))
        step = max(1, int(math.ceil(len(t) / max_n))) if len(t) else 1
        cr = apm.cumulative_length(P)
        cf = apm.cumulative_length(F, summary.get("filter", {}).get("min_step_m", 0.0))
        out[side] = {"t": [round(float(v), 4) for v in t[::step]],
                     "raw": [[_r(v) for v in p] for p in P[::step]],
                     "filt": [[_r(v) for v in p] for p in F[::step]],
                     "cum_raw": [round(float(v), 5) for v in cr[::step]],
                     "cum_filt": [round(float(v), 5) for v in cf[::step]]}
    return out


def list_tasks(out_dir):
    out = []
    try:
        names = sorted(os.listdir(out_dir), reverse=True)
    except FileNotFoundError:
        return out
    for n in names:
        p = os.path.join(out_dir, n, "summary.json")
        if not _TASK_RE.match(n) or not os.path.isfile(p):
            continue
        try:
            s = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        arms = s.get("arms", {})
        out.append({"id": n, "name": s.get("name"), "start_utc": s.get("start_utc"),
                    "n_samples": s.get("n_samples"), "data_duration_s": s.get("data_duration_s"),
                    "rate_hz": s.get("rate_hz_measured"), "point": (s.get("point") or {}).get("label"),
                    "files": [f for f in ("left.csv", "right.csv", "summary.json") if os.path.isfile(os.path.join(out_dir, n, f))],
                    "arms": {side: {k: arms.get(side, {}).get(k) for k in
                                    ("length_raw_m", "length_filtered_m", "net_displacement_m", "mean_speed_m_s",
                                     "max_speed_m_s")} for side, _ in SIDES}})
    return out[:300]


class Sampler:
    """Fixed-rate FK of the latest measured q -> live buffer + recorder."""

    def __init__(self, qstate, points_fn, point_cfg, recorder=None, rate_hz=100.0, buffer_s=60.0,
                 stale_s=0.1, clock=time.monotonic):
        self.qstate = qstate
        self.points_fn = points_fn  # (q14, off_l, off_r) -> (pl, pr)
        self.point_cfg = point_cfg
        self.recorder = recorder
        self.rate_hz = float(rate_hz)
        self.stale_s = stale_s
        self.clock = clock
        self.lock = threading.Lock()
        self.buf = collections.deque(maxlen=int(buffer_s * rate_hz) + 10)
        self.cursor = 0
        self.n_fk_err = 0
        self.n_stale = 0
        self.last_n = -1
        self._times = collections.deque(maxlen=400)
        self._stop = threading.Event()

    def step(self, now=None, t_unix=None):
        now = self.clock() if now is None else now
        q, tq, n = self.qstate.get()
        if self.recorder is not None:
            self.recorder.tick()
        if q is None:
            return None
        age = now - tq
        if age > self.stale_s:
            self.n_stale += 1
            return None
        ol, orr = self.point_cfg.offsets()
        try:
            pl, pr = self.points_fn(q, ol, orr)
        except Exception:
            pl = pr = None
        if pl is None or pr is None:
            self.n_fk_err += 1
            return None
        t_unix = time.time() if t_unix is None else t_unix
        with self.lock:
            self.cursor += 1
            s = {"id": self.cursor, "t": now, "l": [float(v) for v in pl], "r": [float(v) for v in pr]}
            self.buf.append(s)
            self._times.append(now)
        if self.recorder is not None:
            self.recorder.add(now, t_unix, q, pl, pr, age)
        return s

    def run(self):
        dt = 1.0 / self.rate_hz
        nxt = time.monotonic()
        while not self._stop.is_set():
            self.step()
            nxt += dt
            delay = nxt - time.monotonic()
            if delay < -0.5:
                nxt = time.monotonic()
            elif delay > 0:
                self._stop.wait(delay)

    def start(self):
        threading.Thread(target=self.run, name="arm-path-sampler", daemon=True).start()
        return self

    def stop(self):
        self._stop.set()

    def clear(self):
        with self.lock:
            self.buf.clear()

    def samples_since(self, since, max_n=6000):
        with self.lock:
            out = [s for s in self.buf if s["id"] > since][-max_n:]
        return [{"id": s["id"], "t": round(s["t"], 4), "l": [_r(v) for v in s["l"]], "r": [_r(v) for v in s["r"]]}
                for s in out]

    def rate(self):
        now = self.clock()
        with self.lock:
            ts = [t for t in self._times if now - t <= 2.0]
        return round((len(ts) - 1) / (ts[-1] - ts[0]), 1) if len(ts) > 2 and ts[-1] > ts[0] else 0.0

    def status(self):
        age = self.qstate.age()
        _, _, n = self.qstate.get()
        with self.lock:
            cur = self.cursor
        return {"cursor": cur, "rate_hz": self.rate(), "target_hz": self.rate_hz, "n_fk_err": self.n_fk_err,
                "n_stale": self.n_stale, "lowstate_n": n, "lowstate_rate_hz": self.qstate.rate(),
                "lowstate_age_s": None if age is None else round(age, 4)}


def make_handler(sampler, recorder, out_dir, point_cfg, token=None, page_path=PAGE_PATH, info=None):
    info = info or {}

    class Handler(BaseHTTPRequestHandler):
        server_version = "ArmPathWeb/1"

        def log_message(self, fmt, *args):
            pass

        def _auth(self, qs):
            if not token:
                return True
            if (self.headers.get("X-Token") or (qs.get("token") or [None])[0]) == token:
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

        def _status(self):
            return {"stream": sampler.status(), "record": recorder.status(), "point": point_cfg.describe(), **info}

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(u.query)
            if u.path == "/path_plot.js":  # static code, no data: served without token
                try:
                    return self._send(200, open(JS_PATH, "rb").read(), "text/javascript; charset=utf-8")
                except OSError:
                    return self._send(500, {"error": "js ausente"})
            if not self._auth(qs):
                return
            if u.path in ("/", "/index.html"):
                try:
                    return self._send(200, open(page_path, "rb").read(), "text/html; charset=utf-8")
                except OSError:
                    return self._send(500, {"error": "página ausente"})
            if u.path == "/api/status":
                return self._send(200, self._status())
            if u.path == "/api/samples":
                try:
                    since = int((qs.get("since") or ["0"])[0])
                except ValueError:
                    since = 0
                return self._send(200, {"samples": sampler.samples_since(since), **self._status()})
            if u.path == "/api/record/path":
                return self._send(200, {"path": recorder.path_since(0), "record": recorder.status()})
            if u.path == "/api/tasks":
                return self._send(200, {"tasks": list_tasks(out_dir), "dir": out_dir})
            if u.path == "/api/task":
                try:
                    return self._send(200, load_task(out_dir, (qs.get("id") or [""])[0]))
                except (FileNotFoundError, OSError, ValueError) as e:
                    return self._send(404, {"error": f"tarefa não encontrada: {e}"})
            if u.path.startswith("/tasks/"):
                parts = [urllib.parse.unquote(p) for p in u.path[len("/tasks/"):].split("/")]
                if (len(parts) != 2 or not _TASK_RE.match(parts[0])
                        or parts[1] not in ("left.csv", "right.csv", "summary.json")):
                    return self._send(404, {"error": "não encontrado"})
                p = os.path.join(out_dir, parts[0], parts[1])
                if not os.path.isfile(p):
                    return self._send(404, {"error": "não encontrado"})
                ctype = "text/csv; charset=utf-8" if p.endswith(".csv") else "application/json; charset=utf-8"
                return self._send(200, open(p, "rb").read(), ctype,
                                  {"Content-Disposition": f'attachment; filename="{parts[0]}_{parts[1]}"'})
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
                    st = recorder.start(body.get("name", ""), body.get("duration"), body.get("countdown", 0),
                                        meta_extra={"point": point_cfg.describe(), "source": info.get("source"),
                                                    "rate_hz_target": sampler.rate_hz})
                except (RuntimeError, ValueError, TypeError) as e:
                    return self._send(409, {"error": str(e)})
                return self._send(200, {"record": st})
            if u.path == "/api/record/stop":
                res = recorder.stop()
                return self._send(200, {"result": res, "record": recorder.status()})
            if u.path == "/api/config":
                if recorder.status()["state"] != "idle":
                    return self._send(409, {"error": "pare a gravação antes de trocar o ponto"})
                kind = body.get("point")
                if kind not in ("wrist", "hand"):
                    return self._send(400, {"error": "point deve ser 'wrist' ou 'hand'"})
                point_cfg.kind = kind
                sampler.clear()
                return self._send(200, {"point": point_cfg.describe()})
            return self._send(404, {"error": "rota desconhecida"})

    return Handler


def build_points_fn():
    from teleop.utils.arm_fk import G1_29_WristFK
    return G1_29_WristFK().points_xyz


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("ARM_PATH_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("ARM_PATH_PORT", DEFAULT_PORT)))
    ap.add_argument("--iface", default=DEFAULT_IFACE, help="interface DDS (padrão enP8p1s0; '' = padrão do cyclonedds)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--rate", type=float, default=100.0, help="taxa de amostragem/FK (Hz)")
    ap.add_argument("--point", choices=("wrist", "hand"), default="wrist", help="ponto medido inicial")
    ap.add_argument("--hand", choices=sorted(HAND_PROFILES), default=os.environ.get("ARM_PATH_HAND", DEFAULT_HAND),
                    help="mão montada para o 'centro da mão' (padrão dex3 nesta linha; inspire = 0.110,0,0)")
    ap.add_argument("--hand-center-offset", type=parse_xyz, default=None,
                    help="x,y,z (m) no frame wrist_yaw_link, ambas as mãos (sobrepõe --hand)")
    ap.add_argument("--hand-center-offset-left", type=parse_xyz, default=None)
    ap.add_argument("--hand-center-offset-right", type=parse_xyz, default=None)
    ap.add_argument("--filter-window-s", type=float, default=apm.DEFAULT_WINDOW_S)
    ap.add_argument("--min-step-m", type=float, default=apm.DEFAULT_MIN_STEP_M)
    ap.add_argument("--buffer-s", type=float, default=60.0, help="janela ao vivo fora da gravação")
    ap.add_argument("--out-dir", default=os.environ.get("ARM_PATH_DIR", DEFAULT_DIR))
    ap.add_argument("--fake-source", action="store_true", help="q sintético (sem robô/DDS)")
    ap.add_argument("--run-seconds", type=float, default=0.0, help="sair após N s (teste)")
    args = ap.parse_args(argv)
    token = os.environ.get("ARM_PATH_TOKEN") or None

    prof_l, prof_r, prof_src = HAND_PROFILES[args.hand]
    hl = args.hand_center_offset_left or args.hand_center_offset or prof_l
    hr = args.hand_center_offset_right or args.hand_center_offset or prof_r
    custom = any((args.hand_center_offset, args.hand_center_offset_left, args.hand_center_offset_right))
    point_cfg = PointConfig(args.point, hl, hr, "definido pelo usuário (--hand-center-offset)" if custom else prof_src)
    print(f"[arm_path] centro da mão: perfil {args.hand} E {hl} D {hr} (m, wrist_yaw_link)", flush=True)
    points_fn = build_points_fn()
    print("[arm_path] FK G1_29 carregada (mesmo modelo/frames do IK).", flush=True)
    qstate = QState()
    src = FakeSource(qstate) if args.fake_source else LowStateSource(qstate, args.iface, args.domain)
    src.start()
    print(f"[arm_path] fonte: {src.describe()}", flush=True)
    version = _git_version()
    recorder = Recorder(args.out_dir, version=version, window_s=args.filter_window_s, min_step_m=args.min_step_m)
    sampler = Sampler(qstate, points_fn, point_cfg, recorder, rate_hz=args.rate, buffer_s=args.buffer_s).start()
    info = {"version": version, "frame": FRAME_DESC.split("; point")[0], "source": src.describe(),
            "fake": bool(args.fake_source), "filter": apm.filter_desc(args.filter_window_s, args.min_step_m),
            "buffer_s": args.buffer_s}
    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(sampler, recorder, args.out_dir, point_cfg, token=token, info=info))
    httpd.daemon_threads = True
    print(f"[arm_path] tarefas em {args.out_dir} | {args.rate:.0f} Hz | versão {version}", flush=True)
    for label, url in local_urls(httpd.server_address[1], token):
        print(f"[arm_path] página ({label}): {url}", flush=True)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()
    try:
        if args.run_seconds > 0:
            time.sleep(args.run_seconds)
        else:
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        if recorder.status()["state"] != "idle":
            print(f"[arm_path] gravação em andamento salva: {recorder.stop()}", flush=True)
        st = sampler.status()
        print(f"[arm_path] lowstate n={st['lowstate_n']} taxa={st['lowstate_rate_hz']} Hz, "
              f"amostras={st['cursor']} ({st['rate_hz']} Hz), fk_err={st['n_fk_err']}", flush=True)
        httpd.shutdown()
        sampler.stop()
        src.stop()
        print("[arm_path] encerrado.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
