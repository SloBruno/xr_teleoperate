"""Side-channel loop diagnostics (read-only, never touches control).

Everything called from the control loop only does in-memory arithmetic
(perf_counter + dict/list updates).  File/JSON I/O happens in the existing async
sinks; /proc reads happen in a separate sampler thread.  Every public method
swallows its own errors into ``error_count``.
"""
from __future__ import annotations

import gc
import os
import threading
import time
from collections import deque
from typing import Callable, Mapping

STAGES = (
    "controller", "camera", "render", "state_read", "arm_cycle",
    "hand", "locomotion", "telemetry", "record", "sleep",
)
GAP_BUCKETS = (("lt50", 50.0), ("50_100", 100.0), ("100_200", 200.0), ("200_500", 500.0), ("gt500", None))
STALE_AGE_S = 0.25  # matches quest_safety freshness limit
QUEST_PORT = 8012


def _pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))], 3)


def _summ(values):
    return {"p50": _pct(values, 0.5), "p95": _pct(values, 0.95), "max": _pct(values, 1.0)}


def classify_ip(ip: str) -> str:
    try:
        parts = [int(p) for p in ip.split(".")]
        if len(parts) != 4:
            return "other"
    except ValueError:
        return "other"
    a, b = parts[0], parts[1]
    if a == 100 and 64 <= b <= 127:
        return "tailscale"
    if a == 10 or (a == 192 and b == 168) or (a == 172 and 16 <= b <= 31):
        return "lan"
    if a == 127:
        return "loopback"
    return "other"


class CycleTimer:
    """Per-cycle stage timer: begin(); mark(stage)...; end()."""

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._t0 = self._last = 0.0
        self.stages: dict[str, float] = {}
        self.total_ms = 0.0
        self.last: dict = {}

    def begin(self):
        self._t0 = self._last = self._clock()
        self.stages = {}

    def mark(self, stage: str):
        now = self._clock()
        self.stages[stage] = self.stages.get(stage, 0.0) + (now - self._last) * 1000.0
        self._last = now

    def end(self) -> dict:
        self.total_ms = (self._clock() - self._t0) * 1000.0
        self.last = {k: round(v, 3) for k, v in self.stages.items()}
        self.last["total"] = round(self.total_ms, 3)
        return self.last

    def dominant(self):
        best, val = None, -1.0
        for k, v in self.stages.items():
            if k != "sleep" and v > val:
                best, val = k, v
        return best


class ControllerFeedTracker:
    """Unique-sample feed stats + stale bursts, in memory only."""

    def __init__(self, stale_age_s=STALE_AGE_S):
        self.stale_age_s = stale_age_s
        self.last_ts = None
        self.last_arrival = None
        self.unique_total = 0
        self.repeated_total = 0
        self.out_of_order_total = 0
        self.stale_cycles_total = 0
        self.unique_dt = []      # ms between unique sample timestamps (window)
        self.arrival_dt = []     # ms between arrival of unique samples (window)
        self.ages = []           # ms (window)
        self.hist = {k: 0 for k, _ in GAP_BUCKETS}
        self.bursts = deque(maxlen=64)
        self._pending = []       # bursts not yet emitted
        self._burst = None       # {start, cycles, stage_ms}
        self.unique_in_window = 0

    def observe(self, sample_ts, now, stages: Mapping[str, float] | None = None):
        ts = float(sample_ts) if sample_ts else 0.0
        age = now - ts if ts > 0 else float("inf")
        if ts > 0 and self.last_ts is not None and ts == self.last_ts:
            self.repeated_total += 1
        elif ts > 0:
            if self.last_ts is not None and ts < self.last_ts:
                self.out_of_order_total += 1
            else:
                if self.last_ts is not None:
                    gap = (ts - self.last_ts) * 1000.0
                    self.unique_dt.append(gap)
                    for name, limit in GAP_BUCKETS:
                        if limit is None or gap < limit:
                            self.hist[name] += 1
                            break
                if self.last_arrival is not None:
                    self.arrival_dt.append((now - self.last_arrival) * 1000.0)
                self.last_arrival = now
                self.unique_total += 1
                self.unique_in_window += 1
                self.last_ts = ts
        if age != float("inf"):
            self.ages.append(age * 1000.0)
        stale = age > self.stale_age_s
        if stale:
            self.stale_cycles_total += 1
            if self._burst is None:
                self._burst = {"start": now, "cycles": 0, "stage_ms": {}}
            self._burst["cycles"] += 1
            self._burst["last"] = now
            for k, v in (stages or {}).items():
                if k not in ("sleep", "total"):
                    self._burst["stage_ms"][k] = self._burst["stage_ms"].get(k, 0.0) + v
        elif self._burst is not None:
            self._close_burst(now)

    def _close_burst(self, now):
        b, self._burst = self._burst, None
        sm = b["stage_ms"]
        dom = max(sm, key=sm.get) if sm else None
        rec = {"start_mono": round(b["start"], 4), "duration_ms": round((now - b["start"]) * 1000.0, 1),
               "cycles": b["cycles"], "dominant_stage": dom}
        self.bursts.append(rec)
        self._pending.append(rec)

    def snapshot_and_reset(self, window_s):
        snap = {
            "unique_hz": round(self.unique_in_window / window_s, 2) if window_s > 0 else None,
            "gap_ms": _summ(self.unique_dt),
            "arrival_dt_ms": _summ(self.arrival_dt),
            "age_ms": _summ(self.ages),
            "gap_hist": dict(self.hist),
            "unique_total": self.unique_total,
            "repeated_total": self.repeated_total,
            "out_of_order_total": self.out_of_order_total,
            "stale_cycles_total": self.stale_cycles_total,
            "stale_bursts": self._pending[:8],
            "stale_bursts_dropped": max(0, len(self._pending) - 8),
            "burst_in_progress": self._burst is not None,
        }
        self._pending = []
        self.unique_dt, self.arrival_dt, self.ages = [], [], []
        self.unique_in_window = 0
        return snap


class VideoCounter:
    def __init__(self):
        self.frames_total = 0
        self.bytes_total = 0
        self._w_frames = 0
        self._w_bytes = 0
        self.last_shape = None

    def record(self, frame):
        self.frames_total += 1
        self._w_frames += 1
        n = int(getattr(frame, "nbytes", 0) or 0)  # raw pixels handed to render_to_xr
        self.bytes_total += n
        self._w_bytes += n
        self.last_shape = tuple(getattr(frame, "shape", ()) or ())

    def snapshot_and_reset(self, window_s):
        w = window_s if window_s > 0 else 1.0
        snap = {"fps": round(self._w_frames / w, 2), "raw_bytes_per_s": int(self._w_bytes / w),
                "frames_total": self.frames_total, "raw_bytes_total": self.bytes_total,
                "last_shape": list(self.last_shape) if self.last_shape else None}
        self._w_frames = self._w_bytes = 0
        return snap


class GcWatcher:
    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._t0 = 0.0
        self.collections = [0, 0, 0]
        self.win_count = 0
        self.win_ms = 0.0
        self.win_max_ms = 0.0
        self.installed = False

    def _cb(self, phase, info):
        try:
            if phase == "start":
                self._t0 = self._clock()
            else:
                d = (self._clock() - self._t0) * 1000.0
                self.collections[int(info.get("generation", 0))] += 1
                self.win_count += 1
                self.win_ms += d
                self.win_max_ms = max(self.win_max_ms, d)
        except Exception:
            pass

    def install(self):
        if not self.installed:
            gc.callbacks.append(self._cb)
            self.installed = True

    def uninstall(self):
        if self.installed:
            try:
                gc.callbacks.remove(self._cb)
            except ValueError:
                pass
            self.installed = False

    def snapshot_and_reset(self):
        snap = {"collections_by_gen": list(self.collections), "window_collections": self.win_count,
                "window_ms": round(self.win_ms, 3), "window_max_ms": round(self.win_max_ms, 3)}
        self.win_count, self.win_ms, self.win_max_ms = 0, 0.0, 0.0
        return snap


def _read(path):
    with open(path, "r") as f:
        return f.read()


def parse_proc_net_tcp(text: str, port: int = QUEST_PORT):
    """Remote IPv4 addresses of ESTABLISHED connections whose local port is ``port``."""
    peers = []
    for line in text.splitlines()[1:]:
        f = line.split()
        if len(f) < 4 or f[3] != "01":
            continue
        try:
            lport = int(f[1].split(":")[1], 16)
            if lport != port:
                continue
            rhex, rport = f[2].split(":")
            if len(rhex) == 8:
                b = bytes.fromhex(rhex)[::-1]
                ip = ".".join(str(x) for x in b)
            elif len(rhex) == 32:
                raw = bytes.fromhex(rhex)
                words = [raw[i:i + 4][::-1] for i in range(0, 16, 4)]
                b = b"".join(words)
                if b[:12] == b"\x00" * 10 + b"\xff\xff":  # v4-mapped
                    ip = ".".join(str(x) for x in b[12:])
                else:
                    ip = "ipv6:" + b.hex()
            else:
                continue
            peers.append({"ip": ip, "port": int(rport, 16), "class": classify_ip(ip)})
        except (ValueError, IndexError):
            continue
    return peers


class ResourceSampler:
    """~1 Hz /proc + /sys sampler in its own thread; ``latest`` is swapped atomically."""

    def __init__(self, proc="/proc", sys_root="/sys", period_s=1.0, pid=None, port=QUEST_PORT,
                 clock=time.monotonic):
        self.proc, self.sys_root, self.period_s = proc, sys_root, period_s
        self.pid = pid or os.getpid()
        self.port = port
        self._clock = clock
        self.latest: dict = {}
        self.error_count = 0
        self._stop = threading.Event()
        self._thread = None
        self._prev_cpu = None
        self._prev_threads: dict = {}
        self._prev_t = None
        self._img_pid = None
        self._img_scan_at = -1e9
        self._clk_tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100

    def start(self):
        self._thread = threading.Thread(target=self._run, name="loop-diag-sampler", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.period_s)

    def sample_once(self):
        snap = {}
        for name, fn in (("cpu", self._cpu), ("threads", self._threads), ("thermal", self._thermal),
                         ("net", self._net)):
            try:
                snap[name] = fn()
            except Exception:
                self.error_count += 1
        self._prev_t = self._clock()
        self.latest = snap
        return snap

    def _cpu(self):
        first = _read(f"{self.proc}/stat").splitlines()[0].split()
        vals = [int(x) for x in first[1:9]]
        idle = vals[3] + vals[4]
        total = sum(vals)
        out = {"loadavg": [float(x) for x in _read(f"{self.proc}/loadavg").split()[:3]]}
        if self._prev_cpu is not None:
            dt, di = total - self._prev_cpu[0], idle - self._prev_cpu[1]
            out["busy_pct"] = round(100.0 * (dt - di) / dt, 1) if dt > 0 else None
        self._prev_cpu = (total, idle)
        freqs = []
        cpu_dir = f"{self.sys_root}/devices/system/cpu"
        try:
            for d in sorted(os.listdir(cpu_dir)):
                if d.startswith("cpu") and d[3:].isdigit():
                    try:
                        freqs.append(int(_read(f"{cpu_dir}/{d}/cpufreq/scaling_cur_freq")) // 1000)
                    except (OSError, ValueError):
                        pass
        except OSError:
            pass
        if freqs:
            out["cpu_mhz_min"], out["cpu_mhz_max"] = min(freqs), max(freqs)
        return out

    def _proc_threads(self, pid):
        res = {}
        base = f"{self.proc}/{pid}/task"
        for tid in os.listdir(base):
            try:
                s = _read(f"{base}/{tid}/stat")
                name = s[s.index("(") + 1:s.rindex(")")]
                rest = s[s.rindex(")") + 2:].split()
                res[(pid, tid)] = (name, int(rest[11]) + int(rest[12]))  # utime+stime
            except (OSError, ValueError, IndexError):
                continue
        return res

    def _find_image_pid(self, now):
        if now - self._img_scan_at < 10.0:
            return self._img_pid
        self._img_scan_at = now
        self._img_pid = None
        for d in os.listdir(self.proc):
            if d.isdigit() and int(d) != self.pid:
                try:
                    cmd = _read(f"{self.proc}/{d}/cmdline")
                except OSError:
                    continue
                if "image_server" in cmd or "teleimager" in cmd.lower():
                    self._img_pid = int(d)
                    break
        return self._img_pid

    def _threads(self):
        now = self._clock()
        cur = self._proc_threads(self.pid)
        img = self._find_image_pid(now)
        if img:
            try:
                cur.update(self._proc_threads(img))
            except OSError:
                pass
        out = {"image_server_pid": img}
        if self._prev_t is not None and now > self._prev_t:
            dt = now - self._prev_t
            rows = []
            for key, (name, ticks) in cur.items():
                prev = self._prev_threads.get(key)
                if prev is not None:
                    pct = 100.0 * (ticks - prev[1]) / self._clk_tck / dt
                    rows.append((pct, key[0], name))
            rows.sort(reverse=True)
            out["top_threads"] = [{"pid": p, "name": n, "cpu_pct": round(c, 1)} for c, p, n in rows[:6]]
            out["own_cpu_pct"] = round(sum(c for c, p, _ in rows if p == self.pid), 1)
            if img:
                out["image_server_cpu_pct"] = round(sum(c for c, p, _ in rows if p == img), 1)
        self._prev_threads = cur
        return out

    def _thermal(self):
        base = f"{self.sys_root}/class/thermal"
        temps = {}
        for d in os.listdir(base):
            if d.startswith("thermal_zone"):
                try:
                    temps[_read(f"{base}/{d}/type").strip()] = int(_read(f"{base}/{d}/temp")) / 1000.0
                except (OSError, ValueError):
                    pass
        return {"max_c": max(temps.values()) if temps else None, "zones": temps}

    def _net(self):
        peers = []
        for f in ("tcp", "tcp6"):
            try:
                peers += parse_proc_net_tcp(_read(f"{self.proc}/net/{f}"), self.port)
            except OSError:
                pass
        return {"quest_port": self.port, "peers": peers,
                "route": ",".join(sorted({p["class"] for p in peers})) or "none"}


class LoopDiagnostics:
    """Aggregates per-cycle timing; ``cycle_end`` is the only hot-path call."""

    def __init__(self, emit: Callable[[Mapping], object] | None = None, interval_s=1.0,
                 sampler: ResourceSampler | None = None, gc_watcher: GcWatcher | None = None,
                 clock=time.monotonic, perf=time.perf_counter):
        self._emit = emit
        self.interval_s = interval_s
        self.timer = CycleTimer(perf)
        self.feed = ControllerFeedTracker()
        self.video = VideoCounter()
        self.sampler = sampler
        self.gc = gc_watcher
        self._clock = clock
        self._win = {}
        self._win_total = []
        self._cycles = 0
        self._last_emit = None
        self.error_count = 0
        self.emit_drop_count = 0

    def begin(self):
        self.timer.begin()

    def mark(self, stage):
        self.timer.mark(stage)

    def record_video(self, frame):
        try:
            self.video.record(frame)
        except Exception:
            self.error_count += 1

    def pose_fields(self, rate_limiter=None, calibrator=None):
        """Small dict for the pose record: last completed cycle timing + cumulative counters."""
        try:
            diag = {"limiter_bands": dict(getattr(rate_limiter, "band_counts", {}) or {}),
                    "calibration": _calib_counters(calibrator),
                    "feed": {"unique_total": self.feed.unique_total,
                             "repeated_total": self.feed.repeated_total,
                             "stale_cycles_total": self.feed.stale_cycles_total},
                    "video_frames_total": self.video.frames_total}
            return dict(self.timer.last), diag
        except Exception:
            self.error_count += 1
            return None, None

    def cycle_end(self, sample_ts, rate_limiter=None, calibrator=None):
        try:
            last = self.timer.end()
            now = self._clock()
            self.feed.observe(sample_ts, now, self.timer.stages)
            for k, v in self.timer.stages.items():
                self._win.setdefault(k, []).append(v)
            self._win_total.append(last["total"])
            self._cycles += 1
            if self._last_emit is None:
                self._last_emit = now
            elif now - self._last_emit >= self.interval_s:
                self._flush(now, rate_limiter, calibrator)
        except Exception:
            self.error_count += 1

    def _flush(self, now, rate_limiter, calibrator):
        window = now - self._last_emit
        rec = {
            "event": "loop_diagnostics",
            "t_mono": round(now, 4), "wall": round(time.time(), 3),
            "window_s": round(window, 3), "cycles": self._cycles,
            "loop_hz": round(self._cycles / window, 2) if window > 0 else None,
            "stages_ms": {k: _summ(v) for k, v in self._win.items()},
            "total_ms": _summ(self._win_total),
            "feed": self.feed.snapshot_and_reset(window),
            "limiter_bands": dict(getattr(rate_limiter, "band_counts", {}) or {}),
            "calibration": _calib_counters(calibrator),
            "video": self.video.snapshot_and_reset(window),
            "diag_errors": self.error_count + (self.sampler.error_count if self.sampler else 0),
        }
        if self.gc:
            rec["gc"] = self.gc.snapshot_and_reset()
        if self.sampler:
            rec["system"] = self.sampler.latest
        self._win, self._win_total, self._cycles, self._last_emit = {}, [], 0, now
        if self._emit is not None:
            try:
                if self._emit(rec) is False:
                    self.emit_drop_count += 1
            except Exception:
                self.error_count += 1

    def start(self):
        if self.sampler:
            self.sampler.start()
        if self.gc:
            self.gc.install()

    def close(self):
        if self.sampler:
            self.sampler.stop()
        if self.gc:
            self.gc.uninstall()


def _calib_counters(calibrator):
    if calibrator is None:
        return None
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in (
        (n, getattr(calibrator, n, None)) for n in
        ("reanchor_count", "continuity_limit_count", "projection_count", "rejection_counts"))}
