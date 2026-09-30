import builtins
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from teleop.utils.loop_diagnostics import (
    ControllerFeedTracker, CycleTimer, GcWatcher, LoopDiagnostics, ResourceSampler,
    VideoCounter, classify_ip, parse_proc_net_tcp,
)
from teleop.utils.full_pose_telemetry import build_pose_record
import analyze_loop_feed as alf


def test_cycle_timer_stages_fake_clock():
    t = [0.0]
    c = CycleTimer(lambda: t[0])
    c.begin()
    t[0] = 0.010; c.mark("controller")
    t[0] = 0.040; c.mark("render")
    t[0] = 0.041; c.mark("sleep")
    last = c.end()
    assert abs(last["controller"] - 10) < 1e-6 and abs(last["render"] - 30) < 1e-6
    assert abs(last["total"] - 41) < 1e-6
    assert c.dominant() == "render"


def test_feed_hist_repeats_and_stale_burst_with_dominant_stage():
    f = ControllerFeedTracker()
    now = 100.0
    f.observe(now, now)
    f.observe(now + 0.03, now + 0.03)          # 30 ms
    f.observe(now + 0.03, now + 0.06)          # repeated
    f.observe(now + 0.18, now + 0.18)          # 150 ms
    assert f.unique_total == 3 and f.repeated_total == 1
    assert f.hist["lt50"] == 1 and f.hist["100_200"] == 1
    # stale: sample old by > 250 ms for 3 cycles, render dominant
    t = now + 0.18
    for i in range(3):
        t += 0.1
        f.observe(now + 0.18, t, {"render": 80.0, "ik": 5.0, "sleep": 100.0})
    f.observe(t + 0.1, t + 0.1)                # fresh again closes burst
    snap = f.snapshot_and_reset(1.0)
    assert snap["stale_bursts"][0]["dominant_stage"] == "render"
    assert snap["stale_bursts"][0]["cycles"] >= 1
    assert snap["stale_cycles_total"] >= 1


def test_classify_and_proc_net_tcp():
    assert classify_ip("100.101.2.3") == "tailscale"
    assert classify_ip("100.128.0.1") == "other"
    assert classify_ip("192.168.1.5") == "lan" and classify_ip("10.0.0.2") == "lan"
    hdr = "  sl  local_address rem_address   st\n"
    # 8012 = 0x1F4C ; remote 100.64.0.9:5000 -> little endian hex 09004064
    line = "   0: 0100007F:1F4C 09004064:1388 01 00000000:00000000\n"
    other = "   1: 0100007F:1F4C 0100A8C0:1389 06 00000000:00000000\n"  # TIME_WAIT
    wrong = "   2: 0100007F:0050 0100A8C0:1389 01 00000000:00000000\n"
    peers = parse_proc_net_tcp(hdr + line + other + wrong)
    assert peers == [{"ip": "100.64.0.9", "port": 5000, "class": "tailscale"}]


def test_video_counter():
    v = VideoCounter()
    v.record(np.zeros((10, 10, 3), np.uint8)); v.record(np.zeros((10, 10, 3), np.uint8))
    s = v.snapshot_and_reset(2.0)
    assert s["fps"] == 1.0 and s["raw_bytes_per_s"] == 300 and s["frames_total"] == 2


def test_gc_watcher_counts():
    g = GcWatcher(); g.install()
    try:
        gc.collect()
    finally:
        g.uninstall()
    s = g.snapshot_and_reset()
    assert s["window_collections"] >= 1 and sum(s["collections_by_gen"]) >= 1


def test_sampler_reads_fake_proc(tmp_path):
    proc, sysr = tmp_path / "proc", tmp_path / "sys"
    (proc / "net").mkdir(parents=True)
    (proc / "stat").write_text("cpu  100 0 100 800 0 0 0 0\n")
    (proc / "loadavg").write_text("1.50 1.00 0.50 1/100 123\n")
    (proc / "net" / "tcp").write_text("h\n   0: 0100007F:1F4C 09004064:1388 01 0:0\n")
    (proc / "net" / "tcp6").write_text("h\n")
    tk = proc / "42" / "task" / "42"; tk.mkdir(parents=True)
    tk.joinpath("stat").write_text("42 (python) S " + " ".join(["0"] * 11) + " 10 5 " + "0 " * 30)
    z = sysr / "class" / "thermal" / "thermal_zone0"; z.mkdir(parents=True)
    (z / "type").write_text("cpu\n"); (z / "temp").write_text("61500\n")
    t = [0.0]
    s = ResourceSampler(str(proc), str(sysr), pid=42, clock=lambda: t[0])
    s.sample_once()
    (proc / "stat").write_text("cpu  200 0 200 1000 0 0 0 0\n")
    t[0] = 1.0
    snap = s.sample_once()
    assert snap["cpu"]["busy_pct"] == 50.0
    assert snap["thermal"]["max_c"] == 61.5
    assert snap["net"]["route"] == "tailscale"
    assert s.error_count == 0


def test_sampler_errors_are_counted_not_raised(tmp_path):
    s = ResourceSampler(str(tmp_path / "nope"), str(tmp_path / "nope2"), pid=1)
    s.sample_once()
    assert s.error_count >= 1


def test_pose_record_carries_loop_fields():
    import inspect
    assert "loop_timing" in inspect.signature(build_pose_record).parameters


def test_diagnostics_window_emission_and_overhead_no_io(monkeypatch):
    out = []
    t = [1000.0]
    d = LoopDiagnostics(out.append, clock=lambda: t[0])

    class RL:
        band_counts = {"pass": 3, "clamp": 1}
    opened = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda *a, **k: (opened.append(a), real_open(*a, **k))[1])
    frame = np.zeros((4, 4, 3), np.uint8)
    n = 2000
    start = time.perf_counter()
    for i in range(n):
        d.begin()
        for st in ("camera", "render", "controller", "hand", "locomotion", "telemetry", "state_read",
                   "arm_cycle", "hand", "telemetry", "record", "sleep"):
            d.mark(st)
        d.record_video(frame)
        d.pose_fields(RL(), None)
        t[0] += 0.033
        d.cycle_end(t[0] - 0.01, RL(), None)
    per_cycle_ms = (time.perf_counter() - start) * 1000.0 / n
    assert opened == [], "instrument must not do file I/O"
    assert per_cycle_ms < 0.2, per_cycle_ms
    assert out and out[0]["event"] == "loop_diagnostics"
    assert out[0]["limiter_bands"] == {"pass": 3, "clamp": 1}
    assert "p95" in out[0]["stages_ms"]["render"]
    json.dumps(out[0])


def test_emit_failure_is_counted_not_raised():
    t = [0.0]
    def bad(_): raise RuntimeError("x")
    d = LoopDiagnostics(bad, clock=lambda: t[0])
    for _ in range(100):
        d.begin(); d.mark("sleep"); t[0] += 0.05; d.cycle_end(t[0])
    assert d.error_count >= 1


def test_analyzer_verdicts():
    diag = []
    for i in range(10):
        diag.append({"event": "loop_diagnostics", "loop_hz": 12.6,
                     "total_ms": {"p95": 90}, "stages_ms": {"render": {"p95": 40 + i * 5}, "ik": {"p95": 5}, "controller": {"p95": 1}, "sleep": {"p95": 0}},
                     "feed": {"unique_hz": 11.4, "gap_ms": {"p95": 100 + i * 10}, "gap_hist": {"gt500": 1},
                              "stale_bursts": [{"dominant_stage": "render", "duration_ms": 300}]},
                     "system": {"net": {"route": "tailscale"}, "cpu": {"busy_pct": 40}, "threads": {"own_cpu_pct": 99},
                                "thermal": {"max_c": 55}}, "gc": {"window_max_ms": 2}, "video": {"fps": 12, "raw_bytes_per_s": 1000000}})
    res = alf.analyze(diag)
    text = "\n".join(res["verdicts"])
    assert "etapa dominante render" in text
    assert "coincidem com picos de render" in text
    assert "rota do Quest: tailscale" in text
    assert "GIL" in text
    assert "correlação" in text
