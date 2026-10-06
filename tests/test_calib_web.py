"""UI web de calibracao: logica pura, seguranca do offset, HTTP e compatibilidade com o analisador."""
import json
import math
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import analyze_walk_calibration as awc  # noqa: E402
from tools import calib_web as cw  # noqa: E402
from tools import calib_web_core as core  # noqa: E402
from tools import mark_calibration_segments as mcs  # noqa: E402

from tests.test_walk_calibration import T0, attempt, write_jsonl  # noqa: E402

# captured from Unitree Explorer (cog_capture_20261002_223414, identity.id=80)
CAPTURED_SET = '{"name":"imu_offset_json","content":"{\\"imu\\":[-0.1,2.5,0.0]}"}'
CAPTURED_TORSO = '{"name":"secondaryimu_offset_json","content":"{\\"secondaryimu\\":[-0.8,0.0,0.0]}"}'


class Clock:
    def __init__(self):
        self.t = 0.0

    def wall(self):
        return T0 + self.t

    def mono(self):
        return 500.0 + self.t

    def sleep(self, dt):
        self.t += dt


def make(tmp_path, *, mode="normal", values=None, live=None, **bk):
    clock = Clock()
    backend = core.FakeBackend(values={"imu": [0.0, 2.5, 0.0], "secondaryimu": [-0.8, 0.0, 0.0]}
                               if values is None else values, live=live, clock=clock.mono, **bk)
    ctrl = core.CalibController(backend, tmp_path, mode=mode, wall_clock=clock.wall,
                                monotonic_clock=clock.mono, sleep=lambda dt: _advance(clock, backend, dt),
                                stamp="TEST", log=lambda *_: None)
    return ctrl, backend, clock


def _advance(clock, backend, dt):
    """sleep fake: the simulated IMU publishes during the wait."""
    end = clock.t + dt
    while clock.t < end:
        backend.tick_imu(clock.mono())
        clock.t += 0.05


# ------------------------------------------------------------------ payload
def test_set_payload_is_byte_identical_to_capture():
    assert core.build_set_parameter("imu", [-0.1, 2.5, 0]) == CAPTURED_SET
    assert core.build_set_parameter("secondaryimu", [-0.8, 0, -0.0]) == CAPTURED_TORSO
    assert core.build_set_parameter("imu", [0.1 + 0.2, 2.5, 0]).count("0.3,") == 1   # float noise rounded
    assert core.build_get_parameter("imu") == '{"name":"imu_offset_json"}'
    with pytest.raises(ValueError):
        core.build_set_parameter("imu", [math.nan, 0, 0])


def test_parse_responses_and_passive_sources():
    assert core.parse_set_request(CAPTURED_SET) == ("imu", [-0.1, 2.5, 0.0])
    assert core.parse_set_request('{"name":"led_json","content":"{}"}') is None
    assert core.parse_get_response("imu", '{"content":"{\\"imu\\":[0.0,2.5,0.0]}"}') == [0.0, 2.5, 0.0]
    assert core.parse_get_response("imu", '{"imu":[0.1,2.4,0]}') == [0.1, 2.4, 0.0]
    assert core.parse_get_response("imu", "") is None
    assert core.parse_get_response("imu", '{"content":"{\\"imu\\":[1,2]}"}') is None
    assert core.parse_config_change("secondaryimu_offset_json", '{"secondaryimu":[-0.8,0,0]}') == (
        "secondaryimu", [-0.8, 0.0, 0.0])
    assert core.parse_config_change("led_json", "{}") is None


# ------------------------------------------------------------------- policy
def test_policy_limits_step_rate_and_nan():
    p = core.OffsetPolicy()
    base = cur = [0.0, 2.5, 0.0]
    v = lambda tgt, **kw: p.validate(key="imu", current=kw.get("cur", cur), target=tgt, base=base,
                                     last_write_mono=kw.get("last"), now=kw.get("now"))
    assert v([0.5, 2.5, 0.0]) == []
    assert any("passo" in e for e in v([0.6, 2.5, 0.0]))
    assert any("janela" in e for e in v([0.0, 5.6, 0.0], cur=[0.0, 5.3, 0.0]))
    assert any("rigido" in e for e in core.OffsetPolicy(window_deg=20).validate(
        key="imu", current=[0, 9.8, 0], target=[0, 10.2, 0], base=[0, 9.8, 0]))
    assert any("NaN" in e for e in v([math.nan, 2.5, 0.0]))
    assert any("NaN" in e for e in v([math.inf, 2.5, 0.0]))
    assert any("yaw" in e for e in v([0.0, 2.5, 0.1]))
    assert any("intervalo" in e for e in v([0.1, 2.5, 0.0], last=10.0, now=10.5))
    assert v([0.1, 2.5, 0.0], last=10.0, now=11.01) == []
    assert any("base desconhecida" in e for e in p.validate(key="imu", current=cur, target=[0.1, 2.5, 0],
                                                             base=None))
    # hard caps cannot be loosened from the CLI
    loose = core.OffsetPolicy(max_step_deg=5, min_interval_s=0.1, hard_abs_deg=50)
    assert (loose.max_step_deg, loose.min_interval_s, loose.hard_abs_deg) == (0.5, 1.0, 10.0)


def test_motion_warnings():
    p = core.OffsetPolicy()
    calm = {"wireless": {"age": 0.1, "lx": 0, "ly": 0, "rx": 0, "zero": True},
            "odom": {"age": 0.1, "speed": 0.01, "yaw_rate": 0.0}}
    assert p.motion_warnings(calm) == []
    moving = {"wireless": {"age": 0.1, "lx": 0, "ly": 0.4, "rx": 0, "zero": False},
              "odom": {"age": 0.1, "speed": 0.2, "yaw_rate": 0.0}}
    assert len(p.motion_warnings(moving)) == 2
    assert len(p.motion_warnings({})) == 2           # unknown state warns too


# --------------------------------------------------------------- controller
def test_initial_get_sets_base_and_write_verified(tmp_path):
    ctrl, bk, clock = make(tmp_path)
    ctrl.initial_read()
    assert [c[0] for c in bk.calls] == [core.API_GET, core.API_GET]     # only GETs at startup
    assert ctrl.offsets["imu"]["base"] == [0.0, 2.5, 0.0] and ctrl.offsets["imu"]["source"] == "dds_get"
    for _ in range(20):
        bk.tick_imu(clock.mono())
        clock.t += 0.05
    r = ctrl.apply("imu", [0.1, 2.5, 0.0], expect_current=[0.0, 2.5, 0.0], confirm=True)
    assert r["ok"] and r["code"] == 0 and r["verified"]
    assert r["verification"]["imu"]["verified"] and r["verification"]["get"]["verified"]
    assert bk.calls[2] == (core.API_SET, '{"name":"imu_offset_json","content":"{\\"imu\\":[0.1,2.5,0.0]}"}')
    # rate limit: immediately again is refused, nothing sent
    n = len(bk.calls)
    ctrl.last_write_mono = clock.mono()
    r2 = ctrl.apply("imu", [0.2, 2.5, 0.0], confirm=True)
    assert not r2["ok"] and any("intervalo" in e for e in r2["errors"]) and len(bk.calls) == n
    log = [json.loads(l) for l in ctrl.offset_log_path.read_text().splitlines()]
    assert log[0]["before"] == [0.0, 2.5, 0.0] and log[0]["after"] == [0.1, 2.5, 0.0] and log[0]["verified"]
    assert "timestamp_monotonic" in log[0] and log[0]["timestamp_utc"].endswith("Z")
    lines = ctrl.close()
    assert lines and "difere da base" in lines[0]
    assert [c[0] for c in bk.calls].count(core.API_SET) == 1          # close sends nothing


def test_unknown_base_blocks_writes_until_manual(tmp_path):
    ctrl, bk, _ = make(tmp_path, get_code=core.CODE_TIMEOUT)
    ctrl.initial_read()
    assert ctrl.get_available is False and ctrl.offsets["imu"]["value"] is None
    r = ctrl.apply("imu", [0.1, 2.5, 0.0], confirm=True)
    assert not r["ok"] and any("base desconhecida" in e for e in r["errors"])
    assert all(c[0] == core.API_GET for c in bk.calls)
    assert ctrl.set_manual("imu", [0.0, 2.5, 0.0])["ok"]
    assert ctrl.offsets["imu"]["base"] == [0.0, 2.5, 0.0] and ctrl.offsets["imu"]["base_source"] == "manual"
    assert ctrl.apply("imu", [0.0, 2.6, 0.0], confirm=True)["ok"]


def test_passive_app_value_used_when_get_unavailable(tmp_path):
    ctrl, bk, clock = make(tmp_path, get_code=core.CODE_TIMEOUT)
    ctrl.initial_read()
    bk.passive.append((clock.mono(), "imu", [0.0, 2.4, 0.0], "dds_passive_app"))
    st = ctrl.state()
    assert st["offsets"]["imu"]["value"] == [0.0, 2.4, 0.0]
    assert st["offsets"]["imu"]["source"] == "dds_passive_app" and st["offsets"]["imu"]["base"] == [0.0, 2.4, 0.0]


def test_set_without_get_keeps_ui_set_source(tmp_path):
    ctrl, bk, clock = make(tmp_path, can_get=False)
    ctrl.initial_read()
    ctrl.set_manual("imu", [0.0, 2.5, 0.0])
    assert ctrl.apply("imu", [0.1, 2.5, 0.0], confirm=True)["ok"]
    assert ctrl.offsets["imu"]["source"] == "ui_set"
    assert ctrl.history[-1]["verification"]["get"] is None


def test_set_timeout_marks_uncertain_and_blocks(tmp_path):
    ctrl, bk, clock = make(tmp_path, set_code=core.CODE_TIMEOUT)
    ctrl.initial_read()
    r = ctrl.apply("imu", [0.1, 2.5, 0.0], confirm=True)
    assert not r["ok"] and r["code"] == core.CODE_TIMEOUT and not r["verified"]
    assert ctrl.offsets["imu"]["uncertain"] and ctrl.offsets["imu"]["value"] == [0.0, 2.5, 0.0]
    clock.t += 5
    r2 = ctrl.apply("imu", [0.1, 2.5, 0.0], confirm=True)
    assert not r2["ok"] and any("incerto" in e for e in r2["errors"])
    ctrl.read_get("imu")                     # fresh GET clears the uncertainty
    assert not ctrl.offsets["imu"]["uncertain"]


def test_confirmation_motion_ack_and_stale_expectation(tmp_path):
    moving = {"wireless": {"age": 0.1, "lx": 0.0, "ly": 0.5, "rx": 0.0, "zero": False},
              "odom": {"age": 0.1, "speed": 0.3, "yaw_rate": 0.0}, "sources": {}}
    ctrl, bk, clock = make(tmp_path, live=moving)
    ctrl.initial_read()
    assert not ctrl.apply("imu", [0.1, 2.5, 0.0])["ok"]                      # no confirm
    r = ctrl.apply("imu", [0.1, 2.5, 0.0], confirm=True)
    assert r.get("needs_motion_ack") and not r["ok"]
    r = ctrl.apply("imu", [0.1, 2.5, 0.0], confirm=True, expect_current=[0.0, 2.4, 0.0], ack_motion=True)
    assert not r["ok"] and "mudou" in r["errors"][0]
    assert ctrl.apply("imu", [0.1, 2.5, 0.0], confirm=True, ack_motion=True)["ok"]
    assert ctrl.history[-1]["warnings_acked"]


def test_read_only_and_dry_run_never_write(tmp_path):
    ro, bk, _ = make(tmp_path / "ro", mode="read-only")
    ro.initial_read()
    ro.set_manual("imu", [0.0, 2.5, 0.0])
    assert not ro.apply("imu", [0.1, 2.5, 0.0], confirm=True)["ok"]
    assert bk.calls == []
    dr, bk2, _ = make(tmp_path / "dr", mode="dry-run")
    assert dr.get_enabled
    dbk = cw.DdsBackend("lo", dry_run=True, log=lambda *_: None)    # real backend, never started
    assert dbk.config_set("imu", [0.1, 2.5, 0.0]) == (0, "") and dbk._pub is None
    assert not dbk.can_get and dbk.config_get("imu") == (-1, None)
    rbk = cw.DdsBackend("lo", read_only=True, log=lambda *_: None)
    with pytest.raises(RuntimeError):
        rbk.config_set("imu", [0.1, 2.5, 0.0])


def test_restore_steps_toward_base(tmp_path):
    ctrl, bk, clock = make(tmp_path)
    ctrl.initial_read()
    for tgt in ([0.5, 2.5, 0.0], [0.9, 2.5, 0.0]):
        clock.t += 2
        assert ctrl.apply("imu", tgt, confirm=True)["ok"]
    r = ctrl.restore_step("imu")
    assert r["target"] == [0.4, 2.5, 0.0] and r["steps_remaining"] == 2
    sets = [c for c in bk.calls if c[0] == core.API_SET]
    n = len(bk.calls)
    ctrl.restore_step("imu")
    assert len(bk.calls) == n and "0.9" in sets[-1][1]              # restore_step sends nothing


def test_torso_requires_advanced(tmp_path):
    ctrl, _, _ = make(tmp_path)
    ctrl.initial_read()
    assert not ctrl.preview("secondaryimu", [-0.7, 0, 0])["ok"]
    assert ctrl.preview("secondaryimu", [-0.7, 0, 0], advanced=True)["ok"]


# ------------------------------------------------------------- markers
def test_markers_auto_attempt_checklist_and_offset(tmp_path):
    ctrl, _, clock = make(tmp_path)
    ctrl.initial_read()
    assert not ctrl.marker("segment", kind="reta")["ok"]            # condition first
    ctrl.marker("condition", condition="sem_caixa")
    for k in core.KINDS:
        clock.t += 1
        assert ctrl.marker("segment", kind=k)["ok"]
    clock.t += 1
    ctrl.marker("end")
    clock.t += 1
    ctrl.marker("segment", kind="reta")                              # auto -> attempt 2
    st = ctrl.state()
    assert st["session"]["attempt"] == 2 and st["session"]["open_segment"]["kind"] == "reta"
    assert st["checklist"]["sem_caixa"][0] == {"attempt": 1, "reta": True, "giro_esquerda": True,
                                                "giro_direita": True}
    ctrl.marker("undo")
    assert ctrl.state()["session"]["open_segment"] is None
    ctrl.close()
    lines = [json.loads(l) for l in ctrl.markers_path.read_text().splitlines()]
    seg = [l for l in lines if l["type"] == "segment_start"][0]
    assert seg["imu_offset"] == [0.0, 2.5, 0.0] and seg["imu_offset_source"] == "dds_get"
    assert all(l["event"] == "calib_marker" for l in lines)


def test_analyzer_on_ui_markers_groups_by_offset(tmp_path):
    """UI-generated markers + synthetic telemetry -> analyzer groups by live offset."""
    ctrl, bk, clock = make(tmp_path)
    ctrl.initial_read()
    tel = []
    plan = [(0.0, 0.02), (0.5, 0.04), (1.0, 0.06)]
    for i, (roll, vy) in enumerate(plan):
        ts = 10.0 + i * 40.0
        if roll:
            clock.t = ts - 3
            assert ctrl.apply("imu", [roll, 2.5, 0.0], confirm=True)["ok"]
        clock.t = ts - 1
        ctrl.marker("condition", condition="com_caixa")
        for dt, kind in ((0.0, "reta"), (13.0, "giro_esquerda"), (20.0, "giro_direita")):
            clock.t = ts + dt
            ctrl.marker("segment", kind=kind)
        clock.t = ts + 27
        ctrl.marker("end")
        tel += attempt(ts, vy, 0.0, yaw0=0.0)
    ctrl.close()
    rep = awc.analyze([write_jsonl(tmp_path / "p.jsonl", tel)], ctrl.markers_path, fit_offset=True)
    retas = [s for s in rep["segments"] if s["kind"] == "reta"]
    assert [s["imu_offset"][0] for s in retas] == [0.0, 0.5, 1.0]
    assert [s["attempt"] for s in retas] == [1, 2, 3]
    assert retas[0]["imu_offset_fonte"] == "marcador:dds_get"
    assert retas[1]["imu_offset_fonte"] == "marcador:dds_get"     # SET then re-read by GET
    assert "com_caixa @ roll=+0.50 pitch=+2.50 yaw=+0.00" in rep["aggregate_by_offset"]
    fit = rep["fit_offset"]["com_caixa"]["roll_vs_trim_vy"]
    assert abs(fit["inclinacao"] - 0.04) < 1e-6 and abs(fit["offset_zero"] + 0.5) < 1e-6
    offs = [json.loads(l) for l in ctrl.markers_path.read_text().splitlines() if '"offset_set"' in l]
    assert len(offs) == 2 and offs[0]["offset_after"] == [0.5, 2.5, 0.0]


def test_offset_set_event_drives_segments_for_old_style_markers():
    ev = [{"event": "calib_marker", "seq": 1, "type": "offset_set", "timestamp": T0, "ok": True,
           "offset_key": "imu", "imu_offset": [0.5, 2.5, 0.0]},
          {"event": "calib_marker", "seq": 2, "type": "offset_set", "timestamp": T0 + 0.5, "ok": True,
           "offset_key": "secondaryimu", "imu_offset": None},
          {"event": "calib_marker", "seq": 3, "type": "segment_start", "timestamp": T0 + 1, "kind": "reta",
           "condition": "com_caixa", "attempt": 1, "imu_offset": None},
          {"event": "calib_marker", "seq": 4, "type": "segment_end", "timestamp": T0 + 2}]
    assert mcs.segments_from_markers(ev)[0]["imu_offset"] == [0.5, 2.5, 0.0]
    assert awc.parse_imu_offset({"name": "imu_offset_json", "content": '{"imu":[0.1,2.5,0]}'}) == [0.1, 2.5, 0.0]
    assert awc.parse_imu_offset({"name": "secondaryimu_offset_json",
                                 "content": '{"secondaryimu":[0.1,2.5,0]}'}) is None


# ------------------------------------------------------------------ misc
def test_teleop_detection_exact_argv(tmp_path):
    def proc(pid, argv):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    proc(10, ["/env/bin/python", "teleop/teleop_hand_and_arm.py", "--arm=G1_29"])
    proc(11, ["grep", "teleop_hand_and_arm.py"])
    proc(12, ["python", "tools/calib_web.py"])
    assert cw.teleop_active(str(tmp_path), own_pid=1) == [10]


# ------------------------------------------------------------------ HTTP
@pytest.fixture
def server(tmp_path):
    ctrl, bk, clock = make(tmp_path)
    ctrl.initial_read()
    httpd = cw.serve(ctrl, "127.0.0.1", 0, token="segredo")
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, ctrl, bk, clock
    httpd.shutdown()
    httpd.server_close()
    ctrl.close()


def _req(url, body=None, token="segredo"):
    h = {"Content-Type": "application/json"}
    if token:
        h["X-Calib-Token"] = token
    r = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_http_endpoints(server):
    base, ctrl, bk, clock = server
    assert _req(base + "/api/state", token=None)[0] == 401
    code, html = _req(base + "/?token=segredo", token=None)
    assert code == 200 and b"Calibra" in html
    st = json.loads(_req(base + "/api/state")[1])
    assert st["offsets"]["imu"]["value"] == [0.0, 2.5, 0.0] and st["mode"] == "normal"
    assert json.loads(_req(base + "/api/marker", {"action": "condition", "condition": "com_caixa"})[1])["ok"]
    assert json.loads(_req(base + "/api/marker", {"action": "segment", "kind": "reta"})[1])["ok"]
    pv = json.loads(_req(base + "/api/offset/preview", {"key": "imu", "target": [0.1, 2.5, 0]})[1])
    assert pv["ok"] and pv["parameter"].startswith('{"name":"imu_offset_json"')
    bad = json.loads(_req(base + "/api/offset/apply", {"key": "imu", "target": [0.1, 2.5, 0]})[1])
    assert not bad["ok"]                                     # missing confirm:true
    ok = json.loads(_req(base + "/api/offset/apply", {"key": "imu", "target": [0.1, 2.5, 0],
                                                      "confirm": True})[1])
    assert ok["ok"] and ok["code"] == 0
    assert _req(base + "/api/nope", {})[0] == 404
    st = json.loads(_req(base + "/api/state")[1])
    assert st["history"][0]["after"] == [0.1, 2.5, 0.0] and st["session"]["open_segment"]["kind"] == "reta"


def test_cli_help_and_sim_run(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        cw.main(["--help"])
    assert e.value.code == 0
    assert cw.main(["--sim", "--read-only", "--port", "0", "--host", "127.0.0.1",
                    "--out-dir", str(tmp_path), "--run-seconds", "0.3"]) == 0
    out = capsys.readouterr().out
    assert "read-only" in out and "encerrado" in out
