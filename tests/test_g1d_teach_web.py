"""Offline web control panel for G1D teach/replay (real HTTP on 127.0.0.1, fake IO)."""
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from teleop.utils import teach_replay as tr  # noqa: E402
from teleop.utils import teach_web as tw  # noqa: E402
from test_g1d_teach_replay_session import DT, make, write_recording  # noqa: E402

TOKEN = "test-token-0123456789"


# ------------------------------------------------------------------ helpers
class FakeSession:
    """Records lifecycle calls; optional gate makes request_* block like a held lock."""

    def __init__(self):
        self.calls = []
        self.mode = tr.Mode.IDLE
        self.last_message = "pronto"
        self.gate = threading.Event()
        self.gate.set()
        self.quit_flag = threading.Event()
        self.deadman_feeds = 0
        self.busy = {"replay": None, "recording": []}

    def request_teach(self):
        self.gate.wait(10)
        self.calls.append("teach")
        return True

    def request_save(self):
        self.gate.wait(10)
        self.calls.append("save")
        return True

    def request_replay(self, name, *, origin="terminal"):
        self.gate.wait(10)
        self.calls.append(("replay", name, origin))
        return True

    def request_quit(self):
        self.quit_flag.set()
        self.calls.append("quit")
        return True

    @property
    def quit_requested(self):
        return self.quit_flag.is_set()

    def feed_web_deadman(self, timeout_s):
        self.deadman_feeds += 1

    def status_view(self):
        return {"mode": self.mode.value, "message": self.last_message, "messages": [],
                "replay_origin": None, "recording": None, "replay": None,
                "busy_recordings": [n for n in [self.busy["replay"]] + self.busy["recording"] if n]}


def start_panel(session, teach_dir, **kw):
    kw.setdefault("token", TOKEN)
    panel = tw.WebPanel(session, teach_dir, host="127.0.0.1", port=0,
                        freshness_fn=lambda: {"arm": 0.01, "left": 0.02, "right": None}, **kw)
    panel.start()
    return panel


def call(panel, method, path, body=None, token=TOKEN, client="A", timeout=3.0):
    headers = {"X-Client-Id": client}
    if token is not None:
        headers["X-Token"] = token
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"http://127.0.0.1:{panel.port}{path}", data=data,
                                 headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw.decode()


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def fake(tmp_path):
    session = FakeSession()
    panel = start_panel(session, tmp_path)
    yield session, panel
    session.gate.set()
    panel.stop()


def saved_file(tmp_path, name="20261019T100000_000000Z", n=10):
    t = np.arange(n) * 0.02
    arm = np.zeros((n, 14))
    hands = {"left": np.zeros((n, 7)), "right": np.zeros((n, 7))}
    return write_recording(tmp_path / f"{name}.jsonl", t, arm, hands)


# ------------------------------------------------------------------ token
@pytest.mark.parametrize("token", [None, "wrong", TOKEN + "x", ""])
def test_api_requires_the_token(fake, token):
    session, panel = fake
    for method, path, body in [("GET", "/api/status", None), ("POST", "/api/teach", {}),
                               ("POST", "/api/stop", {}), ("GET", "/api/recordings", None),
                               ("POST", "/api/delete", {"name": "x.jsonl"})]:
        status, _ = call(panel, method, path, body, token=token)
        assert status == 401, (path, token)
    time.sleep(0.1)
    assert session.calls == []


def test_token_is_random_and_compared_in_constant_time(tmp_path, monkeypatch):
    seen = []
    real = tw.hmac.compare_digest

    def spy(a, b):
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(tw.hmac, "compare_digest", spy)
    a, b = tw.WebPanel(FakeSession(), tmp_path), tw.WebPanel(FakeSession(), tmp_path)
    assert a.token != b.token and len(a.token) >= 16
    session = FakeSession()
    panel = start_panel(session, tmp_path)
    try:
        call(panel, "GET", "/api/status", token="nope")
        assert seen, "token check must use hmac.compare_digest"
    finally:
        panel.stop()


def test_token_accepted_as_query_parameter(fake):
    _, panel = fake
    status, body = call(panel, "GET", f"/api/status?t={TOKEN}", token=None)
    assert status == 200 and body["mode"] == "idle"


def test_page_without_token_shows_only_token_required(fake):
    _, panel = fake
    status, html = call(panel, "GET", "/", token=None)
    assert status == 401
    assert "token" in html.lower()
    assert "Ensinar" not in html and "/api/teach" not in html


def test_page_with_token_is_self_contained_and_offline(fake):
    _, panel = fake
    status, html = call(panel, "GET", f"/?t={TOKEN}", token=None)
    assert status == 200
    for label in ("Ensinar", "Salvar", "Reproduzir", "PARAR"):
        assert label in html
    assert not re.search(r"https?://|//cdn|<link[^>]+href|<script[^>]+src", html, re.I)
    assert "<style>" in html and "<script>" in html
    assert TOKEN not in html            # token comes from the URL, never embedded


# ------------------------------------------------------------------ routing
def test_mutations_route_to_the_session_lifecycle_methods(fake, tmp_path):
    session, panel = fake
    path = saved_file(tmp_path)
    assert call(panel, "POST", "/api/teach", {})[0] == 202
    assert wait_until(lambda: session.calls == ["teach"])
    assert call(panel, "POST", "/api/save", {})[0] == 202
    assert wait_until(lambda: session.calls[-1] == "save")
    assert call(panel, "POST", "/api/replay", {"name": path.name})[0] == 202
    assert wait_until(lambda: isinstance(session.calls[-1], tuple))
    kind, name, origin = session.calls[-1]
    assert kind == "replay" and origin == "web"
    assert Path(name) == tmp_path / path.name     # resolved inside the teach dir only
    assert call(panel, "POST", "/api/stop", {})[0] == 200
    assert session.calls[-1] == "quit"


def test_status_reports_mode_message_freshness_and_controller(fake):
    session, panel = fake
    session.mode = tr.Mode.HOLD
    session.last_message = "HOLD: x"
    time.sleep(0.25)
    status, body = call(panel, "GET", "/api/status")
    assert status == 200
    assert body["mode"] == "hold" and body["message"] == "HOLD: x"
    assert body["fresh"]["arm"] == pytest.approx(0.01) and body["fresh"]["right"] is None
    assert body["controller"] is True


def test_get_does_not_mutate(fake):
    session, panel = fake
    for path in ("/api/teach", "/api/stop", "/api/replay"):
        assert call(panel, "GET", path)[0] in (404, 405)
    time.sleep(0.1)
    assert session.calls == []


# ------------------------------------------------------------------ stop / non-blocking
def test_handler_never_blocks_when_the_session_lock_is_held(tmp_path):
    session, io, clock = make(tmp_path)
    panel = start_panel(session, tmp_path)
    try:
        with session.lock:            # a long control tick / blocking activation
            t0 = time.monotonic()
            status, body = call(panel, "POST", "/api/teach", {})
            assert status in (202, 503) and time.monotonic() - t0 < 0.5
            t0 = time.monotonic()
            assert call(panel, "GET", "/api/status")[0] == 200
            assert time.monotonic() - t0 < 0.5
            t0 = time.monotonic()
            assert call(panel, "POST", "/api/stop", {})[0] == 200
            assert time.monotonic() - t0 < 0.5
            assert session.quit_requested
        # stop wins: the queued teach is refused or dropped, never started
        clock.t += DT
        session.tick()
        time.sleep(0.2)
        clock.t += DT
        session.tick()
        assert session.mode in (tr.Mode.DONE, tr.Mode.QUITTING)
        assert io.activations == 0 and io.writes == []
    finally:
        panel.stop()


def test_queue_full_answers_busy_quickly(fake):
    session, panel = fake
    session.gate.clear()
    codes = []
    t0 = time.monotonic()
    for _ in range(12):
        codes.append(call(panel, "POST", "/api/teach", {})[0])
    assert time.monotonic() - t0 < 2.0
    assert 503 in codes and 202 in codes


def test_stop_wins_over_pending_web_actions(fake):
    session, panel = fake
    session.gate.clear()                 # dispatcher stuck inside a lifecycle call
    for _ in range(3):
        call(panel, "POST", "/api/teach", {})
    assert call(panel, "POST", "/api/stop", {})[0] == 200
    assert session.calls == ["quit"]
    session.gate.set()
    time.sleep(0.3)
    # at most the call that was already in flight; nothing queued after stop
    assert session.calls.count("teach") <= 1
    assert session.calls[0] == "quit"


def test_stop_accepted_from_a_read_only_client(fake):
    session, panel = fake
    call(panel, "GET", "/api/status", client="A")
    status, body = call(panel, "POST", "/api/teach", {}, client="B")
    assert status == 409
    assert call(panel, "POST", "/api/stop", {}, client="B")[0] == 200
    assert session.quit_requested


# ------------------------------------------------------------------ single controller
def test_second_client_is_read_only_until_the_first_stops_polling(tmp_path):
    session = FakeSession()
    now = [100.0]
    panel = start_panel(session, tmp_path, clock=lambda: now[0])
    try:
        assert call(panel, "GET", "/api/status", client="A")[1]["controller"] is True
        assert call(panel, "GET", "/api/status", client="B")[1]["controller"] is False
        assert call(panel, "POST", "/api/teach", {}, client="B")[0] == 409
        now[0] += 2.0
        call(panel, "GET", "/api/status", client="A")
        now[0] += 2.5
        assert call(panel, "GET", "/api/status", client="B")[1]["controller"] is False
        now[0] += 3.1
        assert call(panel, "GET", "/api/status", client="B")[1]["controller"] is True
        assert call(panel, "POST", "/api/teach", {}, client="B")[0] == 202
        assert call(panel, "GET", "/api/status", client="A")[1]["controller"] is False
    finally:
        panel.stop()


# ------------------------------------------------------------------ recordings
def test_recording_list_newest_first_saved_only_with_duration(fake, tmp_path):
    _, panel = fake
    saved_file(tmp_path, "20261019T100000_000000Z", n=11)
    saved_file(tmp_path, "20261019T110000_000000Z", n=51)
    (tmp_path / "20261019T120000_000000Z.aborted.jsonl").write_text("{}\n")
    (tmp_path / "20261019T130000_000000Z.jsonl.part").write_text("{}\n")
    status, body = call(panel, "GET", "/api/recordings")
    assert status == 200
    names = [r["name"] for r in body["recordings"]]
    assert names == ["20261019T110000_000000Z.jsonl", "20261019T100000_000000Z.jsonl"]
    assert body["recordings"][0]["duration_s"] == pytest.approx(1.0)
    assert body["recordings"][0]["date"]
    _, body = call(panel, "GET", "/api/recordings?all=1")
    assert "20261019T120000_000000Z.aborted.jsonl" in [r["name"] for r in body["recordings"]]


@pytest.mark.parametrize("bad", ["../x.jsonl", "/etc/passwd", "a/b.jsonl", "..", "x.txt",
                                 "x.jsonl.part", ".hidden.jsonl", "x\x00.jsonl", ""])
def test_names_are_sanitized_against_path_traversal(fake, tmp_path, bad):
    session, panel = fake
    outside = tmp_path.parent / "x.jsonl"
    outside.write_text("keep")
    for path, body in [("/api/replay", {"name": bad}), ("/api/delete", {"name": bad}),
                       ("/api/rename", {"name": bad, "label": "ok"})]:
        assert call(panel, "POST", path, body)[0] == 400
    time.sleep(0.1)
    assert session.calls == []
    assert outside.read_text() == "keep"


def test_rename_keeps_utc_prefix_and_sanitizes_label(fake, tmp_path):
    _, panel = fake
    path = saved_file(tmp_path)
    status, body = call(panel, "POST", "/api/rename", {"name": path.name, "label": "aceno"})
    assert status == 200
    assert body["name"] == "20261019T100000_000000Z_aceno.jsonl"
    assert (tmp_path / body["name"]).is_file() and not path.exists()
    for label in ("../evil", "a/b", "", "x" * 65, "dot.dot"):
        assert call(panel, "POST", "/api/rename", {"name": body["name"], "label": label})[0] == 400


def test_rename_refuses_existing_target(fake, tmp_path):
    _, panel = fake
    path = saved_file(tmp_path)
    saved_file(tmp_path, "20261019T100000_000000Z_a")
    assert call(panel, "POST", "/api/rename", {"name": path.name, "label": "a"})[0] == 409
    assert path.exists()


def test_delete_moves_into_trash_inside_teach_dir(fake, tmp_path):
    _, panel = fake
    path = saved_file(tmp_path)
    assert call(panel, "POST", "/api/delete", {"name": path.name})[0] == 200
    assert not path.exists()
    assert (tmp_path / "trash" / path.name).is_file()
    assert tr.latest_recording(tmp_path) is None
    assert call(panel, "POST", "/api/delete", {"name": path.name})[0] == 404


def test_delete_and_rename_refused_while_replayed_or_recorded(fake, tmp_path):
    session, panel = fake
    path = saved_file(tmp_path)
    session.busy["replay"] = path.stem
    assert call(panel, "POST", "/api/delete", {"name": path.name})[0] == 409
    assert call(panel, "POST", "/api/rename", {"name": path.name, "label": "x"})[0] == 409
    session.busy["replay"] = None
    session.busy["recording"] = [path.stem]
    assert call(panel, "POST", "/api/delete", {"name": path.name})[0] == 409
    assert path.exists()


def test_session_status_view_names_busy_recordings(tmp_path):
    session, io, clock = make(tmp_path)
    session.request_teach()
    clock.t += DT
    session.tick()
    view = session.status_view()
    assert view["mode"] == "teach"
    assert view["recording"]["name"] in view["busy_recordings"]
    clock.t += 0.5
    assert session.status_view()["recording"]["elapsed_s"] == pytest.approx(0.5)


# ------------------------------------------------------------------ deadman
def _long_recording(tmp_path, session, io):
    n = 200                                    # 4 s, slow enough to play at recorded speed
    t = np.arange(n) * 0.02
    arm = np.tile(io.arm_q, (n, 1))
    arm[:, 0] += 0.2 * np.sin(np.linspace(0, np.pi, n))
    hands = {s: np.tile(io.hand_q[s], (n, 1)) for s in ("left", "right")}
    return write_recording(tmp_path / "20261019T100000_000000Z.jsonl", t, arm, hands)


def _ticks(session, clock, seconds):
    for _ in range(int(round(seconds / DT))):
        clock.t += DT
        session.tick()


def test_web_replay_holds_when_heartbeat_is_missed(tmp_path):
    session, io, clock = make(tmp_path)
    path = _long_recording(tmp_path, session, io)
    panel = start_panel(session, tmp_path)
    try:
        call(panel, "GET", "/api/status")
        assert call(panel, "POST", "/api/replay", {"name": path.name})[0] == 202
        assert wait_until(lambda: session._pending is not None)
        _ticks(session, clock, 0.05)
        assert session.mode == tr.Mode.REPLAY_RAMP
        for _ in range(4):                     # browser polling keeps it alive
            call(panel, "GET", "/api/status")
            _ticks(session, clock, 0.5)
            assert session.mode in (tr.Mode.REPLAY_RAMP, tr.Mode.REPLAY_PLAY)
        _ticks(session, clock, 1.1)            # browser gone
        assert session.mode == tr.Mode.HOLD
        assert "web connection lost" in session.last_message
        assert not session.quit_requested and session.active and io.released == 0
        held = io.arm_writes()[-1]
        assert np.all(held.kp > 0)             # hold, never release
    finally:
        panel.stop()


def test_heartbeat_endpoint_also_feeds_the_deadman(tmp_path):
    session, io, clock = make(tmp_path)
    path = _long_recording(tmp_path, session, io)
    panel = start_panel(session, tmp_path)
    try:
        call(panel, "POST", "/api/replay", {"name": path.name})
        assert wait_until(lambda: session._pending is not None)
        for _ in range(4):
            assert call(panel, "POST", "/api/heartbeat", {})[0] == 200
            _ticks(session, clock, 0.5)
        assert session.mode in (tr.Mode.REPLAY_RAMP, tr.Mode.REPLAY_PLAY)
    finally:
        panel.stop()


def test_read_only_client_polling_does_not_feed_the_deadman(tmp_path):
    session, io, clock = make(tmp_path)
    path = _long_recording(tmp_path, session, io)
    panel = start_panel(session, tmp_path)
    try:
        call(panel, "GET", "/api/status", client="A")
        call(panel, "POST", "/api/replay", {"name": path.name}, client="A")
        assert wait_until(lambda: session._pending is not None)
        for _ in range(4):
            call(panel, "GET", "/api/status", client="B")
            _ticks(session, clock, 0.3)
        assert session.mode == tr.Mode.HOLD and "web connection lost" in session.last_message
    finally:
        panel.stop()


def test_terminal_replay_is_unaffected_by_the_web_deadman(tmp_path):
    session, io, clock = make(tmp_path)
    path = _long_recording(tmp_path, session, io)
    panel = start_panel(session, tmp_path)
    try:
        call(panel, "GET", "/api/status")     # a web client exists but never feeds again
        assert session.request_replay(str(path))
        modes = []
        for _ in range(int(3.0 / DT)):
            clock.t += DT
            session.tick()
            modes.append(session.mode)
        assert all(m in (tr.Mode.REPLAY_RAMP, tr.Mode.REPLAY_PLAY) for m in modes)
        assert not any("web connection lost" in m for m in session.messages)
    finally:
        panel.stop()


def test_queued_web_replay_refused_if_connection_already_lost(tmp_path):
    session, io, clock = make(tmp_path)
    path = _long_recording(tmp_path, session, io)
    assert session.request_replay(str(path), origin="web")
    session.feed_web_deadman(1.0)
    clock.t += 2.0                     # dispatcher was slow; browser already gone
    session.tick()
    assert session.mode == tr.Mode.IDLE and io.activations == 0 and io.writes == []
    assert "web connection lost" in session.last_message


def test_deadman_in_session_has_no_effect_on_teach(tmp_path):
    session, io, clock = make(tmp_path)
    session.feed_web_deadman(1.0)
    session.request_teach()
    _ticks(session, clock, 2.0)
    assert session.mode == tr.Mode.TEACH


# ------------------------------------------------------------------ entrypoint wiring
def test_entrypoint_web_flag_defaults_and_port():
    import importlib
    entry = importlib.import_module("teleop.teach_replay_g1d")
    parser = entry.build_parser()
    assert parser.parse_args([]).web is None
    assert parser.parse_args(["--web"]).web == 8095
    assert parser.parse_args(["--web", "9000"]).web == 9000


def test_start_web_panel_prints_tokenized_url_and_survives_port_in_use(tmp_path):
    import importlib
    import socket
    entry = importlib.import_module("teleop.teach_replay_g1d")
    lines = []

    class IO:
        clock = staticmethod(time.monotonic)

        def arm_state(self):
            return tr.ArmState(q=np.zeros(14), dq=np.zeros(14), timestamp=time.monotonic())

        def hand_state(self, side):
            return None

    session = FakeSession()
    panel = entry.start_web_panel(session, IO(), ("left",), tmp_path, 0, print_fn=lines.append)
    try:
        assert panel is not None
        assert any(f":{panel.port}/?t={panel.token}" in line for line in lines)
        _, body = call(panel, "GET", "/api/status", token=panel.token)
        assert body["fresh"]["arm"] is not None and body["fresh"]["left"] is None
    finally:
        panel.stop()
    busy = socket.socket()
    busy.bind(("0.0.0.0", 0))
    busy.listen(1)
    try:
        lines.clear()
        assert entry.start_web_panel(session, IO(), (), tmp_path, busy.getsockname()[1],
                                     print_fn=lines.append) is None
        assert "NOT started" in lines[-1]
    finally:
        busy.close()


def test_launcher_forwards_web_flag(tmp_path):
    import subprocess
    out = subprocess.run(["bash", str(ROOT / "teleop" / "run_g1d_teach_replay.sh"), "--web", "8095"],
                         env={"PATH": "/usr/bin:/bin", "G1D_TEACH_SKIP_RUN": "1", "HOME": str(tmp_path)},
                         capture_output=True, text=True, timeout=10)
    assert out.returncode == 0
    assert "teach_replay_g1d.py --network-interface enP2p1s0 --web 8095" in out.stdout


def test_panel_url_contains_ip_port_and_token(tmp_path):
    panel = tw.WebPanel(FakeSession(), tmp_path, port=8095, token="abc")
    assert panel.url("10.0.0.7") == "http://10.0.0.7:8095/?t=abc"


def test_web_module_uses_stdlib_only():
    import ast
    tree = ast.parse((ROOT / "teleop" / "utils" / "teach_web.py").read_text())
    mods = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    allowed = set(sys.stdlib_module_names) | {"__future__", "teleop"}
    assert mods <= allowed, mods - allowed
