"""Offline web control panel for G1D teach/replay (real HTTP on 127.0.0.1, fake IO)."""
import http.client
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
        self.busy = {"replay": None, "recording": []}

    def request_teach(self):
        self.gate.wait(10)
        self.calls.append("teach")
        return True

    def request_save(self):
        self.gate.wait(10)
        self.calls.append("save")
        return True

    def request_replay(self, name, *, origin="terminal", repeat=1, continuous=False):
        self.gate.wait(10)
        self.calls.append(("replay", name, origin, repeat, continuous))
        return True

    def request_quit(self):
        self.quit_flag.set()
        self.calls.append("quit")
        return True

    @property
    def quit_requested(self):
        return self.quit_flag.is_set()

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


def upload_call(panel, name, payload, token=TOKEN, client="A"):
    headers = {"X-Client-Id": client, "X-Token": token, "Content-Type": "application/jsonl",
               "X-Filename": name}
    req = urllib.request.Request(f"http://127.0.0.1:{panel.port}/api/upload", data=payload,
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


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


def test_session_endpoint_is_token_protected_read_only_and_exact(fake):
    _, panel = fake
    before = panel._controller
    status, body = call(panel, "GET", "/api/session")
    assert status == 200 and body == {"ok": True, "session_id": panel.session_id}
    assert panel._controller == before
    assert call(panel, "GET", "/api/session", token="wrong")[0] == 401


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
    kind, name, origin, repeat, continuous = session.calls[-1]
    assert kind == "replay" and origin == "web"
    assert repeat == 1 and not continuous
    assert Path(name) == tmp_path / path.name     # resolved inside the teach dir only
    assert call(panel, "POST", "/api/stop", {})[0] == 200
    assert session.calls[-1] == "quit"


def test_web_replay_exposes_finite_repeat_and_continuous_modes(fake, tmp_path):
    session, panel = fake
    path = saved_file(tmp_path)
    assert call(panel, "POST", "/api/replay", {"name": path.name, "repeat": 3})[0] == 202
    assert wait_until(lambda: len(session.calls) == 1)
    assert session.calls[-1][3:] == (3, False)
    assert call(panel, "POST", "/api/replay", {"name": path.name, "continuous": True})[0] == 202
    assert wait_until(lambda: len(session.calls) == 2)
    assert session.calls[-1][3:] == (1, True)


@pytest.mark.parametrize("repeat", [True, False, 1.0, "1", 0, 101])
def test_web_replay_rejects_non_integral_or_out_of_range_repeat(fake, tmp_path, repeat):
    _session, panel = fake
    path = write_recording(tmp_path / "repeat.jsonl", [0.0, 0.02], np.zeros((2, 14)),
                           {"left": np.zeros((2, 7)), "right": np.zeros((2, 7))})

    assert call(panel, "POST", "/api/replay", {"name": path.name, "repeat": repeat})[0] == 400


@pytest.mark.parametrize("repeat", [1, 100])
def test_web_replay_accepts_repeat_boundaries(fake, tmp_path, repeat):
    session, panel = fake
    path = write_recording(tmp_path / "repeat.jsonl", [0.0, 0.02], np.zeros((2, 14)),
                           {"left": np.zeros((2, 7)), "right": np.zeros((2, 7))})

    assert call(panel, "POST", "/api/replay", {"name": path.name, "repeat": repeat})[0] == 202
    assert wait_until(lambda: len(session.calls) == 1)
    assert session.calls[-1][3:] == (repeat, False)


def test_web_replay_continuous_remains_unbounded_until_stop(fake, tmp_path):
    session, panel = fake
    path = write_recording(tmp_path / "repeat.jsonl", [0.0, 0.02], np.zeros((2, 14)),
                           {"left": np.zeros((2, 7)), "right": np.zeros((2, 7))})

    assert call(panel, "POST", "/api/replay", {"name": path.name, "continuous": True})[0] == 202
    assert wait_until(lambda: len(session.calls) == 1)
    assert session.calls[-1][3:] == (1, True)


def test_web_page_documents_and_limits_finite_repeats(fake):
    _session, panel = fake
    status, body = call(panel, "GET", "/")

    assert status == 200
    assert 'type="number" min="1" max="100"' in body
    assert "repetições finitas: de 1 a 100" in body.lower()


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


def test_cross_panel_delete_uses_unique_trash_destination(tmp_path):
    session = FakeSession()
    first = start_panel(session, tmp_path); second = start_panel(session, tmp_path)
    try:
        path = saved_file(tmp_path); (tmp_path / "trash").mkdir()
        (tmp_path / "trash" / path.name).write_bytes(b"existing")
        results = []; barrier = threading.Barrier(2)
        def remove(panel):
            barrier.wait(); results.append(panel.delete(path.name))
        threads = [threading.Thread(target=remove, args=(panel,)) for panel in (first, second)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        assert sum(status == 200 for status, _ in results) == 1
        assert (tmp_path / "trash" / path.name).read_bytes() == b"existing"
        assert len(list((tmp_path / "trash").glob("*.jsonl"))) == 2
    finally:
        first.stop(); second.stop()


def test_upload_valid_recording_promotes_atomically_and_never_queues_replay(fake, tmp_path):
    session, panel = fake
    source = saved_file(tmp_path / "source")
    payload = source.read_bytes()
    status, body = upload_call(panel, "phone_take.jsonl", payload)
    assert status == 201 and body["ok"]
    target = tmp_path / body["name"]
    assert target.is_file() and target.read_bytes() == payload
    assert not list(tmp_path.glob("*.part"))
    assert session.calls == []


def test_concurrent_uploads_same_name_never_overwrite_each_other(tmp_path):
    payload_a = saved_file(tmp_path / "a", "20261019T100000_000000Z", n=10).read_bytes()
    payload_b = saved_file(tmp_path / "b", "20261019T110000_000000Z", n=10).read_bytes()
    p1 = start_panel(FakeSession(), tmp_path)
    p2 = start_panel(FakeSession(), tmp_path)
    results = []
    barrier = threading.Barrier(2)

    def upload(panel, payload):
        barrier.wait()
        results.append(upload_call(panel, "same.jsonl", payload))

    try:
        threads = [threading.Thread(target=upload, args=(p1, payload_a)),
                   threading.Thread(target=upload, args=(p2, payload_b))]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        assert sorted(code for code, _ in results) == [201, 201]
        names = [body["name"] for _, body in results]
        assert len(set(names)) == 2
        assert {(tmp_path / name).read_bytes() for name in names} == {payload_a, payload_b}
    finally:
        p1.stop(); p2.stop()


@pytest.mark.parametrize("name", ["../evil.jsonl", "/tmp/evil.jsonl", "a/b.jsonl", "evil.txt"])
def test_upload_rejects_traversal_and_bad_filename_without_partial(fake, tmp_path, name):
    session, panel = fake
    source = saved_file(tmp_path / "source")
    status, _ = upload_call(panel, name, source.read_bytes())
    assert status == 400
    assert session.calls == []
    assert not list(tmp_path.glob("*.part"))


def test_upload_rejects_malformed_recording_and_cleans_partial(fake, tmp_path):
    session, panel = fake
    status, body = upload_call(panel, "bad.jsonl", b'{"type":"header"}\n')
    assert status == 400 and not body["ok"]
    assert session.calls == []
    assert not list(tmp_path.glob("*.part"))


def test_upload_rejects_declared_oversize_without_reading_body_or_leaving_partial(fake, tmp_path):
    session, panel = fake
    conn = http.client.HTTPConnection("127.0.0.1", panel.port, timeout=3.0)
    conn.putrequest("POST", "/api/upload")
    conn.putheader("X-Token", TOKEN)
    conn.putheader("X-Client-Id", "A")
    conn.putheader("X-Filename", "too_large.jsonl")
    conn.putheader("Content-Length", str(tw.MAX_UPLOAD_BYTES + 1))
    conn.endheaders()
    response = conn.getresponse()
    payload = json.loads(response.read())
    conn.close()
    assert response.status == 413 and not payload["ok"]
    assert session.calls == []
    assert not (tmp_path / "too_large.jsonl").exists()
    assert not list(tmp_path.glob(".g1d-upload-*.part"))


def test_upload_rejects_truncated_http_body_and_removes_partial(fake, tmp_path):
    session, panel = fake
    source = saved_file(tmp_path / "source").read_bytes()
    conn = http.client.HTTPConnection("127.0.0.1", panel.port, timeout=3.0)
    conn.putrequest("POST", "/api/upload")
    conn.putheader("X-Token", TOKEN)
    conn.putheader("X-Client-Id", "A")
    conn.putheader("X-Filename", "truncated.jsonl")
    conn.putheader("Content-Length", str(len(source) + 100))
    conn.endheaders(source[:100])
    conn.sock.shutdown(1)
    response = conn.getresponse()
    payload = json.loads(response.read())
    conn.close()
    assert response.status == 400 and not payload["ok"]
    assert session.calls == []
    assert not (tmp_path / "truncated.jsonl").exists()
    assert not list(tmp_path.glob(".g1d-upload-*.part"))


@pytest.mark.parametrize("mutate", [
    lambda records: records[0].__setitem__("profile", "OTHER"),
    lambda records: records[0].__setitem__("robot", "OTHER"),
    lambda records: records[0].__setitem__("arm_joint_names", list(reversed(tr.ARM_JOINT_NAMES))),
    lambda records: records[0].__setitem__("hand_sides", ["right", "left"]),
    lambda records: records[1].__setitem__("arm_q", [0.0]),
    lambda records: records[1].__setitem__("arm_q", [99.0] + [0.0] * 13),
    lambda records: records[1].__setitem__("left_hand_q", [99.0] + [0.0] * 6),
    lambda records: records[1].__setitem__("t", float("nan")),
])
def test_upload_endpoint_rejects_schema_nonfinite_vectors_and_limits(fake, tmp_path, mutate):
    fake[1].arm_lower = np.full(14, -2.0)
    fake[1].arm_upper = np.full(14, 2.0)
    source = saved_file(tmp_path / "source")
    records = [json.loads(line) for line in source.read_text().splitlines()]
    mutate(records)
    payload = ("\n".join(json.dumps(record, allow_nan=True) for record in records) + "\n").encode()
    status, body = upload_call(fake[1], "rejected.jsonl", payload)
    assert status == 400 and not body["ok"]
    assert not (tmp_path / "rejected.jsonl").exists()
    assert not list(tmp_path.glob(".g1d-upload-*.part"))


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


# ------------------------------------------------------------------ replay web authority
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


def test_read_only_client_polling_does_not_stop_replay(tmp_path):
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
        assert session.mode in (tr.Mode.REPLAY_RAMP, tr.Mode.REPLAY_PLAY)
    finally:
        panel.stop()


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
        assert panel.canonical_url in "\n".join(lines)
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


def test_unauthorized_api_error_is_portuguese(tmp_path):
    panel = start_panel(FakeSession(), tmp_path)
    try:
        status, body = call(panel, "POST", "/api/status", token=None)
        assert status == 401 and body["error"] == "token necessário"
    finally:
        panel.stop()


def test_web_module_uses_stdlib_only():
    import ast
    tree = ast.parse((ROOT / "teleop" / "utils" / "teach_web.py").read_text())
    mods = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    allowed = set(sys.stdlib_module_names) | {"__future__", "teleop"}
    assert mods <= allowed, mods - allowed
