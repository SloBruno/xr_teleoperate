"""Captura passiva de CoG: fakes apenas, sem DDS/robo."""
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_cog_capture as an  # noqa: E402
import capture_app_cog_adjust as cc  # noqa: E402


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def adv(self, dt):
        self.t += dt


def req(api_id, param="", rid=1):
    return NS(header=NS(identity=NS(id=rid, api_id=api_id), lease=NS(id=0), policy=NS(priority=0, noreply=False)),
              parameter=param, binary=[])


def resp(api_id, code=0, data="", rid=1):
    return NS(header=NS(identity=NS(id=rid, api_id=api_id), status=NS(code=code)), data=data, binary=[1, 2])


def test_source_is_passive():
    for f in ("capture_app_cog_adjust.py", "analyze_cog_capture.py"):
        src = (ROOT / "tools" / f).read_text()
        code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        for bad in ("ChannelPublisher", "DataWriter", "CreateSendChannel", "SetWriter", ".Write(", ".write_sample",
                    "ChannelSubscriber", "SportClient", "LocoClient", "_Call("):
            if bad in ("ChannelSubscriber",):
                continue
            assert bad not in code, (f, bad)
        assert not re.search(r"\bSet[A-Z]\w*\(", code), f


def test_api_names_and_unknown():
    assert cc.api_name("rt/api/sport/request", 7108).startswith("SET_MOTION")
    assert cc.api_name("rt/api/sport/request", 9999) is None
    assert cc.api_name("rt/api/xyz/request", 102) == "LEASE_RENEWAL"


def test_on_api_records_and_redacts(tmp_path):
    c = cc.Capture(tmp_path, clock=Clock())
    c.mark("antes")
    c.on_api("rt/api/sport/request", "request", req(7777, '{"token": "abc", "x": [1,2]}'))
    c.on_api("rt/api/sport/response", "response", resp(7777, 3, "oops"))
    c.close()
    ev = [json.loads(l) for l in open(tmp_path / "api_events.jsonl")]
    assert ev[0]["api_id"] == 7777 and ev[0]["known"] is False and ev[0]["phase"] == "antes"
    assert "abc" not in ev[0]["parameter"] and "<redacted>" in ev[0]["parameter"]
    assert ev[1]["code"] == 3 and ev[1]["binary_hex"] == "0102"


def test_size_limit_and_errors_are_counters(tmp_path):
    c = cc.Capture(tmp_path, api_max_bytes=300)
    for i in range(20):
        c.on_api("rt/api/sport/request", "request", req(7108, "x" * 50, i))
    assert c.api.dropped > 0 and c.api.size <= 300
    c.on_api("rt/api/sport/request", "request", object())  # sem header: nao levanta
    c.on_state("rt/sportmodestate", None)  # None: nao levanta
    c.close()


def test_markers_and_phase_counts(tmp_path):
    clk = Clock()
    c = cc.Capture(tmp_path, clock=clk)
    c.on_state("rt/sportmodestate", {"a": 1}, "T", None, 0)
    assert [c.mark(), c.mark(), c.mark()] == ["antes", "mexendo", "depois"]
    c.on_api("rt/api/arm/request", "request", req(7106))
    assert c.counts["rt/api/arm/request"]["phase"] == {"depois": 1}
    assert c.counts["rt/sportmodestate"]["phase"] == {"inicio": 1}


def test_state_decimation_and_projection(tmp_path):
    clk = Clock()
    c = cc.Capture(tmp_path, clock=clk)

    @dataclass
    class Imu:
        rpy: list = field(default_factory=lambda: [0.1, 0.2, 0.3])
        quaternion: list = field(default_factory=lambda: [1, 0, 0, 0])

    @dataclass
    class Low:
        imu_state: Imu = field(default_factory=Imu)
        motor_state: list = field(default_factory=lambda: [1] * 5)

    for _ in range(10):
        c.on_state("rt/lowstate", Low(), "T", ("imu_state",), 0.5)
        clk.adv(0.1)
    c.close()
    rows = [json.loads(l) for l in open(tmp_path / "state_decimated.jsonl")]
    assert len(rows) == 2 and set(rows[0]["data"]) == {"imu_state"}
    assert rows[0]["data"]["imu_state"]["rpy"] == [0.1, 0.2, 0.3]
    assert c.counts["rt/lowstate"]["total"] == 10


def test_project_nested_list():
    d = {"motor_state": [{"q": 1, "dq": 2}, {"q": 3, "dq": 4}]}
    assert cc.project(d, ("motor_state.q",)) == {"motor_state.q": [1, 3]}


def test_wanted_subscription_filters():
    assert cc.wanted_subscription("rt/api/config/request", "unitree_api::msg::dds_::Request_")[0] == "api_request"
    assert cc.wanted_subscription("rt/api/x/response", "unitree_api.msg.dds_.Response_")[0] == "api_response"
    assert cc.wanted_subscription("rt/lowstate", "unitree_hg.msg.dds_.LowState_")[1] == ("imu_state", "wireless_remote")
    assert cc.wanted_subscription("rt/random", "foo.Bar") is None
    assert cc.wanted_subscription("rt/lf/whatever", "unknown.T") is None


def test_discovery_and_subscription_with_fakes(tmp_path):
    c = cc.Capture(tmp_path)
    queue = {"rt/api/config/request": [req(8001, "[1,2,3]")], "rt/sportmodestate": [{"mode": 1}]}
    opened = []

    def opener(topic, tn):
        opened.append(topic)
        return lambda n: queue.pop(topic, [])

    subs = cc.Subscriptions(c, opener)
    fetch = lambda: [
        {"topic": "rt/api/config/request", "type": "unitree_api::msg::dds_::Request_", "kind": "pub", "participant": "p1"},
        {"topic": "rt/mystery", "type": "foo.Bar", "kind": "pub"},
        {"topic": "rt/sportmodestate", "type": "unitree_go.msg.dds_.SportModeState_", "kind": "pub"},
    ]
    assert cc.discover_once(c, fetch, subs) == 3
    assert "rt/mystery" in c.topics and "rt/mystery" not in opened
    assert subs.poll_once() == 2
    assert c.counts["rt/api/config/request"]["total"] == 1
    cc.discover_once(c, fetch, subs)
    assert opened.count("rt/api/config/request") == 1  # nao reabre


def test_discovery_failures_are_counters(tmp_path):
    c = cc.Capture(tmp_path)
    subs = cc.Subscriptions(c, lambda t, ty: (_ for _ in ()).throw(RuntimeError("x")))
    assert cc.discover_once(c, lambda: (_ for _ in ()).throw(RuntimeError), subs) == 0
    cc.discover_once(c, lambda: [{"topic": "rt/api/a/request", "type": "unitree_api.msg.dds_.Request_",
                                  "kind": "pub"}], subs)
    assert c.errors == {"discover_fail": 1, "subscribe_fail": 1}


def test_file_diff_and_watcher(tmp_path):
    cfg = tmp_path / "balance_config.json"
    cfg.write_text('{"cog": 0.0}\n')
    other = tmp_path / "unrelated.txt"
    other.write_text("x")
    cap = cc.Capture(tmp_path / "out", clock=Clock())
    fw = cc.FileWatcher(cap, roots=(str(tmp_path),))
    fw.start()
    assert str(cfg) in fw.paths and str(other) not in fw.paths
    import os
    cfg.write_text('{"cog": 0.02}\n')
    os.utime(cfg, (cap.t0 + 5, cap.t0 + 5))
    fw.finish()
    ch = json.load(open(cap.dir / "file_changes.json"))
    assert ch["changes"][0]["change"] == "modified"
    d = next((cap.dir / "diffs").iterdir()).read_text()
    assert '-{"cog": 0.0}' in d and '+{"cog": 0.02}' in d


def test_file_search_skips_secrets(tmp_path):
    (tmp_path / "token_config.json").write_text("{}")
    (tmp_path / "com_offset.yaml").write_text("a: 1")
    paths, _ = cc.find_candidates((str(tmp_path),))
    assert [Path(p).name for p in paths] == ["com_offset.yaml"]


def test_netwatcher_skips_without_tcpdump(tmp_path):
    cap = cc.Capture(tmp_path)
    nw = cc.NetWatcher(cap, run=lambda *a, **k: NS(stdout="", returncode=1), which=lambda n: None)
    nw.start()
    assert nw.status == "tcpdump ausente"
    nw2 = cc.NetWatcher(cap, run=lambda *a, **k: NS(stdout="", returncode=1), which=lambda n: "/usr/bin/tcpdump")
    nw2.start_tcpdump()
    assert "sudo" in nw2.status and nw2.proc is None


def test_full_run_and_report(tmp_path):
    clk = Clock()
    args = cc.parse_args(["--out-base", str(tmp_path), "--no-files", "--no-net", "--no-stdin", "--duration", "3"])
    q = {"rt/api/config/request": [], "rt/sportmodestate": []}
    state = {"n": 0}

    def opener(topic, tn):
        def rd(n):
            if topic == "rt/api/config/request" and state["n"] == 2:
                state["n"] += 1
                return [req(8123, "[0.01, 0.0]")]
            return []
        return rd

    fetch = lambda: [{"topic": "rt/api/config/request", "type": "unitree_api.msg.dds_.Request_", "kind": "pub"}]
    real_poll = cc.Subscriptions.poll_once

    def poll(self, max_n=64):
        clk.adv(1.0)
        state["n"] = min(state["n"] + 1, 2) if state["n"] < 2 else state["n"]
        if state["n"] == 1:
            self.cap.mark()
        if state["n"] == 2 and not getattr(self, "_m2", False):
            self._m2 = True
            self.cap.mark()
        return real_poll(self, max_n)

    cc.Subscriptions.poll_once = poll
    try:
        out = cc.run_capture(args, fetch=fetch, open_reader=opener, clock=clk)
    finally:
        cc.Subscriptions.poll_once = real_poll
    rep = (out / "report.md").read_text()
    assert "8123" in rep and "DESCONHECIDO" in rep
    assert (out / "meta.json").exists() and json.load(open(out / "meta.json"))["passive"] is True


def test_analyzer_deltas():
    snaps = [{"phase": "antes", "imu_state": {"rpy": [0.0, 0.0, 0.0]}},
             {"phase": "antes", "imu_state": {"rpy": [0.0, 0.0, 0.0]}},
             {"phase": "depois", "imu_state": {"rpy": [0.0, 0.1, 0.0]}}]
    rows = an.snapshot_deltas(snaps)
    assert rows[0][0] == "imu_state.rpy.1" and abs(rows[0][3] - 0.1) < 1e-9


def test_stdin_markers(tmp_path):
    import threading
    c = cc.Capture(tmp_path)
    stop = threading.Event()
    th = cc.stdin_marker_thread(c, stop, io.StringIO("\n\n\n"))
    th.join(2)
    assert [m["name"] for m in c.markers] == ["antes", "mexendo", "depois"]


def test_string_topics_hash_only_for_webrtc(tmp_path):
    c = cc.Capture(tmp_path)
    assert cc.wanted_subscription("rt/xfk_webrtcreq", "std_msgs::msg::dds_::String_")[0] == "string"
    c.on_string("rt/xfk_webrtcreq", NS(data="v=0 SECRET sdp"))
    c.on_string("rt/gpt_state", NS(data='{"token": "zzz", "a": 1}'))
    c.close()
    ev = [json.loads(l) for l in open(tmp_path / "api_events.jsonl")]
    assert "text" not in ev[0] and ev[0]["len"] == 14
    assert "zzz" not in ev[1]["text"]


def test_missing_idl_type_is_unsupported_not_error(tmp_path):
    c = cc.Capture(tmp_path)
    def opener(t, tn):
        raise AttributeError("no such type")
    subs = cc.Subscriptions(c, opener)
    subs.ensure("rt/lf/dex3/left/state", "unitree_hg::msg::dds_::HandState_")
    assert c.unsupported and not c.errors and subs.poll_once() == 0


def test_tcpdump_requires_passwordless_sudo(tmp_path):
    calls = []
    def run(cmd, **k):
        calls.append(cmd)
        return NS(stdout="", returncode=1)
    nw = cc.NetWatcher(cap=cc.Capture(tmp_path), run=run, which=lambda n: "/usr/bin/tcpdump")
    nw.start_tcpdump()
    assert calls == [["sudo", "-n", "true"]] and nw.proc is None
