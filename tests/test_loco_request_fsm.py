"""Opt-in --loco-request-fsm 500: one SetFsmId(500) in the preflight only. Fakes only."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from teleop.utils import loco_preflight as lp
from test_loco_botbrain_parity import FakeWrapper

ROOT = Path(__file__).resolve().parents[1]


class Clk:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def go(w, request="500", **kw):
    c = Clk()
    kw.setdefault("sleep", c.sleep)
    kw.setdefault("clock", c)
    return lp.run_loco_preflight(w, request_fsm=request, **kw)


def sent(w):
    return [c for c in w.calls if c[0] == "set_fsm"]


@pytest.mark.parametrize("req", ["none", None])
def test_none_sends_nothing(req):
    w = FakeWrapper(fsm_seq=(501,))
    r = go(w, req)
    assert r["loco_enabled"] and sent(w) == [] and r["fsm_requested"] is None


def test_default_param_is_none():
    w = FakeWrapper(fsm_seq=(501,))
    r = lp.run_loco_preflight(w, sleep=lambda s: None)
    assert r["loco_enabled"] and sent(w) == []


def test_500_when_already_500_sends_nothing():
    w = FakeWrapper(fsm_seq=(500,))
    r = go(w)
    assert r["loco_enabled"] and sent(w) == []
    assert r["fsm_before"] == 500 and r["fsm_after"] == 500 and r["fsm_requested"] == 500
    assert "500" in r["fsm_banner"]


def test_501_to_500_confirms():
    w = FakeWrapper(fsm_seq=(501, 501, 501, 500))
    r = go(w)
    assert sent(w) == [("set_fsm", 500)]
    assert r["loco_enabled"] and r["refusal_reason"] is None
    assert (r["fsm_before"], r["fsm_requested"], r["fsm_after"], r["set_fsm_rc"]) == (501, 500, 500, 0)
    assert r["fsm_confirm_s"] is not None and r["fsm_confirm_s"] <= 3.0
    assert r["fsm_banner"] == "FSM solicitado: 501 -> 500 (confirmado)"
    assert ("zero",) in w.calls


def test_501_not_confirmed_times_out_disables_without_retry():
    w = FakeWrapper(fsm_seq=(501,))
    r = go(w)
    assert len(sent(w)) == 1
    assert not r["loco_enabled"] and r["refusal_reason"].startswith("fsm_request_failed:")
    assert r["fsm_after"] == 501 and "501" in r["refusal_reason"]
    assert not any(c[0] in ("zero", "speed") for c in w.calls)
    assert "FALHOU" in r["fsm_banner"] or "falh" in r["fsm_banner"].lower()
    reads = [c for c in w.calls if c[0] == "read_fsm"]
    assert 3 <= len(reads) < 60   # bounded polling


def test_set_fsm_rc_error_disables():
    w = FakeWrapper(fsm_seq=(501,), set_fsm_rc=3102)
    r = go(w)
    assert len(sent(w)) == 1
    assert not r["loco_enabled"] and r["refusal_reason"] == "fsm_request_failed:rc=3102"
    assert r["set_fsm_rc"] == 3102


@pytest.mark.parametrize("fsm", [0, 1, 2, 4, 801, 200])
def test_other_fsm_refused_without_sending(fsm):
    w = FakeWrapper(fsm_seq=(fsm,))
    r = go(w)
    assert sent(w) == [] and not r["loco_enabled"]
    assert r["refusal_reason"] == f"fsm_not_walk:{fsm}"


def test_unreadable_fsm_refused_without_sending():
    w = FakeWrapper(fsm_seq=(None,))
    r = go(w)
    assert sent(w) == [] and r["refusal_reason"] == "fsm_unreadable"


@pytest.mark.parametrize("bad", ["801", "501", "0", "abc", 801, 501, ""])
def test_invalid_value_rejected(bad):
    w = FakeWrapper(fsm_seq=(501,))
    with pytest.raises(ValueError, match="500"):
        go(w, bad)
    assert w.calls == []


def test_parse_request_fsm():
    assert lp.parse_request_fsm("none") is None
    assert lp.parse_request_fsm("500") == 500
    assert lp.parse_request_fsm(500) == 500


def test_wrapper_set_fsm_id_uses_cfg_client_7101(monkeypatch):
    from test_loco_explicit_stop import _load_switcher, FakeClient
    calls = []

    class C(FakeClient):
        def _Call(self, api, p):
            calls.append((api, p))
            return 0, ""
    w = _load_switcher(monkeypatch, C).LocoClientWrapper()
    assert w.set_fsm_id(500, timeout=0.5) == 0
    assert calls == [(7101, '{"data": 500}')]
    assert w._cfg_client is not w.client and w._cfg_client.timeouts[-1] == 0.5
    with pytest.raises(ValueError):
        w.set_fsm_id(801)
    assert len(calls) == 1


def test_cli_and_launcher_wiring():
    src = (ROOT / "teleop/teleop_hand_and_arm.py").read_text()
    assert "--loco-request-fsm" in src and "default='none'" in src.split("--loco-request-fsm")[1].split("\n")[0]
    assert "choices=['none', '500']" in src.split("--loco-request-fsm")[1].split("\n")[0]
    assert "request_fsm=args.loco_request_fsm" in src
    sh = (ROOT / "teleop/run_g1_quest_dex3.sh").read_text()
    assert '--loco-request-fsm "${G1_LOCO_REQUEST_FSM:-none}"' in sh
    assert 'echo "FSM solicitado (G1_LOCO_REQUEST_FSM): ${G1_LOCO_REQUEST_FSM:-none}"' in sh
