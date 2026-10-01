import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teleop.utils import quest_controls
from teleop.utils import loco_preflight as lp


class FakeWrapper:
    def __init__(self, fsm_seq=(500,), speed_rc=(0,), balance_rc=0, zero_rc=0, set_fsm_rc=0):
        self.fsm_seq = list(fsm_seq)
        self.speed_rc = list(speed_rc)
        self.balance_rc = balance_rc
        self.zero_rc = zero_rc
        self.set_fsm_rc = set_fsm_rc
        self.calls = []

    def read_fsm_id(self, timeout=0.3):
        self.calls.append(("read_fsm", timeout))
        if len(self.fsm_seq) > 1:
            return self.fsm_seq.pop(0)
        return self.fsm_seq[0]

    def set_speed_mode(self, mode):
        self.calls.append(("speed", mode))
        return self.speed_rc.pop(0) if len(self.speed_rc) > 1 else self.speed_rc[0]

    def set_balance_mode(self, mode):
        self.calls.append(("balance", mode))
        return self.balance_rc

    def set_fsm_id(self, fsm_id):
        self.calls.append(("set_fsm", fsm_id))
        return self.set_fsm_rc

    def checked_zero(self):
        self.calls.append(("zero",))
        return self.zero_rc


def names(w):
    return [c[0] for c in w.calls]


def run(w, **kw):
    kw.pop('auto_fsm', None)
    kw.setdefault("sleep", lambda s: None)
    return lp.run_loco_preflight(w, **kw)


# --- preflight ---------------------------------------------------------------

def test_only_regular_fsm_ids_accepted():
    assert lp.ACCEPTED_FSM_IDS == frozenset({500, 501})


@pytest.mark.parametrize("fsm", [500, 501])
def test_preflight_ok_in_regular_walk(fsm):
    w = FakeWrapper(fsm_seq=(fsm,))
    r = run(w)
    assert r["loco_enabled"] and r["refusal_reason"] is None
    assert ("speed", 0) in w.calls and ("balance", 0) in w.calls
    assert not any(c[0] == "set_fsm" for c in w.calls)
    assert names(w).index("speed") < names(w).index("zero")


@pytest.mark.parametrize("fsm", [801, 1, 4, 0, 706])
def test_preflight_refuses_non_regular_and_sends_nothing(fsm):
    w = FakeWrapper(fsm_seq=(fsm,))
    r = run(w)
    assert not r["loco_enabled"] and r["refusal_reason"] == f"fsm_not_walk:{fsm}"
    assert "R1+X" in r["message"] and str(fsm) in r["message"]
    assert set(names(w)) == {"read_fsm"}


def test_preflight_refuses_unreadable_fsm():
    w = FakeWrapper(fsm_seq=(None,))
    r = run(w)
    assert r["refusal_reason"] == "fsm_unreadable" and not r["loco_enabled"]
    assert set(names(w)) == {"read_fsm"}


def test_speed_mode_retries_then_ok():
    w = FakeWrapper(speed_rc=(3104, 3104, 0))
    r = run(w)
    assert r["loco_enabled"] and r["set_speed_mode_rc"] == 0
    assert names(w).count("speed") == 3


def test_speed_mode_rejected_is_best_effort_not_fatal(caplog):
    w = FakeWrapper(fsm_seq=(501,), speed_rc=(3103,))
    with caplog.at_level("WARNING"):
        r = run(w)
    assert r["loco_enabled"] and r["preflight_ok"] and r["refusal_reason"] is None
    assert r["set_speed_mode_rc"] == 3103 and r["set_speed_mode_ok"] is False
    assert names(w).count("speed") == lp.SPEED_MODE_ATTEMPTS
    assert "zero" in names(w) and "balance" in names(w)
    msgs = [m for m in caplog.messages if "SetSpeedMode não aceito pelo firmware (rc=3103)" in m]
    assert len(msgs) == 1 and "perfil padrão do robô" in msgs[0]


def test_speed_mode_ok_flag_true_on_success():
    r = run(FakeWrapper())
    assert r["set_speed_mode_ok"] is True and r["set_speed_mode_rc"] == 0


def test_speed_mode_rejected_still_gated_by_zero_move():
    r = run(FakeWrapper(speed_rc=(3103,), zero_rc=3104))
    assert not r["loco_enabled"] and r["refusal_reason"] == "zero_move_failed:3104"


def test_speed_mode_rejected_still_gated_by_fsm():
    w = FakeWrapper(fsm_seq=(801,), speed_rc=(3103,))
    r = run(w)
    assert not r["loco_enabled"] and r["refusal_reason"] == "fsm_not_walk:801"


def test_zero_move_failure_refuses():
    r = run(FakeWrapper(zero_rc=3104))
    assert r["refusal_reason"] == "zero_move_failed:3104" and not r["loco_enabled"]


def test_continuous_gait_false_failure_is_not_fatal():
    r = run(FakeWrapper(balance_rc=3104))
    assert r["loco_enabled"]


# --- caps ------------------------------------------------------------------------

def test_defaults_are_botbrain_frontend_profile():
    assert quest_controls.MIN_OPERATOR_WALK_SPEED_MPS == 0.5
    assert quest_controls.MIN_OPERATOR_TURN_RATE_RADPS == 0.3
    assert quest_controls.MAX_WALK_SPEED_CAP_MPS == 0.6 and quest_controls.MAX_TURN_RATE_CAP_RADPS == 1.0
    assert quest_controls.joystick_to_locomotion((0.0, -1.0), (0.0, 0.0)) == (0.5, 0.0, 0.0)


def test_cap_scales_full_stick_and_keeps_curve():
    cmd = quest_controls.joystick_to_locomotion((0.0, -1.0), (1.0, 0.0), walk_cap=0.35, turn_cap=0.2)
    assert cmd == (0.35, 0.0, -0.2)
    half = quest_controls.joystick_to_locomotion((0.0, -0.56), (0, 0), walk_cap=0.30)[0]
    base = quest_controls.joystick_to_locomotion((0.0, -0.56), (0, 0))[0]
    assert half == pytest.approx(base * 0.6)


def test_resolve_caps_defaults_env_and_validation():
    assert quest_controls.resolve_speed_caps(None, None, {}) == (0.5, 0.3)
    assert quest_controls.resolve_speed_caps(None, None, {"G1_WALK_SPEED_CAP": "0.25", "G1_TURN_RATE_CAP": "0.4"}) == (0.25, 0.4)
    assert quest_controls.resolve_speed_caps(0.15, 0.1, {"G1_WALK_SPEED_CAP": "0.25"}) == (0.15, 0.1)
    for bad in (0.0, -0.1, 0.61, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            quest_controls.resolve_speed_caps(bad, None, {})
    with pytest.raises(ValueError):
        quest_controls.resolve_speed_caps(None, 1.01, {})
    with pytest.raises(ValueError):
        quest_controls.resolve_speed_caps(None, None, {"G1_WALK_SPEED_CAP": "abc"})


def test_cap_banner_shows_active_values():
    b = quest_controls.speed_cap_banner(0.35, 0.5)
    assert "0.35" in b and "0.50" in b and "ACTIVE" in b


def test_dispatch_uses_caps_and_loco_stick_timeout():
    class W:
        def __init__(self):
            self.moves = []
            self.stops = []

        def Move(self, *c):
            self.moves.append(c)

        def StopMove(self, reason):
            self.stops.append(reason)
    w = W()
    out = quest_controls.dispatch_joystick_locomotion(w, True, True, (0, -1), (0, 0), walk_cap=0.3)
    assert out == (0.3, 0.0, 0.0)


def test_loco_stick_freshness_is_0_2s():
    assert quest_controls.LOCO_STICK_TIMEOUT_S == 0.2
    assert quest_controls.loco_stick_is_fresh(10.0, now=10.19)
    assert not quest_controls.loco_stick_is_fresh(10.0, now=10.21)
    assert not quest_controls.loco_stick_is_fresh(0.0, now=1.0)


# --- watchdog ----------------------------------------------------------------------

def test_watchdog_stops_once_when_stale_after_nonzero():
    stops = []
    wd = lp.LocoWatchdog(lambda reason: stops.append(reason) or 0, timeout_s=0.2)
    wd.feed(True, now=1.0)       # nonzero command dispatched
    assert wd.check(now=1.15) is False
    assert wd.check(now=1.25) is True and stops == ["watchdog_timeout"]
    assert wd.check(now=1.5) is False and len(stops) == 1


def test_watchdog_retries_until_ack_and_rearms_on_new_command():
    rcs = [3104, 0]
    stops = []
    wd = lp.LocoWatchdog(lambda r: stops.append(r) or rcs.pop(0), timeout_s=0.2)
    wd.feed(True, now=0.0)
    wd.check(now=0.3)
    wd.check(now=0.4)
    assert len(stops) == 2 and wd.last_rc == 0
    wd.check(now=0.5)
    assert len(stops) == 2
    wd.feed(True, now=1.0)
    wd.check(now=1.3)
    assert len(stops) == 3 and wd.trips == 2


def test_watchdog_idle_when_zero_command():
    stops = []
    wd = lp.LocoWatchdog(lambda r: stops.append(r) or 0, timeout_s=0.2)
    wd.feed(False, now=0.0)
    assert wd.check(now=5.0) is False and not stops


def test_watchdog_stop_failure_never_raises():
    def boom(r):
        raise RuntimeError("x")
    wd = lp.LocoWatchdog(boom, timeout_s=0.2)
    wd.feed(True, now=0.0)
    assert wd.check(now=1.0) is True and wd.stop_failures == 1


# --- async move sender -------------------------------------------------------------

def test_move_sender_keeps_only_latest_and_records_rc():
    gate = threading.Event()
    sent = []

    def send(vx, vy, w):
        sent.append((vx, vy, w))
        gate.wait(1.0)
        return 0
    s = lp.LatestMoveSender(send)
    s.start()
    s.submit((0.1, 0, 0))
    deadline = time.time() + 1
    while not sent and time.time() < deadline:
        time.sleep(0.001)
    for v in (0.11, 0.12, 0.0):
        s.submit((v, 0, 0))
    gate.set()
    deadline = time.time() + 1
    while len(sent) < 2 and time.time() < deadline:
        time.sleep(0.001)
    s.stop()
    assert sent == [(0.1, 0, 0), (0.0, 0, 0)]
    assert s.last_rc == 0 and s.sent == 2


# --- wrapper wiring ------------------------------------------------------------

def test_wrapper_move_goes_through_sender_and_config_uses_blocking_client(monkeypatch):
    from tests.test_loco_explicit_stop import _load_switcher, FakeClient
    FakeClient._Call = lambda self, api, p: (0, None)
    FakeClient.SetBalanceMode = lambda self, m: 0
    ms = _load_switcher(monkeypatch, FakeClient)
    w = ms.LocoClientWrapper()
    assert w.set_speed_mode(0) == 0 and w._cfg_client.timeouts[-1] == 0.3
    assert w.set_balance_mode(0) == 0
    assert w.checked_zero() == 0
    w.start_move_sender()
    try:
        w.Move(0.1, 0.0, 0.0)
        deadline = time.time() + 1
        while w.sender.sent < 1 and time.time() < deadline:
            time.sleep(0.005)
        assert w.sender.sent == 1 and w.client.calls == []  # main client untouched
    finally:
        w.stop_move_sender()
