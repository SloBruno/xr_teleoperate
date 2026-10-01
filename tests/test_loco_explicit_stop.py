import importlib
import json
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teleop.utils import quest_controls

MAIN = Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py"


def _load_switcher(monkeypatch, client_cls):
    loco_module = types.ModuleType("unitree_sdk2py.g1.loco.g1_loco_client")
    loco_module.LocoClient = client_cls
    motion_module = types.ModuleType("unitree_sdk2py.comm.motion_switcher.motion_switcher_client")
    motion_module.MotionSwitcherClient = object
    channel_module = types.ModuleType("unitree_sdk2py.core.channel")
    channel_module.ChannelFactoryInitialize = object
    for name, mod in {
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": channel_module,
        "unitree_sdk2py.g1": types.ModuleType("unitree_sdk2py.g1"),
        "unitree_sdk2py.g1.loco": types.ModuleType("unitree_sdk2py.g1.loco"),
        "unitree_sdk2py.g1.loco.g1_loco_client": loco_module,
        "unitree_sdk2py.comm": types.ModuleType("unitree_sdk2py.comm"),
        "unitree_sdk2py.comm.motion_switcher": types.ModuleType("unitree_sdk2py.comm.motion_switcher"),
        "unitree_sdk2py.comm.motion_switcher.motion_switcher_client": motion_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return importlib.reload(importlib.import_module("teleop.utils.motion_switcher"))


class FakeClient:
    def __init__(self):
        self.calls = []
        self.timeouts = []
        self.codes = []
        self.raise_on_call = False

    def SetTimeout(self, t):
        self.timeouts.append(t)

    def Init(self):
        pass

    def SetVelocity(self, vx, vy, w, d):
        if self.raise_on_call:
            raise RuntimeError("dds down")
        self.calls.append((vx, vy, w, d))
        return self.codes.pop(0) if self.codes else 0


def _wrapper(monkeypatch):
    return _load_switcher(monkeypatch, FakeClient).LocoClientWrapper()


# --- StopMove on the wrapper -------------------------------------------------

def test_stop_move_is_zero_velocity_same_wire_as_sdk(monkeypatch):
    w = _wrapper(monkeypatch)
    w.client.codes = [0]
    assert w.StopMove("release") == 0
    assert w.client.calls == [(0.0, 0.0, 0.0, 1.0)]
    assert w.stop_count == 1 and w.last_stop_reason == "release"


def test_stop_move_never_raises_and_counts_failure(monkeypatch):
    w = _wrapper(monkeypatch)
    w.client.raise_on_call = True
    assert w.StopMove("exception") is None
    assert w.stop_failures == 1


def test_stop_move_blocking_retries_until_ack_and_restores_timeout(monkeypatch):
    w = _wrapper(monkeypatch)
    w.client.codes = [3104, 3104, 0]
    assert w.StopMove("shutdown", timeout=0.25, attempts=3) == 0
    assert len(w.client.calls) == 3
    assert w.client.timeouts[-2:] == [0.25, 0.0001]


def test_stop_move_blocking_bounded_attempts(monkeypatch):
    w = _wrapper(monkeypatch)
    w.client.codes = [3104] * 10
    assert w.StopMove("shutdown", timeout=0.25, attempts=2) == 3104
    assert len(w.client.calls) == 2


def test_stop_move_does_not_touch_move_code_counters(monkeypatch):
    w = _wrapper(monkeypatch)
    w.client.codes = [3104]
    w.StopMove("release")
    assert w.last_move_code is None and w.nonzero_move_codes == 0
    assert w.last_stop_code == 3104


# --- transition logic --------------------------------------------------------

class Rec:
    def __init__(self):
        self.events = []

    def Move(self, *c):
        self.events.append(("move", c))

    def StopMove(self, reason="", **kw):
        self.events.append(("stop", reason))


def _d(rec, fresh=True, left=(0.0, 0.0), right=(0.0, 0.0), enabled=True):
    return quest_controls.dispatch_joystick_locomotion(rec, enabled, fresh, left, right)


def test_stop_once_on_nonzero_to_zero_transition_not_each_cycle():
    r = Rec()
    _d(r, left=(0.0, -1.0))
    _d(r, left=(0.0, -1.0))
    _d(r)
    _d(r)
    _d(r)
    assert [e for e in r.events if e[0] == "stop"] == [("stop", "release")]
    assert sum(1 for e in r.events if e == ("move", (0.0, 0.0, 0.0))) == 3


def test_no_stop_when_never_moving():
    r = Rec()
    _d(r)
    _d(r)
    assert not [e for e in r.events if e[0] == "stop"]


def test_stale_controller_while_moving_triggers_stop_with_reason():
    r = Rec()
    _d(r, left=(0.0, -1.0))
    _d(r, fresh=False, left=(0.0, -1.0))
    assert ("stop", "stale") in r.events


def test_second_push_after_release_stops_again():
    r = Rec()
    _d(r, left=(0.0, -1.0))
    _d(r)
    _d(r, right=(-1.0, 0.0))
    _d(r)
    assert [e for e in r.events if e[0] == "stop"] == [("stop", "release")] * 2


def test_stop_failure_in_dispatch_does_not_propagate():
    class Bad(Rec):
        def StopMove(self, reason="", **kw):
            raise RuntimeError("boom")

    r = Bad()
    _d(r, left=(0.0, -1.0))
    _d(r)


def test_motion_disabled_while_moving_stops():
    r = Rec()
    _d(r, left=(0.0, -1.0))
    assert _d(r, enabled=False) == (0.0, 0.0, 0.0)
    assert ("stop", "motion_disabled") in r.events


def test_dispatch_with_legacy_wrapper_without_stopmove_still_works():
    class Old:
        def __init__(self):
            self.c = []

        def Move(self, *c):
            self.c.append(c)

    o = Old()
    _d(o, left=(0.0, -1.0))
    _d(o)
    assert o.c[-1] == (0.0, 0.0, 0.0)


def test_dispatch_none_wrapper_when_disabled_still_ok():
    assert quest_controls.dispatch_joystick_locomotion(None, False, True, (0, -1), (0, 0)) == (0.0, 0.0, 0.0)


def test_stick_curve_and_caps_unchanged():
    assert quest_controls.joystick_to_locomotion((0.0, -1.0), (-1.0, 0.0)) == (0.3, 0.0, 0.3)


# --- robot state monitor -----------------------------------------------------

def _mon(**kw):
    from teleop.utils import robot_state_monitor as rsm
    return rsm, rsm.RobotStateMonitor(**kw)


class Msg:
    def __init__(self, mode=1, gait=2, v=(0.1, 0.0, 0.02)):
        self.mode, self.gait_type, self.velocity = mode, gait, list(v)


def test_monitor_snapshot_empty_is_safe_and_json_ready():
    rsm, m = _mon()
    s = m.snapshot(now=10.0)
    json.dumps(s)
    assert s["sport_samples"] == 0 and s["sport_velocity"] is None and s["fsm_id"] is None


def test_monitor_records_sample_via_callback_without_io():
    rsm, m = _mon()
    m.on_sport_state(Msg(), now=5.0)
    s = m.snapshot(now=5.5)
    assert s["sport_mode"] == 1 and s["sport_gait_type"] == 2
    assert s["sport_velocity"] == [0.1, 0.0, 0.02]
    assert s["sport_age_ms"] == 500.0 and s["sport_samples"] == 1


def test_monitor_bad_message_becomes_error_counter():
    rsm, m = _mon()
    m.on_sport_state(object(), now=1.0)
    assert m.snapshot(now=1.0)["sport_errors"] == 1


def test_monitor_fsm_poll_thread_limited_rate_and_off_caller_thread():
    seen = []
    main = threading.get_ident()

    def reader():
        seen.append(threading.get_ident())
        return 500

    rsm, m = _mon(fsm_reader=reader, fsm_period_s=0.05)
    m.start()
    time.sleep(0.3)
    m.close()
    assert seen and all(t != main for t in seen)
    assert len(seen) <= 8
    assert m.snapshot()["fsm_id"] == 500


def test_monitor_fsm_reader_exception_counts_and_survives():
    def reader():
        raise RuntimeError("x")

    rsm, m = _mon(fsm_reader=reader, fsm_period_s=0.02)
    m.start()
    time.sleep(0.15)
    m.close()
    assert m.snapshot()["fsm_errors"] >= 2


def test_monitor_subscription_failure_is_counted_not_raised():
    def bad_factory(topic, handler):
        raise RuntimeError("no dds")

    rsm, m = _mon(subscriber_factory=bad_factory)
    m.start()
    m.close()
    assert m.snapshot()["sub_failures"] >= 1


def test_monitor_close_is_idempotent_and_closes_subscribers():
    closed = []

    class Sub:
        def Close(self):
            closed.append(1)

    rsm, m = _mon(subscriber_factory=lambda topic, handler: Sub(), topics=("rt/sportmodestate",))
    m.start()
    m.close()
    m.close()
    assert len(closed) == 1


def test_stop_locomotion_best_effort_never_raises_and_handles_none():
    from teleop.utils.robot_state_monitor import stop_locomotion_best_effort
    stop_locomotion_best_effort(None, "x")

    class Boom:
        def StopMove(self, *a, **k):
            raise BaseException("z")

    stop_locomotion_best_effort(Boom(), "x")


# --- main wiring (static) ----------------------------------------------------

def _main_src():
    return MAIN.read_text()


def test_main_finally_stops_move_before_and_after_graceful_arm_shutdown():
    src = _main_src()
    fin = src[src.index("    finally:\n        # Explicit StopMove first"):]
    first = fin.index("stop_locomotion_best_effort")
    graceful = fin.index("graceful_g1_29_shutdown(")
    last = fin.rindex("stop_locomotion_best_effort")
    assert first < graceful < last


def test_main_initializes_loco_wrapper_and_monitor_before_try():
    src = _main_src()
    head = src[:src.index("    try:\n        # setup dds communication")]
    assert "loco_wrapper = None" in head and "robot_monitor = None" in head


def test_main_logs_robot_state_with_stick_record():
    assert 'stick_log["robot_state"]' in _main_src()


def test_main_closes_monitor_in_cleanup():
    assert "robot_monitor.close()" in _main_src()


def test_monitor_factory_handler_feeds_state():
    holder = {}
    rsm, m = _mon(subscriber_factory=lambda topic, handler: holder.setdefault("h", handler) and None)
    m.start()
    holder["h"](Msg())
    assert m.snapshot()["sport_samples"] == 1
    m.close()


def test_make_fsm_reader_uses_separate_client(monkeypatch):
    created = []

    class C(FakeClient):
        def __init__(self):
            super().__init__()
            created.append(self)

        def _Call(self, api, p):
            return 0, '{"data": 500}'

    w = _load_switcher(monkeypatch, C).LocoClientWrapper()
    read = w.make_fsm_reader()
    assert len(created) == 2 and created[1] is not w.client
    assert read() == 500 and created[1].timeouts == [1.0]
    assert w.client.timeouts == [0.0001]
