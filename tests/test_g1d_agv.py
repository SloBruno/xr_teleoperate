"""G1-D AGV base + lift column: mapping, client wire format, height feedback,
and the 20 Hz sender's fail-safe behaviour.  Fake RPC client only; nothing here
touches DDS or a robot."""
import json
import math
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teleop.utils import g1d_agv as agv  # noqa: E402

NAN, INF = float("nan"), float("inf")


# ------------------------------------------------------------------ mapping
def test_released_sticks_command_zero():
    assert agv.map_sticks((0.0, 0.0), (0.0, 0.0), 0.6, 0.6) == (0.0, 0.0, 0.0)


def test_deadzone_suppresses_small_deflection():
    assert agv.map_sticks((0.1, -0.1), (0.05, -0.1), 0.6, 0.6) == (0.0, 0.0, 0.0)


def test_left_stick_forward_is_positive_vx_same_sign_as_g1_walk():
    # Validated G1 convention: Quest stick forward reports y < 0 -> vx > 0.
    vx, vyaw, vz = agv.map_sticks((0.0, -1.0), (0.0, 0.0), 0.6, 0.6)
    assert (vx, vyaw, vz) == (0.6, 0.0, 0.0)
    from teleop.utils.quest_controls import joystick_to_locomotion
    assert math.copysign(1, joystick_to_locomotion((0.0, -1.0), (0.0, 0.0))[0]) == math.copysign(1, vx)
    assert agv.map_sticks((0.0, 1.0), (0.0, 0.0), 0.6, 0.6)[0] == -0.6


def test_left_stick_left_is_positive_ccw_vyaw():
    assert agv.map_sticks((-1.0, 0.0), (0.0, 0.0), 0.6, 0.6) == (0.0, 0.6, 0.0)
    assert agv.map_sticks((1.0, 0.0), (0.0, 0.0), 0.6, 0.6) == (0.0, -0.6, 0.0)


def test_right_stick_forward_raises_column_and_right_x_is_ignored():
    assert agv.map_sticks((0.0, 0.0), (0.0, -1.0), 0.6, 0.6) == (0.0, 0.0, 1.0)
    assert agv.map_sticks((0.0, 0.0), (0.0, 1.0), 0.6, 0.6) == (0.0, 0.0, -1.0)
    assert agv.map_sticks((0.0, 0.0), (1.0, 0.0), 0.6, 0.6) == (0.0, 0.0, 0.0)


def test_midpoint_uses_existing_deadzone_and_cubic_shaping():
    from teleop.utils.quest_controls import _shape_stick_value
    vx, vyaw, vz = agv.map_sticks((0.5, -0.5), (0.0, -0.5), 0.6, 0.6)
    s = _shape_stick_value(0.5)
    assert vx == pytest.approx(0.6 * s) and vyaw == pytest.approx(-0.6 * s) and vz == pytest.approx(s)
    assert 0.0 < s < 0.5


def test_overrange_is_clamped_to_caps():
    assert agv.map_sticks((-5.0, -5.0), (0.0, -5.0), 0.6, 0.6) == (0.6, 0.6, 1.0)


def test_left_and_right_are_isolated():
    assert agv.map_sticks((0.0, 0.0), (-1.0, -1.0), 0.6, 0.6)[:2] == (0.0, 0.0)
    assert agv.map_sticks((-1.0, -1.0), (0.0, 0.0), 0.6, 0.6)[2] == 0.0


@pytest.mark.parametrize("left,right", [
    ((NAN, -1.0), (0.0, 0.0)), ((0.0, NAN), (0.0, INF)), ((INF, -INF), (NAN, NAN)),
    (None, None), ("xy", object()), ((1.0,), (1.0, 2.0, 3.0)),
])
def test_nonfinite_or_malformed_input_is_zero_on_that_axis(left, right):
    vx, vyaw, vz = agv.map_sticks(left, right, 0.6, 0.6)
    assert all(math.isfinite(v) for v in (vx, vyaw, vz))
    if left is None or not isinstance(left, tuple) or len(left) != 2:
        assert (vx, vyaw) == (0.0, 0.0)
    if right is None or not isinstance(right, tuple) or len(right) != 2:
        assert vz == 0.0


def test_nan_axis_zeroes_only_itself():
    assert agv.map_sticks((NAN, -1.0), (0.0, NAN), 0.6, 0.6) == (0.6, 0.0, 0.0)


# ------------------------------------------------------------------ caps
def test_default_vx_cap_is_0_6_and_yaw_is_native_0_6():
    assert agv.resolve_agv_max_vx(None) == 0.6
    assert agv.AGV_MAX_VYAW_RADPS == 0.6
    assert agv.AGV_HARD_MAX_VX_MPS == 1.5


@pytest.mark.parametrize("value", ["0.3", 1.0, "1.5", 0.05])
def test_vx_cap_override_accepted_up_to_hard_max(value):
    assert agv.resolve_agv_max_vx(value) == float(value)


@pytest.mark.parametrize("value", ["1.51", 2, 0, -0.1, "nan", "inf", "abc", ""])
def test_vx_cap_override_rejected_outside_range(value):
    with pytest.raises(ValueError):
        agv.resolve_agv_max_vx(value)


# ------------------------------------------------------------------ client
class FakeBase:
    def __init__(self, service, lease):
        self.service, self.lease = service, lease
        self.version, self.apis, self.calls, self.timeout = None, {}, [], None
        self.code = 0

    def _SetApiVerson(self, v):
        self.version = v

    def _RegistApi(self, api, prio):
        self.apis[api] = prio

    def SetTimeout(self, t):
        self.timeout = t

    def _Call(self, api, param):
        self.calls.append((api, json.loads(param)))
        return self.code, None


def _client():
    cls = agv.build_agv_client_class(FakeBase)
    c = cls()
    c.Init()
    return c


def test_client_registers_agv_service_version_and_apis():
    c = _client()
    assert (c.service, c.lease) == ("agv", False)
    assert c.version == "1.0.0.1"
    assert c.apis == {1001: 0, 1002: 0}


def test_move_wire_format_vy_always_zero_and_clamped_at_native_limits():
    c = _client()
    assert c.Move(0.4, 0.9, -0.2) == 0
    assert c.calls[-1] == (1001, {"vx": 0.4, "vy": 0.0, "vyaw": -0.2})
    c.Move(9.0, 0.0, 9.0)
    assert c.calls[-1] == (1001, {"vx": 1.5, "vy": 0.0, "vyaw": 0.6})
    c.Move(NAN, INF, -INF)
    assert c.calls[-1] == (1001, {"vx": 0.0, "vy": 0.0, "vyaw": 0.0})


def test_height_adjust_wire_format_and_clamp():
    c = _client()
    c.HeightAdjust(0.5)
    assert c.calls[-1] == (1002, {"data": 0.5})
    c.HeightAdjust(-3.0)
    assert c.calls[-1] == (1002, {"data": -1.0})
    c.HeightAdjust(NAN)
    assert c.calls[-1] == (1002, {"data": 0.0})


def test_client_returns_rpc_code():
    c = _client()
    c.code = 9102
    assert c.Move(0.1, 0.0, 0.0) == 9102
    c.code = 9103
    assert c.HeightAdjust(0.1) == 9103


# ------------------------------------------------------------------ height feedback
class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def _height_monitor(clock):
    holder = {}

    def factory(topic, handler):
        holder["topic"], holder["handler"] = topic, handler
        return type("Sub", (), {"Close": lambda self: holder.setdefault("closed", True)})()
    mon = agv.HeightMonitor(subscriber_factory=factory, clock=clock)
    mon.start()
    return mon, holder


def test_height_monitor_subscribes_hispeed_state_and_reads_y():
    clock = Clock()
    mon, h = _height_monitor(clock)
    assert h["topic"] == "rt/hispeed_state"
    assert mon.snapshot()["fresh"] is False and mon.snapshot()["height_m"] is None
    h["handler"](type("P", (), {"x": 0.0, "y": 0.1803, "z": 0.0})())
    snap = mon.snapshot()
    assert snap["fresh"] is True and snap["height_m"] == pytest.approx(0.1803)
    clock.t += 0.5
    assert mon.is_fresh() is True
    clock.t += 0.01
    assert mon.is_fresh() is False and mon.snapshot()["fresh"] is False
    mon.close()
    assert h["closed"] is True


def test_height_monitor_ignores_nonfinite_samples():
    clock = Clock()
    mon, h = _height_monitor(clock)
    h["handler"](type("P", (), {"y": NAN})())
    assert mon.is_fresh() is False
    h["handler"](type("P", (), {})())
    assert mon.is_fresh() is False


# ------------------------------------------------------------------ sender
class FakeAgvClient:
    def __init__(self, move_code=0, height_code=0, delay=0.0):
        self.calls = []
        self.move_code, self.height_code, self.delay = move_code, height_code, delay

    def Move(self, vx, vy, vyaw):
        if self.delay:
            time.sleep(self.delay)
        self.calls.append(("move", vx, vy, vyaw))
        return self.move_code

    def HeightAdjust(self, vz):
        self.calls.append(("height", vz))
        return self.height_code


class FakeHeight:
    def __init__(self, fresh=True):
        self.fresh, self.closed = fresh, False

    def start(self):
        pass

    def is_fresh(self, now=None):
        return self.fresh

    def snapshot(self, now=None):
        return {"height_m": 0.18 if self.fresh else None, "age_s": 0.0 if self.fresh else None, "fresh": self.fresh}

    def close(self):
        self.closed = True


def _ctrl(armed=True, height_fresh=True, client=None, clock=None, warn=None):
    state = {"armed": armed}
    client = client or FakeAgvClient()
    height = FakeHeight(height_fresh)
    clock = clock or Clock()
    ctrl = agv.G1dAgvController(
        max_vx=0.6, client_factory=lambda: client, height_factory=lambda: height,
        armed_fn=lambda: state["armed"], clock=clock, warn=warn or (lambda m: None))
    return ctrl, client, height, state, clock


def _last(client, kind):
    return [c for c in client.calls if c[0] == kind][-1]


def test_nothing_is_constructed_until_feedback_and_activate():
    made = []
    ctrl = agv.G1dAgvController(max_vx=0.6, client_factory=lambda: made.append("client"),
                                height_factory=lambda: made.append("height"), armed_fn=lambda: True)
    assert made == [] and ctrl.sender_alive() is False
    snap = ctrl.submit((0.0, -1.0), (0.0, -1.0), time.monotonic())
    assert made == [] and snap["target"] == [0.0, 0.0, 0.0] and snap["reason"] == "inactive"
    ctrl.shutdown("test")  # never raises without a client
    assert made == []


def test_armed_fresh_target_is_sent_by_tick():
    ctrl, client, _, _, clock = _ctrl()
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    snap = ctrl.submit((-1.0, -1.0), (0.0, -1.0), clock.t)
    assert snap["target"] == [0.6, 0.6, 1.0]
    assert snap["raw_left_xy"] == [-1.0, -1.0] and snap["raw_right_xy"] == [0.0, -1.0]
    ctrl.tick()
    assert _last(client, "move") == ("move", 0.6, 0.0, 0.6)
    assert _last(client, "height") == ("height", 1.0)
    assert ctrl.snapshot()["sent"] == [0.6, 0.0, 0.6, 1.0]


def test_not_armed_sends_zero():
    ctrl, client, _, state, clock = _ctrl(armed=False)
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    snap = ctrl.submit((0.0, -1.0), (0.0, -1.0), clock.t)
    assert snap["target"] == [0.0, 0.0, 0.0] and snap["reason"] == "not_armed"
    ctrl.tick()
    assert _last(client, "move") == ("move", 0.0, 0.0, 0.0)
    assert _last(client, "height") == ("height", 0.0)


def test_sender_rechecks_arming_itself_after_q():
    ctrl, client, _, state, clock = _ctrl()
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    ctrl.submit((0.0, -1.0), (0.0, -1.0), clock.t)
    state["armed"] = False          # q pressed; control loop posts nothing more
    ctrl.tick()
    assert _last(client, "move") == ("move", 0.0, 0.0, 0.0)
    assert _last(client, "height") == ("height", 0.0)
    assert ctrl.snapshot()["reason"] == "not_armed"


def test_stale_controller_sample_is_zero_in_submit_and_in_sender():
    ctrl, client, _, _, clock = _ctrl()
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    snap = ctrl.submit((0.0, -1.0), (0.0, -1.0), clock.t - 0.3)
    assert snap["target"] == [0.0, 0.0, 0.0] and snap["reason"] == "controller_stale"
    ctrl.submit((0.0, -1.0), (0.0, -1.0), clock.t)
    clock.t += 0.26                 # loop stalled: sender ages out the target itself
    ctrl.tick()
    assert _last(client, "move") == ("move", 0.0, 0.0, 0.0)
    assert _last(client, "height") == ("height", 0.0)
    assert ctrl.snapshot()["reason"] == "controller_stale"


def test_column_stops_without_fresh_height_but_base_still_moves():
    ctrl, client, height, _, clock = _ctrl(height_fresh=False)
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    ctrl.submit((0.0, -1.0), (0.0, -1.0), clock.t)
    ctrl.tick()
    assert _last(client, "move") == ("move", 0.6, 0.0, 0.0)
    assert _last(client, "height") == ("height", 0.0)
    assert ctrl.snapshot()["column_reason"] == "height_stale"
    height.fresh = True
    ctrl.tick()
    assert _last(client, "height") == ("height", 1.0)


def test_rpc_errors_are_counted_and_first_ones_logged_from_sender():
    warnings = []
    client = FakeAgvClient(move_code=9102, height_code=9103)
    ctrl, client, _, _, clock = _ctrl(client=client, warn=warnings.append)
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    for _ in range(20):
        ctrl.tick()
    snap = ctrl.snapshot()
    assert snap["move_code"] == 9102 and snap["height_code"] == 9103
    assert snap["nonzero_move_codes"] == 20 and snap["nonzero_height_codes"] == 20
    assert 1 <= len(warnings) <= 2 * agv.MAX_LOGGED_RPC_ERRORS  # per-kind budget
    assert any("9102" in w for w in warnings)


def test_rpc_exception_is_contained_and_counted():
    class Boom(FakeAgvClient):
        def Move(self, *a):
            raise RuntimeError("dds gone")
    ctrl, client, _, _, clock = _ctrl(client=Boom())
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    ctrl.tick()
    assert ctrl.snapshot()["rpc_exceptions"] >= 1
    assert _last(client, "height") == ("height", 0.0)  # column still zeroed after base failure


def test_shutdown_sends_zero_burst_joins_and_is_idempotent():
    ctrl, client, height, _, clock = _ctrl()
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    ctrl.submit((0.0, -1.0), (0.0, -1.0), clock.t)
    ctrl.tick()
    n = len(client.calls)
    ctrl.shutdown("q", burst_gap_s=0.0)
    tail = client.calls[n:]
    assert ("move", 0.0, 0.0, 0.0) in tail and ("height", 0.0) in tail
    assert all(c[1:] in ((0.0, 0.0, 0.0), (0.0,)) for c in tail)
    assert height.closed is True
    m = len(client.calls)
    ctrl.shutdown("again")
    assert len(client.calls) == m
    assert ctrl.submit((0.0, -1.0), (0.0, -1.0), clock.t)["target"] == [0.0, 0.0, 0.0]


def test_real_thread_runs_about_20hz_and_submit_never_blocks_on_rpc():
    client = FakeAgvClient(delay=0.2)    # robot replies slowly
    state = {"armed": True}
    ctrl = agv.G1dAgvController(max_vx=0.6, client_factory=lambda: client, height_factory=lambda: FakeHeight(),
                                armed_fn=lambda: state["armed"])
    ctrl.start_feedback()
    ctrl.activate()
    try:
        assert ctrl.sender_alive()
        worst = 0.0
        for _ in range(20):
            t0 = time.perf_counter()
            ctrl.submit((0.0, -1.0), (0.0, -1.0), time.monotonic())
            worst = max(worst, time.perf_counter() - t0)
            time.sleep(0.01)
        assert worst < 0.05, worst   # RPC takes 0.2 s; submit must not wait for it
    finally:
        t0 = time.monotonic()
        ctrl.shutdown("test", join_timeout_s=0.5, burst_gap_s=0.0)
        assert time.monotonic() - t0 < 2.0
    assert not ctrl.sender_alive()
    assert client.calls[-1] == ("height", 0.0)


def test_real_thread_zeroes_quickly_when_controller_goes_stale():
    client = FakeAgvClient()
    ctrl = agv.G1dAgvController(max_vx=0.6, client_factory=lambda: client, height_factory=lambda: FakeHeight(),
                                armed_fn=lambda: True)
    ctrl.start_feedback()
    ctrl.activate()
    try:
        ctrl.submit((0.0, -1.0), (0.0, 0.0), time.monotonic())
        time.sleep(0.15)
        assert ("move", 0.6, 0.0, 0.0) in client.calls
        time.sleep(0.25 + 0.15)           # no new samples: sample ages past 0.25 s
        assert _last(client, "move") == ("move", 0.0, 0.0, 0.0)
    finally:
        ctrl.shutdown("test", burst_gap_s=0.0)


def test_snapshot_is_json_serializable():
    ctrl, client, _, _, clock = _ctrl()
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    ctrl.submit((NAN, -1.0), (0.0, -1.0), clock.t)
    ctrl.tick()
    json.dumps(ctrl.snapshot(), allow_nan=False)


def test_client_works_over_the_real_sdk_client_class(monkeypatch):
    pytest.importorskip("cyclonedds")
    import unitree_sdk2py.rpc.client_base as client_base
    sent = []

    class Stub:
        def __init__(self, name):
            sent.append(("service", name))

        def Init(self):
            pass

        def SendRequest(self, request, timeout):
            sent.append((request.header.identity.api_id, json.loads(request.parameter)))
            return None   # no reply -> SDK maps to an error code, never raises

    monkeypatch.setattr(client_base, "ClientStub", Stub)
    from unitree_sdk2py.rpc.client import Client
    c = agv.build_agv_client_class(Client)()
    c.SetTimeout(0.01)
    c.Init()
    assert c.GetApiVersion() == "1.0.0.1"
    code = c.Move(0.3, 0.0, 0.1)
    c.HeightAdjust(-0.4)
    assert ("service", "agv") in sent
    assert (1001, {"vx": 0.3, "vy": 0.0, "vyaw": 0.1}) in sent
    assert (1002, {"data": -0.4}) in sent
    assert isinstance(code, int) and code != 0


# ------------------------------------------------------------------ review fixes
def test_sender_blocked_past_shutdown_timeout_still_ends_with_zeros():
    """F1: an in-flight sender RPC must not be the last word after shutdown."""
    gate = threading.Event()
    entered = threading.Event()
    calls = []

    class Blocking(FakeAgvClient):
        def Move(self, vx, vy, vyaw):
            if vx != 0.0 and not gate.is_set():
                entered.set()
                gate.wait(2.0)            # robot/DDS hang longer than the lock wait
            calls.append(("move", vx, vy, vyaw))
            return 0

        def HeightAdjust(self, vz):
            calls.append(("height", vz))
            return 0

    clock = Clock()
    ctrl = agv.G1dAgvController(max_vx=0.6, client_factory=Blocking, height_factory=lambda: FakeHeight(),
                                armed_fn=lambda: True, clock=clock)
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    ctrl.submit((0.0, -1.0), (0.0, -1.0), clock.t)
    t = threading.Thread(target=ctrl.tick)
    t.start()
    assert entered.wait(1.0)
    stopper = threading.Thread(target=lambda: ctrl.shutdown("q", join_timeout_s=0.05, burst_gap_s=0.0))
    stopper.start()
    time.sleep(0.2)
    gate.set()                            # stuck sender resumes after the burst
    t.join(2.0)
    stopper.join(2.0)
    assert not t.is_alive() and not stopper.is_alive()
    moves = [c for c in calls if c[0] == "move"]
    heights = [c for c in calls if c[0] == "height"]
    assert moves[-1] == ("move", 0.0, 0.0, 0.0), calls
    assert heights[-1] == ("height", 0.0), calls
    assert ("height", 1.0) not in heights  # column RPC after a closed decision is replaced by zero


def test_keyboard_interrupt_during_shutdown_burst_still_sends_all_zeros_and_closes():
    """F2: a second Ctrl-C must not cut the zero burst short (shutdown never raises
    so teleop's arm cleanup that follows is not skipped)."""
    raised = []

    class Interrupting(FakeAgvClient):
        def Move(self, vx, vy, vyaw):
            if not raised:
                raised.append(1)
                raise KeyboardInterrupt
            return super().Move(vx, vy, vyaw)

    ctrl, client, height, _, clock = _ctrl(client=Interrupting())
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    ctrl.shutdown("q", burst_gap_s=0.0)
    assert raised
    assert ("move", 0.0, 0.0, 0.0) in client.calls
    assert client.calls[-1] == ("height", 0.0)
    assert height.closed is True


def test_keyboard_interrupt_in_burst_sleep_is_contained(monkeypatch):
    ctrl, client, height, _, clock = _ctrl()
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    real_sleep = time.sleep

    def sleep(s):
        raise KeyboardInterrupt
    monkeypatch.setattr(agv.time, "sleep", sleep)
    ctrl.shutdown("q", burst_gap_s=0.05)
    monkeypatch.setattr(agv.time, "sleep", real_sleep)
    assert [c for c in client.calls if c[0] == "move"][-1] == ("move", 0.0, 0.0, 0.0)
    assert height.closed is True


def test_tick_failure_falls_back_to_zero_send():
    """F5: a tick that raises before sending must still command zero."""
    ctrl, client, _, state, clock = _ctrl()
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    ctrl.submit((0.0, -1.0), (0.0, -1.0), clock.t)
    ctrl.tick()
    assert _last(client, "move") == ("move", 0.6, 0.0, 0.0)

    class BadHeight(FakeHeight):
        def is_fresh(self, now=None):
            raise RuntimeError("boom")
    ctrl.height = BadHeight()
    ctrl._tick_safely()
    assert _last(client, "move") == ("move", 0.0, 0.0, 0.0)
    assert _last(client, "height") == ("height", 0.0)
    assert ctrl.snapshot()["rpc_exceptions"] >= 1


@pytest.mark.parametrize("offset", [+0.05, +5.0])
def test_future_dated_controller_sample_is_stale_in_submit_and_tick(offset):
    ctrl, client, _, _, clock = _ctrl()
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    snap = ctrl.submit((0.0, -1.0), (0.0, -1.0), clock.t + offset)
    assert snap["reason"] == "controller_stale" and snap["target"] == [0.0, 0.0, 0.0]
    ctrl._target, ctrl._sample_t = (0.6, 0.0, 1.0), clock.t + offset   # force a bad stored sample
    ctrl.tick()
    assert _last(client, "move") == ("move", 0.0, 0.0, 0.0)
    assert _last(client, "height") == ("height", 0.0)


def test_error_log_budget_is_per_rpc_kind():
    warnings = []
    client = FakeAgvClient(move_code=9102, height_code=0)
    ctrl, client, _, _, clock = _ctrl(client=client, warn=warnings.append)
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)
    for _ in range(20):
        ctrl.tick()
    client.height_code = 9103
    ctrl.tick()
    assert any("9103" in w for w in warnings), warnings


def test_shutdown_still_closes_feedback_if_an_interrupt_escapes_the_burst():
    warnings = []
    ctrl, client, height, _, clock = _ctrl(warn=warnings.append)
    ctrl.start_feedback()
    ctrl.activate(start_thread=False)

    def bad_send(*a):
        raise KeyboardInterrupt
    ctrl._send = bad_send
    ctrl.shutdown("q", burst_gap_s=0.0)
    assert height.closed is True
    assert ctrl.snapshot()["sent"] == [0.0, 0.0, 0.0, 0.0]
    assert ctrl.snapshot()["shutdown_interrupted"] is True
    assert any("interrupted" in w for w in warnings)


def test_real_thread_stuck_rpc_past_join_is_reported_and_ends_with_zeros():
    gate = threading.Event()
    entered = threading.Event()
    calls = []
    warnings = []

    class Stuck(FakeAgvClient):
        def Move(self, vx, vy, vyaw):
            if vx != 0.0 and not gate.is_set():
                entered.set()
                gate.wait(3.0)
            calls.append(("move", vx, vy, vyaw))
            return 0

        def HeightAdjust(self, vz):
            calls.append(("height", vz))
            return 0

    ctrl = agv.G1dAgvController(max_vx=0.6, client_factory=Stuck, height_factory=lambda: FakeHeight(),
                                armed_fn=lambda: True, warn=warnings.append)
    ctrl.start_feedback()
    ctrl.activate()
    stop = threading.Event()

    def feed():
        while not stop.is_set():
            ctrl.submit((0.0, -1.0), (0.0, -1.0), time.monotonic())
            time.sleep(0.01)
    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    try:
        assert entered.wait(2.0)
        ctrl.shutdown("q", join_timeout_s=0.05, burst_gap_s=0.0)
        assert ctrl.snapshot()["sender_alive_after_join"] is True
        assert any("still blocked" in w for w in warnings)
    finally:
        stop.set()
        gate.set()
        feeder.join(1.0)
        ctrl._thread.join(2.0)
    assert not ctrl.sender_alive()
    assert [c for c in calls if c[0] == "move"][-1] == ("move", 0.0, 0.0, 0.0)
    assert [c for c in calls if c[0] == "height"][-1] == ("height", 0.0)
