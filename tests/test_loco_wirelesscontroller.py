"""Opt-in --loco-backend wirelesscontroller: continuous 20 Hz normalized
rt/wirelesscontroller state, client ramp, immediate zero on safety. All fakes;
nothing is sent to a robot."""
import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teleop.utils import quest_controls
from teleop.utils.loco_wireless import (
    PUBLISH_HZ, RAMP_DOWN, RAMP_UP, ROBOT_MAX_LINEAR_MPS, ROBOT_MAX_YAW_RADPS,
    WirelessControllerPublisher, normalize_command,
)

MAIN = Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py"


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(cap=0.3, turn=0.3, stale_s=0.3):
    sent = []
    clk = Clock()
    pub = WirelessControllerPublisher(lambda lx, ly, rx, ry, keys: sent.append((lx, ly, rx, ry, keys)) or True,
                                      walk_cap=cap, turn_cap=turn, clock=clk, stale_s=stale_s)
    return pub, sent, clk


def tick(pub, clk, n=1):
    for _ in range(n):
        clk.t += 1.0 / PUBLISH_HZ
        pub.tick()


# --- mapping / normalization ---------------------------------------------------

def test_constants_match_reference_client():
    assert PUBLISH_HZ == 20.0 and RAMP_UP == 0.15 and RAMP_DOWN == 0.30
    assert ROBOT_MAX_LINEAR_MPS == 1.0


def test_mapping_is_ly_minus_lx_minus_rx_inverse():
    # robot reads [ly, -lx, -rx] -> [vx, vy, omega]; so ly=vx, lx=-vy, rx=-omega
    lx, ly, rx = normalize_command(0.2, 0.1, 0.3, walk_cap=0.3, turn_cap=0.3)
    assert ly == 0.2 / ROBOT_MAX_LINEAR_MPS
    assert lx == -0.1 / ROBOT_MAX_LINEAR_MPS
    assert abs(rx - (-0.3 / ROBOT_MAX_YAW_RADPS)) < 1e-9


def test_normalized_values_never_exceed_cap_scale():
    lx, ly, rx = normalize_command(5.0, -5.0, 9.0, walk_cap=0.3, turn_cap=0.3)
    assert abs(ly) <= 0.3 and abs(lx) <= 0.3 and abs(rx) <= 0.3 / ROBOT_MAX_YAW_RADPS + 1e-9


def test_nonfinite_is_zero():
    assert normalize_command(float("nan"), float("inf"), 0.0, 0.3, 0.3) == (0.0, 0.0, 0.0)


# --- continuous publishing ---------------------------------------------------

def test_publishes_zero_every_tick_when_idle():
    pub, sent, clk = make()
    tick(pub, clk, 5)
    assert len(sent) == 5 and all(s == (0.0, 0.0, 0.0, 0.0, 0) for s in sent)


def test_ramp_up_is_per_tick_limited_and_reaches_target():
    pub, sent, clk = make()
    pub.set_command(0.3, 0.0, 0.0)
    tick(pub, clk)
    assert abs(sent[-1][1] - 0.15) < 1e-9          # first tick limited to RAMP_UP
    tick(pub, clk)
    assert abs(sent[-1][1] - 0.30) < 1e-9          # target = cap/1.0
    tick(pub, clk, 3)
    assert abs(sent[-1][1] - 0.30) < 1e-9          # holds, keeps publishing
    assert len(sent) == 5


def test_ramp_down_is_limited_on_release_not_a_step():
    pub, sent, clk = make(cap=0.6)
    pub.set_command(0.6, 0.0, 0.0)
    tick(pub, clk, 6)
    assert abs(sent[-1][1] - 0.6) < 1e-9
    pub.set_command(0.0, 0.0, 0.0)
    tick(pub, clk)
    assert abs(sent[-1][1] - 0.30) < 1e-9
    tick(pub, clk)
    assert sent[-1][1] == 0.0
    tick(pub, clk)
    assert sent[-1] == (0.0, 0.0, 0.0, 0.0, 0)


def test_stale_command_goes_to_immediate_zero_without_ramp():
    pub, sent, clk = make(cap=0.6, stale_s=0.3)
    pub.set_command(0.6, 0.0, 0.0)
    tick(pub, clk, 6)
    assert sent[-1][1] > 0.5
    clk.t += 1.0           # loop stalled: no set_command
    pub.tick()
    assert sent[-1] == (0.0, 0.0, 0.0, 0.0, 0)
    assert pub.last_reason == "stale"


def test_zero_now_is_immediate_synchronous_and_resets_ramp():
    pub, sent, clk = make(cap=0.6)
    pub.set_command(0.6, 0.0, 0.0)
    tick(pub, clk, 6)
    n = len(sent)
    assert pub.zero_now("q") is True
    assert len(sent) == n + 1 and sent[-1] == (0.0, 0.0, 0.0, 0.0, 0)
    pub.set_command(0.0, 0.0, 0.0)
    tick(pub, clk)
    assert sent[-1] == (0.0, 0.0, 0.0, 0.0, 0)   # ramp state was reset, not resumed


def test_zero_now_overrides_pending_command_in_same_tick():
    pub, sent, clk = make()
    pub.set_command(0.3, 0.0, 0.0)
    pub.zero_now("emergency")
    tick(pub, clk)
    assert sent[-1] == (0.0, 0.0, 0.0, 0.0, 0)  # target cleared by zero_now


def test_writer_failure_counted_never_raises():
    clk = Clock()

    def boom(*a):
        raise RuntimeError("dds down")
    pub = WirelessControllerPublisher(boom, walk_cap=0.3, turn_cap=0.3, clock=clk)
    clk.t += 0.05
    pub.tick()
    assert pub.write_failures == 1
    assert pub.zero_now("x") is False


def test_telemetry_reports_values_and_actual_rate():
    pub, sent, clk = make()
    for _ in range(10):
        pub.set_command(0.3, 0.0, 0.0)   # control loop keeps refreshing
        tick(pub, clk)
    t = pub.telemetry()
    assert t["backend"] == "wirelesscontroller"
    assert abs(t["published"][1] - 0.30) < 1e-9
    assert abs(t["actual_hz"] - 20.0) < 0.5
    assert t["sent"] == 10 and t["write_failures"] == 0


def test_stop_publishes_final_zeros_and_stops_thread():
    pub, sent, clk = make()
    pub.set_command(0.3, 0.0, 0.0)
    tick(pub, clk, 3)
    pub.start()
    pub.stop()
    assert sent[-1] == (0.0, 0.0, 0.0, 0.0, 0)
    assert pub._thread is None or not pub._thread.is_alive()


def test_dedicated_thread_publishes_at_20hz_real_clock():
    import time
    sent = []
    pub = WirelessControllerPublisher(lambda *a: sent.append(a) or True, walk_cap=0.3, turn_cap=0.3)
    pub.start()
    time.sleep(1.1)
    pub.stop()
    assert 15 <= len(sent) <= 30, len(sent)
    assert all(s == (0.0, 0.0, 0.0, 0.0, 0) for s in sent)


# --- wrapper integration -------------------------------------------------------

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

    def SetTimeout(self, t):
        pass

    def Init(self):
        pass

    def SetVelocity(self, vx, vy, w, d):
        self.calls.append((vx, vy, w, d))
        return 0


def test_default_backend_is_setvelocity_and_unchanged(monkeypatch):
    ms = _load_switcher(monkeypatch, FakeClient)
    w = ms.LocoClientWrapper()
    assert w.backend == "setvelocity" and w.wireless is None
    w.StopMove("x")
    assert w.client.calls == [(0.0, 0.0, 0.0, 1.0)]


def test_wireless_backend_routes_move_and_never_uses_setvelocity_or_damp(monkeypatch):
    ms = _load_switcher(monkeypatch, FakeClient)
    sent = []
    w = ms.LocoClientWrapper(backend="wirelesscontroller", walk_cap=0.3, turn_cap=0.3,
                             wireless_writer=lambda *a: sent.append(a) or True)
    assert w.backend == "wirelesscontroller"
    w.start_move_sender()
    w.Move(0.3, 0.0, 0.0)
    assert w.wireless._target[1] > 0
    assert w.StopMove("release") == 0
    assert sent[-1] == (0.0, 0.0, 0.0, 0.0, 0)
    assert w.stop_count == 1 and w.last_stop_reason == "release"
    w.stop_move_sender()
    assert w.client.calls == []  # SetVelocity never used for walking/stop in this backend


def test_wireless_stale_dispatch_gives_immediate_zero_via_wrapper(monkeypatch):
    ms = _load_switcher(monkeypatch, FakeClient)
    sent = []
    w = ms.LocoClientWrapper(backend="wirelesscontroller", walk_cap=0.3, turn_cap=0.3,
                             wireless_writer=lambda *a: sent.append(a) or True)
    w.start_move_sender()
    quest_controls.dispatch_joystick_locomotion(w, True, True, (0.0, -1.0), (0.0, 0.0), 0.3, 0.3)
    quest_controls.dispatch_joystick_locomotion(w, True, False, (0.0, -1.0), (0.0, 0.0), 0.3, 0.3)
    assert sent[-1] == (0.0, 0.0, 0.0, 0.0, 0)
    w.stop_move_sender()


def test_wireless_telemetry_exposed_by_wrapper(monkeypatch):
    ms = _load_switcher(monkeypatch, FakeClient)
    w = ms.LocoClientWrapper(backend="wirelesscontroller", wireless_writer=lambda *a: True)
    assert w.backend_telemetry()["backend"] == "wirelesscontroller"
    w2 = ms.LocoClientWrapper()
    assert w2.backend_telemetry() == {"backend": "setvelocity"}


# --- CLI / default -------------------------------------------------------------

def test_cli_default_is_wirelesscontroller_and_choice_present():
    src = MAIN.read_text()
    assert "--loco-backend" in src
    assert "choices=['setvelocity', 'wirelesscontroller']" in src
    assert "default='wirelesscontroller'" in src.split("--loco-backend")[1].split("\n")[0]


def test_launcher_defaults_to_wirelesscontroller_and_passes_flag():
    sh = (MAIN.parent / "run_g1_quest_dex3.sh").read_text()
    assert "loco_backend=${G1_LOCO_BACKEND:-wirelesscontroller}" in sh
    assert '--loco-backend "$loco_backend"' in sh
    assert 'echo "Backend de caminhada: ${loco_backend}"' in sh


def test_wireless_init_failure_disables_locomotion_without_fallback(monkeypatch):
    from teleop.utils.loco_preflight import run_loco_preflight
    ms = _load_switcher(monkeypatch, FakeClient)

    def boom():
        raise ImportError("no WirelessController_ IDL")
    monkeypatch.setattr("teleop.utils.loco_wireless.make_dds_writer", boom)
    w = ms.LocoClientWrapper(backend="wirelesscontroller")
    assert w.backend == "wirelesscontroller" and w.backend_error
    res = run_loco_preflight(w)
    assert res["loco_enabled"] is False
    assert res["refusal_reason"].startswith("backend_unavailable:")
    w.start_move_sender()
    assert w.wireless is None and w.sender is None
    assert w.client.calls == [] and w._cfg_client is None   # no SetVelocity/SetSpeedMode fallback
    assert w.backend_telemetry()["backend_error"]


def test_forced_setvelocity_still_uses_rpc_preflight(monkeypatch):
    ms = _load_switcher(monkeypatch, FakeClient)
    w = ms.LocoClientWrapper(backend="setvelocity")
    assert w.backend_error is None and w.wireless is None


def test_wireless_preflight_skips_speedmode_and_gait(monkeypatch):
    from teleop.utils.loco_preflight import run_loco_preflight
    ms = _load_switcher(monkeypatch, FakeClient)
    w = ms.LocoClientWrapper(backend="wirelesscontroller", wireless_writer=lambda *a: True)
    w.read_fsm_id = lambda timeout=0.3: 500
    w.set_speed_mode = lambda m=0: (_ for _ in ()).throw(AssertionError("SetSpeedMode called"))
    w.set_balance_mode = lambda m=0: (_ for _ in ()).throw(AssertionError("ContinuousGait called"))
    res = run_loco_preflight(w)
    assert res["loco_enabled"] and res["backend"] == "wirelesscontroller"


def test_wireless_preflight_still_refuses_non_regular_fsm(monkeypatch):
    from teleop.utils.loco_preflight import run_loco_preflight
    ms = _load_switcher(monkeypatch, FakeClient)
    w = ms.LocoClientWrapper(backend="wirelesscontroller", wireless_writer=lambda *a: True)
    w.read_fsm_id = lambda timeout=0.3: 801
    assert not run_loco_preflight(w)["loco_enabled"]


def test_shutdown_burst_of_three_zeros_even_after_exception(monkeypatch):
    pub, sent, clk = make()
    pub.set_command(0.3, 0, 0)
    tick(pub, clk, 5)
    n = len(sent)
    pub.stop(burst_gap_s=0)
    assert sent[n:] == [(0.0, 0.0, 0.0, 0.0, 0)] * 3


def test_writer_exception_never_propagates_and_counts_failures():
    def bad(*a):
        raise RuntimeError("dds down")
    pub = WirelessControllerPublisher(bad, clock=Clock())
    assert pub.zero_now("x") is False
    pub.stop(burst_gap_s=0)
    assert pub.telemetry()["write_failures"] >= 4


def test_idle_publishes_zero_continuously_and_telemetry_fields():
    pub, sent, clk = make()
    tick(pub, clk, 5)
    assert sent == [(0.0, 0.0, 0.0, 0.0, 0)] * 5
    t = pub.telemetry()
    assert t["actual_hz"] and abs(t["actual_hz"] - 20.0) < 1e-6
    assert {"backend", "published", "write_failures", "actual_hz"} <= set(t)


def test_teleop_logs_loco_enabled_and_reason():
    src = MAIN.read_text()
    assert 'stick_log["loco_enabled"]' in src and 'stick_log["loco_disabled_reason"]' in src
