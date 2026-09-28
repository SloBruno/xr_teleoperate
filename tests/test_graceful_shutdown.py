"""Graceful G1_29 shutdown: q/B -> no new IK, bounded return home, Dex3 open,
arm_sdk weight ramp 1 -> 0, writer deactivated. All fakes; no DDS/actuators."""

import importlib
import runpy
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teleop.utils.arm_graceful_shutdown import (  # noqa: E402
    DEFAULT_MAX_JOINT_VELOCITY,
    plan_return_duration,
    run_graceful_arm_shutdown,
)
from teleop.utils.arm_tracking_orchestration import run_arm_tracking_cycle  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += float(seconds)


class SimArm:
    """Records every command; measured q tracks the last command (ideal servo)."""

    arm_joint_split = (7, 7)

    def __init__(self, clock, start_q=None, motion_mode=True):
        self.clock = clock
        self.motion_mode = motion_mode
        self.q = np.linspace(-0.8, 0.9, 14) if start_q is None else np.asarray(start_q, float)
        self.commanded_tau = np.full(14, 2.0)
        self.commands = []          # (t, q, tau)
        self.weights = []           # (t, weight)
        self.weight = 1.0
        self.active = True
        self.deactivated = 0
        self.follow = True
        self.state_age = 0.0
        self.writer_dead = False
        self.log = []

    def get_dual_arm_q_snapshot(self):
        return self.q.copy(), self.state_age

    def get_current_dual_arm_q(self):
        return self.q.copy()

    def get_arm_command(self):
        return self.q.copy(), self.commanded_tau.copy()

    def ctrl_dual_arm(self, q, tau):
        assert self.active, "command after deactivate"
        self.commands.append((self.clock(), np.asarray(q, float).copy(), np.asarray(tau, float).copy()))
        self.log.append("cmd")
        if self.follow:
            self.q = np.asarray(q, float).copy()
        return len(self.commands)

    def set_motion_authority_weight(self, weight):
        assert self.active, "weight after deactivate"
        self.weight = float(weight)
        self.weights.append((self.clock(), self.weight))
        self.log.append("weight")

    def get_publication_status(self):
        return {
            "active": self.active and not self.writer_dead,
            "last_publish_monotonic": None if self.writer_dead else self.clock(),
            "last_published_weight": self.weight if self.motion_mode else None,
        }

    def deactivate(self):
        self.active = False
        self.deactivated += 1
        self.log.append("deactivate")


def shutdown(arm, clock, **kwargs):
    return run_graceful_arm_shutdown(arm, clock=clock, sleep=clock.sleep, **kwargs)


def test_return_is_monotonic_velocity_limited_then_weight_ramps_down_then_deactivates():
    clock = FakeClock()
    arm = SimArm(clock)
    start = arm.q.copy()
    events = []
    hands = []
    result = shutdown(arm, clock, emit=lambda name, detail: events.append(name),
                      open_hands=lambda: hands.append(clock()))

    assert result.returned_home and result.arrival_confirmed
    qs = np.array([q for _, q, _ in arm.commands])
    ts = np.array([t for t, _, _ in arm.commands])
    # starts at measured pose, ends exactly at all-zero
    np.testing.assert_allclose(qs[0], start)
    np.testing.assert_allclose(qs[-1], np.zeros(14))
    # monotonic toward zero per joint (no overshoot)
    dist = np.abs(qs)
    assert np.all(np.diff(dist, axis=0) <= 1e-12)
    # velocity limit between waypoints
    velocity = np.abs(np.diff(qs, axis=0)) / np.diff(ts)[:, None]
    assert velocity.max() <= DEFAULT_MAX_JOINT_VELOCITY + 1e-9
    # duration proportional to distance
    expected = plan_return_duration(start, np.zeros(14))
    assert ts[-1] - ts[0] == pytest.approx(expected, abs=0.03)
    # feed-forward blends from last commanded tau to zero without gravity model
    np.testing.assert_allclose(arm.commands[0][2], arm.commanded_tau)
    np.testing.assert_allclose(arm.commands[-1][2], np.zeros(14))

    weights = [w for _, w in arm.weights]
    wts = [t for t, _ in arm.weights]
    assert weights[0] == 1.0 and weights[-1] == 0.0
    assert np.all(np.diff(weights) <= 0.0)
    assert np.max(-np.diff(weights)) <= 0.011          # no abrupt drop
    assert wts[-1] - wts[0] == pytest.approx(2.0, abs=0.03)
    assert result.weight_released and result.release_confirmed
    # order: return -> hands open -> weight ramp -> deactivate
    assert hands and ts[-1] <= hands[0] <= wts[0]
    assert arm.log.index("weight") > max(i for i, e in enumerate(arm.log) if e == "cmd")
    assert arm.log[-1] == "deactivate" and arm.deactivated == 1
    assert events == [
        "shutdown_return_started", "shutdown_return_finished",
        "weight_release_started", "weight_release_finished", "shutdown_arm_released",
    ]


def test_tauff_blends_continuously_from_last_command_to_gravity_model():
    clock = FakeClock()
    arm = SimArm(clock)
    shutdown(arm, clock, gravity_tauff=lambda q: np.asarray(q) * 10.0 + 1.0)
    taus = np.array([tau for _, _, tau in arm.commands])
    np.testing.assert_allclose(taus[0], arm.commanded_tau)        # continuous at start
    np.testing.assert_allclose(taus[-1], np.ones(14))             # gravity(0) at the end
    assert np.max(np.abs(np.diff(taus, axis=0))) < 0.1            # no torque step


def test_trajectory_starts_from_commanded_q_when_close_to_measured():
    clock = FakeClock()
    arm = SimArm(clock)
    commanded = arm.q + 0.05                                      # gravity-sag offset
    arm.get_arm_command = lambda: (commanded.copy(), np.zeros(14))
    shutdown(arm, clock)
    np.testing.assert_allclose(arm.commands[0][1], commanded)


def test_trajectory_ignores_far_or_nonfinite_command_and_starts_from_measured():
    for bad in (np.full(14, 2.0), np.full(14, np.nan)):
        clock = FakeClock()
        arm = SimArm(clock)
        measured = arm.q.copy()
        arm.get_arm_command = lambda bad=bad: (bad.copy(), np.full(14, np.inf))
        shutdown(arm, clock)
        np.testing.assert_allclose(arm.commands[0][1], measured)
        np.testing.assert_allclose(arm.commands[0][2], np.zeros(14))


def test_repeated_stop_is_idempotent_and_commands_nothing_new():
    clock = FakeClock()
    arm = SimArm(clock)
    first = shutdown(arm, clock)
    n_cmd, n_w = len(arm.commands), len(arm.weights)
    second = shutdown(arm, clock)
    assert second is first
    assert (len(arm.commands), len(arm.weights), arm.deactivated) == (n_cmd, n_w, 1)


@pytest.mark.parametrize("mutate,reason", [
    (lambda arm: setattr(arm, "q", np.full(14, np.nan)), "state_invalid"),
    (lambda arm: setattr(arm, "q", np.zeros(13)), "state_invalid"),
    (lambda arm: setattr(arm, "state_age", 5.0), "state_stale"),
    (lambda arm: setattr(arm, "q", np.full(14, 5.0)), "return_too_far"),
])
def test_invalid_state_skips_return_and_only_releases_weight(mutate, reason):
    clock = FakeClock()
    arm = SimArm(clock)
    mutate(arm)
    result = shutdown(arm, clock)
    assert arm.commands == []
    assert result.return_skipped_reason == reason
    assert [w for _, w in arm.weights][-1] == 0.0
    assert result.deactivated


def test_dead_writer_skips_motion_and_release_but_still_deactivates():
    clock = FakeClock()
    arm = SimArm(clock)
    arm.writer_dead = True
    result = shutdown(arm, clock)
    assert arm.commands == [] and arm.weights == []
    assert result.return_skipped_reason == "writer_not_publishing"
    assert not result.weight_released and result.deactivated


def test_arrival_timeout_is_bounded_then_release_happens():
    clock = FakeClock()
    arm = SimArm(clock)
    arm.follow = False            # arm stuck (e.g. obstruction)
    t0 = clock()
    result = shutdown(arm, clock, settle_timeout=1.0)
    assert result.returned_home and not result.arrival_confirmed
    assert result.weight_released and result.deactivated
    total = clock() - t0
    duration = plan_return_duration(np.linspace(-0.8, 0.9, 14), np.zeros(14))
    assert total <= duration + 1.0 + 2.0 + 0.6 + 0.2


def test_ctrl_c_during_return_cancels_motion_but_still_releases_and_deactivates():
    clock = FakeClock()
    arm = SimArm(clock)
    calls = {"n": 0}

    def sleep(seconds):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt
        clock.sleep(seconds)

    result = run_graceful_arm_shutdown(arm, clock=clock, sleep=sleep)
    assert result.cancelled and not result.returned_home
    assert len(arm.commands) == 5
    assert [w for _, w in arm.weights][-1] == 0.0
    assert arm.deactivated == 1


def test_attempt_return_false_releases_only():
    clock = FakeClock()
    arm = SimArm(clock)
    result = shutdown(arm, clock, attempt_return=False)
    assert arm.commands == [] and result.return_skipped_reason == "return_not_requested"
    assert result.weight_released and result.deactivated


def test_non_motion_mode_returns_home_without_weight_writes():
    clock = FakeClock()
    arm = SimArm(clock, motion_mode=False)
    result = shutdown(arm, clock)
    assert result.returned_home and arm.weights == [] and result.deactivated


def test_failing_hand_open_does_not_block_arm_release():
    clock = FakeClock()
    arm = SimArm(clock)
    result = shutdown(arm, clock, open_hands=lambda: (_ for _ in ()).throw(RuntimeError("dds")))
    assert not result.hands_opened
    assert result.weight_released and result.deactivated


# ---------------------------------------------------------------- tracking gate
class FakeIK:
    def __init__(self):
        self.calls = 0

    def solve_ik(self, *args):
        self.calls += 1
        return np.full(14, 9.0), np.zeros(14)


def test_q_during_tracking_produces_no_new_ik_and_no_enqueued_command():
    clock = FakeClock()
    arm = SimArm(clock)
    ik = FakeIK()
    calibrator = types.SimpleNamespace(calibrated=True, targets=lambda *a, **k: (np.eye(4), np.eye(4)))
    result = run_arm_tracking_cycle(
        arm_ctrl=arm, arm_ik=ik, calibrator=calibrator,
        controller_poses=(np.eye(4), np.eye(4)), sample_timestamp=time.monotonic(),
        current_q=arm.q, current_dq=np.zeros(14),
        lifecycle_lock=threading.Lock(), is_started=lambda: False, is_stopped=lambda: True,
    )
    assert ik.calls == 0 and arm.commands == [] and arm.active
    assert result.publication is None and result.decision_reason == "lifecycle_stop_hold"


# ---------------------------------------------------------- real G1_29 writer
def _load_robot_arm(monkeypatch):
    logging_mp = types.ModuleType("logging_mp")
    logging_mp.getLogger = lambda name: types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None)
    monkeypatch.setitem(sys.modules, "logging_mp", logging_mp)
    channel = types.ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelPublisher = channel.ChannelSubscriber = object
    channel.ChannelFactoryInitialize = lambda *a: None
    for name in ("unitree_sdk2py", "unitree_sdk2py.core", "unitree_sdk2py.idl", "unitree_sdk2py.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", channel)
    for name in ("unitree_hg", "unitree_go"):
        msg = types.ModuleType(f"unitree_sdk2py.idl.{name}.msg.dds_")
        msg.LowCmd_ = msg.LowState_ = object
        monkeypatch.setitem(sys.modules, f"unitree_sdk2py.idl.{name}.msg.dds_", msg)
    default = types.ModuleType("unitree_sdk2py.idl.default")
    default.unitree_hg_msg_dds__LowCmd_ = default.unitree_go_msg_dds__LowCmd_ = object
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.default", default)
    crc = types.ModuleType("unitree_sdk2py.utils.crc")
    crc.CRC = object
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.utils.crc", crc)
    sys.modules.pop("teleop.robot_control.robot_arm", None)
    try:
        return importlib.import_module("teleop.robot_control.robot_arm")
    finally:
        sys.modules.pop("teleop.robot_control.robot_arm", None)


def _writer_controller(module, published, stop_after):
    ctrl = module.G1_29_ArmController.__new__(module.G1_29_ArmController)
    ctrl.motion_mode = True
    ctrl.simulation_mode = True
    ctrl.control_dt = 0.0
    ctrl._speed_gradual_max = False
    ctrl.ctrl_lock = threading.Lock()
    ctrl._init_arm_publication_state()
    ctrl.q_target = np.zeros(14)
    ctrl.tauff_target = np.zeros(14)
    ctrl._motion_authority_weight = 1.0
    ctrl._last_publish_monotonic = None
    ctrl._last_published_weight = None
    ctrl.output_enabled = threading.Event()
    ctrl.output_enabled.set()
    ctrl.publish_thread = threading.Thread(target=lambda: None)
    ctrl.msg = types.SimpleNamespace(
        motor_cmd=[types.SimpleNamespace(q=0.0, dq=0.0, tau=0.0) for _ in range(35)], crc=0)
    ctrl.crc = types.SimpleNamespace(Crc=lambda msg: 0)

    class Publisher:
        def Write(self, msg):
            published.append(msg.motor_cmd[module.G1_29_JointIndex.kNotUsedJoint0].q)
            if len(published) == 2:
                ctrl.set_motion_authority_weight(0.4)
            if len(published) >= stop_after:
                ctrl.output_enabled.clear()

    ctrl.lowcmd_publisher = Publisher()
    return ctrl


def test_real_writer_publishes_commanded_weight_every_frame_and_reports_it(monkeypatch):
    module = _load_robot_arm(monkeypatch)
    published = []
    ctrl = _writer_controller(module, published, stop_after=4)
    ctrl._ctrl_motor_state()
    assert published == [1.0, 1.0, 0.4, 0.4]
    status = ctrl.get_publication_status()
    assert status["last_published_weight"] == 0.4
    assert status["last_publish_monotonic"] is not None
    with pytest.raises(ValueError):
        ctrl.set_motion_authority_weight(float("nan"))
    ctrl.set_motion_authority_weight(-3.0)
    assert ctrl.get_publication_status()["commanded_weight"] == 0.0


def test_real_writer_rejects_nonfinite_shutdown_waypoint(monkeypatch):
    module = _load_robot_arm(monkeypatch)
    published = []
    ctrl = _writer_controller(module, published, stop_after=1)
    ctrl.ctrl_dual_arm(np.full(14, np.nan), np.zeros(14))
    calls = {"n": 0}
    real_sleep = module.time.sleep

    def sleep(seconds):
        calls["n"] += 1
        if calls["n"] >= 3:
            ctrl.output_enabled.clear()
        real_sleep(0)

    monkeypatch.setattr(module.time, "sleep", sleep)
    ctrl._ctrl_motor_state()
    assert published == []
    receipts = ctrl.drain_arm_publication_receipts()
    assert receipts and all(r.published_q is None for r in receipts)


# -------------------------------------------------------------------- Dex3
def test_dex3_force_open_overrides_closed_trigger(monkeypatch):
    for name in ("unitree_sdk2py", "unitree_sdk2py.core", "unitree_sdk2py.idl",
                 "unitree_sdk2py.idl.unitree_hg", "unitree_sdk2py.idl.unitree_hg.msg",
                 "unitree_sdk2py.idl.unitree_go", "unitree_sdk2py.idl.unitree_go.msg"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    channel = types.ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelPublisher = channel.ChannelSubscriber = channel.ChannelFactoryInitialize = object
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", channel)
    hg = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    hg.HandCmd_ = hg.HandState_ = object
    go = types.ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_")
    go.MotorCmds_ = go.MotorStates_ = object
    default = types.ModuleType("unitree_sdk2py.idl.default")
    default.unitree_hg_msg_dds__HandCmd_ = default.unitree_go_msg_dds__MotorCmd_ = object
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.unitree_hg.msg.dds_", hg)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.unitree_go.msg.dds_", go)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.default", default)
    logging_mp = types.ModuleType("logging_mp")
    logging_mp.getLogger = lambda name: types.SimpleNamespace(info=lambda *a: None)
    monkeypatch.setitem(sys.modules, "logging_mp", logging_mp)
    sys.modules.pop("teleop.robot_control.robot_hand_unitree", None)
    try:
        module = importlib.import_module("teleop.robot_control.robot_hand_unitree")
    finally:
        sys.modules.pop("teleop.robot_control.robot_hand_unitree", None)

    published = []
    controller = module.Dex3_1_Controller.__new__(module.Dex3_1_Controller)
    controller.ctrl_dual_hand = lambda left, right, *ts: published.append((left.copy(), right.copy()))
    from multiprocessing import Array
    left = Array("d", [1.0, time.monotonic()])
    right = Array("d", [1.0, time.monotonic()])
    controller.control_step(None, None, left_ctrl_sample_in=left, right_ctrl_sample_in=right)
    assert np.abs(published[-1][0]).max() > 0.5          # trigger closes
    controller.outputs_activated = False
    controller.hand_control_process = None
    controller.open_and_deactivate(open_hold_s=0.0)
    controller.control_step(None, None, left_ctrl_sample_in=left, right_ctrl_sample_in=right)
    np.testing.assert_allclose(published[-1][0], np.zeros(7))
    np.testing.assert_allclose(published[-1][1], np.zeros(7))
    assert controller.running is False


# ------------------------------------------------ launcher finally integration
def _run_launcher(monkeypatch, *, raise_in_tele_data=None, press_q=True):
    calls = []

    class FakeArmController:
        arm_joint_split = (7, 7)

        def __init__(self, **kwargs):
            self.clock = time.monotonic
            self.motion_mode = kwargs.get("motion_mode", False)
            self.q = np.full(14, 0.2)
            self.weight = 1.0
            self.active = False

        def activate(self):
            self.active = True
            calls.append("activate")

        def ctrl_dual_arm_go_home(self, release_motion_authority=False):
            calls.append("prepare")
            return True

        def get_current_dual_arm_q(self):
            return self.q.copy()

        def get_dual_arm_q_snapshot(self):
            return self.q.copy(), 0.0

        def get_arm_command(self):
            return self.q.copy(), np.zeros(14)

        def ctrl_dual_arm(self, q, tau):
            calls.append("return_cmd")
            self.q = np.asarray(q, float).copy()

        def set_motion_authority_weight(self, w):
            calls.append(("weight", round(float(w), 3)))
            self.weight = float(w)

        def get_publication_status(self):
            return {"active": self.active, "last_publish_monotonic": time.monotonic(),
                    "last_published_weight": self.weight}

        def deactivate(self):
            self.active = False
            calls.append("deactivate")

        def speed_gradual_max(self):
            pass

    class FakeArmIK:
        def forward_kinematics(self, q):
            return np.eye(4), np.eye(4)

    callbacks = []

    class FakeTeleVuerWrapper:
        def __init__(self, **kwargs):
            pass

        def get_tele_data(self):
            if raise_in_tele_data is not None:
                raise raise_in_tele_data
            if callbacks and press_q:
                callbacks.pop()("q")
            return types.SimpleNamespace(
                head_pose=np.eye(4), left_wrist_pose=np.eye(4), right_wrist_pose=np.eye(4),
                controller_sample_timestamp=time.monotonic(),
                right_ctrl_aButton=False, right_ctrl_bButton=False)

        def close(self):
            calls.append("tv_close")

    class FakeImageClient:
        def __init__(self, **kwargs):
            pass

        def get_cam_config(self):
            camera = {"enable_webrtc": False, "enable_zmq": False, "image_shape": [2, 2],
                      "binocular": False, "webrtc_port": 0}
            return {"head_camera": camera, "left_wrist_camera": camera, "right_wrist_camera": camera}

        def close(self):
            pass

    class FakeIPCServer:
        def __init__(self, on_press, get_state):
            self.on_press = on_press

        def start(self):
            callbacks.append(self.on_press)

        def stop(self):
            calls.append("ipc_stop")

    logger = types.SimpleNamespace(debug=lambda *a: None, info=lambda *a: None,
                                   warning=lambda *a: None, error=lambda *a: None)
    modules = {
        "logging_mp": types.SimpleNamespace(basicConfig=lambda **kwargs: None, getLogger=lambda name: logger, INFO=20),
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": types.SimpleNamespace(ChannelFactoryInitialize=lambda *a, **k: None, ChannelPublisher=object),
        "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.std_msgs": types.ModuleType("unitree_sdk2py.idl.std_msgs"),
        "unitree_sdk2py.idl.std_msgs.msg": types.ModuleType("unitree_sdk2py.idl.std_msgs.msg"),
        "unitree_sdk2py.idl.std_msgs.msg.dds_": types.SimpleNamespace(String_=object),
        "televuer": types.SimpleNamespace(TeleVuerWrapper=FakeTeleVuerWrapper),
        "teleimager": types.ModuleType("teleimager"),
        "teleimager.image_client": types.SimpleNamespace(ImageClient=FakeImageClient),
        "sshkeyboard": types.SimpleNamespace(listen_keyboard=lambda **kwargs: None, stop_listening=lambda: None),
        "teleop.robot_control.robot_arm": types.SimpleNamespace(
            G1_29_ArmController=FakeArmController, G1_23_ArmController=FakeArmController,
            H1_2_ArmController=FakeArmController, H1_ArmController=FakeArmController, H2_ArmController=FakeArmController),
        "teleop.robot_control.robot_arm_ik": types.SimpleNamespace(
            G1_29_ArmIK=FakeArmIK, G1_23_ArmIK=FakeArmIK, H1_2_ArmIK=FakeArmIK, H1_ArmIK=FakeArmIK, H2_ArmIK=FakeArmIK),
        "teleop.utils.episode_writer": types.SimpleNamespace(EpisodeWriter=object),
        "teleop.utils.ipc": types.SimpleNamespace(IPC_Server=FakeIPCServer),
        "teleop.utils.motion_switcher": types.SimpleNamespace(MotionSwitcher=object, LocoClientWrapper=lambda: None),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("XR_TELEOP_STATUS_LOG", "/nonexistent-dir-for-test/status.jsonl")
    monkeypatch.setenv("XR_TELEOP_POSE_LOG_DIR", "/nonexistent-dir-for-test")
    monkeypatch.setattr(sys, "argv", ["teleop_hand_and_arm.py", "--motion", "--input-mode", "hand",
                                      "--camera-layout", "head", "--ipc", "--arm", "G1_29"])
    import teleop.utils.arm_graceful_shutdown as shutdown_module
    fast = FakeClock()
    real = shutdown_module.run_graceful_arm_shutdown
    monkeypatch.setattr(shutdown_module, "run_graceful_arm_shutdown",
                        lambda arm, **kw: real(arm, **{**kw, "clock": fast, "sleep": fast.sleep}))
    runpy.run_path(str(ROOT / "teleop" / "teleop_hand_and_arm.py"), run_name="__main__")
    return calls


def _assert_graceful_sequence(calls):
    assert calls[:2] == ["activate", "prepare"]
    first_return = calls.index("return_cmd")
    weights = [c[1] for c in calls if isinstance(c, tuple)]
    first_weight = next(i for i, c in enumerate(calls) if isinstance(c, tuple))
    assert first_return < first_weight < calls.index("deactivate") < calls.index("ipc_stop")
    assert weights[0] == 1.0 and weights[-1] == 0.0 and np.all(np.diff(weights) <= 0)
    assert calls.count("deactivate") == 1


def test_launcher_q_runs_graceful_return_then_release_before_teardown(monkeypatch):
    _assert_graceful_sequence(_run_launcher(monkeypatch))


def test_launcher_exception_path_runs_graceful_shutdown_in_finally(monkeypatch):
    _assert_graceful_sequence(_run_launcher(monkeypatch, raise_in_tele_data=RuntimeError("xr died")))


def test_launcher_ctrl_c_path_runs_graceful_shutdown_in_finally(monkeypatch):
    _assert_graceful_sequence(_run_launcher(monkeypatch, raise_in_tele_data=KeyboardInterrupt()))
