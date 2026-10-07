"""dev-inspire graceful shutdown (ported from main-line edfd901). Fakes only:
no DDS, no actuators, no network.

Covers: velocity-limited return + rt/arm_sdk weight ramp 1->0, idempotence,
Ctrl+C during return, dead writer, real G1_29 writer weight/stop, Inspire
command process stop (no hand command after shutdown, bounded join), TeleVuer
without orphan, and the real teleop_hand_and_arm.py finally block for q before
r, q after r, exception, Ctrl+C and SIGTERM.
"""

import importlib
import multiprocessing
import os
import runpy
import signal
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
from teleop.utils import session_shutdown  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += float(seconds)


class SimArm:
    """Ideal servo: measured q follows the last command. Records everything."""

    arm_joint_split = (7, 7)

    def __init__(self, clock, start_q=None, motion_mode=True):
        self.clock = clock
        self.motion_mode = motion_mode
        self.q = np.linspace(-0.8, 0.9, 14) if start_q is None else np.asarray(start_q, float)
        self.commands, self.weights, self.log = [], [], []
        self.weight = 1.0
        self.active = True
        self.deactivated = 0
        self.writer_dead = False
        self.state_age = 0.0

    def get_dual_arm_q_snapshot(self):
        return self.q.copy(), self.state_age

    def get_arm_command(self):
        return self.q.copy(), np.full(14, 2.0)

    def ctrl_dual_arm(self, q, tau):
        assert self.active, "command after deactivate"
        self.commands.append((self.clock(), np.asarray(q, float).copy()))
        self.log.append("cmd")
        self.q = np.asarray(q, float).copy()

    def set_motion_authority_weight(self, weight):
        assert self.active, "weight after deactivate"
        self.weight = float(weight)
        self.weights.append((self.clock(), self.weight))
        self.log.append("weight")

    def get_publication_status(self):
        return {"active": self.active and not self.writer_dead,
                "last_publish_monotonic": None if self.writer_dead else self.clock(),
                "last_published_weight": self.weight if self.motion_mode else None}

    def deactivate(self, join_timeout=1.0):
        self.active = False
        self.deactivated += 1
        self.log.append("deactivate")


def shutdown(arm, clock, **kw):
    return run_graceful_arm_shutdown(arm, clock=clock, sleep=clock.sleep, **kw)


# ------------------------------------------------------------- arm core
def test_return_velocity_limited_then_weight_ramp_ends_at_zero_then_deactivate():
    clock = FakeClock()
    arm = SimArm(clock)
    start = arm.q.copy()
    events = []
    result = shutdown(arm, clock, emit=lambda name, detail: events.append((name, detail)))
    qs = np.array([q for _, q in arm.commands])
    ts = np.array([t for t, _ in arm.commands])
    np.testing.assert_allclose(qs[0], start)
    np.testing.assert_allclose(qs[-1], np.zeros(14))
    velocity = np.abs(np.diff(qs, axis=0)) / np.diff(ts)[:, None]
    assert velocity.max() <= DEFAULT_MAX_JOINT_VELOCITY + 1e-9
    assert ts[-1] - ts[0] == pytest.approx(plan_return_duration(start, np.zeros(14)), abs=0.03)
    weights = [w for _, w in arm.weights]
    assert weights[0] == 1.0 and weights[-1] == 0.0 and np.all(np.diff(weights) <= 0)
    assert np.max(-np.diff(weights)) <= 0.011
    assert result.returned_home and result.release_confirmed and result.deactivated
    assert arm.log[-1] == "deactivate" and arm.deactivated == 1
    started = dict(events)["shutdown_return_started"]
    assert started["max_joint_velocity"] == 0.5
    assert started["max_distance_rad"] == pytest.approx(0.9)


def test_shutdown_is_idempotent():
    clock = FakeClock()
    arm = SimArm(clock)
    first = shutdown(arm, clock)
    n = (len(arm.commands), len(arm.weights), arm.deactivated)
    assert shutdown(arm, clock) is first
    assert (len(arm.commands), len(arm.weights), arm.deactivated) == n


def test_ctrl_c_during_return_still_releases_and_deactivates():
    clock = FakeClock()
    arm = SimArm(clock)
    calls = {"n": 0}

    def sleep(seconds):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt
        clock.sleep(seconds)

    result = run_graceful_arm_shutdown(arm, clock=clock, sleep=sleep)
    assert result.cancelled and [w for _, w in arm.weights][-1] == 0.0 and arm.deactivated == 1


@pytest.mark.parametrize("mutate,reason", [
    (lambda a: setattr(a, "q", np.full(14, np.nan)), "state_invalid"),
    (lambda a: setattr(a, "state_age", 5.0), "state_stale"),
    (lambda a: setattr(a, "q", np.full(14, 5.0)), "return_too_far"),
])
def test_invalid_state_skips_motion_and_only_releases(mutate, reason):
    clock = FakeClock()
    arm = SimArm(clock)
    mutate(arm)
    result = shutdown(arm, clock)
    assert arm.commands == [] and result.return_skipped_reason == reason
    assert [w for _, w in arm.weights][-1] == 0.0 and result.deactivated


def test_dead_writer_only_deactivates():
    clock = FakeClock()
    arm = SimArm(clock)
    arm.writer_dead = True
    result = shutdown(arm, clock)
    assert arm.commands == [] and arm.weights == [] and result.deactivated


# ------------------------------------------------ real G1_29 writer (no DDS)
def _stub_sdk(monkeypatch):
    logging_mp = types.ModuleType("logging_mp")
    logging_mp.getLogger = lambda name: types.SimpleNamespace(
        info=lambda *a: None, warning=lambda *a: None, error=lambda *a: None, debug=lambda *a: None)
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


def _import_fresh(monkeypatch, name):
    sys.modules.pop(name, None)
    try:
        return importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)


def _writer(module, published):
    ctrl = module.G1_29_ArmController.__new__(module.G1_29_ArmController)
    ctrl.motion_mode = True
    ctrl.simulation_mode = True
    ctrl.control_dt = 0.001
    ctrl._speed_gradual_max = False
    ctrl.ctrl_lock = threading.Lock()
    ctrl.q_target = np.zeros(14)
    ctrl.tauff_target = np.zeros(14)
    ctrl._motion_authority_weight = 1.0
    ctrl._last_publish_monotonic = None
    ctrl._last_published_weight = None
    ctrl.output_enabled = threading.Event()
    ctrl.output_enabled.set()
    ctrl.msg = types.SimpleNamespace(
        motor_cmd=[types.SimpleNamespace(q=0.0, dq=0.0, tau=0.0) for _ in range(35)], crc=0)
    ctrl.crc = types.SimpleNamespace(Crc=lambda msg: 0)
    ctrl.lowcmd_publisher = types.SimpleNamespace(
        Write=lambda msg: published.append(msg.motor_cmd[module.G1_29_JointIndex.kNotUsedJoint0].q))
    ctrl.publish_thread = threading.Thread(target=ctrl._ctrl_motor_state, daemon=True)
    return ctrl


def test_real_writer_publishes_weight_each_frame_and_stops_on_deactivate(monkeypatch):
    _stub_sdk(monkeypatch)
    module = _import_fresh(monkeypatch, "teleop.robot_control.robot_arm")
    published = []
    ctrl = _writer(module, published)
    ctrl.publish_thread.start()
    time.sleep(0.05)
    assert published and published[0] == 1.0
    ctrl.set_motion_authority_weight(0.0)
    time.sleep(0.05)
    assert published[-1] == 0.0
    assert ctrl.get_publication_status()["last_published_weight"] == 0.0
    t0 = time.monotonic()
    assert ctrl.deactivate(join_timeout=1.0) is True
    assert time.monotonic() - t0 < 1.0
    n = len(published)
    time.sleep(0.05)
    assert len(published) == n and not ctrl.get_publication_status()["active"]
    assert ctrl.deactivate() is True  # idempotent
    with pytest.raises(ValueError):
        ctrl.set_motion_authority_weight(float("nan"))


def test_real_writer_shutdown_end_to_end_ramp_reaches_zero(monkeypatch):
    _stub_sdk(monkeypatch)
    module = _import_fresh(monkeypatch, "teleop.robot_control.robot_arm")
    published = []
    ctrl = _writer(module, published)
    state = {"q": np.full(14, 0.1)}
    ctrl.get_dual_arm_q_snapshot = lambda: (state["q"].copy(), 0.0)
    real_ctrl = ctrl.ctrl_dual_arm

    def follow(q, tau):
        real_ctrl(q, tau)
        state["q"] = np.asarray(q, float).copy()

    ctrl.ctrl_dual_arm = follow
    ctrl.publish_thread.start()
    time.sleep(0.02)
    result = run_graceful_arm_shutdown(ctrl, clock=time.monotonic, sleep=time.sleep,
                                       release_duration=0.1, release_dt=0.01, min_return_duration=0.05)
    assert result.returned_home and result.release_confirmed and result.deactivated
    assert published[-1] == 0.0 and not ctrl.publish_thread.is_alive()


# ------------------------------------------------------------- Inspire hand
def _stub_inspire_deps(monkeypatch):
    _stub_sdk(monkeypatch)
    retarget = types.ModuleType("teleop.robot_control.hand_retargeting")
    retarget.HandRetargeting = object
    retarget.HandType = types.SimpleNamespace(INSPIRE_HAND=0, INSPIRE_HAND_Unit_Test=1)
    monkeypatch.setitem(sys.modules, "teleop.robot_control.hand_retargeting", retarget)
    for name in ("cyclonedds", "cyclonedds.idl", "cyclonedds.idl.annotations", "cyclonedds.idl.types"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class IdlStruct:
        def __init_subclass__(cls, **kw):
            pass

        def __init__(self, **kw):
            self.__dict__.update(kw)

    sys.modules["cyclonedds.idl"].IdlStruct = IdlStruct
    ann = sys.modules["cyclonedds.idl.annotations"]
    ann.final = lambda c: c
    ann.autoid = lambda *_: (lambda c: c)
    t = sys.modules["cyclonedds.idl.types"]
    t.sequence = type("S", (), {"__class_getitem__": classmethod(lambda c, k: list)})
    t.int16 = t.int8 = t.uint8 = int
    sys.modules["cyclonedds"].idl = sys.modules["cyclonedds.idl"]
    sys.modules["cyclonedds.idl"].annotations = ann
    sys.modules["cyclonedds.idl"].types = t
    return _import_fresh(monkeypatch, "teleop.robot_control.robot_hand_inspire")


def test_inspire_dfx_no_hand_command_after_deactivate(monkeypatch):
    module = _stub_inspire_deps(monkeypatch)
    ctrl = module.Inspire_Controller_DFX.__new__(module.Inspire_Controller_DFX)
    ctrl.fps = 500.0
    ctrl._stop_event = multiprocessing.Event()
    writes = []
    pub = types.SimpleNamespace(Write=lambda msg: writes.append(list(msg.angle_set)))
    ctrl.LeftHandCmd_publisher = ctrl.RightHandCmd_publisher = pub
    left = multiprocessing.Array("d", 75)
    right = multiprocessing.Array("d", 75)
    state = multiprocessing.Array("d", 6)
    # Run the command loop in a thread here (a process in production).
    worker = threading.Thread(target=ctrl.control_process, args=(left, right, state, state), daemon=True)
    worker.start()
    time.sleep(0.05)
    assert writes
    ctrl.hand_control_process = worker
    t0 = time.monotonic()
    ctrl.deactivate(join_timeout=1.0)
    assert time.monotonic() - t0 < 1.0 and not worker.is_alive()
    n = len(writes)
    time.sleep(0.05)
    assert len(writes) == n                   # nothing published after shutdown
    # the shutdown path itself never adds an open/close command
    assert all(w == writes[0] for w in writes)


def _ignore_stop_forever(_event):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # also ignores terminate()
    while True:
        time.sleep(0.05)


def test_stop_hand_process_timeout_does_not_hang(monkeypatch):
    module = _stub_inspire_deps(monkeypatch)
    event = multiprocessing.Event()
    proc = multiprocessing.Process(target=_ignore_stop_forever, args=(event,), daemon=True)
    proc.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    assert module.stop_hand_process(proc, event, join_timeout=0.2, kill_timeout=0.5) is True
    assert time.monotonic() - t0 < 2.0
    assert event.is_set() and not proc.is_alive()


# ------------------------------------------------------------- TeleVuer
def _vuer_like(pid_file):
    # Vuer's own helper child: survives its parent's SIGTERM (orphan risk).
    helper = multiprocessing.Process(target=_ignore_stop_forever, args=(None,))
    helper.start()
    Path(pid_file).write_text(str(helper.pid))
    while True:
        time.sleep(0.05)


def test_close_televuer_leaves_no_orphan(tmp_path):
    pid_file = tmp_path / "helper.pid"
    proc = multiprocessing.Process(target=_vuer_like, args=(str(pid_file),), daemon=False)
    proc.start()
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text():
            break
        time.sleep(0.02)
    helper_pid = int(pid_file.read_text())

    class Wrapper:  # like TeleVuerWrapper.close(): terminate + join(0.5), no kill
        def __init__(self):
            self.tvuer = types.SimpleNamespace(process=proc)

        def close(self):
            time.sleep(5)  # hangs: must be bounded by close_televuer

    t0 = time.monotonic()
    assert session_shutdown.close_televuer(Wrapper(), timeout=0.5) is True
    assert time.monotonic() - t0 < 3.0
    assert not proc.is_alive()
    time.sleep(0.2)
    import psutil
    assert not psutil.pid_exists(helper_pid) or psutil.Process(helper_pid).status() == psutil.STATUS_ZOMBIE


# --------------------------------------------- session order / robustness
def test_session_shutdown_order_and_exceptions_do_not_skip_steps():
    clock = FakeClock()
    arm = SimArm(clock)
    order = []

    class Hand:
        def deactivate(self, join_timeout=1.0):
            order.append("hand")
            raise RuntimeError("hand boom")

    class TV:
        tvuer = types.SimpleNamespace(process=None)

        def close(self):
            order.append("tv")
            raise RuntimeError("tv boom")

    real_deactivate = arm.deactivate
    arm.deactivate = lambda **kw: (order.append("arm"), real_deactivate())
    steps = session_shutdown.run_session_shutdown(
        arm_kind="G1_29", arm_ctrl=arm, hand_ctrl=Hand(), tv_wrapper=TV(),
        clock=clock, sleep=clock.sleep, reap=False)
    assert order == ["hand", "arm", "tv"]
    assert steps == ["hand_stop_failed", "arm_released", "televuer_closed"]
    assert arm.weights[-1][1] == 0.0
    # idempotent: nothing re-run
    session_shutdown.run_session_shutdown(arm_kind="G1_29", arm_ctrl=arm, hand_ctrl=Hand(),
                                          tv_wrapper=TV(), clock=clock, sleep=clock.sleep, reap=False)
    assert order.count("arm") == 1


# ------------------------------- real teleop_hand_and_arm.py finally block
def _run_teleop(monkeypatch, *, keys=("r", "q"), raise_in_tele_data=None, sigterm=False):
    calls = []
    callbacks = []

    class FakeArm:
        arm_joint_split = (7, 7)

        def __init__(self, **kw):
            self.motion_mode = kw.get("motion_mode", False)
            self.q = np.full(14, 0.2)
            self.weight = 1.0
            self.active = True
            calls.append("arm_init")

        def get_dual_arm_q_snapshot(self):
            return self.q.copy(), 0.0

        def get_current_dual_arm_q(self):
            return self.q.copy()

        def get_current_dual_arm_dq(self):
            return np.zeros(14)

        def get_arm_command(self):
            return self.q.copy(), np.zeros(14)

        def ctrl_dual_arm(self, q, tau):
            assert self.active
            calls.append("cmd")
            self.q = np.asarray(q, float).copy()

        def set_motion_authority_weight(self, w):
            assert self.active
            calls.append(("weight", round(float(w), 3)))
            self.weight = float(w)

        def get_publication_status(self):
            return {"active": self.active, "last_publish_monotonic": time.monotonic(),
                    "last_published_weight": self.weight}

        def deactivate(self, join_timeout=1.0):
            self.active = False
            calls.append("arm_deactivate")

        def speed_gradual_max(self):
            calls.append("tracking")

        def ctrl_dual_arm_go_home(self):
            calls.append("legacy_go_home")

    class FakeIK:
        def solve_ik(self, *a):
            return np.full(14, 0.2), np.zeros(14)

    class FakeHand:
        def __init__(self, *a, **kw):
            calls.append("hand_init")

        def deactivate(self, join_timeout=1.0):
            calls.append("hand_stop")

    tele = {"n": 0}

    class FakeTV:
        def __init__(self, **kw):
            self.tvuer = types.SimpleNamespace(process=None)

        def render_to_xr(self, img):
            pass

        def get_tele_data(self):
            tele["n"] += 1
            if raise_in_tele_data is not None:
                raise raise_in_tele_data
            if sigterm:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.05)
            if tele["n"] == 2 and callbacks and "q" in keys:
                callbacks[0]("q")
            return types.SimpleNamespace(
                left_hand_pos=np.zeros((25, 3)), right_hand_pos=np.zeros((25, 3)),
                left_wrist_pose=np.eye(4), right_wrist_pose=np.eye(4))

        def close(self):
            calls.append("tv_close")

    class FakeImageClient:
        def __init__(self, **kw):
            pass

        def get_cam_config(self):
            cam = {"enable_webrtc": False, "enable_zmq": False, "image_shape": [480, 640],
                   "binocular": False, "webrtc_port": 0, "fps": 30}
            return {"head_camera": cam, "left_wrist_camera": dict(cam), "right_wrist_camera": dict(cam)}

        def close(self):
            calls.append("img_close")

    class FakeIPC:
        def __init__(self, on_press, get_state):
            self.on_press = on_press

        def start(self):
            callbacks.append(self.on_press)
            if "r" in keys:
                threading.Timer(0.05, lambda: self.on_press("r")).start()
            elif "q" in keys:
                threading.Timer(0.05, lambda: self.on_press("q")).start()

        def stop(self):
            calls.append("ipc_stop")

    logger = types.SimpleNamespace(debug=lambda *a: None, info=lambda *a: None,
                                   warning=lambda *a: None, error=lambda *a: calls.append(("error", a)))
    modules = {
        "logging_mp": types.SimpleNamespace(basicConfig=lambda **k: None, getLogger=lambda n: logger, INFO=20),
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": types.SimpleNamespace(ChannelFactoryInitialize=lambda *a, **k: None,
                                                             ChannelPublisher=object),
        "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.std_msgs": types.ModuleType("unitree_sdk2py.idl.std_msgs"),
        "unitree_sdk2py.idl.std_msgs.msg": types.ModuleType("unitree_sdk2py.idl.std_msgs.msg"),
        "unitree_sdk2py.idl.std_msgs.msg.dds_": types.SimpleNamespace(String_=object),
        "televuer": types.SimpleNamespace(TeleVuerWrapper=FakeTV),
        "teleimager": types.ModuleType("teleimager"),
        "teleimager.image_client": types.SimpleNamespace(ImageClient=FakeImageClient),
        "sshkeyboard": types.SimpleNamespace(listen_keyboard=lambda **k: None, stop_listening=lambda: None),
        "cv2": types.ModuleType("cv2"),
        "teleop.robot_control.robot_arm": types.SimpleNamespace(
            G1_29_ArmController=FakeArm, G1_23_ArmController=FakeArm, H1_2_ArmController=FakeArm,
            H1_ArmController=FakeArm, H2_ArmController=FakeArm),
        "teleop.robot_control.robot_arm_ik": types.SimpleNamespace(
            G1_29_ArmIK=FakeIK, G1_23_ArmIK=FakeIK, H1_2_ArmIK=FakeIK, H1_ArmIK=FakeIK, H2_ArmIK=FakeIK),
        "teleop.robot_control.robot_hand_inspire": types.SimpleNamespace(Inspire_Controller_DFX=FakeHand),
        "teleop.utils.episode_writer": types.SimpleNamespace(EpisodeWriter=object),
        "teleop.utils.ipc": types.SimpleNamespace(IPC_Server=FakeIPC),
        "teleop.utils.motion_switcher": types.SimpleNamespace(MotionSwitcher=object, LocoClientWrapper=object),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", ["teleop_hand_and_arm.py", "--motion", "--input-mode", "hand", "--ipc",
                                      "--arm", "G1_29", "--ee", "inspire_dfx", "--camera-layout", "head"])
    fast = FakeClock()
    real = session_shutdown.run_graceful_arm_shutdown
    monkeypatch.setattr(session_shutdown, "run_graceful_arm_shutdown",
                        lambda arm, **kw: real(arm, **{**kw, "clock": fast, "sleep": fast.sleep}))
    monkeypatch.setattr(session_shutdown, "reap_child_processes", lambda **kw: calls.append("reap") or [])
    old_term, old_hup = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)
    try:
        with pytest.raises(SystemExit):
            runpy.run_path(str(ROOT / "teleop" / "teleop_hand_and_arm.py"), run_name="__main__")
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGHUP, old_hup)
    return calls


def _assert_sequence(calls, tracked):
    i_hand = calls.index("hand_stop")
    assert ("tracking" in calls) == (tracked is not None)
    if tracked is not None:
        assert ("cmd" in calls[:i_hand]) == tracked  # IK target(s) published before q
    weights = [c[1] for c in calls if isinstance(c, tuple) and c[0] == "weight"]
    i_w0 = next(i for i, c in enumerate(calls) if isinstance(c, tuple) and c[0] == "weight")
    i_deact = calls.index("arm_deactivate")
    assert i_hand < i_w0 < i_deact < calls.index("ipc_stop") < calls.index("tv_close") < calls.index("reap")
    assert weights[0] == 1.0 and weights[-1] == 0.0 and np.all(np.diff(weights) <= 0)
    assert calls.count("arm_deactivate") == 1 and calls.count("hand_stop") == 1
    assert calls.count("tv_close") == 1 and "legacy_go_home" not in calls
    # no command after the writer was deactivated
    assert "cmd" not in calls[i_deact:] and not any(isinstance(c, tuple) and c[0] == "weight" for c in calls[i_deact:])


def test_q_before_r_runs_graceful_shutdown(monkeypatch):
    calls = _run_teleop(monkeypatch, keys=("q",))
    # q before r: no IK target was ever sent; speed_gradual_max still runs
    # (pre-existing behaviour, unchanged), so only the return+release is checked.
    assert "cmd" not in calls[:calls.index("hand_stop")]
    _assert_sequence(calls, tracked=False)


def test_q_after_r_runs_graceful_shutdown(monkeypatch):
    _assert_sequence(_run_teleop(monkeypatch, keys=("r", "q")), tracked=True)


def test_exception_in_loop_runs_graceful_shutdown(monkeypatch):
    _assert_sequence(_run_teleop(monkeypatch, keys=("r",), raise_in_tele_data=RuntimeError("xr died")),
                     tracked=False)


def test_ctrl_c_runs_graceful_shutdown(monkeypatch):
    _assert_sequence(_run_teleop(monkeypatch, keys=("r",), raise_in_tele_data=KeyboardInterrupt()),
                     tracked=False)


def test_sigterm_runs_graceful_shutdown(monkeypatch):
    _assert_sequence(_run_teleop(monkeypatch, keys=("r",), sigterm=True), tracked=False)
