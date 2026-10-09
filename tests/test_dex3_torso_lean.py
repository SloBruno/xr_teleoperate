"""Torso lean + pose stream on the Dex3 line (controllers, deferred activate()).

* geometry: world-fixed controller targets -> torso frame (pinocchio check);
* run_arm_tracking_cycle(target_transform=...) default is unchanged;
* the REAL teleop_hand_and_arm.py main loop with --ee dex3 and fakes:
  - feature off: no waist API call, IK gets the calibrated targets unchanged,
    no gravity change, identical call sequence to the off baseline;
  - feature on: waist taken only after r (after activate()), yaw stays at the
    neutral even with the headset turning, waist back to neutral before the
    arm_sdk weight ramp, Dex3 graceful order kept;
  - q before r never takes the waist.
No DDS, no actuators.
"""

import math
import os
import runpy
import sys
import threading
import time
import types
from multiprocessing import Array
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teleop.utils import torso_lean as tl  # noqa: E402
from teleop.utils.arm_tracking_orchestration import run_arm_tracking_cycle  # noqa: E402

DEG = math.pi / 180.0
URDF = ROOT / "assets" / "g1" / "g1_body29_hand14.urdf"


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def head(pivot=(0.0, 0.0, 1.6), yaw=0.0):
    R = _rot("z", yaw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pivot, float) + R @ tl.NECK_TO_EYE
    return T


# ------------------------------------------------------------- geometry
def test_world_fixed_retarget_puts_leaning_wrist_on_the_world_target():
    """Full URDF FK with waist roll/pitch: the reduced-model solution for the
    retargeted target places the real wrist exactly on the original (pelvis
    frame) target, i.e. the Dex3 hand stays where the controller says."""
    pin = pytest.importorskip("pinocchio")
    from teleop.utils.arm_fk import G1_29_WristFK
    robot = pin.RobotWrapper.BuildFromURDF(str(URDF), str(URDF.parent))
    model, data = robot.model, robot.model.createData()
    fk = G1_29_WristFK()
    rd = fk.model.createData()
    rng = np.random.default_rng(3)
    jl = fk.model.getJointId("left_wrist_yaw_joint")
    for _ in range(5):
        q_arm = np.clip(rng.normal(0, 0.3, 14), fk.model.lowerPositionLimit, fk.model.upperPositionLimit)
        lean = (0.0, rng.uniform(-10, 10) * DEG, rng.uniform(-10, 10) * DEG)
        q_full = np.zeros(model.nq)
        for name, v in (("waist_yaw_joint", lean[0]), ("waist_roll_joint", lean[1]), ("waist_pitch_joint", lean[2])):
            q_full[model.joints[model.getJointId(name)].idx_q] = v
        for k, name in enumerate(fk.model.names[1:]):
            q_full[model.joints[model.getJointId(name)].idx_q] = q_arm[k]
        pin.forwardKinematics(model, data, q_full)
        W_full = data.oMi[model.getJointId("left_wrist_yaw_joint")].homogeneous.copy()   # real wrist, pelvis frame
        pin.forwardKinematics(fk.model, rd, q_arm)
        W_red = rd.oMi[jl].homogeneous.copy()                                             # reduced model, same q
        # The world target W_full, retargeted, must equal what the reduced model reaches with q_arm.
        np.testing.assert_allclose(tl.retarget_world_fixed_to_torso(W_full, tl.waist_rotation(lean)), W_red,
                                   atol=1e-9)
    T = np.eye(4)
    T[:3, 3] = (0.3, 0.2, 0.1)
    np.testing.assert_array_equal(tl.retarget_world_fixed_to_torso(T, np.eye(3)), T)


# --------------------------------------------------------- orchestration
class _Arm:
    arm_joint_split = (7, 7)

    def __init__(self):
        self.published = []

    def get_current_dual_arm_q(self):
        return np.zeros(14)

    def ctrl_dual_arm(self, q, tau):
        self.published.append(np.asarray(q).copy())
        return len(self.published)


class _IK:
    def __init__(self):
        self.targets = []

    def solve_ik(self, l, r, q, dq):
        self.targets.append((np.array(l), np.array(r)))
        return np.zeros(14), np.zeros(14)

    def forward_kinematics(self, q):
        return self.targets[-1]


def _pair():
    a, b = np.eye(4), np.eye(4)
    a[:3, 3] = (0.3, 0.2, 0.1)
    b[:3, 3] = (0.3, -0.2, 0.1)
    return a, b


def _cycle(transform):
    ik = _IK()
    res = run_arm_tracking_cycle(
        arm_ctrl=_Arm(), arm_ik=ik, calibrator=None, controller_poses=_pair(),
        sample_timestamp=time.monotonic(), current_q=np.zeros(14), current_dq=np.zeros(14),
        candidate_targets=_pair(), lifecycle_lock=threading.Lock(),
        is_started=lambda: True, is_stopped=lambda: False, target_transform=transform)
    return res, ik


def test_cycle_without_transform_is_unchanged_and_with_transform_feeds_ik():
    res, ik = _cycle(None)
    assert res.published and res.decision_reason == "ik_command_selected"
    np.testing.assert_array_equal(ik.targets[0][0], _pair()[0])
    np.testing.assert_array_equal(res.ik_target[0], _pair()[0])
    R = tl.waist_rotation((0.0, 0.0, 5 * DEG))
    res, ik = _cycle(lambda p: tl.retarget_world_fixed_to_torso(p, R))
    assert res.published
    np.testing.assert_allclose(ik.targets[0][0], tl.retarget_world_fixed_to_torso(_pair()[0], R))
    np.testing.assert_array_equal(res.target[0], _pair()[0])          # requested/accepted target untouched
    res, ik = _cycle(lambda p: np.full((4, 4), np.nan))                # bad transform: fail closed, no IK
    assert not res.published and ik.targets == [] and res.ik_target is None


# --------------------------------------------- real teleop main loop (Dex3)
W_LEFT = np.eye(4)
W_LEFT[:3, 3] = (0.25, 0.20, 0.10)
W_RIGHT = np.eye(4)
W_RIGHT[:3, 3] = (0.25, -0.20, 0.10)


def _run_dex3(monkeypatch, env, *, keys=("r", "q"), cycles_before_q=30, head_fn=None,
              waist_follows=True):
    calls = []
    state = {"ik_targets": [], "gravity": [], "waist_targets": [], "arm_cmds": 0}
    callbacks = []

    class FakeArm:
        arm_joint_split = (7, 7)

        def __init__(self, **kw):
            self.motion_mode = kw.get("motion_mode", False)
            self.q = np.full(14, 0.2)
            self.weight = 1.0
            self.active = False
            self.waist_held = np.array([0.0, 0.01, -0.02])
            self.waist_cfg = None
            self.waist_target = None

        def activate(self):
            self.active = True
            calls.append("activate")

        def ctrl_dual_arm_go_home(self, release_motion_authority=False):
            calls.append("prepare")
            return True

        def get_current_dual_arm_q(self):
            return self.q.copy()

        def get_current_dual_arm_dq(self):
            return np.zeros(14)

        def get_dual_arm_q_snapshot(self):
            return self.q.copy(), 0.0

        def get_arm_command(self):
            return self.q.copy(), np.zeros(14)

        def ctrl_dual_arm(self, q, tau):
            assert self.active
            state["arm_cmds"] += 1
            calls.append("cmd")
            return state["arm_cmds"]

        def set_motion_authority_weight(self, w):
            calls.append(("weight", round(float(w), 3)))
            self.weight = float(w)

        def get_publication_status(self):
            return {"active": self.active, "last_publish_monotonic": time.monotonic(),
                    "last_published_weight": self.weight}

        def drain_arm_publication_receipts(self):
            return ()

        publication_receipt_drop_count = 0

        def deactivate(self):
            self.active = False
            calls.append("arm_deactivate")

        def speed_gradual_max(self):
            calls.append("tracking")

        # waist API (only used by G1_TORSO_LEAN)
        def get_waist_q_snapshot(self):
            calls.append("waist_read")
            return self.waist_held.copy(), 0.0

        def get_waist_command_written(self):
            assert self.active, "waist read before activate()"
            return self.waist_held.copy()

        def configure_waist_command(self, lower, upper, rate, initial):
            assert self.active, "waist configured before activate()"
            calls.append("waist_configure")
            self.waist_cfg = (np.array(lower), np.array(upper), rate, np.array(initial))
            self.waist_target = np.array(initial)

        def set_waist_target(self, q):
            calls.append("waist_target")
            self.waist_target = np.array(q)
            if waist_follows:
                self.waist_held = self.waist_target.copy()
            state["waist_targets"].append(np.array(q))

        def get_waist_command(self):
            if self.waist_cfg is None:
                return {"enabled": False}
            return {"enabled": True, "target": self.waist_target.copy(), "written": self.waist_target.copy(),
                    "neutral": self.waist_cfg[3].copy(), "max_rate": self.waist_cfg[2]}

        def waist_return_to_neutral(self):
            calls.append("waist_neutral")
            self.waist_target = self.waist_cfg[3].copy()
            return True

        # waist gravity feed-forward API (G1_TORSO_LEAN_WAIST_FF, default on with the lean)
        ff_model = None
        ff_out = False

        def configure_waist_gravity_ff(self, model):
            assert self.waist_cfg is not None, "feed-forward before the waist was configured"
            calls.append("ff_configure")
            self.ff_model = model

        def get_waist_gravity_ff(self):
            if self.ff_model is None:
                return {"configured": False}
            return {"configured": True, "gain": 0.0 if self.ff_out else 1.0, "reason": "ok",
                    "tau_nm": [0.0, -0.5, -8.0], "raw_nm": [0.0, -0.5, -8.0], "finished": self.ff_out}

        def waist_gravity_ff_ramp_out(self):
            calls.append("ff_ramp_out")
            self.ff_out = True
            return True

    class FakeIK:
        def __init__(self):
            self.last = (W_LEFT.copy(), W_RIGHT.copy())

        def forward_kinematics(self, q):
            return self.last[0].copy(), self.last[1].copy()

        def solve_ik(self, l, r, *a):
            state["ik_targets"].append((np.array(l), np.array(r)))
            self.last = (np.array(l), np.array(r))
            return np.full(14, 0.2), np.zeros(14)

        def gravity_tauff(self, q):
            return np.zeros(14)

        def reset_warm_start(self):
            pass

        def set_torso_rotation(self, R):
            state["gravity"].append(np.array(R))

    class FakeDex3:
        def __init__(self, *a, **kw):
            pass

        def activate(self):
            calls.append("hand_activate")

        def get_pressure_samples(self):
            return (None, 0.0), (None, 0.0)

        def get_pose_samples(self):
            return None, None, None

        def get_extended_samples(self):
            return None

        def open_and_deactivate(self, *a, **kw):
            calls.append("hand_open")

        def deactivate(self, *a, **kw):
            calls.append("hand_stop")

    tele = {"n": 0}

    class FakeTV:
        def __init__(self, **kw):
            self.tvuer = types.SimpleNamespace(process=None)

        def render_to_xr(self, img):
            pass

        def set_pressure_samples(self, *a):
            pass

        def get_tele_data(self):
            tele["n"] += 1
            if tele["n"] == cycles_before_q and callbacks and "q" in keys:
                callbacks[0]("q")
            hp = head_fn(tele["n"]) if head_fn is not None else head()
            return types.SimpleNamespace(
                head_pose=hp, left_wrist_pose=W_LEFT.copy(), right_wrist_pose=W_RIGHT.copy(),
                left_hand_pos=np.zeros((25, 3)), right_hand_pos=np.zeros((25, 3)),
                controller_sample_timestamp=time.monotonic(),
                head_pose_sample_timestamp=time.monotonic(), head_pose_is_fallback=False,
                left_ctrl_triggerValue=0.0, right_ctrl_triggerValue=0.0,
                left_ctrl_squeezeValue=0.0, right_ctrl_squeezeValue=0.0,
                left_ctrl_thumbstickValue=np.zeros(2), right_ctrl_thumbstickValue=np.zeros(2),
                left_hand_pinchValue=0.0, right_hand_pinchValue=0.0,
                right_ctrl_aButton=False, right_ctrl_bButton=False, left_ctrl_xButton=False,
                motion_data_ready=True)

        def close(self):
            calls.append("tv_close")

    class FakeImageClient:
        def __init__(self, **kw):
            pass

        def get_cam_config(self):
            cam = {"enable_webrtc": False, "enable_zmq": False, "image_shape": [2, 2],
                   "binocular": False, "webrtc_port": 0}
            return {"head_camera": cam, "left_wrist_camera": dict(cam), "right_wrist_camera": dict(cam)}

        def close(self):
            pass

    class FakeIPC:
        def __init__(self, on_press, get_state):
            self.on_press = on_press

        def start(self):
            callbacks.append(self.on_press)
            first = "r" if "r" in keys else "q"
            threading.Timer(0.05, lambda: self.on_press(first)).start()

        def stop(self):
            calls.append("ipc_stop")

    logger = types.SimpleNamespace(debug=lambda *a: None, info=lambda *a: calls.append(("info", a[0])),
                                   warning=lambda *a: calls.append(("warning", a[0])),
                                   error=lambda *a: calls.append(("error", a[0])))
    def loco(**kw):
        return types.SimpleNamespace(
            read_fsm_id=lambda timeout=0.3: 500, set_speed_mode=lambda m: 0, set_balance_mode=lambda m: 0,
            set_fsm_id=lambda f: 0, checked_zero=lambda: 0, start_move_sender=lambda: None,
            stop_move_sender=lambda: None, last_move_code=None, nonzero_move_codes=0,
            make_fsm_reader=lambda: (lambda: 500), Move=lambda *a, **k: 0, StopMove=lambda *a, **k: 0,
            backend="wirelesscontroller", stop_count=0, last_stop_code=None, last_stop_reason=None,
            sender=None, _was_moving=False, backend_telemetry=lambda: None)
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
        "teleop.robot_control.robot_arm": types.SimpleNamespace(
            G1_29_ArmController=FakeArm, G1_23_ArmController=FakeArm, H1_2_ArmController=FakeArm,
            H1_ArmController=FakeArm, H2_ArmController=FakeArm),
        "teleop.robot_control.robot_arm_ik": types.SimpleNamespace(
            G1_29_ArmIK=FakeIK, G1_23_ArmIK=FakeIK, H1_2_ArmIK=FakeIK, H1_ArmIK=FakeIK, H2_ArmIK=FakeIK),
        "teleop.robot_control.robot_hand_unitree": types.SimpleNamespace(Dex3_1_Controller=FakeDex3),
        "teleop.utils.episode_writer": types.SimpleNamespace(EpisodeWriter=object),
        "teleop.utils.ipc": types.SimpleNamespace(IPC_Server=FakeIPC),
        "cv2": sys.modules.get("cv2") or types.ModuleType("cv2"),          # cameras disabled in this test
        "teleop.utils.motion_switcher": types.SimpleNamespace(MotionSwitcher=object, LocoClientWrapper=loco,
                                                              is_walk_fsm=lambda fsm_id: False),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for k in [k for k in os.environ if k.startswith(("G1_TORSO_LEAN", "XR_POSE_STREAM", "G1_BALANCE", "G1_COM"))]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("XR_TELEOP_STATUS_LOG", "/nonexistent-dir-for-test/status.jsonl")
    monkeypatch.setenv("XR_TELEOP_POSE_LOG_DIR", "/nonexistent-dir-for-test")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "argv", ["teleop_hand_and_arm.py", "--motion", "--input-mode", "hand",
                                      "--camera-layout", "head", "--ipc", "--arm", "G1_29", "--ee", "dex3"])
    import teleop.utils.waist_gravity_ff as wff_module

    class FakeGravityModel:
        def __init__(self, *a, **k):
            calls.append("ff_model_load")

        def waist_tau(self, waist_q, arm_q, quat):
            return np.array([0.0, -8.0])

    monkeypatch.setattr(wff_module, "WaistGravityModel", FakeGravityModel)
    status_blocks = state.setdefault("status_torso", [])
    real_status_block = wff_module.status_block
    monkeypatch.setattr(wff_module, "status_block",
                        lambda *a, **k: status_blocks.append(real_status_block(*a, **k)) or status_blocks[-1])
    import teleop.utils.arm_graceful_shutdown as shutdown_module

    class FakeClock:
        def __init__(self):
            self.now = 100.0

        def __call__(self):
            return self.now

        def sleep(self, s):
            self.now += float(s)

    fast = FakeClock()
    real = shutdown_module.run_graceful_arm_shutdown
    monkeypatch.setattr(shutdown_module, "run_graceful_arm_shutdown",
                        lambda arm, **kw: real(arm, **{**kw, "clock": fast, "sleep": fast.sleep}))
    runpy.run_path(str(ROOT / "teleop" / "teleop_hand_and_arm.py"), run_name="__main__")
    return calls, state


def _waist_calls(calls):
    return [c for c in calls if isinstance(c, str) and c.startswith("waist")]


def _seq(calls):
    """Call sequence without logging and per-cycle commands (stable across runs)."""
    out = []
    for c in calls:
        if isinstance(c, tuple) and c[0] in ("info", "warning", "error"):
            continue
        if c == "cmd" and out and out[-1] == "cmd":
            continue
        out.append(c if not isinstance(c, tuple) else c[0])
    return out


def test_dex3_feature_off_never_touches_waist_and_keeps_calibrated_targets(monkeypatch):
    calls, state = _run_dex3(monkeypatch, {})
    assert _waist_calls(calls) == [] and state["gravity"] == []
    assert ("info", "Inclinação do tronco: DESLIGADA") in calls
    assert state["ik_targets"], "tracking never reached IK"
    for l, r in state["ik_targets"]:
        np.testing.assert_allclose(l, W_LEFT, atol=1e-12)     # stationary controller: calibrated target = FK at r
        np.testing.assert_allclose(r, W_RIGHT, atol=1e-12)
    # Dex3 order kept: arm activate/prepare before r, Dex3 activation after r, graceful release
    assert calls.index("activate") < calls.index("prepare") < calls.index("hand_activate")
    s = _seq(calls)
    assert s.index("weight") < s.index("arm_deactivate") < s.index("ipc_stop")


def test_dex3_feature_off_and_stream_off_sequence_matches_pose_stream_on(monkeypatch):
    """XR_POSE_STREAM only adds UDP packets: the actuator call sequence is the same."""
    off, _ = _run_dex3(monkeypatch, {})
    on, _ = _run_dex3(monkeypatch, {"XR_POSE_STREAM": "1", "XR_POSE_STREAM_PORT": "47999"})
    assert [c for c in _seq(off) if c != "cmd"] == [c for c in _seq(on) if c != "cmd"]


def test_dex3_feature_on_q_before_r_never_takes_waist(monkeypatch):
    calls, _ = _run_dex3(monkeypatch, {"G1_TORSO_LEAN": "1"}, keys=("q",))
    assert _waist_calls(calls) == []


def test_dex3_feature_on_pitch_roll_only_yaw_fixed_and_neutral_before_weight_ramp(monkeypatch):
    # after r the operator steps 20 cm forward AND turns the head 60 deg:
    # the waist must lean forward (pitch) but its yaw must stay at the neutral.
    calls, state = _run_dex3(
        monkeypatch, {"G1_TORSO_LEAN": "1", "G1_TORSO_LEAN_MAX_DEG": "3"}, cycles_before_q=60,
        head_fn=lambda n: head(pivot=(min(max(n - 8, 0) * 0.01, 0.20), 0.0, 1.6),
                               yaw=min(max(n - 8, 0) * 3 * DEG, 60 * DEG)))   # gradual: no jump gate
    w = _waist_calls(calls)
    assert w and w[0] == "waist_read" and "waist_configure" in w and "waist_target" in w
    assert calls.index("activate") < calls.index("waist_configure")
    targets = np.array(state["waist_targets"])
    neutral = np.array([0.0, 0.01, -0.02])
    assert np.all(targets[:, 0] == neutral[0])                                   # yaw never follows the headset
    assert np.max(np.abs(targets[:, 1:] - neutral[1:])) <= 3 * DEG + 1e-9        # <= max deg
    assert np.max(targets[:, 2] - neutral[2]) > 0.2 * DEG                         # it did lean forward
    i_neutral = calls.index("waist_neutral")
    i_w0 = next(i for i, c in enumerate(calls) if isinstance(c, tuple) and c[0] == "weight")
    assert calls.index("hand_activate") < i_neutral < i_w0 < calls.index("arm_deactivate")
    assert "waist_target" not in calls[i_neutral:]
    assert state["gravity"] and not np.allclose(state["gravity"][-2], np.eye(3))
    np.testing.assert_allclose(state["gravity"][-1], np.eye(3))
    # the IK got torso-frame targets once leaning
    assert not np.allclose(state["ik_targets"][-1][0], W_LEFT)


def test_nonfollowing_measured_waist_warns_degraded_but_keeps_limited_command_and_measured_compensation(monkeypatch):
    calls, state = _run_dex3(
        monkeypatch, {"G1_TORSO_LEAN": "1", "G1_TORSO_LEAN_MAX_DEG": "3"},
        cycles_before_q=75, waist_follows=False,
        # sub-mm jitter = a live headset (identical frames would be "stale" and,
        # after decay_after_s, the target decays; with no low-pass/second ramp
        # that decay is no longer hidden by lag)
        head_fn=lambda n: head(pivot=(min(max(n - 8, 0) * 0.01, 0.20) + 1e-5 * (n % 2), 0.0, 1.6)))

    warnings = [c[1] for c in calls if isinstance(c, tuple) and c[0] == "warning"]
    assert any("cintura não acompanhou" in message for message in warnings)
    # Measured waist remained neutral, so compensation/gravity must remain
    # identity even while the commanded target briefly leaned.
    assert state["gravity"] and all(np.allclose(R, np.eye(3)) for R in state["gravity"])
    assert state["ik_targets"] and all(np.allclose(pair[0], W_LEFT) for pair in state["ik_targets"])
    targets = np.asarray(state["waist_targets"])
    neutral = np.array([0.0, 0.01, -0.02])
    assert np.max(np.abs(targets[:, 1:] - neutral[1:])) > 0.2 * DEG
    assert np.max(np.abs(targets[:, 1:] - neutral[1:])) <= 3 * DEG + 1e-9
    assert np.linalg.norm(targets[-1, 1:] - neutral[1:]) > 0.2 * DEG


def test_dex3_feature_rejects_max_above_20_and_stays_off(monkeypatch):
    calls, _ = _run_dex3(monkeypatch, {"G1_TORSO_LEAN": "1", "G1_TORSO_LEAN_MAX_DEG": "21"})
    assert _waist_calls(calls) == []
    assert any(isinstance(c, tuple) and c[0] == "error" and "rejeitada" in c[1] for c in calls)


def test_waist_yaw_has_no_path_from_head_yaw():
    """Complements tests/test_quest_controls.py::test_headset_waist_yaw_follow_has_no_cli_or_robot_actuator_path:
    the only waist writer is torso_lean, whose command keeps yaw == neutral."""
    cfg = tl.TorsoLeanConfig(max_deg=10.0)
    cmd = tl.WaistLeanCommand(cfg, [0.3, 0.0, 0.0], [0.3, 0.0, 0.0], 0.0)
    t = 0.0
    for _ in range(200):
        t += 0.033
        q = cmd.step((10 * DEG, -10 * DEG), t)
        assert q[0] == 0.3
    assert cmd.lower[0] == cmd.upper[0] == 0.3          # yaw box has zero width


# ------------------------------------------------- waist gravity feed-forward
def _ff_calls(calls):
    return [c for c in calls if isinstance(c, str) and c.startswith("ff_")]


def test_waist_ff_off_when_torso_lean_off(monkeypatch):
    calls, state = _run_dex3(monkeypatch, {"G1_TORSO_LEAN_WAIST_FF": "1"})
    assert _ff_calls(calls) == []
    assert all(b == {"configured": False, "enabled": False, "status": "off"} for b in state["status_torso"])


def test_waist_ff_default_on_after_r_ramps_out_before_weight_release(monkeypatch):
    calls, state = _run_dex3(
        monkeypatch, {"G1_TORSO_LEAN": "1", "G1_TORSO_LEAN_MAX_DEG": "3"}, cycles_before_q=40,
        head_fn=lambda n: head(pivot=(min(max(n - 8, 0) * 0.01, 0.20), 0.0, 1.6)))
    assert _ff_calls(calls) == ["ff_model_load", "ff_configure", "ff_ramp_out"]
    assert calls.index("waist_configure") < calls.index("ff_configure")          # only after the waist is ours (r)
    assert calls.index("hand_activate") < calls.index("ff_configure")
    i_w0 = next(i for i, c in enumerate(calls) if isinstance(c, tuple) and c[0] == "weight")
    assert calls.index("waist_neutral") < calls.index("ff_ramp_out") < i_w0 < calls.index("arm_deactivate")
    tracked = [b for b in state["status_torso"] if b.get("configured") and "waist_ff" in b]
    assert tracked and tracked[-1]["waist_ff"]["tau_nm"] == {"yaw": 0.0, "roll": -0.5, "pitch": -8.0}
    assert any(isinstance(c, tuple) and c[0] == "info" and "feed-forward de gravidade da cintura ativo" in c[1]
               for c in calls)


def test_waist_ff_kill_switch_keeps_lean_and_never_configures_ff(monkeypatch):
    on, _ = _run_dex3(monkeypatch, {"G1_TORSO_LEAN": "1", "G1_TORSO_LEAN_MAX_DEG": "3"}, cycles_before_q=40)
    off, state = _run_dex3(monkeypatch, {"G1_TORSO_LEAN": "1", "G1_TORSO_LEAN_MAX_DEG": "3",
                                         "G1_TORSO_LEAN_WAIST_FF": "0"}, cycles_before_q=40)
    assert _ff_calls(off) == []
    # lean behaviour identical apart from the feed-forward calls
    strip = lambda cs: [c for c in _seq(cs) if c != "cmd" and not str(c).startswith("ff_")]
    assert strip(on) == strip(off)
    tracked = [b for b in state["status_torso"] if b.get("configured")]
    assert tracked and all(b["waist_ff"] == {"enabled": False, "reason": "kill_switch"} for b in tracked)


def test_waist_ff_invalid_kill_switch_value_fails_closed(monkeypatch):
    calls, _ = _run_dex3(monkeypatch, {"G1_TORSO_LEAN": "1", "G1_TORSO_LEAN_MAX_DEG": "3",
                                       "G1_TORSO_LEAN_WAIST_FF": "yes"}, cycles_before_q=30)
    assert _ff_calls(calls) == []
    assert "waist_configure" in calls                                              # the lean itself still runs
    assert any(isinstance(c, tuple) and c[0] == "error" and "G1_TORSO_LEAN_WAIST_FF" in c[1] for c in calls)
