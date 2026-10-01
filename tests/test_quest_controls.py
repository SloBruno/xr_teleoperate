import ast
import math
import sys
from pathlib import Path
import importlib
import runpy
import time
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teleop.utils import quest_controls

# Curve/mapping tests use an explicit 0.5 m/s cap (the historical default) so the
# stick math stays covered independent of the operator default (now 0.3).
_CURVE_CAP = 0.5


def joystick_to_locomotion(left_xy, right_xy, walk_cap=_CURVE_CAP, turn_cap=quest_controls.MIN_OPERATOR_TURN_RATE_RADPS):
    return quest_controls.joystick_to_locomotion(left_xy, right_xy, walk_cap, turn_cap)


def test_joystick_locomotion_uses_minimum_reviewed_operator_speed_caps():
    assert quest_controls.MIN_OPERATOR_WALK_SPEED_MPS == 0.3
    assert quest_controls.MIN_OPERATOR_TURN_RATE_RADPS == 0.3

    assert joystick_to_locomotion((0.0, 1.0), (1.0, 0.0)) == (-0.5, 0.0, -0.3)
    assert joystick_to_locomotion((-1.0, -1.0), (-1.0, 0.0)) == (0.5, 0.5, 0.3)


def test_full_forward_stick_uses_minimum_physical_walk_cap():
    # Operator default = 0.3 m/s (lowered from 0.5 after the 0.5 m/s session).
    assert quest_controls.joystick_to_locomotion((0.0, -1.0), (0.0, 0.0)) == (0.3, 0.0, 0.0)


def test_locomotion_dispatch_preserves_enable_and_stale_release_to_zero_gates():
    class FakeLocoWrapper:
        def __init__(self):
            self.calls = []

        def Move(self, *command):
            self.calls.append(command)

    disabled = FakeLocoWrapper()
    assert quest_controls.dispatch_joystick_locomotion(
        disabled, motion_enabled=False, controller_is_fresh=True,
        left_xy=(1.0, 1.0), right_xy=(1.0, 0.0),
    ) == (0.0, 0.0, 0.0)
    assert disabled.calls == []

    stale = FakeLocoWrapper()
    assert quest_controls.dispatch_joystick_locomotion(
        stale, motion_enabled=True, controller_is_fresh=False,
        left_xy=(1.0, 1.0), right_xy=(1.0, 0.0),
    ) == (0.0, 0.0, 0.0)
    assert stale.calls == [(0.0, 0.0, 0.0)]

    fresh = FakeLocoWrapper()
    assert quest_controls.dispatch_joystick_locomotion(
        fresh, motion_enabled=True, controller_is_fresh=True,
        left_xy=(-1.0, -1.0), right_xy=(-1.0, 0.0),
    ) == (0.3, 0.3, 0.3)
    assert fresh.calls == [(0.3, 0.3, 0.3)]


@pytest.mark.parametrize(
    ("left_xy", "right_xy", "expected"),
    [
        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 1.0), (0.0, 0.0), (-0.5, 0.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0), (0.0, -0.5, 0.0)),
        ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0, -0.3)),
        ((-1.0, 1.0), (0.0, 0.0), (-0.5, 0.5, 0.0)),
        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 0.0)),
    ],
)
def test_joystick_to_locomotion_maps_normalized_sticks(left_xy, right_xy, expected):
    assert joystick_to_locomotion(left_xy, right_xy) == expected


def test_joystick_small_movements_use_deadzone_and_precision_curve():
    assert joystick_to_locomotion((0.10, -0.10), (0.10, 0.0)) == (0.0, 0.0, 0.0)

    shaped_half = ((0.50 - 0.12) / (1.0 - 0.12)) ** 3
    forward, lateral, yaw = joystick_to_locomotion((0.50, -0.50), (0.50, 0.0))
    assert forward == pytest.approx(0.5 * shaped_half)
    assert lateral == pytest.approx(-0.5 * shaped_half)
    assert yaw == pytest.approx(-0.3 * shaped_half)

    assert joystick_to_locomotion((1.0, -1.0), (1.0, 0.0)) == (0.5, -0.5, -0.3)


def test_joystick_to_locomotion_clamps_finite_values():
    assert joystick_to_locomotion((2.0, -2.0), (1.5, -1.5)) == (0.5, -0.5, -0.3)


@pytest.mark.parametrize(
    ("left_xy", "right_xy"),
    [
        ((math.nan, 0.0), (0.0, 0.0)),
        ((math.inf, 0.0), (0.0, 0.0)),
        ((-math.inf, 0.0), (0.0, 0.0)),
    ],
)
def test_joystick_to_locomotion_handles_nonfinite_values(left_xy, right_xy):
    result = joystick_to_locomotion(left_xy, right_xy)
    assert all(math.isfinite(value) for value in result)
    assert all(-1.0 <= value <= 1.0 for value in result)


def _expected_precision(value):
    if not math.isfinite(value) or abs(value) <= 0.12:
        return 0.0
    magnitude = (min(abs(value), 1.0) - 0.12) / (1.0 - 0.12)
    return math.copysign(magnitude ** 3, value)


@pytest.mark.parametrize(
    ("left_xy", "right_xy", "expected"),
    [
        ((math.nan, 0.25), (0.5, 0.0), (-0.5 * _expected_precision(0.25), 0.0, -0.3 * _expected_precision(0.5))),
        ((0.25, math.inf), (0.5, 0.0), (0.0, -0.5 * _expected_precision(0.25), -0.3 * _expected_precision(0.5))),
        ((0.25, -0.5), (-math.inf, 0.0), (0.5 * _expected_precision(0.5), -0.5 * _expected_precision(0.25), 0.0)),
    ],
)
def test_each_nonfinite_stick_axis_has_exact_zero_locomotion_contribution(left_xy, right_xy, expected):
    assert joystick_to_locomotion(left_xy, right_xy) == pytest.approx(expected)


def test_loco_wrapper_passes_exact_native_tuple_to_client(monkeypatch):
    calls = []

    class FakeLocoClient:
        def SetTimeout(self, timeout):
            pass

        def Init(self):
            pass

        def SetVelocity(self, vx, vy, vyaw, duration):
            calls.append((vx, vy, vyaw, duration))
            return 0

    loco_module = types.ModuleType("unitree_sdk2py.g1.loco.g1_loco_client")
    loco_module.LocoClient = FakeLocoClient
    monkeypatch.setitem(sys.modules, "unitree_sdk2py", types.ModuleType("unitree_sdk2py"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core", types.ModuleType("unitree_sdk2py.core"))
    channel_module = types.ModuleType("unitree_sdk2py.core.channel")
    channel_module.ChannelFactoryInitialize = object
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", channel_module)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.g1", types.ModuleType("unitree_sdk2py.g1"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.g1.loco.g1_loco_client", loco_module)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.g1.loco", types.ModuleType("unitree_sdk2py.g1.loco"))
    motion_module = types.ModuleType("unitree_sdk2py.comm.motion_switcher.motion_switcher_client")
    motion_module.MotionSwitcherClient = object
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.comm", types.ModuleType("unitree_sdk2py.comm"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.comm.motion_switcher", types.ModuleType("unitree_sdk2py.comm.motion_switcher"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.comm.motion_switcher.motion_switcher_client", motion_module)

    switcher_module = importlib.import_module("teleop.utils.motion_switcher")
    switcher_module = importlib.reload(switcher_module)
    wrapper = switcher_module.LocoClientWrapper()
    locomotion = joystick_to_locomotion((0.25, -0.75), (-0.5, 0.0))

    assert wrapper.Move(*locomotion) == 0

    # duration=1.0 is the SDK's Move(continous_move=False) contract, but the
    # RPC status is now observable instead of discarded.
    assert calls == [(*locomotion, 1.0)]
    assert wrapper.last_move_code == 0


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


def test_loco_wrapper_surfaces_nonzero_rpc_status_without_raising(monkeypatch):
    class FakeLocoClient:
        def SetTimeout(self, timeout): pass
        def Init(self): pass
        def SetVelocity(self, vx, vy, vyaw, duration): return 3104

    wrapper = _load_switcher(monkeypatch, FakeLocoClient).LocoClientWrapper()

    assert wrapper.Move(0.1, 0.0, 0.0) == 3104
    assert wrapper.last_move_code == 3104
    assert wrapper.nonzero_move_codes == 1


def test_loco_wrapper_reads_fsm_id_and_flags_modes_that_cannot_walk(monkeypatch):
    class FakeLocoClient:
        fsm = 500
        def SetTimeout(self, timeout): pass
        def Init(self): pass
        def _Call(self, api_id, parameter):
            assert api_id == 7001
            return 0, '{"data": %d}' % FakeLocoClient.fsm

    mod = _load_switcher(monkeypatch, FakeLocoClient)
    wrapper = mod.LocoClientWrapper()
    assert wrapper.read_fsm_id() == 500
    assert mod.is_walk_fsm(500) and mod.is_walk_fsm(501)
    assert not mod.is_walk_fsm(1) and not mod.is_walk_fsm(801) and not mod.is_walk_fsm(None)


def test_controller_lifecycle_edges_use_the_right_controller_and_no_damping():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py").read_text()

    assert ".Damp(" not in source
    assert "right_ctrl_aButton" in source
    assert "right_ctrl_bButton" in source
    assert "left_ctrl_bButton" not in source


def test_controller_lifecycle_poll_uses_fresh_rising_edges_and_shared_requests(monkeypatch):
    stubs = {
        "logging_mp": types.ModuleType("logging_mp"),
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": types.ModuleType("unitree_sdk2py.core.channel"),
        "televuer": types.ModuleType("televuer"),
        "teleimager": types.ModuleType("teleimager"),
        "teleimager.image_client": types.ModuleType("teleimager.image_client"),
        "sshkeyboard": types.ModuleType("sshkeyboard"),
        "teleop.robot_control.robot_arm": types.ModuleType("teleop.robot_control.robot_arm"),
        "teleop.robot_control.robot_arm_ik": types.ModuleType("teleop.robot_control.robot_arm_ik"),
        "teleop.utils.episode_writer": types.ModuleType("teleop.utils.episode_writer"),
        "teleop.utils.ipc": types.ModuleType("teleop.utils.ipc"),
        "teleop.utils.motion_switcher": types.ModuleType("teleop.utils.motion_switcher"),
        "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.std_msgs": types.ModuleType("unitree_sdk2py.idl.std_msgs"),
        "unitree_sdk2py.idl.std_msgs.msg": types.ModuleType("unitree_sdk2py.idl.std_msgs.msg"),
        "unitree_sdk2py.idl.std_msgs.msg.dds_": types.ModuleType("unitree_sdk2py.idl.std_msgs.msg.dds_"),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
        module.__getattr__ = lambda name: object
    stubs["logging_mp"].basicConfig = lambda **kwargs: None
    stubs["logging_mp"].getLogger = lambda name: types.SimpleNamespace(warning=lambda message: None)
    stubs["logging_mp"].INFO = 20
    stubs["unitree_sdk2py.core.channel"].ChannelFactoryInitialize = object
    stubs["unitree_sdk2py.core.channel"].ChannelPublisher = object
    stubs["unitree_sdk2py.idl.std_msgs.msg.dds_"].String_ = object
    stubs["sshkeyboard"].listen_keyboard = object
    stubs["sshkeyboard"].stop_listening = object

    module = importlib.reload(importlib.import_module("teleop.teleop_hand_and_arm"))
    module.START = False
    module.STOP = False
    module.PREPARATION_COMPLETE = False

    sample = types.SimpleNamespace(
        right_ctrl_aButton=True,
        right_ctrl_bButton=False,
        controller_sample_timestamp=time.monotonic(),
    )
    edge_state = module.poll_controller_lifecycle(sample, False, False)
    assert edge_state == (True, False)
    assert (module.START, module.STOP) == (False, False)

    module.PREPARATION_COMPLETE = True
    module.poll_controller_lifecycle(sample, True, False)
    assert (module.START, module.STOP) == (False, False)
    released = types.SimpleNamespace(
        right_ctrl_aButton=False,
        right_ctrl_bButton=False,
        controller_sample_timestamp=time.monotonic(),
    )
    module.poll_controller_lifecycle(released, True, False)
    module.poll_controller_lifecycle(sample, False, False)
    assert module.START is True

    stop_sample = types.SimpleNamespace(
        right_ctrl_aButton=True,
        right_ctrl_bButton=True,
        controller_sample_timestamp=time.monotonic(),
    )
    module.poll_controller_lifecycle(stop_sample, True, False)
    assert (module.START, module.STOP) == (False, True)

    module.START = False
    module.STOP = False
    module.PREPARATION_COMPLETE = True
    stale = types.SimpleNamespace(
        right_ctrl_aButton=True,
        right_ctrl_bButton=True,
        controller_sample_timestamp=0.0,
    )
    module.poll_controller_lifecycle(stale, False, False)
    assert (module.START, module.STOP) == (False, False)


def test_repeat_start_while_tracking_preserves_calibration(monkeypatch):
    class Calibration:
        def __init__(self):
            self.calibrated = True
            self.reset_count = 0

        def reset_for_start_request(self, _timestamp):
            self.calibrated = False
            self.reset_count += 1

    source_path = Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py"
    tree = ast.parse(source_path.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_request_start_locked"
    )
    calibration = Calibration()
    namespace = {
        "time": time,
        "logger_mp": types.SimpleNamespace(warning=lambda *_args: None),
        "PREPARATION_COMPLETE": True,
        "START": True,
        "STOP": False,
        "ARM_REQUEST_TIMESTAMP": 1.0,
        "arm_calibration": calibration,
        "LIFECYCLE_EVENTS": [],
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)

    assert namespace["_request_start_locked"]()
    assert namespace["START"] is True
    assert calibration.calibrated is True
    assert calibration.reset_count == 0


def test_locomotion_is_not_gated_to_controller_mode_and_uses_freshness():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py").read_text()

    assert "if args.motion:" in source
    assert "controller_sample_is_fresh" in source
    assert "dispatch_joystick_locomotion(" in source


def test_hand_mode_teledata_carries_controller_sticks_and_sample_timestamp():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "televuer" / "src" / "televuer" / "tv_wrapper.py").read_text()

    hand_return = source.split("if self.use_hand_tracking:", 1)[1].split("# controller tracking", 1)[0]
    assert "left_ctrl_thumbstickValue=self.tvuer.left_ctrl_thumbstickValue" in hand_return
    assert "right_ctrl_thumbstickValue=self.tvuer.right_ctrl_thumbstickValue" in hand_return
    assert "controller_sample_timestamp=controller_sample_timestamp" in hand_return


def test_hand_mode_teledata_carries_right_controller_lifecycle_buttons():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "televuer" / "src" / "televuer" / "tv_wrapper.py").read_text()

    hand_return = source.split("if self.use_hand_tracking:", 1)[1].split("# controller tracking", 1)[0]
    assert "right_ctrl_aButton=self.tvuer.right_ctrl_aButton" in hand_return
    assert "right_ctrl_bButton=self.tvuer.right_ctrl_bButton" in hand_return


def test_headset_waist_yaw_follow_has_no_cli_or_robot_actuator_path():
    root = Path(__file__).resolve().parents[1]
    teleop_source = (root / "teleop" / "teleop_hand_and_arm.py").read_text()
    arm_source = (root / "teleop" / "robot_control" / "robot_arm.py").read_text()

    for forbidden in ("waist-yaw-follow", "waist_yaw_follow", "head_yaw_from_pose", "wrapped_angle_difference", "ctrl_waist_yaw"):
        assert forbidden not in teleop_source
    for forbidden in ("ctrl_waist_yaw", "waist_yaw_target", "waist_yaw_command", "waist_yaw_velocity_limit", "get_current_waist_yaw"):
        assert forbidden not in arm_source


def test_terminal_keys_are_the_only_lifecycle_authority(monkeypatch):
    stubs = {
        "logging_mp": types.ModuleType("logging_mp"),
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": types.ModuleType("unitree_sdk2py.core.channel"),
        "televuer": types.ModuleType("televuer"),
        "teleimager": types.ModuleType("teleimager"),
        "teleimager.image_client": types.ModuleType("teleimager.image_client"),
        "sshkeyboard": types.ModuleType("sshkeyboard"),
        "teleop.robot_control.robot_arm": types.ModuleType("teleop.robot_control.robot_arm"),
        "teleop.robot_control.robot_arm_ik": types.ModuleType("teleop.robot_control.robot_arm_ik"),
        "teleop.utils.episode_writer": types.ModuleType("teleop.utils.episode_writer"),
        "teleop.utils.ipc": types.ModuleType("teleop.utils.ipc"),
        "teleop.utils.motion_switcher": types.ModuleType("teleop.utils.motion_switcher"),
        "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.std_msgs": types.ModuleType("unitree_sdk2py.idl.std_msgs"),
        "unitree_sdk2py.idl.std_msgs.msg": types.ModuleType("unitree_sdk2py.idl.std_msgs.msg"),
        "unitree_sdk2py.idl.std_msgs.msg.dds_": types.ModuleType("unitree_sdk2py.idl.std_msgs.msg.dds_"),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
        module.__getattr__ = lambda name: object
    stubs["logging_mp"].basicConfig = lambda **kwargs: None
    stubs["logging_mp"].getLogger = lambda name: types.SimpleNamespace(warning=lambda message: None)
    stubs["logging_mp"].INFO = 20
    stubs["unitree_sdk2py.core.channel"].ChannelFactoryInitialize = object
    stubs["unitree_sdk2py.core.channel"].ChannelPublisher = object
    stubs["unitree_sdk2py.idl.std_msgs.msg.dds_"].String_ = object
    stubs["sshkeyboard"].listen_keyboard = object
    stubs["sshkeyboard"].stop_listening = object

    module = importlib.reload(importlib.import_module("teleop.teleop_hand_and_arm"))
    module.START = False
    module.STOP = False
    module.PREPARATION_COMPLETE = False
    module.on_press("x")
    assert (module.START, module.STOP) == (False, False)
    module.on_press("r")
    assert (module.START, module.STOP) == (False, False)
    module.PREPARATION_COMPLETE = True
    module.on_press("r")
    assert (module.START, module.STOP) == (True, False)
    module.on_press("q")
    assert (module.START, module.STOP) == (False, True)


def test_hand_motion_initializes_locomotion_before_first_move(monkeypatch):
    moves = []
    arm_request_callbacks = []

    class StopAfterMove(Exception):
        pass

    class FakeLocoWrapper:
        def __init__(self, **kw):
            moves.append("initialized")

        def read_fsm_id(self, timeout=0.3):
            return 500

        def set_speed_mode(self, mode):
            return 0

        def set_balance_mode(self, mode):
            return 0

        def checked_zero(self):
            return 0

        def start_move_sender(self):
            pass

        def Move(self, *locomotion):
            moves.append(locomotion)
            raise StopAfterMove

    class FakeArmController:
        def __init__(self, **kwargs):
            pass

        def activate(self):
            pass

        def deactivate(self):
            pass

        def speed_gradual_max(self):
            pass

        def ctrl_dual_arm_go_home(self, release_motion_authority=False):
            return True

    class FakeArmIK:
        pass

    class FakeTeleVuerWrapper:
        def __init__(self, **kwargs):
            pass

        def get_tele_data(self):
            if arm_request_callbacks:
                arm_request_callbacks.pop()("r")
            return types.SimpleNamespace(
                head_pose=__import__("numpy").eye(4),
                controller_sample_timestamp=time.monotonic(),
                left_ctrl_thumbstickValue=(0.0, 0.0),
                right_ctrl_thumbstickValue=(0.0, 0.0),
                motion_data_ready=False,
            )

        def close(self):
            pass

    class FakeImageClient:
        def __init__(self, **kwargs):
            pass

        def get_cam_config(self):
            camera = {
                "enable_webrtc": False,
                "enable_zmq": False,
                "image_shape": [2, 2],
                "binocular": False,
                "webrtc_port": 0,
            }
            return {"head_camera": camera, "left_wrist_camera": camera, "right_wrist_camera": camera}

        def close(self):
            pass

    class FakeIPCServer:
        def __init__(self, on_press, get_state):
            self.on_press = on_press

        def start(self):
            arm_request_callbacks.append(self.on_press)

        def stop(self):
            pass

    logger = types.SimpleNamespace(debug=lambda *args: None, info=lambda *args: None,
                                   warning=lambda *args: None, error=lambda *args: None)
    modules = {
        "logging_mp": types.SimpleNamespace(basicConfig=lambda **kwargs: None, getLogger=lambda name: logger, INFO=20),
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": types.SimpleNamespace(ChannelFactoryInitialize=lambda *args, **kwargs: None, ChannelPublisher=object),
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
        "teleop.utils.motion_switcher": types.SimpleNamespace(MotionSwitcher=object, LocoClientWrapper=FakeLocoWrapper, is_walk_fsm=lambda fsm_id: fsm_id in (500, 501)),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", [
        "teleop_hand_and_arm.py", "--motion", "--input-mode", "hand", "--camera-layout", "head", "--ipc",
        "--arm", "G1_23"
    ])
    monkeypatch.setattr("builtins.exit", lambda code: None)

    script = Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py"
    runpy.run_path(str(script), run_name="__main__")

    assert moves == ["initialized", (0.0, 0.0, 0.0)]


def test_stick_snapshot_records_raw_before_clamp_deadzone_and_curve():
    snap = quest_controls.stick_snapshot((0.05, -2.0), (0.5, 0.3))
    assert snap["raw_left_xy"] == [0.05, -2.0]  # not clamped
    assert snap["raw_right_xy"] == [0.5, 0.3]
    assert snap["shaped_left_xy"] == [0.0, -1.0]
    assert snap["command"] == list(quest_controls.joystick_to_locomotion((0.05, -2.0), (0.5, 0.3)))
    assert snap["command"][0] == 0.3  # sign unchanged: stick y<0 -> vx>0


def test_stick_snapshot_nonfinite_raw_is_null_and_never_raises():
    snap = quest_controls.stick_snapshot((float("nan"), float("inf")), (float("-inf"), 0.0))
    assert snap["raw_left_xy"] == [None, None]
    assert snap["raw_right_xy"] == [None, 0.0]
    assert snap["command"] == [0.0, 0.0, 0.0]
    assert quest_controls.stick_snapshot((0, 0), None)["raw_right_xy"] is None
    assert quest_controls.stick_snapshot(object(), 5)["raw_left_xy"] is None


def test_stick_snapshot_includes_wrapper_move_code_when_available():
    wrapper = types.SimpleNamespace(last_move_code=3104, nonzero_move_codes=7)
    snap = quest_controls.stick_snapshot((0, -1), (0, 0), loco_wrapper=wrapper)
    assert snap["last_move_code"] == 3104 and snap["nonzero_move_codes"] == 7
    assert quest_controls.stick_snapshot((0, -1), (0, 0))["last_move_code"] is None
