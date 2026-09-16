import math
import sys
from pathlib import Path
import importlib
import runpy
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teleop.utils.quest_controls import joystick_to_locomotion


@pytest.mark.parametrize(
    ("left_xy", "right_xy", "expected"),
    [
        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 1.0), (0.0, 0.0), (-1.0, 0.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0), (0.0, -1.0, 0.0)),
        ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0, -1.0)),
        ((-1.0, 1.0), (0.0, 0.0), (-1.0, 1.0, 0.0)),
        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 0.0)),
    ],
)
def test_joystick_to_locomotion_maps_normalized_sticks(left_xy, right_xy, expected):
    assert joystick_to_locomotion(left_xy, right_xy) == expected


def test_joystick_to_locomotion_clamps_finite_values():
    assert joystick_to_locomotion((2.0, -2.0), (1.5, -1.5)) == (1.0, -1.0, -1.0)


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


@pytest.mark.parametrize(
    ("left_xy", "right_xy", "expected"),
    [
        ((math.nan, 0.25), (0.5, 0.0), (-0.25, 0.0, -0.5)),
        ((0.25, math.inf), (0.5, 0.0), (0.0, -0.25, -0.5)),
        ((0.25, -0.5), (-math.inf, 0.0), (0.5, -0.25, 0.0)),
    ],
)
def test_each_nonfinite_stick_axis_has_exact_zero_locomotion_contribution(left_xy, right_xy, expected):
    assert joystick_to_locomotion(left_xy, right_xy) == expected


def test_loco_wrapper_passes_exact_native_tuple_to_client(monkeypatch):
    calls = []

    class FakeLocoClient:
        def SetTimeout(self, timeout):
            pass

        def Init(self):
            pass

        def Move(self, vx, vy, vyaw, continous_move):
            calls.append((vx, vy, vyaw, continous_move))

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

    wrapper.Move(*locomotion)

    assert calls == [(*locomotion, False)]


def test_ordinary_controller_input_cannot_damp_or_change_lifecycle():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py").read_text()

    assert ".Damp(" not in source
    assert "right_ctrl_aButton" not in source
    assert "left_ctrl_bButton" not in source
    assert "right_ctrl_bButton" not in source


def test_locomotion_is_not_gated_to_controller_mode_and_uses_freshness():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py").read_text()

    assert "if args.motion:" in source
    assert "controller_sample_is_fresh" in source
    assert "locomotion = (0.0, 0.0, 0.0)" in source


def test_hand_mode_teledata_carries_controller_sticks_and_sample_timestamp():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "televuer" / "src" / "televuer" / "tv_wrapper.py").read_text()

    hand_return = source.split("if self.use_hand_tracking:", 1)[1].split("# controller tracking", 1)[0]
    assert "left_ctrl_thumbstickValue=self.tvuer.left_ctrl_thumbstickValue" in hand_return
    assert "right_ctrl_thumbstickValue=self.tvuer.right_ctrl_thumbstickValue" in hand_return
    assert "controller_sample_timestamp=self.tvuer.controller_sample_timestamp" in hand_return


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
    module.on_press("x")
    assert (module.START, module.STOP) == (False, False)
    module.on_press("r")
    assert (module.START, module.STOP) == (True, False)
    module.on_press("q")
    assert (module.START, module.STOP) == (False, True)


def test_hand_motion_initializes_locomotion_before_first_move(monkeypatch):
    moves = []

    class StopAfterMove(Exception):
        pass

    class FakeLocoWrapper:
        def __init__(self):
            moves.append("initialized")

        def Move(self, *locomotion):
            moves.append(locomotion)
            raise StopAfterMove

    class FakeArmController:
        def __init__(self, **kwargs):
            pass

        def speed_gradual_max(self):
            pass

        def ctrl_dual_arm_go_home(self):
            pass

    class FakeArmIK:
        pass

    class FakeTeleVuerWrapper:
        def __init__(self, **kwargs):
            pass

        def get_tele_data(self):
            return types.SimpleNamespace(
                head_pose=__import__("numpy").eye(4),
                controller_sample_timestamp=0.0,
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
            self.on_press("r")

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
        "teleop.utils.motion_switcher": types.SimpleNamespace(MotionSwitcher=object, LocoClientWrapper=FakeLocoWrapper),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", [
        "teleop_hand_and_arm.py", "--motion", "--input-mode", "hand", "--camera-layout", "head", "--ipc"
    ])
    monkeypatch.setattr("builtins.exit", lambda code: None)

    script = Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py"
    runpy.run_path(str(script), run_name="__main__")

    assert moves == ["initialized", (0.0, 0.0, 0.0)]
