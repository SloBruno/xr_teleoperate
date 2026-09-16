import math
import sys
from pathlib import Path
import importlib
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
