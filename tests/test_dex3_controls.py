import importlib
import sys
import types

import numpy as np


sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from teleop.utils.dex3_controls import trigger_to_dex3_targets


def test_released_trigger_returns_open_pose_for_all_seven_slots():
    open_pose = np.arange(7, dtype=float)
    closed_pose = open_pose + 10.0

    result = trigger_to_dex3_targets(0.0, open_pose, closed_pose)

    np.testing.assert_allclose(result, open_pose)


def test_midpoint_trigger_interpolates_all_seven_slots():
    open_pose = np.zeros(7)
    closed_pose = np.arange(1, 8, dtype=float)

    result = trigger_to_dex3_targets(0.5, open_pose, closed_pose)

    np.testing.assert_allclose(result, closed_pose / 2.0)


def test_full_trigger_returns_closed_pose_for_all_seven_slots():
    open_pose = np.arange(7, dtype=float)
    closed_pose = open_pose - 3.0

    result = trigger_to_dex3_targets(1.0, open_pose, closed_pose)

    np.testing.assert_allclose(result, closed_pose)


def test_trigger_is_clamped_and_nonfinite_values_fail_open():
    open_pose = np.arange(7, dtype=float)
    closed_pose = open_pose + 10.0

    np.testing.assert_allclose(
        trigger_to_dex3_targets(2.0, open_pose, closed_pose), closed_pose
    )
    np.testing.assert_allclose(
        trigger_to_dex3_targets(-1.0, open_pose, closed_pose), open_pose
    )
    np.testing.assert_allclose(
        trigger_to_dex3_targets(np.nan, open_pose, closed_pose), open_pose
    )


def test_dex3_publisher_receives_side_specific_seven_slot_commands(monkeypatch):
    published = {"left": [], "right": []}

    class FakePublisher:
        def __init__(self, side):
            self.side = side

        def Write(self, message):
            published[self.side].append(
                np.array([motor.q for motor in message.motor_cmd], dtype=float)
            )

    class Motor:
        def __init__(self):
            self.q = 0.0

    class Message:
        def __init__(self):
            self.motor_cmd = [Motor() for _ in range(7)]

    # The controller module has SDK imports at module load; provide only the
    # small surface needed to exercise its fake-DDS publish path.
    sdk_channel = types.ModuleType("unitree_sdk2py.core.channel")
    sdk_channel.ChannelPublisher = object
    sdk_channel.ChannelSubscriber = object
    sdk_channel.ChannelFactoryInitialize = object
    sdk_hand = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    sdk_hand.HandCmd_ = object
    sdk_hand.HandState_ = object
    sdk_default = types.ModuleType("unitree_sdk2py.idl.default")
    sdk_default.unitree_hg_msg_dds__HandCmd_ = Message
    for name, module in {
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": sdk_channel,
        "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.unitree_hg": types.ModuleType("unitree_sdk2py.idl.unitree_hg"),
        "unitree_sdk2py.idl.unitree_hg.msg": types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg"),
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": sdk_hand,
        "unitree_sdk2py.idl.default": sdk_default,
        "unitree_sdk2py.idl.unitree_go": types.ModuleType("unitree_sdk2py.idl.unitree_go"),
        "unitree_sdk2py.idl.unitree_go.msg": types.ModuleType("unitree_sdk2py.idl.unitree_go.msg"),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": types.ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_"),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["unitree_sdk2py.idl.unitree_go.msg.dds_"].MotorCmds_ = object
    sys.modules["unitree_sdk2py.idl.unitree_go.msg.dds_"].MotorStates_ = object
    sdk_default.unitree_go_msg_dds__MotorCmd_ = object

    retargeting = types.ModuleType("teleop.robot_control.hand_retargeting")
    retargeting.HandRetargeting = object
    retargeting.HandType = object
    monkeypatch.setitem(sys.modules, "teleop.robot_control.hand_retargeting", retargeting)
    logging_mp = types.ModuleType("logging_mp")
    logging_mp.getLogger = lambda name: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "logging_mp", logging_mp)

    module = importlib.import_module("teleop.robot_control.robot_hand_unitree")
    controller = module.Dex3_1_Controller.__new__(module.Dex3_1_Controller)
    controller.left_msg = Message()
    controller.right_msg = Message()
    controller.LeftHandCmb_publisher = FakePublisher("left")
    controller.RightHandCmb_publisher = FakePublisher("right")

    left_command = trigger_to_dex3_targets(0.25, np.zeros(7), np.arange(1, 8))
    right_command = trigger_to_dex3_targets(0.75, np.zeros(7), np.arange(10, 17))
    controller.ctrl_dual_hand(left_command, right_command)

    np.testing.assert_allclose(published["left"][0], left_command)
    np.testing.assert_allclose(published["right"][0], right_command)
