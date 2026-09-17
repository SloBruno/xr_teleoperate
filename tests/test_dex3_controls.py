import importlib
from pathlib import Path
import sys
import types

import numpy as np


sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from teleop.utils.dex3_controls import compose_dex3_targets, trigger_to_dex3_targets



def test_control_process_does_not_read_xr_hand_readiness():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "robot_control" / "robot_hand_unitree.py").read_text()
    process_body = source[source.index("    def control_process("):source.index("class Dex3_1_Left_JointIndex")]

    assert "xr_motion_data_ready_in.get_lock()" not in process_body


def test_dex3_controller_does_not_construct_hand_retargeting():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "robot_control" / "robot_hand_unitree.py").read_text()
    dex3_body = source[source.index("class Dex3_1_Controller"):source.index("class Dex3_1_Left_JointIndex")]

    assert "HandRetargeting(" not in dex3_body


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
    np.testing.assert_allclose(
        trigger_to_dex3_targets(np.inf, open_pose, closed_pose), open_pose
    )
    np.testing.assert_allclose(
        trigger_to_dex3_targets(-np.inf, open_pose, closed_pose), open_pose
    )


def test_released_trigger_preserves_retargeted_vector():
    base = np.array([0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 1.4])
    closed = np.arange(1, 8, dtype=float)

    np.testing.assert_allclose(compose_dex3_targets(base, 0.0, closed), base)


def test_midpoint_trigger_blends_each_retargeted_slot_toward_closed_pose():
    base = np.array([0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 1.4])
    closed = np.arange(1, 8, dtype=float)

    np.testing.assert_allclose(
        compose_dex3_targets(base, 0.5, closed), (base + closed) / 2.0
    )


def test_full_trigger_commands_closed_pose_for_each_slot():
    base = np.array([0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 1.4])
    closed = np.arange(1, 8, dtype=float)

    np.testing.assert_allclose(compose_dex3_targets(base, 1.0, closed), closed)


def test_nonfinite_overlay_trigger_preserves_retargeted_vector():
    base = np.arange(7, dtype=float) + 0.25
    closed = np.arange(1, 8, dtype=float)

    for trigger in (np.nan, np.inf, -np.inf):
        np.testing.assert_allclose(compose_dex3_targets(base, trigger, closed), base)


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


def test_control_step_uses_only_side_specific_triggers_when_hand_tracking_is_unavailable(monkeypatch):
    # Reuse the module import fakes from the publisher test's shape, but exercise
    # the controller's one-cycle mapping and publish path.
    sdk_channel = types.ModuleType("unitree_sdk2py.core.channel")
    sdk_channel.ChannelPublisher = object
    sdk_channel.ChannelSubscriber = object
    sdk_channel.ChannelFactoryInitialize = object
    sdk_hand = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    sdk_hand.HandCmd_ = object
    sdk_hand.HandState_ = object
    sdk_default = types.ModuleType("unitree_sdk2py.idl.default")

    class Motor:
        def __init__(self):
            self.q = 0.0

    class Message:
        def __init__(self):
            self.motor_cmd = [Motor() for _ in range(7)]

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
    sdk_default.unitree_hg_msg_dds__HandCmd_ = Message
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
    published = []
    controller = module.Dex3_1_Controller.__new__(module.Dex3_1_Controller)
    controller.left_msg = Message()
    controller.right_msg = Message()
    controller.LeftHandCmb_publisher = types.SimpleNamespace(
        Write=lambda msg: published.append(("left", [motor.q for motor in msg.motor_cmd]))
    )
    controller.RightHandCmb_publisher = types.SimpleNamespace(
        Write=lambda msg: published.append(("right", [motor.q for motor in msg.motor_cmd]))
    )
    controller.hand_retargeting = types.SimpleNamespace(
        left_indices=np.array([[0], [1]]), right_indices=np.array([[0], [1]]),
        left_retargeting=types.SimpleNamespace(retarget=lambda ref: np.array([0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 1.4])),
        right_retargeting=types.SimpleNamespace(retarget=lambda ref: np.array([1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2])),
        left_dex_retargeting_to_hardware=list(range(7)),
        right_dex_retargeting_to_hardware=list(range(7)),
    )
    # No hand-tracking object is available. Dex3 must still operate from
    # controller triggers rather than attempting to read hand data/readiness.
    left_input = object()
    right_input = object()
    left_trigger = types.SimpleNamespace(value=0.25, get_lock=lambda: __import__("contextlib").nullcontext())
    right_trigger = types.SimpleNamespace(value=0.75, get_lock=lambda: __import__("contextlib").nullcontext())
    controller.control_step(left_input, right_input, left_trigger, right_trigger, xr_motion_data_ready=False)

    assert published[0][0] == "left"
    assert published[1][0] == "right"
    # Retargeted vectors deliberately differ from trigger commands: physical
    # finger targets must depend only on their corresponding controller trigger.
    np.testing.assert_allclose(
        published[0][1], trigger_to_dex3_targets(0.25, module.Dex3_Open_Pose, module.Dex3_Closed_Pose)
    )
    np.testing.assert_allclose(
        published[1][1], trigger_to_dex3_targets(0.75, module.Dex3_Open_Pose, module.Dex3_Closed_Pose)
    )
