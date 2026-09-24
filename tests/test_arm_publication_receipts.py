import sys
import time
import types

import numpy as np


def _load_robot_arm(monkeypatch):
    logging_mp = types.ModuleType("logging_mp")
    logging_mp.getLogger = lambda name: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "logging_mp", logging_mp)

    channel = types.ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelPublisher = object
    channel.ChannelSubscriber = object
    channel.ChannelFactoryInitialize = lambda *args: None
    monkeypatch.setitem(sys.modules, "unitree_sdk2py", types.ModuleType("unitree_sdk2py"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core", types.ModuleType("unitree_sdk2py.core"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", channel)

    for name in ("unitree_hg", "unitree_go"):
        msg = types.ModuleType(f"unitree_sdk2py.idl.{name}.msg.dds_")
        msg.LowCmd_ = object
        msg.LowState_ = object
        monkeypatch.setitem(sys.modules, f"unitree_sdk2py.idl.{name}.msg.dds_", msg)
    default = types.ModuleType("unitree_sdk2py.idl.default")
    default.unitree_hg_msg_dds__LowCmd_ = object
    default.unitree_go_msg_dds__LowCmd_ = object
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl", types.ModuleType("unitree_sdk2py.idl"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.default", default)
    crc = types.ModuleType("unitree_sdk2py.utils.crc")
    crc.CRC = object
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.utils", types.ModuleType("unitree_sdk2py.utils"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.utils.crc", crc)

    sys.modules.pop("teleop.robot_control.robot_arm", None)
    from teleop.robot_control import robot_arm
    return robot_arm


def test_receipts_are_immutable_ordered_and_bounded(monkeypatch):
    module = _load_robot_arm(monkeypatch)

    class Controller(module._ArmPublicationMixin):
        arm_joint_split = (2, 2)

    controller = Controller()
    controller._init_arm_publication_state()
    for request_id in range(70):
        controller._record_arm_publication(request_id, np.array([request_id, 1, 2, 3]), "published")

    receipts = controller.drain_arm_publication_receipts()
    assert [receipt.request_id for receipt in receipts] == list(range(6, 70))
    assert controller.publication_receipt_drop_count == 6
    assert receipts[0].arm_joint_split == (2, 2)
    assert isinstance(receipts[0].published_q, tuple)
    assert receipts[0].timestamp_monotonic <= receipts[-1].timestamp_monotonic


def test_failed_receipt_has_null_q_and_reason(monkeypatch):
    module = _load_robot_arm(monkeypatch)

    class Controller(module._ArmPublicationMixin):
        arm_joint_split = (1, 1)

    controller = Controller()
    controller._init_arm_publication_state()
    controller._record_failed_arm_publication(3, RuntimeError("write failed"))
    receipt = controller.drain_arm_publication_receipts()[0]
    assert receipt.request_id == 3
    assert receipt.published_q is None
    assert receipt.reason == "arm_command_publication_failed:RuntimeError"
