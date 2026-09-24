import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def test_submission_does_not_drain_delayed_or_unrelated_receipts(monkeypatch):
    from teleop.utils.full_pose_telemetry import publish_arm_command_for_telemetry

    class Controller:
        arm_joint_split = (1, 1)

        def __init__(self):
            self.receipts = []

        def ctrl_dual_arm(self, q_target, tauff_target):
            return 3

        def drain_arm_publication_receipts(self):
            receipts, self.receipts = tuple(self.receipts), []
            return receipts

    controller = Controller()
    controller.receipts.append({"request_id": 1, "published_q": (1.0, 2.0), "reason": "published"})
    result = publish_arm_command_for_telemetry(controller, np.zeros(2), np.zeros(2))
    assert result.request_id == 3
    assert controller.receipts, "submission must not drain receipts"


def test_publication_bridge_emits_delayed_out_of_order_and_multiple_receipts_once(monkeypatch):
    from teleop.utils.full_pose_telemetry import ArmPublicationTelemetryBridge

    class Sink:
        def __init__(self):
            self.records = []

        def emit(self, record):
            self.records.append(record)
            return True

    class Controller:
        arm_joint_split = (1, 1)
        publication_receipt_drop_count = 0

        def __init__(self):
            self.cycles = [
                (),
                (
                    {"request_id": 2, "published_q": (2.0, 2.5), "reason": "published", "timestamp_monotonic": 2.2, "arm_joint_split": (1, 1)},
                    {"request_id": 1, "published_q": (1.0, 1.5), "reason": "published", "timestamp_monotonic": 1.2, "arm_joint_split": (1, 1)},
                ),
                ({"request_id": 3, "published_q": None, "reason": "arm_command_publication_failed:RuntimeError", "timestamp_monotonic": 3.2, "arm_joint_split": (1, 1)},),
            ]

        def drain_arm_publication_receipts(self):
            return self.cycles.pop(0)

    sink = Sink()
    controller = Controller()
    bridge = ArmPublicationTelemetryBridge(sink, profile="G1_29")
    for _ in range(3):
        bridge.emit_cycle({}, controller)
    publications = [record for record in sink.records if record.get("event") == "arm_publication"]
    assert [record["request_id"] for record in publications] == [2, 1, 3]
    assert publications[0]["published_q"] == [2.0, 2.5]
    assert publications[2]["published_q"] is None
    assert publications[0]["profile"] == "G1_29"
    assert len(publications) == len({record["request_id"] for record in publications})


def test_publication_bridge_retains_failed_writes_and_counts_pending_overflow():
    from teleop.utils.full_pose_telemetry import ArmPublicationTelemetryBridge

    class Sink:
        def __init__(self):
            self.accept = False
            self.records = []

        def emit(self, record):
            if not self.accept:
                return False
            self.records.append(record)
            return True

    class Controller:
        arm_joint_split = (1, 1)
        publication_receipt_drop_count = 0

        def __init__(self):
            self.drained = False

        def drain_arm_publication_receipts(self):
            if self.drained:
                return ()
            self.drained = True
            return tuple({"request_id": n, "published_q": (float(n), float(n)), "reason": "published", "timestamp_monotonic": float(n + 1), "arm_joint_split": (1, 1)} for n in range(4))

    sink = Sink()
    bridge = ArmPublicationTelemetryBridge(sink, profile="G1_29", pending_capacity=2)
    controller = Controller()
    bridge.emit_cycle({}, controller)
    assert bridge.pending_buffer_drop_count == 2
    sink.accept = True
    bridge.emit_cycle({}, controller)
    assert [record["request_id"] for record in sink.records if record.get("event") == "arm_publication"] == [0, 1]
    assert bridge.pending_buffer_drop_count == 2


def test_lifecycle_warning_callback_is_swallowed(monkeypatch):
    from teleop.utils.full_pose_telemetry import emit_lifecycle_event_best_effort

    class Sink:
        def emit(self, record):
            raise RuntimeError("sink failure")

    def raising_warn(message):
        raise RuntimeError("logger failure")

    assert emit_lifecycle_event_best_effort(Sink(), "shutdown_finalization", warn=raising_warn) is False


def test_lifecycle_helper_swallow_all_telemetry_component_failures():
    from teleop.utils.full_pose_telemetry import emit_lifecycle_event_best_effort

    class Sink:
        def emit(self, record):
            raise SystemExit("telemetry component failed")

    assert emit_lifecycle_event_best_effort(
        Sink(), "shutdown_finalization", warn=lambda message: (_ for _ in ()).throw(KeyboardInterrupt())
    ) is False


def test_publication_bridge_swallow_all_boundaries_and_count_invalid_receipts():
    from teleop.utils.full_pose_telemetry import ArmPublicationTelemetryBridge

    warnings = []

    class Sink:
        def __init__(self):
            self.records = []

        def emit(self, record):
            self.records.append(record)
            return True

    class Controller:
        publication_receipt_drop_count = 0

        def __init__(self):
            self.calls = 0

        def drain_arm_publication_receipts(self):
            self.calls += 1
            if self.calls == 1:
                return (
                    {"request_id": 4, "published_q": (1.0, 2.0), "timestamp_monotonic": float("nan"), "arm_joint_split": (1, 1)},
                    {"request_id": 5, "published_q": (1.0,), "timestamp_monotonic": 1.0, "arm_joint_split": (1, 1)},
                )
            raise SystemExit("receipt drain failed")

    sink = Sink()
    bridge = ArmPublicationTelemetryBridge(
        sink, profile="G1_29", warn=lambda message: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    controller = Controller()
    bridge.emit_cycle({}, controller)
    assert bridge.invalid_receipt_count == 2
    assert len([record for record in sink.records if record.get("event") == "arm_publication_invalid"]) == 2

    bridge.emit_cycle({}, controller)
    assert bridge.receipt_drain_failure_count == 1


def test_publication_bridge_swallow_all_pending_and_sink_failures():
    from teleop.utils.full_pose_telemetry import ArmPublicationTelemetryBridge

    class Sink:
        def emit(self, record):
            raise KeyboardInterrupt("sink failure")

    class Controller:
        publication_receipt_drop_count = 0

        def drain_arm_publication_receipts(self):
            return ({"request_id": 1, "published_q": (1.0, 2.0), "timestamp_monotonic": 1.0, "arm_joint_split": (1, 1)},)

    bridge = ArmPublicationTelemetryBridge(Sink(), profile="G1_29")
    assert bridge.emit_cycle({}, Controller()) is False
    assert bridge.telemetry_sink_overflow_count == 2
    assert bridge.pending_buffer_drop_count == 0


def test_pose_record_best_effort_swallow_builder_sink_and_warning_failures(monkeypatch):
    from teleop.utils import full_pose_telemetry

    def raising_warn(message):
        raise SystemExit("logger failed")

    monkeypatch.setattr(
        full_pose_telemetry,
        "build_pose_record",
        lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt("builder failed")),
    )
    assert full_pose_telemetry.emit_pose_record_best_effort(
        lambda record: (_ for _ in ()).throw(RuntimeError("sink failed")),
        warn=raising_warn,
        timestamp=1.0,
        timestamp_monotonic=1.0,
        lifecycle="tracking",
        controller_sample_timestamp=1.0,
        left_wrist_pose=None,
        right_wrist_pose=None,
        measured_arm_q=(0.0, 0.0),
        commanded_arm_q=(0.0, 0.0),
    ) is False
