import json
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_build_record_preserves_controller_wrist_matrices_and_joint_snapshots():
    from teleop.utils.full_pose_telemetry import build_pose_record

    left_wrist = np.arange(16, dtype=float).reshape(4, 4)
    right_wrist = (np.arange(16, dtype=float) + 100).reshape(4, 4)
    record = build_pose_record(
        timestamp=12.5,
        timestamp_monotonic=12.5,
        lifecycle="tracking",
        controller_sample_timestamp=12.4,
        left_wrist_pose=left_wrist,
        right_wrist_pose=right_wrist,
        measured_arm_q=np.arange(14, dtype=float),
        commanded_arm_q=np.arange(14, dtype=float) + 0.5,
        dex3_measured_q=np.arange(14, dtype=float) + 10,
        dex3_commanded_q=np.arange(14, dtype=float) + 20,
        dex3_sample_metadata={
            "left": {"state_valid": True, "state_timestamp": 10.0, "action_valid": True, "action_timestamp": 10.0},
            "right": {"state_valid": True, "state_timestamp": 10.0, "action_valid": True, "action_timestamp": 10.0},
        },
        drop_count=3,
        now=12.5,
    )

    assert record["controller"]["wrist_pose"] == {
        "left": left_wrist.tolist(),
        "right": right_wrist.tolist(),
    }
    assert record["arm"] == {
        "left": {"measured_q": list(range(7)), "commanded_q": [x + 0.5 for x in range(7)]},
        "right": {"measured_q": list(range(7, 14)), "commanded_q": [x + 0.5 for x in range(7, 14)]},
    }
    assert record["dex3"]["left"]["measured_q"] == [float(x) for x in range(10, 17)]
    assert record["dex3"]["right"]["commanded_q"] == [float(x) for x in range(27, 34)]
    assert record["achieved_cartesian_pose"] == {"left": None, "right": None}
    assert record["achieved_cartesian_pose_reason"] == "not_available_from_controller_state"
    assert record["drop_count"] == 3
    assert record["controller"]["sample_timestamp"] == 12.4
    assert record["controller"]["fresh"] is True


def test_build_record_uses_explicit_null_and_reason_when_dex3_is_unavailable():
    from teleop.utils.full_pose_telemetry import build_pose_record

    record = build_pose_record(
        timestamp=2.0,
        timestamp_monotonic=2.0,
        lifecycle="ready",
        controller_sample_timestamp=0.0,
        left_wrist_pose=None,
        right_wrist_pose=None,
        measured_arm_q=np.zeros(14),
        commanded_arm_q=np.zeros(14),
        dex3_measured_q=None,
        dex3_commanded_q=None,
        drop_count=0,
        now=2.0,
    )

    assert record["controller"]["wrist_pose"] == {"left": None, "right": None}
    assert record["controller"]["fresh"] is False
    assert record["dex3"] == {
        "available": False,
        "reason": "dex3_not_configured",
        "left": {
            "state_valid": False,
            "state_timestamp": None,
            "action_valid": False,
            "action_timestamp": None,
            "measured_q": None,
            "commanded_q": None,
        },
        "right": {
            "state_valid": False,
            "state_timestamp": None,
            "action_valid": False,
            "action_timestamp": None,
            "measured_q": None,
            "commanded_q": None,
        },
    }


def test_build_record_distinguishes_configured_dex3_without_a_sample():
    from teleop.utils.full_pose_telemetry import build_pose_record

    record = build_pose_record(
        timestamp=2.0,
        timestamp_monotonic=8.0,
        lifecycle="ready",
        controller_sample_timestamp=0.0,
        left_wrist_pose=None,
        right_wrist_pose=None,
        measured_arm_q=np.zeros(14),
        commanded_arm_q=np.zeros(14),
        dex3_configured=True,
        dex3_measured_q=None,
        dex3_commanded_q=None,
        dex3_sample_metadata={
            "left": {"state_valid": False, "state_timestamp": None, "action_valid": False, "action_timestamp": None},
            "right": {"state_valid": False, "state_timestamp": None, "action_valid": False, "action_timestamp": None},
        },
        drop_count=0,
        now=8.0,
    )

    assert record["timestamp_utc"].endswith("Z")
    assert record["timestamp_monotonic"] == 8.0
    assert record["clock_domain"] == {
        "timestamp": "wall_clock_utc",
        "timestamp_monotonic": "monotonic",
        "controller_sample_timestamp": "monotonic",
    }
    assert record["dex3"]["available"] is False
    assert record["dex3"]["reason"] == "dex3_configured_no_sample"


def test_build_record_does_not_treat_zero_arrays_as_dex3_samples():
    from teleop.utils.full_pose_telemetry import build_pose_record

    record = build_pose_record(
        timestamp=2.0,
        timestamp_monotonic=8.0,
        lifecycle="ready",
        controller_sample_timestamp=0.0,
        left_wrist_pose=None,
        right_wrist_pose=None,
        measured_arm_q=np.zeros(14),
        commanded_arm_q=np.zeros(14),
        dex3_configured=True,
        dex3_measured_q=np.zeros(14),
        dex3_commanded_q=np.zeros(14),
        dex3_sample_metadata={
            "left": {"state_valid": False, "state_timestamp": None, "action_valid": False, "action_timestamp": None},
            "right": {"state_valid": False, "state_timestamp": None, "action_valid": False, "action_timestamp": None},
        },
        drop_count=0,
        now=8.0,
    )

    assert record["dex3"]["available"] is False
    assert record["dex3"]["reason"] == "dex3_configured_no_sample"
    assert record["dex3"]["left"] == {
        "state_valid": False,
        "state_timestamp": None,
        "action_valid": False,
        "action_timestamp": None,
        "measured_q": None,
        "commanded_q": None,
    }


def test_build_record_distinguishes_partial_and_complete_dex3_sampling():
    from teleop.utils.full_pose_telemetry import build_pose_record

    common = dict(
        timestamp=2.0,
        timestamp_monotonic=8.0,
        lifecycle="ready",
        controller_sample_timestamp=0.0,
        left_wrist_pose=None,
        right_wrist_pose=None,
        measured_arm_q=np.zeros(14),
        commanded_arm_q=np.zeros(14),
        dex3_configured=True,
        dex3_measured_q=np.arange(14, dtype=float),
        dex3_commanded_q=np.arange(14, dtype=float) + 1,
        drop_count=0,
        now=8.0,
    )
    partial = build_pose_record(
        **common,
        dex3_sample_metadata={
            "left": {"state_valid": True, "state_timestamp": 7.0, "action_valid": False, "action_timestamp": None},
            "right": {"state_valid": False, "state_timestamp": None, "action_valid": False, "action_timestamp": None},
        },
    )
    sampled = build_pose_record(
        **common,
        dex3_sample_metadata={
            "left": {"state_valid": True, "state_timestamp": 7.0, "action_valid": True, "action_timestamp": 7.0},
            "right": {"state_valid": True, "state_timestamp": 7.0, "action_valid": True, "action_timestamp": 7.0},
        },
    )

    assert partial["dex3"]["reason"] == "dex3_partially_sampled"
    assert partial["dex3"]["left"]["measured_q"] == list(range(7))
    assert partial["dex3"]["left"]["commanded_q"] is None
    assert partial["dex3"]["right"]["measured_q"] is None
    assert partial["dex3"]["right"]["commanded_q"] is None
    assert sampled["dex3"]["reason"] == "dex3_sampled"


def test_build_record_requires_an_explicit_valid_monotonic_timestamp():
    from teleop.utils.full_pose_telemetry import build_pose_record

    kwargs = dict(
        timestamp=100.0,
        lifecycle="tracking",
        controller_sample_timestamp=1.0,
        left_wrist_pose=None,
        right_wrist_pose=None,
        measured_arm_q=np.zeros(14),
        commanded_arm_q=np.zeros(14),
    )
    with unittest.TestCase().assertRaises(TypeError):
        build_pose_record(**kwargs)
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with unittest.TestCase().assertRaises(ValueError):
            build_pose_record(**kwargs, timestamp_monotonic=invalid)


def test_lifecycle_event_record_is_structured_without_serialization():
    from teleop.utils.full_pose_telemetry import build_lifecycle_event

    event = build_lifecycle_event(
        "tracking_started", timestamp=10.0, timestamp_monotonic=20.0
    )

    assert event["event"] == "tracking_started"
    assert event["timestamp_utc"].endswith("Z")
    assert event["timestamp_monotonic"] == 20.0


def test_lifecycle_event_rejects_nonpositive_or_nonfinite_monotonic_timestamps():
    from teleop.utils.full_pose_telemetry import build_lifecycle_event

    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with unittest.TestCase().assertRaises(ValueError):
            build_lifecycle_event(
                "tracking_started", timestamp=10.0, timestamp_monotonic=invalid
            )


def test_arm_telemetry_uses_the_exact_published_target_from_the_controller():
    from teleop.utils.full_pose_telemetry import (
        build_pose_record,
        publish_arm_command_for_telemetry,
    )

    requested = np.arange(14, dtype=float) + 100.0
    published = np.arange(14, dtype=float) + 0.25

    class FakeLimiter:
        def ctrl_dual_arm(self, q_target, tauff_target):
            assert np.array_equal(q_target, requested)
            return {"published_q": published, "reason": "published"}

    published_q, reason = publish_arm_command_for_telemetry(
        FakeLimiter(), requested, np.zeros(14)
    )
    record = build_pose_record(
        timestamp=10.0,
        timestamp_monotonic=10.0,
        lifecycle="tracking",
        controller_sample_timestamp=9.9,
        left_wrist_pose=None,
        right_wrist_pose=None,
        measured_arm_q=np.zeros(14),
        commanded_arm_q=published_q,
        commanded_arm_q_reason=reason,
    )

    assert record["arm"]["left"]["commanded_q"] == published[:7].tolist()
    assert record["arm"]["right"]["commanded_q"] == published[7:].tolist()
    assert record["arm"]["left"]["commanded_q"] != requested[:7].tolist()


@pytest.mark.parametrize(
    "profile, split",
    [
        ("G1_29", (7, 7)),
        ("G1_23", (5, 5)),
        ("H1_2", (7, 7)),
        ("H1", (4, 4)),
        ("H2", (7, 7)),
    ],
)
def test_arm_telemetry_uses_selected_profile_split_and_left_right_order(profile, split):
    from teleop.utils.full_pose_telemetry import build_pose_record

    left_count, right_count = split
    values = np.arange(left_count + right_count, dtype=float) + 100.0
    record = build_pose_record(
        timestamp=10.0,
        timestamp_monotonic=10.0,
        lifecycle="tracking",
        controller_sample_timestamp=9.9,
        left_wrist_pose=None,
        right_wrist_pose=None,
        measured_arm_q=values,
        commanded_arm_q=values + 0.5,
        arm_joint_split=split,
    )

    assert record["arm"]["left"]["measured_q"] == values[:left_count].tolist()
    assert record["arm"]["right"]["measured_q"] == values[left_count:].tolist()
    assert record["arm"]["left"]["commanded_q"] == (values[:left_count] + 0.5).tolist()
    assert record["arm"]["right"]["commanded_q"] == (values[left_count:] + 0.5).tolist()


@pytest.mark.parametrize("split", [(7, 7), (5, 5), (4, 4)])
def test_arm_publication_preserves_exact_profile_sized_q(split):
    from teleop.utils.full_pose_telemetry import publish_arm_command_for_telemetry

    requested = np.arange(sum(split), dtype=float)
    published = requested + 0.25

    class ProfileController:
        arm_joint_split = split

        def ctrl_dual_arm(self, q_target, tauff_target):
            return {"published_q": published, "reason": "published"}

    published_q, reason = publish_arm_command_for_telemetry(
        ProfileController(), requested, np.zeros(sum(split))
    )

    np.testing.assert_array_equal(published_q, published)
    assert reason == "published"


def test_pending_arm_publication_keeps_request_id_without_calling_it_published():
    from teleop.utils.full_pose_telemetry import publish_arm_command_for_telemetry

    class PendingController:
        arm_joint_split = (5, 5)

        def ctrl_dual_arm(self, q_target, tauff_target):
            return 41

        def drain_arm_publication_receipts(self):
            return ()

    publication = publish_arm_command_for_telemetry(
        PendingController(), np.ones(10), np.zeros(10)
    )

    assert publication.published_q is None
    assert publication.reason == "arm_command_publication_pending"
    assert publication.request_id == 41
    np.testing.assert_array_equal(publication.requested_q, np.ones(10))


def test_build_pose_record_separates_requested_q_from_exact_published_q():
    from teleop.utils.full_pose_telemetry import build_pose_record

    record = build_pose_record(
        timestamp=10.0,
        timestamp_monotonic=10.0,
        lifecycle="tracking",
        controller_sample_timestamp=9.9,
        left_wrist_pose=None,
        right_wrist_pose=None,
        measured_arm_q=np.zeros(10),
        commanded_arm_q=None,
        commanded_arm_q_reason="arm_command_publication_pending",
        arm_joint_split=(5, 5),
        arm_command_request_id=41,
        requested_arm_q=np.ones(10),
        selected_arm_q=np.ones(10),
    )

    assert record["arm"]["request_id"] == 41
    assert record["arm"]["left"]["requested_q"] == [1.0] * 5
    assert record["arm"]["right"]["selected_q"] == [1.0] * 5
    assert record["arm"]["left"]["commanded_q"] is None


def test_submission_records_request_and_waits_for_later_receipt_event():
    from teleop.utils.full_pose_telemetry import publish_arm_command_for_telemetry

    requested = np.arange(10, dtype=float)
    clipped = requested * 0.1

    class AsyncController:
        arm_joint_split = (5, 5)

        def ctrl_dual_arm(self, q_target, tauff_target):
            return 7

        def drain_arm_publication_receipts(self):
            return ({
                "request_id": 7,
                "published_q": clipped,
                "reason": "published",
            },)

    controller = AsyncController()
    publication = publish_arm_command_for_telemetry(controller, requested, np.zeros(10))

    assert publication.request_id == 7
    assert publication.reason == "arm_command_publication_pending"
    np.testing.assert_array_equal(publication.requested_q, requested)
    np.testing.assert_array_equal(controller.drain_arm_publication_receipts()[0]["published_q"], clipped)


def test_lifecycle_emit_is_best_effort_for_builder_and_sink_failures(monkeypatch):
    from teleop.utils import full_pose_telemetry

    warnings = []

    class FailingSink:
        def emit(self, record):
            raise RuntimeError("sink failed")

    assert full_pose_telemetry.emit_lifecycle_event_best_effort(
        FailingSink(), "one", warn=warnings.append
    ) is False
    monkeypatch.setattr(
        full_pose_telemetry,
        "build_lifecycle_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("invalid event")),
    )
    assert full_pose_telemetry.emit_lifecycle_event_best_effort(
        object(), "two", warn=warnings.append
    ) is False
    assert len(warnings) == 2


def test_lifecycle_builder_and_sink_failures_swallow_a_raising_warning_callback(monkeypatch):
    from teleop.utils import full_pose_telemetry

    def fail_builder(*args, **kwargs):
        raise ValueError("builder failed")

    def fail_warn(message):
        raise RuntimeError("logger failed")

    monkeypatch.setattr(full_pose_telemetry, "build_lifecycle_event", fail_builder)
    assert full_pose_telemetry.emit_lifecycle_event_best_effort(
        object(), "cleanup", warn=fail_warn
    ) is False

    class FailingSink:
        def emit(self, record):
            raise OSError("sink failed")

    monkeypatch.setattr(full_pose_telemetry, "build_lifecycle_event", lambda *a, **k: {})
    assert full_pose_telemetry.emit_lifecycle_event_best_effort(
        FailingSink(), "cleanup", warn=fail_warn
    ) is False


def test_arm_publication_rejects_returned_dimension_mismatch_explicitly():
    from teleop.utils.full_pose_telemetry import publish_arm_command_for_telemetry

    class ProfileController:
        arm_joint_split = (5, 5)

        def ctrl_dual_arm(self, q_target, tauff_target):
            return {"published_q": np.zeros(14), "reason": "published"}

    published_q, reason = publish_arm_command_for_telemetry(
        ProfileController(), np.zeros(10), np.zeros(10)
    )

    assert published_q is None
    assert reason == "arm_command_publication_dimension_mismatch:expected=10:actual=14"


def test_arm_publication_rejects_invalid_returned_q_without_published_reason():
    from teleop.utils.full_pose_telemetry import publish_arm_command_for_telemetry

    class ProfileController:
        arm_joint_split = (5, 5)

        def ctrl_dual_arm(self, q_target, tauff_target):
            return {"published_q": np.full(10, np.nan), "reason": "published"}

    published_q, reason = publish_arm_command_for_telemetry(
        ProfileController(), np.zeros(10), np.zeros(10)
    )

    assert published_q is None
    assert reason == "arm_command_publication_invalid"


def test_arm_publication_rejects_requested_dimension_mismatch_explicitly():
    from teleop.utils.full_pose_telemetry import publish_arm_command_for_telemetry

    class ProfileController:
        arm_joint_split = (4, 4)

        def ctrl_dual_arm(self, q_target, tauff_target):
            raise AssertionError("controller must not receive an invalid target")

    published_q, reason = publish_arm_command_for_telemetry(
        ProfileController(), np.zeros(10), np.zeros(10)
    )

    assert published_q is None
    assert reason == "arm_command_target_dimension_mismatch:expected=8:actual=10"


def test_arm_telemetry_records_null_and_reason_when_publication_is_unavailable():
    from teleop.utils.full_pose_telemetry import (
        build_pose_record,
        publish_arm_command_for_telemetry,
    )

    class UnavailableController:
        def ctrl_dual_arm(self, q_target, tauff_target):
            return None

    published_q, reason = publish_arm_command_for_telemetry(
        UnavailableController(), np.ones(14), np.zeros(14)
    )
    record = build_pose_record(
        timestamp=10.0,
        timestamp_monotonic=10.0,
        lifecycle="tracking",
        controller_sample_timestamp=9.9,
        left_wrist_pose=None,
        right_wrist_pose=None,
        measured_arm_q=np.zeros(14),
        commanded_arm_q=published_q,
        commanded_arm_q_reason=reason,
    )

    assert record["arm"]["left"]["commanded_q"] is None
    assert record["arm"]["right"]["commanded_q"] is None
    assert record["arm"]["left"]["commanded_q_reason"] == "arm_command_publication_unavailable"
    assert record["arm"]["right"]["commanded_q_reason"] == "arm_command_publication_unavailable"


def test_failed_telemetry_initialization_returns_nonblocking_sink(monkeypatch, tmp_path):
    from teleop.utils import full_pose_telemetry

    warnings = []
    monkeypatch.setattr(
        full_pose_telemetry,
        "PoseTelemetryJsonlSink",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("thread unavailable")),
    )

    sink = full_pose_telemetry.create_pose_telemetry_sink(tmp_path, warnings.append)
    assert sink.emit({"event": "ignored"}) is False
    sink.close()
    assert warnings == ["Could not initialize pose telemetry sink: thread unavailable"]


def test_failed_status_initialization_returns_nonblocking_sink(monkeypatch):
    from teleop.utils import teleop_status

    warnings = []
    monkeypatch.setattr(
        teleop_status,
        "AsyncStatusFileSink",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("thread unavailable")),
    )

    sink = teleop_status.create_status_sink("/unavailable/status.jsonl", warnings.append)
    sink.emit({"event": "ignored"})
    sink.close()
    assert warnings == ["Could not initialize teleop status sink: thread unavailable"]


def test_jsonl_sink_is_bounded_nonblocking_and_uses_unique_session_paths(tmp_path):
    from teleop.utils.full_pose_telemetry import PoseTelemetryJsonlSink

    first = PoseTelemetryJsonlSink(tmp_path, queue_size=1, close_timeout_s=0.2)
    second = PoseTelemetryJsonlSink(tmp_path, queue_size=1, close_timeout_s=0.2)
    assert first.path != second.path
    assert first.emit({"event": "one"}) is True
    assert first.emit({"event": "two"}) is False
    assert first.drop_count == 1
    first.close()
    second.close()

    payloads = [json.loads(line) for line in first.path.read_text().splitlines()]
    assert payloads == [{"event": "one"}]


def test_jsonl_sink_snapshots_nested_mapping_before_enqueue(tmp_path):
    from teleop.utils.full_pose_telemetry import PoseTelemetryJsonlSink

    sink = PoseTelemetryJsonlSink(tmp_path, close_timeout_s=0.2)
    payload = {"nested": {"items": [1, {"value": "before"}]}}
    assert sink.emit(payload) is True
    payload["nested"]["items"][1]["value"] = "after"
    payload["nested"]["items"].append(2)
    sink.close()

    payloads = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert payloads == [{"nested": {"items": [1, {"value": "before"}]}}]


def test_jsonl_sink_warns_and_stays_usable_when_storage_fails(monkeypatch, tmp_path):
    from teleop.utils import full_pose_telemetry

    warnings = []
    monkeypatch.setattr(full_pose_telemetry.os, "makedirs", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    sink = full_pose_telemetry.PoseTelemetryJsonlSink(tmp_path / "blocked", warnings.append)
    assert sink.emit({"event": "still_nonblocking"}) is True
    sink.close()
    assert warnings == ["Could not initialize pose telemetry log: read-only"]


def test_jsonl_sink_close_is_bounded_when_writer_is_stuck(tmp_path):
    from teleop.utils.full_pose_telemetry import PoseTelemetryJsonlSink

    sink = PoseTelemetryJsonlSink(tmp_path, close_timeout_s=0.01)
    sink._thread.join = lambda timeout: None
    start = time.monotonic()
    sink.close()
    assert time.monotonic() - start < 0.2
