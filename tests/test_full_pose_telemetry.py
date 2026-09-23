import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_build_record_preserves_controller_wrist_matrices_and_joint_snapshots():
    from teleop.utils.full_pose_telemetry import build_pose_record

    left_wrist = np.arange(16, dtype=float).reshape(4, 4)
    right_wrist = (np.arange(16, dtype=float) + 100).reshape(4, 4)
    record = build_pose_record(
        timestamp=12.5,
        lifecycle="tracking",
        controller_sample_timestamp=12.4,
        left_wrist_pose=left_wrist,
        right_wrist_pose=right_wrist,
        measured_arm_q=np.arange(14, dtype=float),
        commanded_arm_q=np.arange(14, dtype=float) + 0.5,
        dex3_measured_q=np.arange(14, dtype=float) + 10,
        dex3_commanded_q=np.arange(14, dtype=float) + 20,
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
        "left": {"measured_q": None, "commanded_q": None},
        "right": {"measured_q": None, "commanded_q": None},
    }


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
