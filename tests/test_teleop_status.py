import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_status_file_sink_writes_asynchronously_without_control_path_file_io(tmp_path):
    from teleop.utils.teleop_status import AsyncStatusFileSink

    warnings = []
    target = tmp_path / "status" / "teleop.jsonl"
    sink = AsyncStatusFileSink(str(target), warnings.append)
    sink.emit('{"event":"teleop_status"}')
    sink.close()

    assert target.read_text() == '{"event":"teleop_status"}\n'
    assert warnings == []


def test_status_file_sink_degrades_to_noop_when_storage_setup_fails(monkeypatch):
    from teleop.utils import teleop_status

    warnings = []
    def fail_makedirs(*args, **kwargs):
        raise OSError("read-only")
    monkeypatch.setattr(teleop_status.os, "makedirs", fail_makedirs)
    sink = teleop_status.AsyncStatusFileSink("/blocked/teleop.jsonl", warnings.append)
    for _ in range(128):
        sink.emit('{"event":"teleop_status"}')
    sink.close()

    # The control path must not invoke logging even if a failed writer leaves
    # its bounded queue full; only the writer thread reports setup failure.
    assert warnings == ["Could not initialize teleop status log: read-only"]



def test_status_monitor_emits_structured_heartbeat_with_control_ages():
    from teleop.utils.teleop_status import TeleopStatusMonitor

    records = []
    monitor = TeleopStatusMonitor(records.append, interval_s=1.0)
    status = monitor.observe(
        now=10.0,
        lifecycle="tracking",
        controller_sample_timestamp=9.9,
        motion_enabled=True,
        locomotion=(0.4, 0.0, -0.2),
        cameras={"head": True, "left_wrist": True},
        dex3_pressure_timestamps=(9.95, 9.8),
    )

    assert status is not None
    event = json.loads(records[-1])
    assert event["event"] == "teleop_status"
    assert event["lifecycle"] == "tracking"
    assert event["controller"] == {"fresh": True, "age_ms": 100}
    assert event["locomotion"] == {"enabled": True, "command": [0.4, 0.0, -0.2]}
    assert event["cameras"] == {"head": True, "left_wrist": True}
    assert event["dex3_pressure"] == {"left_age_ms": 50, "right_age_ms": 200}


def test_status_monitor_emits_one_warning_on_controller_freshness_transition():
    from teleop.utils.teleop_status import TeleopStatusMonitor

    records = []
    monitor = TeleopStatusMonitor(records.append, interval_s=99.0)

    monitor.observe(now=10.0, lifecycle="tracking", controller_sample_timestamp=9.9)
    monitor.observe(now=10.3, lifecycle="tracking", controller_sample_timestamp=9.9)
    monitor.observe(now=10.4, lifecycle="tracking", controller_sample_timestamp=9.9)

    events = [json.loads(record) for record in records]
    warnings = [event for event in events if event["event"] == "controller_freshness_changed"]
    assert warnings == [{"event": "controller_freshness_changed", "fresh": False, "age_ms": 400}]


def test_camera_frame_is_usable_only_when_the_image_has_pixels():
    from teleop.utils.teleop_status import camera_frame_is_usable

    class Image:
        def __init__(self, bgr):
            self.bgr = bgr

    assert not camera_frame_is_usable(None)
    assert not camera_frame_is_usable(Image(None))
    assert camera_frame_is_usable(Image(object()))


def test_status_monitor_reports_missing_camera_and_nonfinite_timestamps_as_unhealthy():
    from teleop.utils.teleop_status import TeleopStatusMonitor

    records = []
    status = TeleopStatusMonitor(records.append).observe(
        now=10.0,
        lifecycle="ready",
        controller_sample_timestamp=float("nan"),
        cameras={"head": False, "left_wrist": None},
        dex3_pressure_timestamps=(float("nan"), 0.0),
    )

    assert status["controller"] == {"fresh": False, "age_ms": None}
    assert status["cameras"] == {"head": False, "left_wrist": False}
    assert status["dex3_pressure"] == {"left_age_ms": None, "right_age_ms": None}
