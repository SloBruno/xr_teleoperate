import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

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


def test_status_file_sink_snapshots_nested_mapping_before_enqueue(tmp_path):
    import json
    from teleop.utils.teleop_status import AsyncStatusFileSink

    target = tmp_path / "status.jsonl"
    sink = AsyncStatusFileSink(str(target), lambda _: None)
    payload = {"nested": {"items": [1, {"value": "before"}]}}
    assert sink.emit(payload) is True
    payload["nested"]["items"][1]["value"] = "after"
    payload["nested"]["items"].append(2)
    sink.close()

    assert json.loads(target.read_text()) == {"nested": {"items": [1, {"value": "before"}]}}


def test_status_file_sink_rejects_emit_after_close(tmp_path):
    from teleop.utils.teleop_status import AsyncStatusFileSink

    sink = AsyncStatusFileSink(str(tmp_path / "status.jsonl"), lambda _: None)
    sink.close()
    assert sink.emit({"event": "late"}) is False


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


def test_disabled_status_sink_emit_returns_false():
    from teleop.utils.teleop_status import _DisabledStatusSink

    assert _DisabledStatusSink().emit({"event": "ignored"}) is False



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
    event = records[-1]
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

    events = records
    warnings = [event for event in events if event["event"] == "controller_freshness_changed"]
    assert warnings == [{"event": "controller_freshness_changed", "fresh": False, "age_ms": 400}]


def test_head_fallback_over_one_second_is_visible_once_and_recovery_is_reported():
    from teleop.utils.teleop_status import TeleopStatusMonitor

    records, terminal_warnings = [], []
    monitor = TeleopStatusMonitor(records.append, warn=terminal_warnings.append, interval_s=99.0)
    common = {"lifecycle": "ready", "controller_sample_timestamp": 9.9,
              "head_pose_sample_timestamp": 0.0, "head_pose_is_fallback": True}

    monitor.observe(now=10.0, **common)
    monitor.observe(now=11.01, **common)
    monitor.observe(now=12.0, **common)
    monitor.observe(now=12.1, lifecycle="ready", controller_sample_timestamp=12.0,
                    head_pose_sample_timestamp=12.05, head_pose_is_fallback=False)

    assert len(terminal_warnings) == 1
    assert "pose da cabeça" in terminal_warnings[0] and "fallback" in terminal_warnings[0]
    head_events = [event for event in records if event["event"].startswith("head_pose_")]
    assert head_events == [
        {"event": "head_pose_unavailable", "reason": "fallback", "duration_ms": 1010},
        {"event": "head_pose_recovered", "age_ms": 50},
    ]


def test_status_heartbeat_includes_head_stream_health():
    from teleop.utils.teleop_status import TeleopStatusMonitor

    status = TeleopStatusMonitor(lambda payload: None).observe(
        now=20.0, lifecycle="tracking", controller_sample_timestamp=19.9,
        head_pose_sample_timestamp=19.95, head_pose_is_fallback=False,
        torso_lean={"configured": True, "enabled": False, "status": "waist_watchdog_tripped"})

    assert status["head_pose"] == {"available": True, "fallback": False, "age_ms": 50}
    assert status["torso_lean"] == {
        "configured": True, "enabled": False, "status": "waist_watchdog_tripped"}


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


def test_status_monitor_contains_ordinary_sink_and_warning_failures():
    from teleop.utils.teleop_status import TeleopStatusMonitor

    calls = []

    def emit(payload):
        calls.append(payload)
        raise RuntimeError("status sink failed")

    monitor = TeleopStatusMonitor(emit, interval_s=1.0)
    assert monitor.observe(now=10.0, lifecycle="tracking", controller_sample_timestamp=9.9) is not None
    assert monitor.observe(now=10.3, lifecycle="tracking", controller_sample_timestamp=9.9) is None
    assert len(calls) == 2


def test_status_monitor_propagates_keyboard_interrupt_from_normal_emit():
    from teleop.utils.teleop_status import TeleopStatusMonitor

    monitor = TeleopStatusMonitor(lambda payload: (_ for _ in ()).throw(KeyboardInterrupt("operator stop")))
    with pytest.raises(KeyboardInterrupt):
        monitor.observe(now=10.0, lifecycle="tracking", controller_sample_timestamp=9.9)


def test_status_monitor_contains_malformed_status_data():
    from teleop.utils.teleop_status import TeleopStatusMonitor

    monitor = TeleopStatusMonitor(lambda payload: None)
    assert monitor.observe(
        now=10.0,
        lifecycle="tracking",
        controller_sample_timestamp=9.9,
        locomotion=(object(),),
    ) is None


def test_status_file_sink_contains_ordinary_buffer_fault_but_propagates_keyboard_interrupt(tmp_path, monkeypatch):
    from teleop.utils import teleop_status

    sink = teleop_status.AsyncStatusFileSink(str(tmp_path / "status.jsonl"), lambda _: None)
    monkeypatch.setattr(sink._queue, "put_nowait", lambda payload: (_ for _ in ()).throw(OSError("buffer fault")))
    assert sink.emit({"event": "teleop_status"}) is False
    sink.close()

    sink = teleop_status.AsyncStatusFileSink(str(tmp_path / "status-2.jsonl"), lambda _: None)
    monkeypatch.setattr(sink._queue, "put_nowait", lambda payload: (_ for _ in ()).throw(KeyboardInterrupt("operator stop")))
    with pytest.raises(KeyboardInterrupt):
        sink.emit({"event": "teleop_status"})
    monkeypatch.undo()
    sink.close()


def test_head_layout_camera_status_omits_unused_left_wrist():
    from teleop.utils.teleop_status import camera_status_for_layout

    head = SimpleNamespace(bgr=object())
    assert camera_status_for_layout("head", head, None) == {"head": True}
    assert camera_status_for_layout("vertical", head, None) == {"head": True, "left_wrist": False}


def test_status_reports_camera_source(monkeypatch):
    from teleop.utils.teleop_status import TeleopStatusMonitor
    import inspect
    src = inspect.getsource(TeleopStatusMonitor)
    assert "camera_source" in src and "TELEIMAGER_CAMERA_SOURCE" in src
