"""Teach/replay recording: schema, non-blocking writer, validation, latest."""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from teleop.utils import teach_replay as tr  # noqa: E402


def _header(sides=("left", "right")):
    return tr.build_header(hand_sides=sides, rate_hz=50.0)


def _sample(t, sides=("left", "right")):
    s = {"t": t, "t_mono": 100.0 + t, "arm_q": [0.01 * t] * 14, "arm_dq": [0.0] * 14}
    for side in ("left", "right"):
        s[f"{side}_hand_q"] = [0.02 * t] * 7 if side in sides else None
        s[f"{side}_hand_dq"] = [0.0] * 7 if side in sides else None
    return s


def test_writer_persists_header_samples_footer_and_atomic_final_name(tmp_path):
    writer = tr.RecordingWriter(tmp_path, _header(), name="20261019T120000_000000Z")
    for i in range(10):
        assert writer.emit(_sample(i * 0.02))
    result = writer.close(status="saved", timeout=5.0)
    assert result.error is None
    assert result.path == tmp_path / "20261019T120000_000000Z.jsonl"
    assert result.samples == 10 and result.dropped == 0
    lines = [json.loads(line) for line in result.path.read_text().splitlines()]
    assert lines[0]["type"] == "header"
    assert lines[0]["schema"] == tr.RECORDING_SCHEMA
    assert lines[0]["rate_hz"] == 50.0
    assert lines[0]["arm_joint_names"] == list(tr.ARM_JOINT_NAMES)
    assert len(lines[0]["arm_joint_names"]) == 14
    assert lines[0]["hand_sides"] == ["left", "right"]
    assert lines[-1] == {"type": "footer", "status": "saved", "samples": 10, "dropped": 0}
    body = lines[1:-1]
    assert all(item["type"] == "sample" for item in body)
    assert len(body[0]["arm_q"]) == 14 and len(body[0]["left_hand_q"]) == 7
    assert not list(tmp_path.glob("*.part"))


def test_aborted_recording_is_kept_but_never_selected_as_latest(tmp_path):
    good = tr.RecordingWriter(tmp_path, _header(), name="20261019T110000_000000Z")
    good.emit(_sample(0.0)); good.emit(_sample(0.02))
    good.close()
    aborted = tr.RecordingWriter(tmp_path, _header(), name="20261019T120000_000000Z")
    aborted.emit(_sample(0.0)); aborted.emit(_sample(0.02))
    result = aborted.close(status="aborted_stale")
    assert result.path.name == "20261019T120000_000000Z.aborted.jsonl"
    assert tr.latest_recording(tmp_path) == tmp_path / "20261019T110000_000000Z.jsonl"


def test_emit_never_blocks_when_worker_storage_is_stuck_and_counts_drops(tmp_path):
    gate = threading.Event()

    def slow_open(path, mode):
        gate.wait(5.0)
        return open(path, mode)

    writer = tr.RecordingWriter(tmp_path, _header(), maxsize=8, open_file=slow_open,
                                name="20261019T130000_000000Z")
    t0 = time.perf_counter()
    results = [writer.emit(_sample(i * 0.02)) for i in range(50)]
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, f"emit blocked for {elapsed:.3f}s"
    assert results.count(False) >= 40
    assert writer.dropped_count >= 40
    gate.set()
    result = writer.close(timeout=5.0)
    assert result.dropped >= 40
    assert result.samples <= 9


def test_emit_after_close_is_rejected_and_lands_nowhere(tmp_path):
    writer = tr.RecordingWriter(tmp_path, _header(), name="20261019T140000_000000Z")
    writer.emit(_sample(0.0)); writer.emit(_sample(0.02))
    result = writer.close()
    assert writer.emit(_sample(0.04)) is False
    assert result.samples == 2


def test_storage_failure_degrades_to_error_result_not_exception(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    writer = tr.RecordingWriter(blocker / "sub", _header(), name="x")
    writer.emit(_sample(0.0))
    result = writer.close(timeout=2.0)
    assert result.error is not None


def test_load_recording_roundtrip_and_validation(tmp_path):
    writer = tr.RecordingWriter(tmp_path, _header(("left",)), name="20261019T150000_000000Z")
    for i in range(5):
        writer.emit(_sample(i * 0.02, sides=("left",)))
    path = writer.close().path
    rec = tr.load_recording(path)
    assert rec.hand_sides == ("left",)
    assert rec.t.shape == (5,) and rec.t[0] == 0.0
    assert rec.arm_q.shape == (5, 14)
    assert rec.hand_q["left"].shape == (5, 7)
    assert "right" not in rec.hand_q
    np.testing.assert_allclose(rec.duration, 0.08)


@pytest.mark.parametrize("mutate,reason", [
    (lambda lines: lines[:-1], "footer"),
    (lambda lines: [lines[0], lines[2], lines[1], lines[-1]], "increasing"),
    (lambda lines: [lines[0], lines[1], lines[-1]], "samples"),
    (lambda lines: [lines[0], lines[1], lines[2].replace('"arm_q": [', '"arm_q": [NaN, '), lines[-1]], "arm_q"),
])
def test_load_recording_rejects_malformed_files(tmp_path, mutate, reason):
    writer = tr.RecordingWriter(tmp_path, _header(), name="r")
    writer.emit(_sample(0.0)); writer.emit(_sample(0.02))
    path = writer.close().path
    lines = path.read_text().splitlines()
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(mutate(lines)) + "\n")
    with pytest.raises(tr.RecordingError, match=reason):
        tr.load_recording(bad)


def test_load_recording_rejects_large_time_gaps(tmp_path):
    writer = tr.RecordingWriter(tmp_path, _header(), name="gap")
    writer.emit(_sample(0.0)); writer.emit(_sample(0.02)); writer.emit(_sample(1.0))
    path = writer.close().path
    with pytest.raises(tr.RecordingError, match="gap"):
        tr.load_recording(path)


@pytest.mark.parametrize("field,bad", [
    ("robot", "OTHER"),
    ("arm_joint_names", list(reversed(tr.ARM_JOINT_NAMES))),
    ("hand_joint_names", {"left": list(reversed(tr.HAND_JOINT_NAMES["left"])),
                          "right": list(tr.HAND_JOINT_NAMES["right"])}),
])
def test_load_recording_rejects_wrong_robot_or_exact_joint_schema(tmp_path, field, bad):
    writer = tr.RecordingWriter(tmp_path, _header(), name="schema")
    writer.emit(_sample(0.0)); writer.emit(_sample(0.02))
    path = writer.close().path
    lines = path.read_text().splitlines()
    header = json.loads(lines[0]); header[field] = bad; lines[0] = json.dumps(header)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(tr.RecordingError, match="schema|robot|joint"):
        tr.load_recording(path)


def test_load_recording_rejects_out_of_limits(tmp_path):
    writer = tr.RecordingWriter(tmp_path, _header(), name="limits")
    sample = _sample(0.0); sample["arm_q"][0] = 99.0
    writer.emit(sample); writer.emit(_sample(0.02))
    path = writer.close().path
    with pytest.raises(tr.RecordingError, match="limit"):
        tr.load_recording(path, arm_lower=np.full(14, -2.0), arm_upper=np.full(14, 2.0),
                          hand_limits=tr.DEX3_LIMITS)


def test_load_recording_rejects_invalid_rate_and_duration_metadata(tmp_path):
    writer = tr.RecordingWriter(tmp_path, _header(), name="metadata")
    writer.emit(_sample(0.0)); writer.emit(_sample(0.02))
    path = writer.close().path
    lines = path.read_text().splitlines(); header = json.loads(lines[0]); header["rate_hz"] = float("nan")
    lines[0] = json.dumps(header); path.write_text("\n".join(lines) + "\n")
    with pytest.raises(tr.RecordingError, match="rate"):
        tr.load_recording(path)


def test_load_recording_rejects_wrong_velocity_vector_dimensions(tmp_path):
    writer = tr.RecordingWriter(tmp_path, _header(), name="vectors")
    sample = _sample(0.0); sample["arm_dq"] = [0.0]
    writer.emit(sample); writer.emit(_sample(0.02))
    path = writer.close().path
    with pytest.raises(tr.RecordingError, match="arm_dq"):
        tr.load_recording(path)


def test_resolve_replay_path_accepts_name_in_teach_dir_and_latest(tmp_path):
    writer = tr.RecordingWriter(tmp_path, _header(), name="20261019T160000_000000Z")
    writer.emit(_sample(0.0)); writer.emit(_sample(0.02))
    path = writer.close().path
    assert tr.resolve_replay_path(tmp_path, None) == path
    assert tr.resolve_replay_path(tmp_path, path.name) == path
    assert tr.resolve_replay_path(tmp_path, str(path)) == path
    with pytest.raises(tr.RecordingError):
        tr.resolve_replay_path(tmp_path / "empty", None)


def test_default_teach_dir_is_under_local_state():
    assert tr.default_teach_dir(Path("/home/u")) == Path("/home/u/.local/state/xr_teleoperate/teach")
