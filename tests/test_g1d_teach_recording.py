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
    assert lines[0]["profile"] == "G1D"
    assert lines[0]["rate_hz"] == 50.0
    assert lines[0]["arm_joint_names"] == list(tr.ARM_JOINT_NAMES)
    assert len(lines[0]["arm_joint_names"]) == 14
    assert lines[0]["hand_sides"] == ["left", "right"]
    assert lines[-1] == {"type": "footer", "status": "saved", "samples": 10, "dropped": 0,
                          "duration_s": 0.18}
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


def test_same_name_writers_keep_both_payloads_without_part_debris(tmp_path):
    writers = [tr.RecordingWriter(tmp_path, _header(), name="same") for _ in range(2)]
    assert writers[0].emit(_sample(0.0)); assert writers[0].emit(_sample(0.02))
    assert writers[1].emit(_sample(10.0)); assert writers[1].emit(_sample(10.02))
    results = [writer.close() for writer in writers]
    assert all(result.error is None for result in results)
    paths = sorted(tmp_path.glob("same*.jsonl"))
    assert len(paths) == 2
    assert {json.loads(path.read_text().splitlines()[1])["t"] for path in paths} == {0.0, 10.0}
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("mutate", [
    lambda record: record.pop("t_mono"),
    lambda record: record.__setitem__("t_mono", "bad"),
    lambda record: record.__setitem__("t_mono", float("nan")),
    lambda record: record.__setitem__("t_mono", float("inf")),
])
def test_emit_rejects_invalid_timing_before_enqueue(tmp_path, mutate):
    writer = tr.RecordingWriter(tmp_path, _header(), name="timing")
    sample = _sample(0.0); mutate(sample)
    assert writer.emit(sample) is False
    assert writer.close().samples == 0


def test_emit_rejects_nonincreasing_timing_after_an_accepted_sample(tmp_path):
    writer = tr.RecordingWriter(tmp_path, _header(), name="timing-order")
    assert writer.emit(_sample(0.0))
    assert writer.emit(_sample(0.0)) is False
    assert writer.close().samples == 1


@pytest.mark.parametrize("field,value,reason", [
    ("clock", "wall", "clock"),
    ("t_mono", None, "sample time"),
    ("t_mono", float("nan"), "sample time"),
    ("t_mono", float("inf"), "sample time"),
    ("t_mono", "bad", "sample time"),
])
def test_load_recording_validates_monotonic_clock_contract(tmp_path, field, value, reason):
    writer = tr.RecordingWriter(tmp_path, _header(), name="clock")
    writer.emit(_sample(0.0)); writer.emit(_sample(0.02))
    path = writer.close().path
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if field == "clock": records[0][field] = value
    elif value is None: records[1].pop(field)
    else: records[1][field] = value
    bad = tmp_path / "bad-clock.jsonl"
    bad.write_text("\n".join(json.dumps(record, allow_nan=True) for record in records) + "\n")
    with pytest.raises(tr.RecordingError, match=reason): tr.load_recording(bad)


def test_load_recording_rejects_nonincreasing_t_mono_and_allows_unknown_fields(tmp_path):
    writer = tr.RecordingWriter(tmp_path, _header(), name="clock-order")
    writer.emit(_sample(0.0)); writer.emit(_sample(0.02))
    path = writer.close().path
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["harmless_unknown"] = {"x": 1}
    records[2]["t_mono"] = records[1]["t_mono"]
    bad = tmp_path / "bad-order.jsonl"
    bad.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    with pytest.raises(tr.RecordingError, match="sample time"): tr.load_recording(bad)


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


@pytest.mark.parametrize("mutate,reason", [
    (lambda h, f: h.__setitem__("profile", "OTHER"), "profile"),
    (lambda h, f: h.__setitem__("robot", "G1_29"), "profile"),
    (lambda h, f: h.__setitem__("hand_sides", ["right", "left"]), "hand sides"),
    (lambda h, f: h.__setitem__("hand_sides", ["left", "left"]), "hand sides"),
])
def test_load_recording_rejects_incomplete_profile_and_hand_side_schema(tmp_path, mutate, reason):
    writer = tr.RecordingWriter(tmp_path, _header(), name="strict-header")
    writer.emit(_sample(0.0)); writer.emit(_sample(0.02))
    path = writer.close().path
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(lines[0], lines[-1])
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    with pytest.raises(tr.RecordingError, match=reason):
        tr.load_recording(path)


@pytest.mark.parametrize("field,value,reason", [
    ("samples", 99, "samples"),
    ("samples", -1, "samples"),
    ("dropped", -1, "dropped"),
    ("duration_s", float("nan"), "duration"),
    ("duration_s", 0.0, "duration"),
])
def test_load_recording_rejects_bad_footer_metadata(tmp_path, field, value, reason):
    writer = tr.RecordingWriter(tmp_path, _header(), name="strict-footer")
    writer.emit(_sample(0.0)); writer.emit(_sample(0.02))
    path = writer.close().path
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines[-1][field] = value
    path.write_text("\n".join(json.dumps(line, allow_nan=True) for line in lines) + "\n")
    with pytest.raises(tr.RecordingError, match=reason):
        tr.load_recording(path)


def test_load_recording_rejects_unexpected_record_between_header_and_footer(tmp_path):
    writer = tr.RecordingWriter(tmp_path, _header(), name="unexpected")
    writer.emit(_sample(0.0)); writer.emit(_sample(0.02))
    path = writer.close().path
    lines = path.read_text().splitlines()
    lines.insert(-1, json.dumps({"type": "metadata", "x": 1}))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(tr.RecordingError, match="unexpected"):
        tr.load_recording(path)


def test_load_recording_rejects_non_object_json_record(tmp_path):
    path = tmp_path / "non-object.jsonl"
    path.write_text("[]\n")
    with pytest.raises(tr.RecordingError):
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
