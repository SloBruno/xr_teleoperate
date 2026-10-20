"""Regression: real G1D recordings must stay replayable.

* Dex3 encoders report a few centirad past the URDF limits at the mechanical
  stops (observed up to 0.061 rad on the robot). The commanded value is already
  clamped inside limits by the session, so loading must accept that sensor
  overshoot while still rejecting genuinely out-of-range data.
* Recordings saved before the explicit ``profile`` header field existed
  (same v1 schema, robot G1D) must still load.
"""

import json

import numpy as np
import pytest

from teleop.utils import teach_replay as tr


def _write(path, *, left_q, profile="G1D"):
    header = tr.build_header(hand_sides=("left", "right"), rate_hz=50.0)
    if profile is None:
        header.pop("profile", None)
    else:
        header["profile"] = profile
    right_q = [0.0, 0.0, -0.5, 0.5, 0.5, 0.5, 0.5]
    lines = [json.dumps(header)]
    n = 5
    for i in range(n):
        t = i * 0.02
        lines.append(json.dumps({
            "type": "sample", "t": t, "t_mono": t,
            "arm_q": [0.0] * 14, "arm_dq": [0.0] * 14,
            "left_hand_q": list(left_q), "left_hand_dq": [0.0] * 7,
            "right_hand_q": right_q, "right_hand_dq": [0.0] * 7,
        }))
    lines.append(json.dumps({"type": "footer", "status": "saved", "samples": n,
                             "dropped": 0, "duration_s": (n - 1) * 0.02}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


ARM_LO = -np.full(14, 3.0)
ARM_HI = np.full(14, 3.0)


def _load(path):
    return tr.load_recording(path, arm_lower=ARM_LO, arm_upper=ARM_HI,
                             hand_limits=tr.DEX3_LIMITS)


def test_dex3_encoder_overshoot_at_stop_is_accepted(tmp_path):
    # left thumb_2 lower limit is 0.0; real stop reads -0.061
    q = [0.0, 0.0, -0.061, -0.5, -0.5, -0.5, -0.5]
    rec = _load(_write(tmp_path / "a.jsonl", left_q=q))
    assert rec.hand_q["left"].shape == (5, 7)


def test_genuinely_out_of_range_hand_is_still_refused(tmp_path):
    q = [0.0, 0.0, -0.5, -0.5, -0.5, -0.5, -0.5]
    with pytest.raises(tr.RecordingError, match="left hand joint limit violation"):
        _load(_write(tmp_path / "b.jsonl", left_q=q))


def test_recording_without_profile_field_from_g1d_still_loads(tmp_path):
    q = [0.0, 0.0, 0.5, -0.5, -0.5, -0.5, -0.5]
    rec = _load(_write(tmp_path / "c.jsonl", left_q=q, profile=None))
    assert rec.header["robot"] == "G1D"


def test_recording_without_footer_duration_from_older_writer_still_loads(tmp_path):
    q = [0.0, 0.0, 0.5, -0.5, -0.5, -0.5, -0.5]
    path = _write(tmp_path / "e.jsonl", left_q=q, profile=None)
    lines = path.read_text().splitlines()
    footer = json.loads(lines[-1]); footer.pop("duration_s")
    lines[-1] = json.dumps(footer)
    path.write_text("\n".join(lines) + "\n")
    rec = _load(path)
    assert rec.t[-1] == pytest.approx(0.08)


def test_wrong_footer_duration_is_still_refused(tmp_path):
    q = [0.0, 0.0, 0.5, -0.5, -0.5, -0.5, -0.5]
    path = _write(tmp_path / "f.jsonl", left_q=q)
    lines = path.read_text().splitlines()
    footer = json.loads(lines[-1]); footer["duration_s"] = 9.0
    lines[-1] = json.dumps(footer)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(tr.RecordingError, match="duration"):
        _load(path)


def test_wrong_profile_is_still_refused(tmp_path):
    q = [0.0, 0.0, 0.5, -0.5, -0.5, -0.5, -0.5]
    with pytest.raises(tr.RecordingError, match="unsupported robot/profile"):
        _load(_write(tmp_path / "d.jsonl", left_q=q, profile="G1_29"))
