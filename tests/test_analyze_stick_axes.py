import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import analyze_stick_axes as a


def rec(t, y, vx):
    return {"event": "teleop_status", "timestamp": t,
            "locomotion": {"raw_left_xy": [0.0, y], "command": [vx, 0.0, 0.0]}}


def run(recs, user=None):
    return a.summarize(a.extract_samples(recs), user_pushed=user)["windows"]


def test_raw_negative_with_negative_vx_is_codigo():
    w = run([rec(i, -1.0, -0.15) for i in range(3)])
    assert [x["verdict"] for x in w] == ["CODIGO"]


def test_raw_positive_with_negative_vx_is_device_inverted_when_user_pushed_forward():
    w = run([rec(i, 1.0, -0.15) for i in range(3)], user="forward")
    assert w[0]["verdict"] == "EIXO_INVERTIDO_NO_DISPOSITIVO"
    assert "?" in run([rec(0, 1.0, -0.15)])[0]["verdict"]


def test_expected_forward_is_consistent_and_windows_split():
    recs = [rec(0, -1.0, 0.15), rec(1, 0.0, 0.0), rec(2, 1.0, -0.15)]
    assert [x["verdict"] for x in run(recs)][0] == "CONSISTENTE"
    assert len(run(recs)) == 2


def test_nonfinite_and_old_records_are_skipped_and_cli_is_readonly(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    lines = [json.dumps({"event": "teleop_status", "locomotion": {"command": [0.1, 0, 0]}}),
             json.dumps(rec(0, None, 0.1)), "garbage", json.dumps(rec(1, -1.0, 0.15))]
    p.write_text("\n".join(lines) + "\n")
    before = p.read_text()
    assert a.main([str(p)]) == 0
    assert p.read_text() == before
    assert "CONSISTENTE" in capsys.readouterr().out
