"""Launcher contract for the opt-in G1-D waist pitch/yaw (G1D_WAIST=1):
env-only, not user-forwardable, works in every G1D video mode. This flag
always refuses to start at the Python level today (no real backend)."""
import subprocess

import pytest

from tests.test_g1d_quest_dex3_launcher import G1D_DOC, G1D_LAUNCHER, _run_launcher


def _argv(result):
    return next(l for l in result.stdout.splitlines() if l.startswith("ARGV:"))


@pytest.mark.parametrize("mode", ["off", "head", "vertical"])
def test_g1d_waist_env_adds_flag_in_every_video_mode(tmp_path, mode):
    result = _run_launcher(tmp_path, extra_env={"G1D_VIDEO_MODE": mode, "G1D_WAIST": "1",
                                                "XR_TELEOP_VUER_IP": "10.0.0.42"})
    assert result.returncode == 0, result.stderr
    argv = _argv(result)
    assert " --g1d-waist" in argv
    assert "--arm G1D --g1d-ik-geometry-assumption --ee dex3 --input-mode hand" in argv
    assert "--motion" not in argv
    assert "WAIST" in result.stdout


@pytest.mark.parametrize("mode", ["off", "head", "vertical"])
@pytest.mark.parametrize("value", [None, "0", ""])
def test_without_g1d_waist_env_no_flag(tmp_path, mode, value):
    env = {"G1D_VIDEO_MODE": mode, "XR_TELEOP_VUER_IP": "10.0.0.42"}
    if value is not None:
        env["G1D_WAIST"] = value
    result = _run_launcher(tmp_path, extra_env=env)
    assert result.returncode == 0, result.stderr
    assert "--g1d-waist" not in _argv(result)


def test_invalid_g1d_waist_value_is_refused(tmp_path):
    result = _run_launcher(tmp_path, extra_env={"G1D_WAIST": "yes"})
    assert result.returncode == 2
    assert "G1D_WAIST must be 0 or 1" in result.stderr
    assert "ARGV:" not in result.stdout


@pytest.mark.parametrize("forbidden", [["--g1d-waist"], ["--display-mode", "pass-through", "--g1d-waist"],
                                       ["--g1d-waist=1"], ["--g1d-wa"], ["--g1d-w"]])
def test_g1d_waist_flag_is_not_user_forwardable(tmp_path, forbidden):
    for env in ({}, {"G1D_WAIST": "1"}):
        result = _run_launcher(tmp_path, *forbidden, extra_env=env)
        assert result.returncode != 0
        assert "forwarded argument refused" in result.stderr
        assert "ARGV:" not in result.stdout


def test_motion_still_refused_with_waist(tmp_path):
    result = _run_launcher(tmp_path, "--motion", extra_env={"G1D_WAIST": "1"})
    assert result.returncode != 0 and "forwarded argument refused" in result.stderr


def test_g1d_waist_and_g1d_agv_can_coexist_in_argv(tmp_path):
    result = _run_launcher(tmp_path, extra_env={"G1D_WAIST": "1", "G1D_AGV": "1"})
    assert result.returncode == 0, result.stderr
    argv = _argv(result)
    assert "--g1d-agv" in argv
    assert "--g1d-waist" in argv


def test_launcher_syntax_and_doc_mentions_waist():
    assert subprocess.run(["bash", "-n", str(G1D_LAUNCHER)]).returncode == 0
    doc = G1D_DOC.read_text(encoding="utf-8")
    assert "G1D_WAIST=1" in doc
