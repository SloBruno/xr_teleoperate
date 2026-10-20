"""Launcher contract for the opt-in G1-D AGV (G1D_AGV=1): env-only, validated
cap, not user-forwardable, works in every G1D video mode."""
import subprocess

import pytest

from tests.test_g1d_quest_dex3_launcher import G1D_DOC, G1D_LAUNCHER, _run_launcher


def _argv(result):
    return next(l for l in result.stdout.splitlines() if l.startswith("ARGV:"))


@pytest.mark.parametrize("mode", ["off", "head", "vertical"])
def test_g1d_agv_env_adds_flag_in_every_video_mode(tmp_path, mode):
    result = _run_launcher(tmp_path, extra_env={"G1D_VIDEO_MODE": mode, "G1D_AGV": "1",
                                                "XR_TELEOP_VUER_IP": "10.0.0.42"})
    assert result.returncode == 0, result.stderr
    argv = _argv(result)
    assert " --g1d-agv" in argv
    assert "--arm G1D --g1d-ik-geometry-assumption --ee dex3 --input-mode hand" in argv
    assert "--motion" not in argv
    assert "AGV" in result.stdout


@pytest.mark.parametrize("mode", ["off", "head", "vertical"])
@pytest.mark.parametrize("value", [None, "0", ""])
def test_without_g1d_agv_env_no_flag(tmp_path, mode, value):
    env = {"G1D_VIDEO_MODE": mode, "XR_TELEOP_VUER_IP": "10.0.0.42"}
    if value is not None:
        env["G1D_AGV"] = value
    result = _run_launcher(tmp_path, extra_env=env)
    assert result.returncode == 0, result.stderr
    assert "--g1d-agv" not in _argv(result)


def test_invalid_g1d_agv_value_is_refused(tmp_path):
    result = _run_launcher(tmp_path, extra_env={"G1D_AGV": "yes"})
    assert result.returncode == 2
    assert "G1D_AGV must be 0 or 1" in result.stderr
    assert "ARGV:" not in result.stdout


@pytest.mark.parametrize("cap", ["0.6", "1.5", "0.25", "1", "1.0"])
def test_valid_max_vx_is_exported(tmp_path, cap):
    result = _run_launcher(tmp_path, extra_env={"G1D_AGV": "1", "G1D_AGV_MAX_VX": cap})
    assert result.returncode == 0, result.stderr
    assert "--g1d-agv" in _argv(result)
    assert f"G1D_AGV_MAX_VX={cap}" in result.stdout


def test_default_max_vx_is_0_6(tmp_path):
    result = _run_launcher(tmp_path, extra_env={"G1D_AGV": "1"})
    assert result.returncode == 0, result.stderr
    assert "G1D_AGV_MAX_VX=0.6" in result.stdout


@pytest.mark.parametrize("cap", ["1.51", "2", "0", "0.0", "-0.5", "nan", "abc", "1e9", "inf"])
def test_invalid_max_vx_is_refused_before_preflight(tmp_path, cap):
    result = _run_launcher(tmp_path, extra_env={"G1D_AGV": "1", "G1D_AGV_MAX_VX": cap})
    assert result.returncode == 2, (cap, result.stdout, result.stderr)
    assert "G1D_AGV_MAX_VX" in result.stderr
    assert "ARGV:" not in result.stdout


@pytest.mark.parametrize("forbidden", [["--g1d-agv"], ["--display-mode", "pass-through", "--g1d-agv"],
                                       ["--g1d-agv=1"], ["--g1d-ag"], ["--g1d-a"]])
def test_g1d_agv_flag_is_not_user_forwardable(tmp_path, forbidden):
    for env in ({}, {"G1D_AGV": "1"}):
        result = _run_launcher(tmp_path, *forbidden, extra_env=env)
        assert result.returncode != 0
        assert "forwarded argument refused" in result.stderr
        assert "ARGV:" not in result.stdout


def test_motion_still_refused_with_agv(tmp_path):
    result = _run_launcher(tmp_path, "--motion", extra_env={"G1D_AGV": "1"})
    assert result.returncode != 0 and "forwarded argument refused" in result.stderr


def test_launcher_syntax_and_doc_has_exact_command():
    assert subprocess.run(["bash", "-n", str(G1D_LAUNCHER)]).returncode == 0
    doc = G1D_DOC.read_text(encoding="utf-8")
    assert "G1D_AGV=1" in doc and "G1D_AGV_MAX_VX" in doc
    assert "G1D_AGV=1 G1D_VIDEO_MODE=head bash run_g1d_quest_dex3.sh" in doc
