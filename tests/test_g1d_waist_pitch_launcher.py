"""Launcher contract for the opt-in G1-D waist pitch (G1D_WAIST_PITCH=1):
env-only, not user-forwardable, every G1D video mode, composes with G1D_AGV."""
import pytest

from tests.test_g1d_quest_dex3_launcher import _run_launcher


def _argv(result):
    return next(l for l in result.stdout.splitlines() if l.startswith("ARGV:"))


@pytest.mark.parametrize("mode", ["off", "head", "vertical"])
def test_env_adds_flag_in_every_video_mode(tmp_path, mode):
    result = _run_launcher(tmp_path, extra_env={"G1D_VIDEO_MODE": mode, "G1D_WAIST_PITCH": "1",
                                                "XR_TELEOP_VUER_IP": "10.0.0.42"})
    assert result.returncode == 0, result.stderr
    argv = _argv(result)
    assert " --g1d-waist-pitch" in argv
    assert "--arm G1D --g1d-ik-geometry-assumption --ee dex3 --input-mode hand" in argv
    assert "--motion" not in argv
    assert "LEFT Y" in result.stdout and "q=0.0" in result.stdout


@pytest.mark.parametrize("mode", ["off", "head", "vertical"])
@pytest.mark.parametrize("value", [None, "0", ""])
def test_without_env_no_flag(tmp_path, mode, value):
    env = {"G1D_VIDEO_MODE": mode, "XR_TELEOP_VUER_IP": "10.0.0.42"}
    if value is not None:
        env["G1D_WAIST_PITCH"] = value
    result = _run_launcher(tmp_path, extra_env=env)
    assert result.returncode == 0, result.stderr
    assert "--g1d-waist-pitch" not in _argv(result)


def test_invalid_value_is_refused(tmp_path):
    result = _run_launcher(tmp_path, extra_env={"G1D_WAIST_PITCH": "yes"})
    assert result.returncode == 2
    assert "G1D_WAIST_PITCH must be 0 or 1" in result.stderr
    assert "ARGV:" not in result.stdout


@pytest.mark.parametrize("arg", ["--g1d-waist-pitch", "--g1d-waist-pitch=1", "--g1d-w"])
def test_flag_cannot_be_forwarded(tmp_path, arg):
    result = _run_launcher(tmp_path, arg)
    assert result.returncode == 2
    assert "forwarded argument refused" in result.stderr


def test_composes_with_agv(tmp_path):
    result = _run_launcher(tmp_path, extra_env={"G1D_WAIST_PITCH": "1", "G1D_AGV": "1",
                                                "XR_TELEOP_VUER_IP": "10.0.0.42"})
    assert result.returncode == 0, result.stderr
    argv = _argv(result)
    assert " --g1d-agv" in argv and " --g1d-waist-pitch" in argv

