"""No-actuation local-simulation pre-arm smoke contract."""

import os
from pathlib import Path
import subprocess
import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "teleop" / "teleop_hand_and_arm.py"
TV_PYTHON = Path("/home/bruno/miniconda3/envs/tv/bin/python")
LEGACY_TELEIMAGER = Path("/home/bruno/xr_teleoperate/teleop/teleimager/src")


def _teleop_env():
    env = os.environ.copy()
    env.pop("AMENT_PREFIX_PATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((str(LEGACY_TELEIMAGER), str(ROOT)))
    return env


def test_prearm_smoke_without_sim_fails_closed_before_dds_setup():
    """The no-output smoke cannot accidentally select the domain-0 robot path."""
    result = subprocess.run(
        [str(TV_PYTHON), str(SCRIPT), "--prearm-smoke"],
        cwd=ROOT,
        env=_teleop_env(),
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 2
    assert "--prearm-smoke requires --sim" in result.stderr


def test_actual_sim_prearm_smoke_exits_before_outputs_and_tears_down_local_stack():
    """Real subprocess: local MuJoCo + fake TeleImager, no Quest or DDS writer."""
    sim_root = Path("/home/bruno/sim/g1_sim")
    launcher = sim_root / "run_xr_teleop_sim.sh"
    assert launcher.is_file()

    stack = subprocess.Popen(
        ["bash", str(launcher), "--sim-seconds", "20"],
        cwd=sim_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(30):
            health = subprocess.run(
                ["bash", str(launcher), "--health"],
                cwd=sim_root,
                text=True,
                capture_output=True,
                timeout=10,
            )
            if health.returncode == 0:
                break
            assert stack.poll() is None, "local simulation stack exited before becoming healthy"
        else:
            raise AssertionError("local simulation stack did not become healthy")

        result = subprocess.run(
            [
                str(TV_PYTHON), str(SCRIPT), "--sim", "--prearm-smoke",
                "--network-interface", "lo", "--img-server-ip", "127.0.0.1",
                "--camera-layout", "vertical",
            ],
            cwd=ROOT,
            env=_teleop_env(),
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[prearm-smoke] PASS passive DDS + camera protocol" in result.stdout
        assert "'rt/arm_sdk': False" in result.stdout
        assert "'rt/lowcmd': False" in result.stdout
        assert "'dex3_command': False" in result.stdout
        assert "'process_writer': False" in result.stdout
        assert "'motion_control': False" in result.stdout
    finally:
        if stack.poll() is None:
            stack.terminate()
        stack.communicate(timeout=20)

    remaining = subprocess.run(
        ["ps", "-eo", "args="], text=True, capture_output=True, check=True
    ).stdout
    assert "g1_dds_sim.py --domain 1 --iface lo" not in remaining
    assert "fake_teleimager.py --bind-host 127.0.0.1" not in remaining


def test_prearm_boundary_rejects_every_output_handle():
    """Formal boundary proof: a pre-arm run cannot retain actuator authority."""
    from teleop.utils.prearm_smoke import PrearmOutputBoundary

    class PassiveArm:
        lowcmd_publisher = None
        outputs_activated = False
        publish_thread = None

    class PassiveDex3:
        LeftHandCmb_publisher = None
        RightHandCmb_publisher = None
        hand_control_process = None
        outputs_activated = False

    proof = PrearmOutputBoundary()
    proof.assert_passive(PassiveArm(), PassiveDex3())
    assert proof.summary() == {
        "rt/arm_sdk": False,
        "rt/lowcmd": False,
        "dex3_command": False,
        "process_writer": False,
        "motion_control": False,
    }

    class ForbiddenPublisher:
        pass

    class ArmedArm(PassiveArm):
        lowcmd_publisher = ForbiddenPublisher()

    with pytest.raises(RuntimeError, match="output boundary violated"):
        proof.assert_passive(ArmedArm(), PassiveDex3())
