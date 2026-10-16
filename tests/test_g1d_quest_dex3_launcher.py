from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
G1_LAUNCHER = ROOT / "teleop" / "run_g1_quest_dex3.sh"
G1D_LAUNCHER = ROOT / "teleop" / "run_g1d_quest_dex3.sh"
G1D_DOC = ROOT / "teleop" / "G1D_QUEST_DEX3.md"


def test_g1d_launcher_is_a_separate_syntax_valid_operator_entrypoint():
    assert G1_LAUNCHER.exists()
    assert G1D_LAUNCHER.exists(), "G1D must not reuse or alter the G1 launcher"
    assert G1D_LAUNCHER != G1_LAUNCHER
    result = subprocess.run(["bash", "-n", str(G1D_LAUNCHER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_g1d_launcher_defaults_to_jetson_g1d_dds_interface_with_override():
    source = G1D_LAUNCHER.read_text(encoding="utf-8")
    assert "G1D_DDS_INTERFACE=${G1D_DDS_INTERFACE:-enP2p1s0}" in source
    assert 'NetworkInterface name="%s"' in source
    assert '"$g1d_dds_interface"' in source
    assert '--network-interface "$g1d_dds_interface"' in source


def test_g1d_launcher_uses_configurable_remote_teleimager_host_without_starting_services():
    source = G1D_LAUNCHER.read_text(encoding="utf-8")
    assert "G1D_TELEIMAGER_HOST=${G1D_TELEIMAGER_HOST:-192.168.123.164}" in source
    assert "teleimager_host=$G1D_TELEIMAGER_HOST" in source
    for forbidden in ("--detect", "pyrealsense2", "teleimager.image_server", "setsid nohup", "teleimager_dir="):
        assert forbidden not in source


def test_g1d_launcher_is_arm_only_and_preserves_terminal_r_q_lifecycle():
    source = G1D_LAUNCHER.read_text(encoding="utf-8")
    assert "--arm G1D" in source
    assert "--g1d-ik-geometry-assumption" in source
    assert "--ee dex3" in source
    assert "--input-mode hand" in source
    assert "--camera-layout vertical" in source
    assert "--motion" not in source
    doc = G1D_DOC.read_text(encoding="utf-8")
    assert "`r`" in doc and "`q`" in doc


def test_g1d_launcher_fails_closed_for_two_configured_physical_camera_paths():
    source = G1D_LAUNCHER.read_text(encoding="utf-8")
    assert "G1D_TELEIMAGER_HOST" in source
    for port in ("60000", "55555", "55556"):
        assert port in source
    assert "left_wrist_camera" in source
    assert "head_camera" in source
    assert "enable_zmq" in source
    assert "get_left_wrist_frame" in source
    assert "refusing to start" in source
    for forbidden in ("TELEIMAGER_CAMERA_MODE=${TELEIMAGER_CAMERA_MODE:-auto}", "mode == \"any\"", "CÂMERA ÚNICA"):
        assert forbidden not in source


def test_g1d_documentation_describes_teleimager_camera_representation_and_no_fallback():
    doc = G1D_DOC.read_text(encoding="utf-8")
    assert "`head_camera`" in doc
    assert "`left_wrist_camera`" in doc
    assert "`binocular`" in doc
    assert "sem fallback" in doc.lower()
