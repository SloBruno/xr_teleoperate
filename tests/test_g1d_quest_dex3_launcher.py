from pathlib import Path
import os
import subprocess
import tempfile


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
    assert "TELEIMAGER_PYTHON:-/home/unitree/.local/envs/xr-g1d/bin/python" in source
    assert "/home/unitree/miniconda3/envs/tv/bin/python" not in source
    for forbidden in ("--detect", "pyrealsense2", "teleimager.image_server", "setsid nohup", "teleimager_dir="):
        assert forbidden not in source


def test_g1d_launcher_announces_a_quest_url_from_default_route_or_explicit_override(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ip = fake_bin / "ip"
    fake_ip.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $1 == -4 && $2 == route ]]; then echo '1.1.1.1 via 10.0.0.1 dev wlan0 src 10.0.0.42'; fi\n"
        "if [[ $1 == -4 && $2 == -o ]]; then echo '2: wlan0    inet 10.0.0.42/24 brd 10.0.0.255 scope global wlan0'; fi\n",
        encoding="utf-8",
    )
    fake_ip.chmod(0o755)
    shell = (
        f"PATH={fake_bin}:$PATH; "
        f"source {ROOT / 'teleop' / 'lib' / 'vuer_network.sh'}; "
        "xr_net_announce_url_only"
    )
    detected = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, check=False)
    assert detected.returncode == 0
    assert "Interface/IP Quest: wlan0 / 10.0.0.42" in detected.stdout
    assert "https://vuer.ai?grid=False&ws=wss://10.0.0.42:8012" in detected.stdout

    overridden = subprocess.run(
        ["bash", "-c", shell],
        env=os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}", "XR_TELEOP_VUER_IP": "192.168.50.8"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert overridden.returncode == 0
    assert "Interface/IP Quest: XR_TELEOP_VUER_IP / 192.168.50.8" in overridden.stdout
    assert "https://vuer.ai?grid=False&ws=wss://192.168.50.8:8012" in overridden.stdout

    fake_ip.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    no_route = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, check=False)
    assert no_route.returncode == 0
    assert "não foi possível descobrir IPv4/interface pela rota default" in no_route.stderr
    assert "XR_TELEOP_VUER_IP=<IPv4 acessível ao Quest>" in no_route.stderr

    source = G1D_LAUNCHER.read_text(encoding="utf-8")
    announce = source.index("xr_net_announce_url_only")
    tls = source.index("preflight_tls")
    assert announce < tls


def test_g1d_launcher_is_arm_only_and_preserves_terminal_r_q_lifecycle():
    source = G1D_LAUNCHER.read_text(encoding="utf-8")
    assert "--arm G1D" in source
    assert "--g1d-ik-geometry-assumption" in source
    assert "--ee dex3" in source
    assert "--input-mode hand" in source
    assert "--camera-layout vertical" in source
    # The only allowed mention is the refusal guard; the exec line must not enable it.
    assert "--motion" not in source[source.index("exec "):]
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
    # ImageClient owns background ZMQ threads; normal interpreter teardown can abort.
    assert "os._exit(0)" in source
    assert "refusing to start" in source
    for forbidden in ("TELEIMAGER_CAMERA_MODE=${TELEIMAGER_CAMERA_MODE:-auto}", "mode == \"any\"", "CÂMERA ÚNICA"):
        assert forbidden not in source


def test_g1d_documentation_describes_teleimager_camera_representation_and_no_fallback():
    doc = G1D_DOC.read_text(encoding="utf-8")
    assert "`head_camera`" in doc
    assert "`left_wrist_camera`" in doc
    assert "`binocular`" in doc
    assert "sem fallback" in doc.lower()


def test_g1d_launcher_regenerates_the_quest_tls_certificate_before_teleop_exec():
    source = G1D_LAUNCHER.read_text(encoding="utf-8")
    tls_preflight = source.index("preflight_tls\n", source.index("preflight_tls()"))
    cert_sync = source.index("xr_net_announce", tls_preflight)
    exec_teleop = source.index("exec ", cert_sync)
    assert tls_preflight < cert_sync < exec_teleop
    assert "xr_net_announce_url_only" not in source[cert_sync:exec_teleop]


def test_g1d_launcher_rejects_an_unreadable_tls_pair_before_teleimager_import():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cert = root / "cert.pem"
        key = root / "key.pem"
        cert.write_text("not a certificate", encoding="utf-8")
        key.write_text("not a key", encoding="utf-8")
        env = os.environ | {
            "XR_TELEOP_CERT": str(cert),
            "XR_TELEOP_KEY": str(key),
            "TELEIMAGER_PYTHON": "python3",
        }
        result = subprocess.run(
            ["bash", str(G1D_LAUNCHER)], env=env, capture_output=True, text=True, timeout=10
        )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "G1D TLS preflight failed" in output
    assert "G1D Teleimager preflight failed" not in output


def _fake_python(root):
    fake = root / "fakepy"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${2:-}" == teleop_hand_and_arm.py ]]; then printf "ARGV:%s\\n" "$*"; exit 0; fi\n'
        "cat >/dev/null; exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake


def _run_launcher(tmp_path, *args):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("x", encoding="utf-8")
    key.write_text("x", encoding="utf-8")
    env = os.environ | {
        "XR_TELEOP_CERT": str(cert),
        "XR_TELEOP_KEY": str(key),
        "TELEIMAGER_PYTHON": str(_fake_python(tmp_path)),
    }
    env.pop("G1D_LAUNCHER_SKIP_TELEOP", None)
    return subprocess.run(
        ["bash", str(G1D_LAUNCHER), *args], env=env, capture_output=True, text=True, timeout=20
    )


def test_g1d_launcher_forwards_extra_args_after_fixed_safety_flags(tmp_path):
    result = _run_launcher(tmp_path, "--display-mode", "pass-through", "--camera-scale", "0.3")
    assert result.returncode == 0, result.stderr
    argv = next(l for l in result.stdout.splitlines() if l.startswith("ARGV:"))
    assert "--arm G1D --g1d-ik-geometry-assumption --ee dex3 --input-mode hand" in argv
    assert argv.endswith("--display-mode pass-through --camera-scale 0.3")


def test_g1d_launcher_refuses_motion_and_profile_overrides_before_anything_else(tmp_path):
    for forbidden in (
        ["--motion"],
        ["--display-mode", "pass-through", "--motion"],
        ["--arm", "G1_29"],
        ["--arm=G1_29"],
        ["--ee=dex1"],
        ["--ee", "inspire"],
        ["--input-mode", "controller"],
        ["--input-mode=controller"],
        ["--network-interface", "eth0"],
        ["--network-interface=eth0"],
    ):
        result = _run_launcher(tmp_path, *forbidden)
        output = result.stdout + result.stderr
        assert result.returncode != 0, forbidden
        assert "ARGV:" not in output, forbidden
        assert "forwarded argument refused" in output, forbidden


def test_g1d_launcher_source_has_guard_and_appends_args_last():
    text = G1D_LAUNCHER.read_text(encoding="utf-8")
    assert text.index("forwarded argument refused") < text.index("xr_net_announce_url_only")
    assert text.rstrip().endswith('"$@"')
