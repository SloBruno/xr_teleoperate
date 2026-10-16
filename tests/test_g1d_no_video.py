"""G1D no-video default: launcher skips camera preflight; teleop creates no ImageClient."""
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "teleop" / "run_g1d_quest_dex3.sh"
TELEOP = (ROOT / "teleop" / "teleop_hand_and_arm.py").read_text()


def _run(tmp_path, extra_env, *args):
    log = tmp_path / "py.log"
    fake = tmp_path / "fakepy"
    fake.write_text('#!/usr/bin/env bash\necho "ARGV: $*" >> "%s"\ncat >/dev/null 2>&1 <<<"" || true\nexit 0\n' % log)
    fake.chmod(0o755)
    cert = tmp_path / "c.pem"; cert.write_text("x")
    env = {**os.environ, "TELEIMAGER_PYTHON": str(fake), "XR_TELEOP_CERT": str(cert),
           "XR_TELEOP_KEY": str(cert), "G1D_LAUNCHER_SKIP_TELEOP": "1"}
    env.pop("G1D_VIDEO", None)
    env.update(extra_env)
    r = subprocess.run(["bash", str(LAUNCHER), *args], capture_output=True, text=True, env=env, cwd=tmp_path)
    return r, (log.read_text() if log.exists() else "")


def test_default_skips_camera_preflight_and_announces_it(tmp_path):
    r, log = _run(tmp_path, {})
    assert r.returncode == 0, r.stderr
    assert "ImageClient" not in log and "192.168.123.164" not in log
    assert "video disabled" in (r.stdout + r.stderr).lower()
    assert "G1D_VIDEO=1" in (r.stdout + r.stderr)


def test_video_env_restores_camera_preflight(tmp_path):
    r, log = _run(tmp_path, {"G1D_VIDEO": "1"})
    assert "video disabled" not in (r.stdout + r.stderr).lower()
    assert "192.168.123.164" in log  # preflight invoked with the host argv


def test_launcher_source_default_args_and_video_args():
    src = LAUNCHER.read_text()
    assert "--display-mode pass-through" in src and "--no-video" in src
    assert "--camera-layout vertical" in src and '--img-server-ip "$teleimager_host"' in src
    assert 'if [[ "${G1D_VIDEO:-0}" == 1 ]]' in src
    for fixed in ("--arm G1D", "--g1d-ik-geometry-assumption", "--ee dex3", "--input-mode hand",
                  '--network-interface "$g1d_dds_interface"'):
        assert fixed in src
    assert "--motion" not in re.sub(r"--motion\|--motion=\*", "", src).split("exec")[-1]


def test_motion_still_refused(tmp_path):
    r, _ = _run(tmp_path, {}, "--motion")
    assert r.returncode == 2


def test_teleop_has_no_video_flag_and_guards_imageclient():
    assert "'--no-video'" in TELEOP
    assert "if not args.no_video:" in TELEOP
    guarded = TELEOP.split("if not args.no_video:")[1]

    pre = TELEOP.split("if not args.no_video:")[0]
    assert "ImageClient(host=" not in pre.split("try:")[-1]
    assert "--record cannot be combined with --no-video" in TELEOP


def test_no_video_creates_wrapper_before_threads_and_never_derefs_none_config():
    try_pos = TELEOP.index("\n    try:\n        # HYPOTHESIS")
    early = TELEOP.index("tv_wrapper = TeleVuerWrapper(", try_pos)
    assert "args.no_video" in TELEOP[try_pos:early]
    assert early < TELEOP.index("ChannelFactoryInitialize(0", try_pos)
    assert "camera_config = None" in TELEOP
    # remaining unguarded uses live only in the --record block, which --no-video refuses
    for m in re.finditer(r"\n( *)if (camera_config\[[^\n]*):", TELEOP):
        assert len(m.group(1)) >= 20, m.group(0)  # only inside the --record block
    assert len(re.findall(r"if camera_config is not None and camera_config\[", TELEOP)) >= 4
