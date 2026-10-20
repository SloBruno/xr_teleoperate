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


def test_head_video_mode_uses_head_only_preflight(tmp_path):
    r, log = _run(tmp_path, {"G1D_VIDEO_MODE": "head"})
    assert r.returncode == 0, r.stderr
    assert "192.168.123.164 head" in log

    src = LAUNCHER.read_text()
    head_probe = src.split("preflight_remote_head_stream()", 1)[1].split("\n}\n", 1)[0]
    assert "60000" in head_probe and "55555" in head_probe
    assert "head_camera" in head_probe and "get_head_frame" in head_probe
    assert 'camera_names=("head_camera",)' in head_probe
    assert "binocular" in head_probe
    assert "55556" not in head_probe
    assert "left_wrist" not in head_probe and "right_wrist" not in head_probe


def test_head_video_mode_has_safe_lightweight_fixed_args_and_legacy_defaults():
    src = LAUNCHER.read_text()
    assert 'G1D_VIDEO_MODE=${G1D_VIDEO_MODE:-}' in src
    assert 'G1D_HEAD_VIDEO_FPS=${G1D_HEAD_VIDEO_FPS:-10}' in src
    assert 'G1D_HEAD_JPEG_QUALITY=${G1D_HEAD_JPEG_QUALITY:-60}' in src
    assert "--camera-layout head" in src
    assert "--force-zmq-video" in src
    assert '--video-fps "$G1D_HEAD_VIDEO_FPS"' in src
    assert '--jpeg-quality "$G1D_HEAD_JPEG_QUALITY"' in src
    assert "--camera-layout vertical" in src  # G1D_VIDEO=1 compatibility
    assert 'if [[ "${G1D_VIDEO:-0}" == 1 ]]' in src


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


def test_teleop_integrates_nonblocking_video_policy_and_quality_controls():
    assert "'--force-zmq-video'" in TELEOP
    assert "'--video-fps'" in TELEOP
    assert "'--jpeg-quality'" in TELEOP
    assert "VideoFrameGate" in TELEOP
    assert "service_xr_video(" in TELEOP
    assert "force_zmq=args.force_zmq_video" in TELEOP
    assert "display_fps=args.video_fps or 30.0" in TELEOP
    assert "jpeg_quality=args.jpeg_quality" in TELEOP
    assert 'selected_cameras = ("head_camera",) if args.camera_layout == "head" else None' in TELEOP
    assert "camera_names=selected_cameras" in TELEOP
    assert "validate_head_only_camera(camera_config['head_camera'])" in TELEOP
    assert "args.camera_layout != 'head' or video_update.frame_fresh" in TELEOP


def test_no_video_creates_wrapper_before_threads_and_never_derefs_none_config():
    try_pos = TELEOP.index("\n    try:\n        # TeleVuer forks its Vuer server Process")
    early = TELEOP.index("tv_wrapper = TeleVuerWrapper(", try_pos)
    assert "args.no_video" in TELEOP[try_pos:early]
    assert early < TELEOP.index("ChannelFactoryInitialize(0", try_pos)
    assert "camera_config = None" in TELEOP
    # Camera access is centralized and both ready/tracking calls are guarded.
    assert TELEOP.count("if camera_config is not None:\n                video_update = service_xr_video(") == 2
