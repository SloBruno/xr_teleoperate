#!/usr/bin/env bash
# G1D operator launcher: video defaults off. G1D_VIDEO=1 keeps the legacy
# two-camera stack; G1D_VIDEO_MODE=head selects lightweight binocular ZMQ.
set -euo pipefail

# Extra arguments are forwarded to teleop (appended after the fixed safety flags).
# Refuse anything that could change the validated G1D profile or enable motion.
for g1d_extra_arg in "$@"; do
    case "$g1d_extra_arg" in
        --motion|--motion=*|--g1d-a|--g1d-a=*|--g1d-ag|--g1d-ag=*|--g1d-agv|--g1d-agv=*|--arm|--arm=*|--ee|--ee=*|--input-mode|--input-mode=*|--network-interface|--network-interface=*|--camera-layout|--camera-layout=*|--force-zmq-video|--video-fps|--video-fps=*|--jpeg-quality|--jpeg-quality=*)
            echo "G1D launcher: forwarded argument refused: $g1d_extra_arg (safety/profile flags are fixed; use G1D_VIDEO_MODE, G1D_HEAD_* and G1D_AGV settings)." >&2
            exit 2
            ;;
    esac
done

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/home/unitree/cyclonedds/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# Override only when the Jetson's robot-facing DDS NIC is different.
G1D_DDS_INTERFACE=${G1D_DDS_INTERFACE:-enP2p1s0}
g1d_dds_interface=$G1D_DDS_INTERFACE
printf -v CYCLONEDDS_URI '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="%s"/></Interfaces></General></Domain></CycloneDDS>' "$g1d_dds_interface"
export CYCLONEDDS_URI
export PYTHONPATH=$repo:$repo/teleop/televuer/src:$repo/teleop/teleimager/src:$repo/teleop/robot_control/dex-retargeting/src:/home/unitree/unitree_sdk2_python${PYTHONPATH:+:${PYTHONPATH}}
export XR_TELEOP_CERT=${XR_TELEOP_CERT:-/home/unitree/.config/xr_teleoperate/cert.pem}
export XR_TELEOP_KEY=${XR_TELEOP_KEY:-/home/unitree/.config/xr_teleoperate/key.pem}

# Teleimager is already hosted on the robot-side PC. This launcher never starts,
# stops, detects USB devices for, or otherwise manages that remote service.
G1D_TELEIMAGER_HOST=${G1D_TELEIMAGER_HOST:-192.168.123.164}
teleimager_host=$G1D_TELEIMAGER_HOST
teleimager_python=${TELEIMAGER_PYTHON:-/home/unitree/.local/envs/xr-g1d/bin/python}
G1D_TELEIMAGER_PREFLIGHT_TIMEOUT_S=${G1D_TELEIMAGER_PREFLIGHT_TIMEOUT_S:-12}
G1D_VIDEO_MODE=${G1D_VIDEO_MODE:-}
G1D_HEAD_VIDEO_FPS=${G1D_HEAD_VIDEO_FPS:-10}
G1D_HEAD_JPEG_QUALITY=${G1D_HEAD_JPEG_QUALITY:-60}

if [[ -z "$G1D_VIDEO_MODE" ]]; then
    if [[ "${G1D_VIDEO:-0}" == 1 ]]; then
        g1d_video_mode=vertical
    else
        g1d_video_mode=off
    fi
else
    g1d_video_mode=$G1D_VIDEO_MODE
fi
case "$g1d_video_mode" in
    off|head|vertical) ;;
    *) echo "G1D launcher: G1D_VIDEO_MODE must be off, head, or vertical." >&2; exit 2 ;;
esac
if [[ "$g1d_video_mode" == head ]]; then
    if [[ ! "$G1D_HEAD_VIDEO_FPS" =~ ^([1-9]|10)(\.0*)?$ ]]; then
        echo "G1D launcher: G1D_HEAD_VIDEO_FPS must be between 1 and 10." >&2
        exit 2
    fi
    if [[ ! "$G1D_HEAD_JPEG_QUALITY" =~ ^([1-9]|[1-9][0-9]|100)$ ]]; then
        echo "G1D launcher: G1D_HEAD_JPEG_QUALITY must be an integer from 1 to 100." >&2
        exit 2
    fi
fi

# Opt-in G1-D AGV base (left stick) + lift column (right stick). Only this
# environment switch adds --g1d-agv; the flag itself is never user-forwardable.
G1D_AGV=${G1D_AGV:-0}
G1D_AGV_MAX_VX=${G1D_AGV_MAX_VX:-0.6}
g1d_agv_args=()
case "$G1D_AGV" in
    0|"") ;;
    1)
        if [[ ! "$G1D_AGV_MAX_VX" =~ ^([0-9]+(\.[0-9]*)?|\.[0-9]+)$ ]] \
            || ! awk -v v="$G1D_AGV_MAX_VX" 'BEGIN { exit !(v + 0 > 0 && v + 0 <= 1.5) }'; then
            echo "G1D launcher: G1D_AGV_MAX_VX must be a number in (0, 1.5] m/s." >&2
            exit 2
        fi
        export G1D_AGV G1D_AGV_MAX_VX
        g1d_agv_args=(--g1d-agv)
        echo "G1D AGV ENABLED: left stick = base (G1D_AGV_MAX_VX=$G1D_AGV_MAX_VX m/s, turn 0.6 rad/s), right stick = lift column; active only after r."
        ;;
    *) echo "G1D launcher: G1D_AGV must be 0 or 1." >&2; exit 2 ;;
esac

# Print the Quest URL before any Python/DDS preflight.  Unlike the G1 helper,
# this only observes routing and never regenerates TLS material or writes state.
# shellcheck source=lib/vuer_network.sh
source "$repo/teleop/lib/vuer_network.sh"
xr_net_announce_url_only

preflight_tls() {
    if [[ ! -r "$XR_TELEOP_CERT" || ! -r "$XR_TELEOP_KEY" ]]; then
        echo "G1D TLS preflight failed: certificate/key must exist and be readable; refusing to import teleop or DDS." >&2
        return 1
    fi
    if ! "$teleimager_python" -s - "$XR_TELEOP_CERT" "$XR_TELEOP_KEY" <<'PY'
import ssl
import sys

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(sys.argv[1], sys.argv[2])
PY
    then
        echo "G1D TLS preflight failed: certificate/key cannot be loaded; refusing to import teleop or DDS." >&2
        return 1
    fi
}

preflight_remote_head_stream() {
    if ! timeout -k 2 "$G1D_TELEIMAGER_PREFLIGHT_TIMEOUT_S" "$teleimager_python" -s - "$teleimager_host" head <<'PY'
import os
import socket
import sys
import time

from teleimager.image_client import ImageClient

host = sys.argv[1]
for port in (60000, 55555):
    with socket.create_connection((host, port), timeout=1):
        pass
client = ImageClient(host=host, request_bgr=True, camera_names=("head_camera",))
config = client.get_cam_config()
head_config = config.get("head_camera")
if not isinstance(head_config, dict):
    raise RuntimeError("missing configured camera path: head_camera")
if head_config.get("enable_zmq") is not True or head_config.get("zmq_port") != 55555:
    raise RuntimeError("head_camera must publish ZMQ on TCP 55555")
if head_config.get("binocular") is not True:
    raise RuntimeError("head_camera must declare binocular=true")
image_shape = head_config.get("image_shape")
if not (isinstance(image_shape, (list, tuple)) and len(image_shape) == 2):
    raise RuntimeError("head_camera.image_shape must declare [height, width]")
expected_shape = (int(image_shape[0]), int(image_shape[1]), 3)
deadline = time.monotonic() + 6.0
while time.monotonic() < deadline:
    head = client.get_head_frame()
    bgr = getattr(head, "bgr", None)
    if bgr is not None and getattr(bgr, "shape", None) == expected_shape and bgr.nbytes > 0:
        # ImageClient owns background ZMQ threads; avoid teardown aborts.
        os._exit(0)
    time.sleep(0.05)
os._exit(2)
PY
    then
        echo "G1D head Teleimager preflight failed for remote host $teleimager_host; refusing to start." >&2
        return 1
    fi
}

preflight_remote_two_camera_streams() {
    if ! timeout -k 2 "$G1D_TELEIMAGER_PREFLIGHT_TIMEOUT_S" "$teleimager_python" -s - "$teleimager_host" <<'PY'
import os
import socket
import sys
import time

from teleimager.image_client import ImageClient

host = sys.argv[1]
required = (("head_camera", 55555), ("left_wrist_camera", 55556))
for port in (60000, 55555, 55556):
    with socket.create_connection((host, port), timeout=1):
        pass
client = ImageClient(host=host, request_bgr=True)
config = client.get_cam_config()
for name, port in required:
    camera = config.get(name)
    if not isinstance(camera, dict):
        raise RuntimeError(f"missing configured camera path: {name}")
    if camera.get("enable_zmq") is not True or camera.get("zmq_port") != port:
        raise RuntimeError(f"{name} must publish ZMQ on TCP {port}")
    if not isinstance(camera.get("binocular"), bool):
        raise RuntimeError(f"{name}.binocular must be declared in the remote config")
deadline = time.monotonic() + 6.0
while time.monotonic() < deadline:
    frames = (client.get_head_frame(), client.get_left_wrist_frame())
    if all(
        (bgr := getattr(frame, "bgr", None)) is not None
        and getattr(bgr, "ndim", 0) == 3
        and bgr.shape[2] == 3
        and bgr.nbytes > 0
        for frame in frames
    ):
        # ImageClient owns background ZMQ threads; normal interpreter teardown
        # can abort. The preflight is read-only, so exit directly after success.
        os._exit(0)
    time.sleep(0.05)
os._exit(2)
PY
    then
        echo "G1D Teleimager preflight failed for remote host $teleimager_host; refusing to start." >&2
        return 1
    fi
}

preflight_tls
if [[ "$g1d_video_mode" == head ]]; then
    preflight_remote_head_stream
    [[ "${G1D_LAUNCHER_SKIP_TELEOP:-0}" == 1 ]] && exit 0
    xr_net_announce
    cd "$repo/teleop"
    echo "G1D HEAD-ONLY BINOCULAR ZMQ: $G1D_HEAD_VIDEO_FPS Hz, JPEG quality $G1D_HEAD_JPEG_QUALITY; no wrist streams."
    echo "G1D arm-only: press r or right A to arm tracking; right B or q to release and exit."
    exec "$teleimager_python" -s teleop_hand_and_arm.py \
      --arm G1D \
      --g1d-ik-geometry-assumption \
      --ee dex3 \
      --input-mode hand \
      --camera-layout head \
      --force-zmq-video \
      --video-fps "$G1D_HEAD_VIDEO_FPS" \
      --jpeg-quality "$G1D_HEAD_JPEG_QUALITY" \
      --img-server-ip "$teleimager_host" \
      --network-interface "$g1d_dds_interface" \
      ${g1d_agv_args[@]+"${g1d_agv_args[@]}"} \
      "$@"
fi
if [[ "$g1d_video_mode" == vertical ]]; then
    preflight_remote_two_camera_streams
    [[ "${G1D_LAUNCHER_SKIP_TELEOP:-0}" == 1 ]] && exit 0
    xr_net_announce
    cd "$repo/teleop"
    echo "G1D arm-only: remote Teleimager $teleimager_host has two verified streams; press r to arm tracking; q to release and exit."
    exec "$teleimager_python" -s teleop_hand_and_arm.py \
      --arm G1D \
      --g1d-ik-geometry-assumption \
      --ee dex3 \
      --input-mode hand \
      --camera-layout vertical \
      --img-server-ip "$teleimager_host" \
      --network-interface "$g1d_dds_interface" \
      ${g1d_agv_args[@]+"${g1d_agv_args[@]}"} \
      "$@"
fi

echo "G1D video DISABLED (pass-through, no camera preflight, no ImageClient); set G1D_VIDEO_MODE=head for lightweight binocular ZMQ or G1D_VIDEO=1 for the legacy two-camera flow."
[[ "${G1D_LAUNCHER_SKIP_TELEOP:-0}" == 1 ]] && exit 0
xr_net_announce
cd "$repo/teleop"
echo "G1D arm-only (no video): press r or right A to arm tracking; right B or q to release and exit."
exec "$teleimager_python" -s teleop_hand_and_arm.py \
  --arm G1D \
  --g1d-ik-geometry-assumption \
  --ee dex3 \
  --input-mode hand \
  --display-mode pass-through \
  --no-video \
  --network-interface "$g1d_dds_interface" \
  ${g1d_agv_args[@]+"${g1d_agv_args[@]}"} \
  "$@"
