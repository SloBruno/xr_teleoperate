#!/usr/bin/env bash
# G1D operator launcher: requires two remote camera streams; r/q stays in Python.
set -euo pipefail

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
preflight_remote_two_camera_streams
[[ "${G1D_LAUNCHER_SKIP_TELEOP:-0}" == 1 ]] && exit 0
cd "$repo/teleop"
echo "G1D arm-only: remote Teleimager $teleimager_host has two verified streams; press r to arm tracking; q to release and exit."
exec "$teleimager_python" -s teleop_hand_and_arm.py \
  --arm G1D \
  --g1d-ik-geometry-assumption \
  --ee dex3 \
  --input-mode hand \
  --camera-layout vertical \
  --img-server-ip "$teleimager_host" \
  --network-interface "$g1d_dds_interface"
