#!/usr/bin/env bash
# G1D operator launcher: two required physical camera streams; terminal r/q stays in Python.
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

teleimager_dir="$repo/teleop/teleimager"
teleimager_python=${TELEIMAGER_PYTHON:-/home/unitree/miniconda3/envs/tv/bin/python}
teleimager_host=${TELEIMAGER_HOST:-127.0.0.1}
teleimager_state_dir=${G1D_TELEIMAGER_STATE_DIR:-/home/unitree/.local/state/xr_teleoperate/g1d}
teleimager_pid_file="$teleimager_state_dir/teleimager.pid"
teleimager_log="$teleimager_state_dir/teleimager.log"
TELEIMAGER_TIMEOUT_S=${TELEIMAGER_TIMEOUT_S:-30}
TELEIMAGER_DETECT_TIMEOUT_S=${TELEIMAGER_DETECT_TIMEOUT_S:-15}
G1D_HEAD_CAMERA_SERIAL=${G1D_HEAD_CAMERA_SERIAL:-243122072230}
G1D_LEFT_WRIST_CAMERA_SERIAL=${G1D_LEFT_WRIST_CAMERA_SERIAL:-233622070789}
# This profile intentionally has no auto/head/any/single camera mode.
TELEIMAGER_CAMERA_MODE=both
export TELEIMAGER_CAMERA_MODE

fail_closed() {
    echo "G1D Teleimager preflight: $*; refusing to start." >&2
    exit 1
}

require_two_physical_cameras() {
    local detected rc=0 detected_source detected_serial detected_count
    detected=$(cd "$repo" && timeout -k 2 "$TELEIMAGER_DETECT_TIMEOUT_S" setsid "$teleimager_python" -s -m teleop.utils.teleimager_head_only_server --detect 9>&-) || rc=$?
    (( rc == 0 )) || fail_closed "physical-camera detection failed (pyrealsense2 rc=$rc, timeout ${TELEIMAGER_DETECT_TIMEOUT_S}s)"
    read -r detected_source detected_serial detected_count <<<"$detected"
    [[ "$detected_source" == "head" ]] || fail_closed "head camera is absent or the detected primary stream is not head"
    [[ "$detected_serial" == "$G1D_HEAD_CAMERA_SERIAL" ]] || fail_closed "unexpected head serial '$detected_serial' (expected '$G1D_HEAD_CAMERA_SERIAL')"
    [[ "$detected_count" == "both" ]] || fail_closed "both required physical cameras are not present (head '$G1D_HEAD_CAMERA_SERIAL', left wrist '$G1D_LEFT_WRIST_CAMERA_SERIAL')"
}

teleimager_is_healthy() {
    local probe_pid rc=2 deadline=$((SECONDS + 9))
    setsid "$teleimager_python" -s - "$teleimager_host" "$G1D_HEAD_CAMERA_SERIAL" "$G1D_LEFT_WRIST_CAMERA_SERIAL" 9>&- <<'PY' &
import os
import socket
import sys
import time
from teleimager.image_client import ImageClient

host, head_serial, wrist_serial = sys.argv[1:]
for port in (60000, 55555, 55556):
    with socket.create_connection((host, port), timeout=1):
        pass
client = ImageClient(host=host, request_bgr=True)
config = client.get_cam_config()
for name, serial in (("head_camera", head_serial), ("left_wrist_camera", wrist_serial)):
    camera = config.get(name)
    if not isinstance(camera, dict) or not camera.get("enable_zmq"):
        raise RuntimeError(f"{name}.enable_zmq must be true")
    if str(camera.get("serial_number")) != serial:
        raise RuntimeError(f"{name}.serial_number does not match required physical camera")
deadline = time.monotonic() + 6.0
while time.monotonic() < deadline:
    frames = (client.get_head_frame(), client.get_left_wrist_frame())
    if all((bgr := getattr(frame, "bgr", None)) is not None and getattr(bgr, "nbytes", 0) > 0 for frame in frames):
        os._exit(0)
    time.sleep(0.05)
os._exit(2)
PY
    probe_pid=$!
    while kill -0 "$probe_pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.1; done
    if kill -0 "$probe_pid" 2>/dev/null; then rc=124; else wait "$probe_pid" && rc=0 || rc=$?; fi
    kill -KILL -- "-$probe_pid" 2>/dev/null || true
    wait "$probe_pid" 2>/dev/null || true
    return "$rc"
}

ensure_two_camera_teleimager() {
    mkdir -p "$teleimager_state_dir"
    if [[ -s "$teleimager_pid_file" ]] && kill -0 "$(<"$teleimager_pid_file")" 2>/dev/null; then
        teleimager_is_healthy && return 0
        fail_closed "known Teleimager PID $(<"$teleimager_pid_file") is unhealthy; stop it explicitly before retrying"
    fi
    if teleimager_is_healthy; then
        echo "G1D Teleimager: reusing healthy two-camera service."
        return 0
    fi
    echo "G1D Teleimager: starting only after two-camera preflight."
    (
        cd "$teleimager_dir"
        setsid nohup "$teleimager_python" -s -m teleimager.image_server --rs --no-affinity >>"$teleimager_log" 2>&1 < /dev/null 9>&- &
        echo "$!" >"$teleimager_pid_file"
    )
    local deadline=$((SECONDS + TELEIMAGER_TIMEOUT_S))
    while (( SECONDS < deadline )); do
        teleimager_is_healthy && return 0
        [[ -s "$teleimager_pid_file" ]] && ! kill -0 "$(<"$teleimager_pid_file")" 2>/dev/null && break
        sleep 1
    done
    fail_closed "Teleimager did not expose both configured camera streams within ${TELEIMAGER_TIMEOUT_S}s; see $teleimager_log"
}

require_two_physical_cameras
ensure_two_camera_teleimager
[[ "${G1D_LAUNCHER_SKIP_TELEOP:-0}" == 1 ]] && exit 0
cd "$repo/teleop"
echo "G1D arm-only: press r to arm tracking; q to release and exit."
exec "$teleimager_python" -s teleop_hand_and_arm.py \
  --arm G1D \
  --g1d-ik-geometry-assumption \
  --ee dex3 \
  --input-mode hand \
  --camera-layout vertical \
  --network-interface "$g1d_dds_interface"
