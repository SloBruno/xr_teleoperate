#!/usr/bin/env bash
# Operator-run launcher: this process waits for local terminal r/q.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/home/unitree/cyclonedds/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# The Dex3 boards publish on the robot's internal Ethernet bus. Explicitly pin
# Cyclone DDS here so Wi-Fi/Tailscale never becomes the discovery interface.
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="enP8p1s0"/></Interfaces></General></Domain></CycloneDDS>'
export PYTHONPATH=$repo:$repo/teleop/televuer/src:$repo/teleop/teleimager/src:$repo/teleop/robot_control/dex-retargeting/src:/home/unitree/unitree_sdk2_python${PYTHONPATH:+:${PYTHONPATH}}
export XR_TELEOP_CERT=/home/unitree/.config/xr_teleoperate/cert.pem
export XR_TELEOP_KEY=/home/unitree/.config/xr_teleoperate/key.pem

teleimager_dir="$repo/teleop/teleimager"
teleimager_state_dir=/home/unitree/.local/state/xr_teleoperate
teleimager_pid_file="$teleimager_state_dir/teleimager.pid"
teleimager_log="$teleimager_state_dir/teleimager.log"
teleimager_lock="$teleimager_state_dir/teleimager.lock"
teleimager_host=${TELEIMAGER_HOST:-127.0.0.1}
TELEIMAGER_TIMEOUT_S=${TELEIMAGER_TIMEOUT_S:-30}
teleimager_python=/home/unitree/miniconda3/envs/tv/bin/python

teleimager_is_healthy() {
    timeout 8 "$teleimager_python" -s - "$teleimager_host" <<'PY'
import os
import socket
import sys
import time

from teleimager.image_client import ImageClient

host = sys.argv[1]
for port in (60000, 55555, 55556):
    with socket.create_connection((host, port), timeout=1):
        pass

client = ImageClient(host=host, request_bgr=True)
client.get_cam_config()
deadline = time.monotonic() + 6.0
while time.monotonic() < deadline:
    head = client.get_head_frame()
    left_wrist = client.get_left_wrist_frame()
    frames = (head, left_wrist)
    if all(
        (bgr := getattr(frame, "bgr", None)) is not None
        and getattr(bgr, "shape", None) == (720, 1280, 3)
        and bgr.nbytes > 0
        for frame in frames
    ):
        # ImageClient owns background ZMQ threads that can abort during normal
        # interpreter teardown. Exit directly after the read-only probe.
        os._exit(0)
    time.sleep(0.05)
os._exit(2)
PY
}

ensure_teleimager() {
    mkdir -p "$teleimager_state_dir"
    exec 9>"$teleimager_lock"
    flock 9

    if teleimager_is_healthy; then
        echo "Reusing healthy Teleimager on ports 60000, 55555, 55556."
        return 0
    fi

    if [[ -s "$teleimager_pid_file" ]]; then
        local pid
        pid=$(<"$teleimager_pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Teleimager PID $pid is running but unhealthy; refusing a duplicate start." >&2
            return 1
        fi
    fi

    echo "Starting Teleimager from $teleimager_dir."
    (
        cd "$teleimager_dir"
        setsid nohup "$teleimager_python" -s -m teleimager.image_server --rs --no-affinity \
            >>"$teleimager_log" 2>&1 < /dev/null 9>&- &
        echo "$!" >"$teleimager_pid_file"
    )

    local deadline=$((SECONDS + TELEIMAGER_TIMEOUT_S))
    while (( SECONDS < deadline )); do
        if teleimager_is_healthy; then
            echo "Teleimager is healthy."
            return 0
        fi
        if [[ -s "$teleimager_pid_file" ]] && ! kill -0 "$(<"$teleimager_pid_file")" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    echo "teleimager did not become healthy within ${TELEIMAGER_TIMEOUT_S}s; see $teleimager_log." >&2
    return 1
}

ensure_teleimager
exec 9>&-
cd "$repo/teleop"
exec "$teleimager_python" -s teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex3 \
  --input-mode hand \
  --motion \
  --camera-layout vertical
