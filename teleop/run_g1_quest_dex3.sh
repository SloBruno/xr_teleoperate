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
teleimager_state_dir=${TELEIMAGER_STATE_DIR:-/home/unitree/.local/state/xr_teleoperate}
teleimager_pid_file="$teleimager_state_dir/teleimager.pid"
teleimager_log="$teleimager_state_dir/teleimager.log"
teleimager_lock="$teleimager_state_dir/teleimager.lock"
teleimager_host=${TELEIMAGER_HOST:-127.0.0.1}
TELEIMAGER_TIMEOUT_S=${TELEIMAGER_TIMEOUT_S:-30}
TELEIMAGER_LOCK_TIMEOUT_S=${TELEIMAGER_LOCK_TIMEOUT_S:-10}
teleimager_python=${TELEIMAGER_PYTHON:-/home/unitree/miniconda3/envs/tv/bin/python}
# Camera mode. Default "auto": detect the connected RealSense cameras
# (pyrealsense2, read-only) and pick the mode by itself:
#   2 cameras (head + left wrist) -> "both"  (layout vertical)
#   1 camera (either one)         -> "any"   (published as head_camera, layout head)
#   0 cameras / detection error   -> clear failure, nothing is started
# Explicit overrides keep working: both | head | any (alias "single").
#   TELEIMAGER_CAMERA_MODE=any bash teleop/run_g1_quest_dex3.sh
TELEIMAGER_CAMERA_MODE=${TELEIMAGER_CAMERA_MODE:-auto}
TELEIMAGER_DETECT_TIMEOUT_S=${TELEIMAGER_DETECT_TIMEOUT_S:-15}
teleimager_mode_file="$teleimager_state_dir/teleimager.mode"
teleimager_source_file="$teleimager_state_dir/teleimager.source"
# Derived head-only server config lives in state, never inside the submodule.
export TELEIMAGER_HEAD_ONLY_CONFIG="$teleimager_state_dir/cam_config_server.head_only.yaml"
[[ "$TELEIMAGER_CAMERA_MODE" == single ]] && TELEIMAGER_CAMERA_MODE=any
case "$TELEIMAGER_CAMERA_MODE" in
    auto|both|head|any) ;;
    *)
        echo "unsupported TELEIMAGER_CAMERA_MODE='$TELEIMAGER_CAMERA_MODE' (use auto|both|head|any|single)" >&2
        exit 2
        ;;
esac
export TELEIMAGER_CAMERA_SOURCE=head

# Detection runs in its own session with a short timeout and never inherits
# the launcher lock (FD 9). Any failure (exception, timeout, import error) is a
# clear failure: no guessing.
detect_cameras() {
    local rc=0
    detected=$(cd "$repo" && timeout -k 2 "$TELEIMAGER_DETECT_TIMEOUT_S" setsid "$teleimager_python" -s -m teleop.utils.teleimager_head_only_server --detect 9>&-) || rc=$?
    if (( rc == 3 )); then
        echo "TELEIMAGER: nenhuma câmera conectada (cabeça 243122072230 / pulso 233622070789); não iniciando." >&2
        exit 3
    elif (( rc != 0 )); then
        echo "TELEIMAGER: falha ao detectar câmeras (pyrealsense2 rc=$rc, timeout ${TELEIMAGER_DETECT_TIMEOUT_S}s); não iniciando." >&2
        exit 4
    fi
    read -r any_source any_serial any_count <<<"$detected"
    if [[ -z "${any_source:-}" || -z "${any_serial:-}" || -z "${any_count:-}" ]]; then
        echo "TELEIMAGER: saída de detecção inválida ('$detected'); não iniciando." >&2
        exit 4
    fi
}

if [[ "$TELEIMAGER_CAMERA_MODE" == auto ]]; then
    detect_cameras
    if [[ "$any_count" == both ]]; then
        TELEIMAGER_CAMERA_MODE=both
    else
        TELEIMAGER_CAMERA_MODE=any
    fi
    echo "TELEIMAGER: detecção automática -> modo $TELEIMAGER_CAMERA_MODE"
elif [[ "$TELEIMAGER_CAMERA_MODE" == any ]]; then
    detect_cameras
fi
export TELEIMAGER_CAMERA_MODE
teleimager_source=""
if [[ "$TELEIMAGER_CAMERA_MODE" == head ]]; then
    teleop_camera_layout=head
    echo "================================================================"
    echo "TELEIMAGER: modo SOMENTE CABEÇA (pulso esquerdo desativado)"
    echo "================================================================"
elif [[ "$TELEIMAGER_CAMERA_MODE" == any ]]; then
    teleop_camera_layout=head
    export TELEIMAGER_CAMERA_SOURCE="$any_source"
    teleimager_source="$any_source $any_serial"
    if [[ "$any_source" == left_wrist ]]; then any_label="pulso esquerdo"; else any_label="cabeça"; fi
    echo "================================================================"
    echo "TELEIMAGER: modo CÂMERA ÚNICA ($any_label serial $any_serial publicada como imagem principal)"
    if [[ "$any_count" == both ]]; then
        echo "AVISO: cabeça e pulso conectados; modo any usa apenas a cabeça."
    fi
    if [[ "$any_source" == left_wrist ]]; then
        echo "AVISO: a imagem principal vem da câmera do PULSO esquerdo — o ponto de vista é o da mão, não o da cabeça, e isso pode confundir a teleoperação."
    fi
    echo "================================================================"
else
    teleop_camera_layout=vertical
    echo "TELEIMAGER: modo DUAS CÂMERAS (cabeça + pulso esquerdo, layout vertical)"
fi

teleop_is_running() {
    pgrep -f 'python.*[t]eleop_hand_and_arm\.py' >/dev/null 2>&1
}

# Stop the known Teleimager PID (SIGTERM, bounded wait) so it can be restarted
# in the mode matching the cameras connected now. Only when no teleop runs.
stop_teleimager_for_restart() {
    local pid=$1 reason=$2
    if teleop_is_running; then
        echo "Teleimager PID $pid precisa reiniciar ($reason), mas há teleop_hand_and_arm.py em execução; recusando e não mexendo em nada. Encerre o teleop primeiro." >&2
        return 1
    fi
    if ! tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q teleimager; then
        echo "PID $pid do arquivo $teleimager_pid_file não parece ser o Teleimager; recusando matar." >&2
        return 1
    fi
    echo "Reiniciando Teleimager PID $pid ($reason): SIGTERM e aguardando."
    kill -TERM "$pid" 2>/dev/null || true
    local stop_deadline=$((SECONDS + ${TELEIMAGER_STOP_TIMEOUT_S:-10}))
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < stop_deadline )); do
        sleep 0.2
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "Teleimager PID $pid não encerrou após SIGTERM; recusando prosseguir." >&2
        return 1
    fi
    rm -f "$teleimager_mode_file" "$teleimager_source_file" "$teleimager_pid_file"
}

teleimager_is_healthy() {
    # The probe runs in its own session and never inherits the launcher lock
    # (FD 9). Every process it forks (e.g. logging helpers) is killed with its
    # process group, so an orphan can never hold the lock and freeze the next
    # launcher.
    local probe_pid rc=2 deadline=$((SECONDS + 9))
    setsid "$teleimager_python" -s - "$teleimager_host" "$TELEIMAGER_CAMERA_MODE" 9>&- <<'PY' &
import os
import socket
import sys
import time

from teleimager.image_client import ImageClient

host = sys.argv[1]
mode = sys.argv[2]
head_only = mode == "head" or mode == "any"
ports = (60000, 55555) if head_only else (60000, 55555, 55556)
for port in ports:
    with socket.create_connection((host, port), timeout=1):
        pass

client = ImageClient(host=host, request_bgr=True)
client.get_cam_config()
deadline = time.monotonic() + 6.0
while time.monotonic() < deadline:
    head = client.get_head_frame()
    # Head-only mode never touches the (disabled) wrist camera.
    left_wrist = None if head_only else client.get_left_wrist_frame()
    frames = (head,) if head_only else (head, left_wrist)
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
    probe_pid=$!
    while kill -0 "$probe_pid" 2>/dev/null && (( SECONDS < deadline )); do
        sleep 0.1
    done
    if kill -0 "$probe_pid" 2>/dev/null; then
        rc=124
    else
        wait "$probe_pid" && rc=0 || rc=$?
    fi
    kill -KILL -- "-$probe_pid" 2>/dev/null || true
    wait "$probe_pid" 2>/dev/null || true
    return "$rc"
}

ensure_teleimager() {
    mkdir -p "$teleimager_state_dir"
    exec 9>"$teleimager_lock"
    if ! flock -w "$TELEIMAGER_LOCK_TIMEOUT_S" 9; then
        echo "could not acquire Teleimager lock within ${TELEIMAGER_LOCK_TIMEOUT_S}s: $teleimager_lock" >&2
        echo "held by: $(fuser "$teleimager_lock" 2>/dev/null || echo unknown)" >&2
        return 1
    fi

    local running_mode=both running_source="" running_pid="" restart_reason=""
    [[ -s "$teleimager_mode_file" ]] && running_mode=$(<"$teleimager_mode_file")
    [[ -s "$teleimager_source_file" ]] && running_source=$(<"$teleimager_source_file")
    [[ -s "$teleimager_pid_file" ]] && running_pid=$(<"$teleimager_pid_file")
    if [[ -n "$running_pid" ]] && kill -0 "$running_pid" 2>/dev/null; then
        if [[ "$running_mode" != "$TELEIMAGER_CAMERA_MODE" ]]; then
            restart_reason="running in mode $running_mode, câmeras agora pedem $TELEIMAGER_CAMERA_MODE"
        elif [[ -n "$teleimager_source" && -n "$running_source" && "$running_source" != "$teleimager_source" ]]; then
            restart_reason="câmera mudou de '$running_source' para '$teleimager_source'"
        fi
        if [[ -n "$restart_reason" ]]; then
            stop_teleimager_for_restart "$running_pid" "$restart_reason" || return 1
        elif teleimager_is_healthy; then
            echo "Reusing healthy Teleimager (mode $TELEIMAGER_CAMERA_MODE)."
            return 0
        else
            echo "Teleimager PID $running_pid is running but unhealthy; refusing a duplicate start." >&2
            return 1
        fi
    elif teleimager_is_healthy; then
        echo "Reusing healthy Teleimager (mode $TELEIMAGER_CAMERA_MODE)."
        return 0
    fi

    echo "Starting Teleimager from $teleimager_dir (mode $TELEIMAGER_CAMERA_MODE)."
    local server_module=teleimager.image_server
    if [[ "$TELEIMAGER_CAMERA_MODE" == head || "$TELEIMAGER_CAMERA_MODE" == any ]]; then
        server_module=teleop.utils.teleimager_head_only_server
    fi
    echo "$TELEIMAGER_CAMERA_MODE" >"$teleimager_mode_file"
    if [[ -n "$teleimager_source" ]]; then echo "$teleimager_source" >"$teleimager_source_file"; else rm -f "$teleimager_source_file"; fi
    (
        cd "$teleimager_dir"
        setsid nohup "$teleimager_python" -s -m "$server_module" --rs --no-affinity \
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

if ! ensure_teleimager; then
    exec 9>&-
    exit 1
fi
exec 9>&-
if [[ "${G1_LAUNCHER_SKIP_TELEOP:-0}" == 1 ]]; then
    exit 0
fi
cd "$repo/teleop"
# Optional XR video plane (unset = historical 1.0 m at 1.0 m). Ex.: XR_VIDEO_PLANE_HEIGHT=auto
video_plane_args=()
[[ -n "${XR_VIDEO_PLANE_HEIGHT:-}" ]] && video_plane_args+=(--video-plane-height "$XR_VIDEO_PLANE_HEIGHT")
[[ -n "${XR_VIDEO_PLANE_DISTANCE:-}" ]] && video_plane_args+=(--video-plane-distance "$XR_VIDEO_PLANE_DISTANCE")
exec "$teleimager_python" -s teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex3 \
  --input-mode hand \
  --motion \
  --camera-layout "$teleop_camera_layout" \
  ${video_plane_args[@]+"${video_plane_args[@]}"}
