#!/usr/bin/env bash
# Operator-run launcher: this process waits for local terminal r/q.
#
# Torso lean (waist PITCH/ROLL from the operator's head DISPLACEMENT, via
# rt/arm_sdk; yaw stays at the neutral; see docs/torso_lean.md). Default OFF:
#   G1_TORSO_LEAN=1  turns it on. Neutral = head + waist at r.
#   G1_TORSO_LEAN_MAX_DEG=10 (hard ceiling 10; larger values are rejected)
#   G1_TORSO_LEAN_GAIN_DEG_PER_M=66.7 (15 cm past the deadband = 10 deg)
#   G1_TORSO_LEAN_DEADBAND_M=0.03  G1_TORSO_LEAN_RATE_DPS=15  G1_TORSO_LEAN_ACCEL_DPS2=0 (off)
#
# Pose compare web (robot wrist FK x operator wrist/IK target, 2D/3D,
# "Salvar tarefa"; see docs/pose_compare_web.md). Default OFF:
#   XR_POSE_WEB=1  exports XR_POSE_STREAM=1 to the teleop (non-blocking UDP
#     127.0.0.1:${XR_POSE_STREAM_PORT:-47555}, 50 Hz) and starts
#     tools/pose_compare_web.py as a child (setsid; log in $teleimager_state_dir/pose_web.log),
#     stopped on exit. POSE_WEB_PORT=8093  POSE_WEB_TOKEN=<opcional>
#     XR_POSE_STREAM_HZ=50  POSE_WEB_STOP_TIMEOUT_S=5
#
# Dex3 ao encerrar (q / B / Ctrl+C / SIGTERM / erro); see
# teleop/utils/dex3_shutdown_hand.py:
#   DEX3_SHUTDOWN_HAND=close (padrão) fecha em rampa até a pose do gatilho=1
#     (não fecha com fault / estado DDS ausente ou antigo / >=80 C: abre)
#   DEX3_SHUTDOWN_HAND=open  comportamento anterior (abre)
#   DEX3_SHUTDOWN_HAND=hold  mantém o último alvo do gatilho
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/home/unitree/cyclonedds/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# The Dex3 boards publish on the robot's internal Ethernet bus. Explicitly pin
# Cyclone DDS here so Wi-Fi/Tailscale never becomes the discovery interface.
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="enP8p1s0"/></Interfaces></General></Domain></CycloneDDS>'
export PYTHONPATH=$repo:$repo/teleop/televuer/src:$repo/teleop/teleimager/src:$repo/teleop/robot_control/dex-retargeting/src:/home/unitree/unitree_sdk2_python${PYTHONPATH:+:${PYTHONPATH}}
export XR_TELEOP_CERT=${XR_TELEOP_CERT:-/home/unitree/.config/xr_teleoperate/cert.pem}
export XR_TELEOP_KEY=${XR_TELEOP_KEY:-/home/unitree/.config/xr_teleoperate/key.pem}

# Walking cap, same as the BotBrain frontend G1 profile (g1-r1.ts:
# linearSpeed 0.5 m/s, angularSpeed 0.3 rad/s). Passed explicitly to the
# teleop; override with G1_WALK_SPEED_CAP / G1_TURN_RATE_CAP. The Python code
# keeps its hard limits (0.6 m/s, 1.0 rad/s) and validation.
walk_speed_cap=${G1_WALK_SPEED_CAP:-0.3}
turn_rate_cap=${G1_TURN_RATE_CAP:-0.3}
# Walking backend: rt/wirelesscontroller (continuous 20 Hz) by default;
# G1_LOCO_BACKEND=setvelocity forces the legacy SetVelocity RPC.
loco_backend=${G1_LOCO_BACKEND:-wirelesscontroller}

# Torso lean: validated before anything starts and again in Python.
G1_TORSO_LEAN=${G1_TORSO_LEAN:-0}
case "$G1_TORSO_LEAN" in
    0|"") G1_TORSO_LEAN=0; torso_lean_msg="Inclinação do tronco: DESLIGADA (G1_TORSO_LEAN=1 para ligar)" ;;
    1)
        G1_TORSO_LEAN_MAX_DEG=${G1_TORSO_LEAN_MAX_DEG:-10}
        if ! awk -v v="$G1_TORSO_LEAN_MAX_DEG" 'BEGIN { exit !(v ~ /^[0-9]+(\.[0-9]+)?$/ && v > 0 && v <= 10) }'; then
            echo "G1_TORSO_LEAN_MAX_DEG='$G1_TORSO_LEAN_MAX_DEG' rejeitado (0 < máx <= 10 graus); não iniciando." >&2
            exit 2
        fi
        export G1_TORSO_LEAN_MAX_DEG
        for v in G1_TORSO_LEAN_GAIN_DEG_PER_M G1_TORSO_LEAN_DEADBAND_M G1_TORSO_LEAN_RATE_DPS G1_TORSO_LEAN_ACCEL_DPS2; do
            if [[ -n "${!v:-}" ]]; then export "${v?}"; fi
        done
        torso_lean_msg="Inclinação do tronco: LIGADA, máx ${G1_TORSO_LEAN_MAX_DEG}° pitch/roll (yaw fixo), ganho ${G1_TORSO_LEAN_GAIN_DEG_PER_M:-66.7}°/m, zona morta ${G1_TORSO_LEAN_DEADBAND_M:-0.03} m, ${G1_TORSO_LEAN_RATE_DPS:-15}°/s; neutro = postura no r"
        ;;
    *) echo "unsupported G1_TORSO_LEAN='$G1_TORSO_LEAN' (use 0|1)" >&2; exit 2 ;;
esac
export G1_TORSO_LEAN
# Dex3 shutdown hand mode: validated before anything starts and again in Python.
DEX3_SHUTDOWN_HAND=${DEX3_SHUTDOWN_HAND:-close}
case "$DEX3_SHUTDOWN_HAND" in
    close) dex3_shutdown_msg="Dex3 ao encerrar: FECHA (rampa suave até a pose do gatilho=1; DEX3_SHUTDOWN_HAND=open para abrir)" ;;
    open)  dex3_shutdown_msg="Dex3 ao encerrar: ABRE (comportamento anterior)" ;;
    hold)  dex3_shutdown_msg="Dex3 ao encerrar: MANTÉM o último alvo do gatilho" ;;
    *) echo "DEX3_SHUTDOWN_HAND='$DEX3_SHUTDOWN_HAND' não suportado (use close|open|hold); não iniciando." >&2; exit 2 ;;
esac
export DEX3_SHUTDOWN_HAND
# Pose compare web (8093): validated here, started right before the teleop.
XR_POSE_WEB=${XR_POSE_WEB:-0}
case "$XR_POSE_WEB" in
    0|"") XR_POSE_WEB=0 ;;
    1) ;;
    *) echo "unsupported XR_POSE_WEB='$XR_POSE_WEB' (use 0|1)" >&2; exit 2 ;;
esac
POSE_WEB_PORT=${POSE_WEB_PORT:-8093}
POSE_WEB_STOP_TIMEOUT_S=${POSE_WEB_STOP_TIMEOUT_S:-5}

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
# RealSense bandwidth profile. This robot has no USB 3.0, so usb2 is the
# default: both RealSense streams 640x480@6 (no crop/FOV trick), last valid
# frame kept frozen on a glitch. XR_REALSENSE_PROFILE=normal restores 1280x720.
XR_REALSENSE_PROFILE=${XR_REALSENSE_PROFILE:-usb2}
[[ "$XR_REALSENSE_PROFILE" == low-bandwidth || "$XR_REALSENSE_PROFILE" == low_bandwidth ]] && XR_REALSENSE_PROFILE=usb2
case "$XR_REALSENSE_PROFILE" in
    normal|usb2) ;;
    *)
        echo "unsupported XR_REALSENSE_PROFILE='$XR_REALSENSE_PROFILE' (use normal|usb2|low-bandwidth)" >&2
        exit 2
        ;;
esac
export XR_REALSENSE_PROFILE
teleimager_profile_file="$teleimager_state_dir/teleimager.realsense_profile"
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

# Show the robot's current Wi-Fi IP + Quest URL and keep the TLS cert SAN in sync.
# Never blocks the launch (always returns 0; warnings only).
# shellcheck source=lib/vuer_network.sh
source "$repo/teleop/lib/vuer_network.sh"
[[ "${G1_LAUNCHER_SKIP_NET:-0}" == 1 ]] || xr_net_announce || true

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
    rm -f "$teleimager_mode_file" "$teleimager_source_file" "$teleimager_profile_file" "$teleimager_pid_file"
}

teleimager_is_healthy() {
    # The probe runs in its own session and never inherits the launcher lock
    # (FD 9). Every process it forks (e.g. logging helpers) is killed with its
    # process group, so an orphan can never hold the lock and freeze the next
    # launcher.
    local probe_pid rc=2 deadline=$((SECONDS + 9))
    setsid "$teleimager_python" -s - "$teleimager_host" "$TELEIMAGER_CAMERA_MODE" "$XR_REALSENSE_PROFILE" 9>&- <<'PY' &
import os
import socket
import sys
import time

from teleimager.image_client import ImageClient

host = sys.argv[1]
mode = sys.argv[2]
profile = sys.argv[3]
head_only = mode == "head" or mode == "any"
ports = (60000, 55555) if head_only else (60000, 55555, 55556)
expected_shape = (480, 640, 3) if profile == "usb2" else (720, 1280, 3)
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
        and getattr(bgr, "shape", None) == expected_shape
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

    local running_mode=both running_source="" running_profile=normal running_pid="" restart_reason=""
    [[ -s "$teleimager_mode_file" ]] && running_mode=$(<"$teleimager_mode_file")
    [[ -s "$teleimager_source_file" ]] && running_source=$(<"$teleimager_source_file")
    [[ -s "$teleimager_profile_file" ]] && running_profile=$(<"$teleimager_profile_file")
    [[ -s "$teleimager_pid_file" ]] && running_pid=$(<"$teleimager_pid_file")
    if [[ -n "$running_pid" ]] && kill -0 "$running_pid" 2>/dev/null; then
        if [[ "$running_mode" != "$TELEIMAGER_CAMERA_MODE" ]]; then
            restart_reason="running in mode $running_mode, câmeras agora pedem $TELEIMAGER_CAMERA_MODE"
        elif [[ "$running_profile" != "$XR_REALSENSE_PROFILE" ]]; then
            restart_reason="perfil RealSense mudou de $running_profile para $XR_REALSENSE_PROFILE"
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

    echo "Starting Teleimager from $teleimager_dir (mode $TELEIMAGER_CAMERA_MODE, RealSense profile $XR_REALSENSE_PROFILE)."
    local server_module=teleimager.image_server
    if [[ "$TELEIMAGER_CAMERA_MODE" == head || "$TELEIMAGER_CAMERA_MODE" == any ]]; then
        server_module=teleop.utils.teleimager_head_only_server
    fi
    echo "$TELEIMAGER_CAMERA_MODE" >"$teleimager_mode_file"
    echo "$XR_REALSENSE_PROFILE" >"$teleimager_profile_file"
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
echo "$torso_lean_msg"
echo "$dex3_shutdown_msg"
if [[ "$XR_POSE_WEB" == 1 ]]; then
    echo "Página de comparação (XR_POSE_WEB=1): LIGADA, porta ${POSE_WEB_PORT}, XR_POSE_STREAM=1"
else
    echo "Página de comparação: DESLIGADA (XR_POSE_WEB=1 para ligar a 8093)"
fi
if [[ "${G1_LAUNCHER_SKIP_TELEOP:-0}" == 1 ]]; then
    exit 0
fi

# ---- pose compare web (opt-in XR_POSE_WEB=1) ----
pose_web_pid=""
pose_web_log="$teleimager_state_dir/pose_web.log"
pose_web_stop() {
    local pid=$pose_web_pid deadline
    [[ -n "$pid" ]] || return 0
    pose_web_pid=""
    kill -0 "$pid" 2>/dev/null || return 0
    kill -INT "$pid" 2>/dev/null || true
    deadline=$((SECONDS + POSE_WEB_STOP_TIMEOUT_S))
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.1; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi
    echo "POSE WEB: parado." >&2
}
pose_web_start() {
    [[ "$XR_POSE_WEB" == 1 ]] || return 0
    export XR_POSE_STREAM=1
    mkdir -p "$teleimager_state_dir"
    echo "==== $(date -Is) start ====" >>"$pose_web_log"
    # --exit-with-pid: the web exits by itself within ~1 s once this launcher
    # shell is gone (q, Ctrl+C, closed terminal, kill), so no shell signal
    # handler is needed and the persistent Teleimager is never touched.
    (cd "$repo" && exec setsid "$teleimager_python" -u -s tools/pose_compare_web.py --port "$POSE_WEB_PORT" \
        --exit-with-pid "$$" >>"$pose_web_log" 2>&1 </dev/null 9>&-) &
    pose_web_pid=$!
    local ip
    ip=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="src"){print $(i+1); exit}}') || true
    echo "POSE WEB: PID $pose_web_pid, página http://${ip:-<ip-do-robô>}:${POSE_WEB_PORT}/${POSE_WEB_TOKEN:+?token=...} (log $pose_web_log; XR_POSE_STREAM=1)"
}

cd "$repo/teleop"
[[ -n "${XR_TELEOP_VUER_IP:-}" ]] && echo "Quest: https://vuer.ai?ws=wss://${XR_TELEOP_VUER_IP}:${XR_NET_PORT}&grid=False"
echo "Teto de caminhada: ${walk_speed_cap} m/s linear, ${turn_rate_cap} rad/s angular (BotBrain g1-r1)"
echo "Backend de caminhada: ${loco_backend}"
echo "FSM solicitado (G1_LOCO_REQUEST_FSM): ${G1_LOCO_REQUEST_FSM:-none}"
# XR video plane: default auto = plane sized 1:1 to the D435i 69.4 deg HFOV
# (wider view, no crop/upscale). XR_VIDEO_PLANE_HEIGHT=1.0 restores the
# historical 1.0 m plane at 1.0 m.
XR_VIDEO_PLANE_HEIGHT=${XR_VIDEO_PLANE_HEIGHT:-auto}
echo "Plano de vídeo XR: altura ${XR_VIDEO_PLANE_HEIGHT} (XR_VIDEO_PLANE_HEIGHT=1.0 = antigo); perfil RealSense ${XR_REALSENSE_PROFILE}"
video_plane_args=(--video-plane-height "$XR_VIDEO_PLANE_HEIGHT")
[[ -n "${XR_VIDEO_PLANE_DISTANCE:-}" ]] && video_plane_args+=(--video-plane-distance "$XR_VIDEO_PLANE_DISTANCE")
teleop_args=(
  --arm G1_29
  --ee dex3
  --input-mode hand
  --motion
  --camera-layout "$teleop_camera_layout"
  --walk-speed-cap "$walk_speed_cap"
  --turn-rate-cap "$turn_rate_cap"
  --loco-backend "$loco_backend"
  --loco-request-fsm "${G1_LOCO_REQUEST_FSM:-none}"
  ${video_plane_args[@]+"${video_plane_args[@]}"})
pose_web_start
if [[ -z "$pose_web_pid" ]]; then
    # Default (XR_POSE_WEB off): same exec as before.
    exec "$teleimager_python" -s teleop_hand_and_arm.py "${teleop_args[@]}"
fi
# We own the pose web: run the teleop as a child (SIGINT from the terminal
# reaches the foreground teleop, which shuts down gracefully), then stop the
# web. If this shell dies first, the web exits via --exit-with-pid.
teleop_rc=0
"$teleimager_python" -s teleop_hand_and_arm.py "${teleop_args[@]}" || teleop_rc=$?
pose_web_stop
exit "$teleop_rc"
