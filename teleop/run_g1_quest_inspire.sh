#!/usr/bin/env bash
# Operator-run launcher for the G1 29-DoF + Inspire RH56DFQ/DFX hands, Quest HAND
# TRACKING only (no controllers, no controller->wrist calibration, no Quest
# locomotion: the physical remote keeps walking control). This process waits
# for local terminal r/q.
#   G1_EE=inspire_dfx (default; per-hand rt/inspire_hand/{ctrl,state}/{l,r})
#   G1_EE=inspire_ftp (inspire_sdkpy)
#   G1_MOTION=1 (default) passes --motion: robot stays in its locomotion mode and
#   the teleop never sends locomotion. G1_MOTION=0 = debug mode (robot SUSPENDED).
#
# Inspire DFQ RS-485 driver (teleop/robot_control/inspire_dfq_485_driver.py, no sudo):
#   INSPIRE_DRIVER=auto (default for inspire_dfx) | skip | external
#     auto:     reuse a running driver (python argv with Headless_driver_485*.py or
#               inspire_dfq_485_driver.py; never stopped by us) or, if none, check
#               the configured serial ports (exist + rw for this user; else abort,
#               listing /dev/ttyUSB* and /dev/serial/by-id), start the wrapper as a
#               child (setsid), then a PASSIVE DDS health check (subscriber only)
#               on rt/inspire_hand/state/{l,r}. On exit/q/error/Ctrl+C only the
#               driver started here is stopped (SIGINT, 8 s, SIGTERM).
#     external: no start/stop; only the passive health check.
#     skip:     nothing (default for G1_EE=inspire_ftp).
#   INSPIRE_LEFT_PORT=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAYH4GK-if01-port0
#   INSPIRE_RIGHT_PORT=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAYH4GK-if02-port0
#     (this robot's FT4232H; other adapters: set by-id/by-path; discover with the wrapper's --probe; see
#     docs/inspire_dfq_485.md)  INSPIRE_BAUDRATE=115200  INSPIRE_LEFT_ID=1
#     INSPIRE_RIGHT_ID=1  INSPIRE_DDS_IFACE=enP8p1s0  INSPIRE_DRIVER_TIMEOUT_S=15
#     INSPIRE_STATE_DIR=~/.local/state/xr_teleoperate_inspire
#   bash teleop/run_g1_quest_inspire.sh --inspire-preflight
#     read-only: shows the running driver / port check and exits (starts nothing).
#
# Pose compare web (robot wrist FK x operator wrist/IK target, X/Y/Z vs time,
# "Salvar tarefa"; see docs/pose_compare_web.md). Default OFF:
#   XR_POSE_WEB=1  exports XR_POSE_STREAM=1 to the teleop (non-blocking UDP
#     127.0.0.1:${XR_POSE_STREAM_PORT:-47555}, 50 Hz) and starts
#     tools/pose_compare_web.py as a child (setsid; log in $INSPIRE_STATE_DIR/pose_web.log),
#     stopped on exit. POSE_WEB_PORT=8093  POSE_WEB_TOKEN=<opcional>
#     XR_POSE_STREAM_HZ=50  POSE_WEB_STOP_TIMEOUT_S=5
#
# Torso lean (waist pitch/roll from the operator's head DISPLACEMENT, via
# rt/arm_sdk; see docs/torso_lean.md). Default OFF:
#   G1_TORSO_LEAN=1  turns it on (requires G1_MOTION=1). Neutral = head + waist at r.
#   G1_TORSO_LEAN_MAX_DEG=10 (hard ceiling 10; larger values are rejected)
#   G1_TORSO_LEAN_GAIN_DEG_PER_M=66.7 (15 cm past the deadband = 10 deg)
#   G1_TORSO_LEAN_DEADBAND_M=0.03  G1_TORSO_LEAN_RATE_DPS=15  G1_TORSO_LEAN_ACCEL_DPS2=0 (off)
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/home/unitree/cyclonedds/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# The hand service publishes on the robot's internal Ethernet bus. Explicitly pin
# Cyclone DDS here so Wi-Fi/Tailscale never becomes the discovery interface.
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="enP8p1s0"/></Interfaces></General></Domain></CycloneDDS>'
export PYTHONPATH=$repo:$repo/teleop/televuer/src:$repo/teleop/teleimager/src:$repo/teleop/robot_control/dex-retargeting/src:/home/unitree/unitree_sdk2_python${PYTHONPATH:+:${PYTHONPATH}}
export XR_TELEOP_CERT=${XR_TELEOP_CERT:-/home/unitree/.config/xr_teleoperate/cert.pem}
export XR_TELEOP_KEY=${XR_TELEOP_KEY:-/home/unitree/.config/xr_teleoperate/key.pem}

G1_EE=${G1_EE:-inspire_dfx}
case "$G1_EE" in
    inspire_dfx|inspire_ftp) ;;
    *)
        echo "unsupported G1_EE='$G1_EE' for this launcher (use inspire_dfx|inspire_ftp)" >&2
        exit 2
        ;;
esac
G1_MOTION=${G1_MOTION:-1}
case "$G1_MOTION" in
    1) motion_args=(--motion) ;;
    0) motion_args=() ;;
    *) echo "unsupported G1_MOTION='$G1_MOTION' (use 1|0)" >&2; exit 2 ;;
esac
# Torso lean: validated before anything starts and again in Python.
G1_TORSO_LEAN=${G1_TORSO_LEAN:-0}
case "$G1_TORSO_LEAN" in
    0|"") echo "Inclinação do tronco: DESLIGADA (G1_TORSO_LEAN=1 para ligar)" ;;
    1)
        G1_TORSO_LEAN_MAX_DEG=${G1_TORSO_LEAN_MAX_DEG:-10}
        if ! awk -v v="$G1_TORSO_LEAN_MAX_DEG" 'BEGIN { exit !(v ~ /^[0-9]+(\.[0-9]+)?$/ && v > 0 && v <= 10) }'; then
            echo "G1_TORSO_LEAN_MAX_DEG='$G1_TORSO_LEAN_MAX_DEG' rejeitado (0 < máx <= 10 graus); não iniciando." >&2
            exit 2
        fi
        if [[ "$G1_MOTION" != 1 ]]; then
            echo "G1_TORSO_LEAN=1 requer G1_MOTION=1 (rt/arm_sdk); não iniciando." >&2
            exit 2
        fi
        export G1_TORSO_LEAN G1_TORSO_LEAN_MAX_DEG
        for v in G1_TORSO_LEAN_GAIN_DEG_PER_M G1_TORSO_LEAN_DEADBAND_M G1_TORSO_LEAN_RATE_DPS G1_TORSO_LEAN_ACCEL_DPS2; do
            [[ -n "${!v:-}" ]] && export "$v"
        done
        echo "Inclinação do tronco: LIGADA, máx ${G1_TORSO_LEAN_MAX_DEG}° (frente/trás/lados), ganho ${G1_TORSO_LEAN_GAIN_DEG_PER_M:-66.7}°/m, zona morta ${G1_TORSO_LEAN_DEADBAND_M:-0.03} m, ${G1_TORSO_LEAN_RATE_DPS:-15}°/s; neutro = postura no r"
        ;;
    *) echo "unsupported G1_TORSO_LEAN='$G1_TORSO_LEAN' (use 0|1)" >&2; exit 2 ;;
esac
export G1_TORSO_LEAN

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
#   TELEIMAGER_CAMERA_MODE=any bash teleop/run_g1_quest_inspire.sh
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

# ------------------------------------------------------------ Inspire RS-485 driver
if [[ "$G1_EE" == inspire_ftp ]]; then
    INSPIRE_DRIVER=${INSPIRE_DRIVER:-skip}
else
    INSPIRE_DRIVER=${INSPIRE_DRIVER:-auto}
fi
case "$INSPIRE_DRIVER" in
    auto|skip|external) ;;
    *) echo "unsupported INSPIRE_DRIVER='$INSPIRE_DRIVER' (use auto|skip|external)" >&2; exit 2 ;;
esac
# Defaults: this robot's FTDI FT4232H (serial FTAYH4GK); if01 = LEFT, if02 = RIGHT
# (side confirmed on hardware). Override for another adapter.
INSPIRE_LEFT_PORT=${INSPIRE_LEFT_PORT:-/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAYH4GK-if01-port0}
INSPIRE_RIGHT_PORT=${INSPIRE_RIGHT_PORT:-/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAYH4GK-if02-port0}
INSPIRE_BAUDRATE=${INSPIRE_BAUDRATE:-115200}
INSPIRE_LEFT_ID=${INSPIRE_LEFT_ID:-1}
INSPIRE_RIGHT_ID=${INSPIRE_RIGHT_ID:-1}
INSPIRE_DDS_IFACE=${INSPIRE_DDS_IFACE-enP8p1s0}
INSPIRE_DRIVER_TIMEOUT_S=${INSPIRE_DRIVER_TIMEOUT_S:-15}
INSPIRE_DRIVER_STOP_TIMEOUT_S=${INSPIRE_DRIVER_STOP_TIMEOUT_S:-8}
INSPIRE_STATE_DIR=${INSPIRE_STATE_DIR:-$HOME/.local/state/xr_teleoperate_inspire}
INSPIRE_SDK_DIR=${INSPIRE_SDK_DIR:-/home/unitree/inspire_hand_ws/inspire_hand_sdk}
INSPIRE_UNITREE_SDK=${INSPIRE_UNITREE_SDK:-/home/unitree/unitree_sdk2_python}
INSPIRE_PYTHON=${INSPIRE_PYTHON:-${TELEIMAGER_PYTHON:-/home/unitree/miniconda3/envs/tv/bin/python}}
INSPIRE_PROC_ROOT=${INSPIRE_PROC_ROOT:-/proc}
inspire_wrapper="$repo/teleop/robot_control/inspire_dfq_485_driver.py"
inspire_pid_file="$INSPIRE_STATE_DIR/inspire_driver.pid"
inspire_log="$INSPIRE_STATE_DIR/inspire_driver.log"
inspire_lock="$INSPIRE_STATE_DIR/inspire_driver.lock"
inspire_started_pid=""

# Environment of the driver (same as the robot's Inspire service).
inspire_env=(env PYTHONNOUSERSITE=1
    "PYTHONPATH=$INSPIRE_UNITREE_SDK:$INSPIRE_SDK_DIR"
    "LD_LIBRARY_PATH=${INSPIRE_LD_LIBRARY_PATH:-/home/unitree/miniconda3/envs/tv/lib:/home/unitree/cyclonedds/build/lib}")

# PIDs of a python whose argv runs an Inspire 485 driver (exact argv match via
# /proc/<pid>/cmdline; probe/health/help invocations are not drivers).
inspire_find_drivers() {
    local d pid a base0 hit
    local -a argv
    for d in "$INSPIRE_PROC_ROOT"/[0-9]*; do
        [[ -r "$d/cmdline" ]] || continue
        pid=${d##*/}
        [[ "$pid" == "$$" ]] && continue
        mapfile -d '' -t argv <"$d/cmdline" 2>/dev/null || continue
        (( ${#argv[@]} >= 2 )) || continue
        base0=${argv[0]##*/}
        [[ "$base0" == python* ]] || continue
        hit=0
        for a in "${argv[@]:1}"; do
            case "$a" in --probe|--health-check|--help|-h) hit=0; break ;; esac
            case "${a##*/}" in
                Headless_driver_485*.py|inspire_dfq_485_driver.py) hit=1 ;;
            esac
        done
        (( hit )) && echo "$pid"
    done
    return 0
}

inspire_list_adapters() {
    echo "  /dev/ttyUSB*:        $(ls /dev/ttyUSB* 2>/dev/null | tr '\n' ' ' || true)" >&2
    echo "  /dev/serial/by-id:   $(ls /dev/serial/by-id/ 2>/dev/null | tr '\n' ' ' || true)" >&2
    echo "  /dev/serial/by-path: $(ls /dev/serial/by-path/ 2>/dev/null | tr '\n' ' ' || true)" >&2
}

# 0 when both configured ports are character devices readable+writable by us.
inspire_check_ports() {
    local bad=0 label port
    for label in ESQUERDA DIREITA; do
        if [[ $label == ESQUERDA ]]; then port=$INSPIRE_LEFT_PORT; else port=$INSPIRE_RIGHT_PORT; fi
        if [[ ! -e "$port" ]]; then
            echo "INSPIRE: porta $label $port não existe." >&2; bad=1
        elif [[ ! -c "$port" ]]; then
            echo "INSPIRE: porta $label $port não é dispositivo serial." >&2; bad=1
        elif [[ ! -r "$port" || ! -w "$port" ]]; then
            echo "INSPIRE: sem permissão rw em $label $port ($(id -un) no grupo dialout?)." >&2; bad=1
        fi
    done
    if [[ -e "$INSPIRE_LEFT_PORT" && "$(readlink -f "$INSPIRE_LEFT_PORT")" == "$(readlink -f "$INSPIRE_RIGHT_PORT")" ]]; then
        echo "INSPIRE: esquerda e direita apontam para o mesmo dispositivo." >&2; bad=1
    fi
    if (( bad )); then
        echo "INSPIRE: adaptador RS-485 da Inspire não encontrado — mão montada/USB conectado?" >&2
        echo "  configurado: INSPIRE_LEFT_PORT=$INSPIRE_LEFT_PORT INSPIRE_RIGHT_PORT=$INSPIRE_RIGHT_PORT" >&2
        inspire_list_adapters
        echo "  descubra/fixe as portas: python $inspire_wrapper --probe [portas...]; docs/inspire_dfq_485.md" >&2
        return 1
    fi
    case "$INSPIRE_LEFT_PORT $INSPIRE_RIGHT_PORT" in
        */dev/serial/by-*/*" "*/dev/serial/by-*|/dev/inspire_*" "/dev/inspire_*) ;;
        *) echo "INSPIRE: AVISO: portas por número (ttyUSBx) mudam com a ordem de conexão; fixe por /dev/serial/by-id (docs/inspire_dfq_485.md)." >&2 ;;
    esac
    return 0
}

# Passive health check: DDS subscribers only (no ctrl publisher). Watches the
# driver we started (if any) so an early crash fails fast.
inspire_health_check() {
    local watch_pid=${1:-} hc_pid rc=1 deadline=$((SECONDS + INSPIRE_DRIVER_TIMEOUT_S + 15))
    (cd "$repo" && exec setsid "${inspire_env[@]}" "$INSPIRE_PYTHON" -u -s "$inspire_wrapper" --health-check \
        --timeout "$INSPIRE_DRIVER_TIMEOUT_S" --iface "$INSPIRE_DDS_IFACE" </dev/null 7>&- 8>&- 9>&-) &
    hc_pid=$!
    while kill -0 "$hc_pid" 2>/dev/null && (( SECONDS < deadline )); do
        if [[ -n "$watch_pid" ]] && ! kill -0 "$watch_pid" 2>/dev/null; then
            echo "INSPIRE: driver PID $watch_pid terminou durante o health check." >&2
            kill -KILL -- "-$hc_pid" 2>/dev/null || kill -KILL "$hc_pid" 2>/dev/null || true
            wait "$hc_pid" 2>/dev/null || true
            return 1
        fi
        sleep 0.2
    done
    if kill -0 "$hc_pid" 2>/dev/null; then
        echo "INSPIRE: health check excedeu o tempo." >&2
        kill -KILL -- "-$hc_pid" 2>/dev/null || kill -KILL "$hc_pid" 2>/dev/null || true
    else
        wait "$hc_pid" && rc=0 || rc=$?
    fi
    wait "$hc_pid" 2>/dev/null || true
    return "$rc"
}

inspire_pid_is_ours() {
    tr '\0' ' ' <"/proc/$1/cmdline" 2>/dev/null | grep -q 'inspire_dfq_485_driver\.py'
}

# Stops ONLY the driver this launcher started: SIGINT, wait, SIGTERM, wait, SIGKILL.
inspire_stop_started() {
    local pid=$inspire_started_pid deadline
    [[ -n "$pid" ]] || return 0
    inspire_started_pid=""
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$inspire_pid_file"
        return 0
    fi
    if ! inspire_pid_is_ours "$pid"; then
        echo "INSPIRE: PID $pid não é mais o driver iniciado; não matando." >&2
        return 0
    fi
    echo "INSPIRE: parando driver PID $pid (SIGINT)..." >&2
    kill -INT "$pid" 2>/dev/null || true
    deadline=$((SECONDS + INSPIRE_DRIVER_STOP_TIMEOUT_S))
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.2; done
    if kill -0 "$pid" 2>/dev/null; then
        echo "INSPIRE: driver não saiu em ${INSPIRE_DRIVER_STOP_TIMEOUT_S}s; SIGTERM." >&2
        kill -TERM "$pid" 2>/dev/null || true
        deadline=$((SECONDS + 3))
        while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.2; done
        if kill -0 "$pid" 2>/dev/null; then
            echo "INSPIRE: driver ainda vivo; SIGKILL." >&2
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    rm -f "$inspire_pid_file"
    echo "INSPIRE: driver parado." >&2
}

inspire_on_exit() {
    local rc=$?
    trap - EXIT INT TERM HUP
    inspire_stop_started
    pose_web_stop
    exit "$rc"
}

# ---- pose compare web (opt-in XR_POSE_WEB=1) ----
XR_POSE_WEB=${XR_POSE_WEB:-0}
POSE_WEB_PORT=${POSE_WEB_PORT:-8093}
POSE_WEB_STOP_TIMEOUT_S=${POSE_WEB_STOP_TIMEOUT_S:-5}
pose_web_pid=""
pose_web_log="$INSPIRE_STATE_DIR/pose_web.log"

pose_web_stop() {
    local pid=$pose_web_pid deadline
    [[ -n "$pid" ]] || return 0
    pose_web_pid=""
    kill -0 "$pid" 2>/dev/null || return 0
    kill -INT "$pid" 2>/dev/null || true
    deadline=$((SECONDS + POSE_WEB_STOP_TIMEOUT_S))
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.1; done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
    echo "POSE WEB: parado." >&2
}

pose_web_on_exit() {
    local rc=$?
    trap - EXIT INT TERM HUP
    pose_web_stop
    exit "$rc"
}

pose_web_start() {
    [[ "$XR_POSE_WEB" == 1 ]] || return 0
    export XR_POSE_STREAM=1
    mkdir -p "$INSPIRE_STATE_DIR"
    echo "==== $(date -Is) start ====" >>"$pose_web_log"
    (cd "$repo" && exec setsid "$teleimager_python" -u -s tools/pose_compare_web.py --port "$POSE_WEB_PORT" \
        >>"$pose_web_log" 2>&1 </dev/null 7>&- 8>&- 9>&-) &
    pose_web_pid=$!
    if [[ -z "$inspire_started_pid" ]]; then
        trap pose_web_on_exit EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM HUP
    fi
    local ip
    ip=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="src"){print $(i+1); exit}}') || true
    echo "POSE WEB: PID $pose_web_pid, página http://${ip:-<ip-do-robô>}:${POSE_WEB_PORT}/${POSE_WEB_TOKEN:+?token=...} (log $pose_web_log; XR_POSE_STREAM=1)"
}

inspire_fail() {
    echo "INSPIRE: $1" >&2
    if [[ -n "$inspire_started_pid" && -s "$inspire_log" ]]; then
        echo "---- últimas linhas de $inspire_log ----" >&2
        tail -n 30 "$inspire_log" >&2 || true
        echo "----------------------------------------" >&2
    fi
    inspire_stop_started
    exec 7>&- 2>/dev/null || true
    exit 6
}

ensure_inspire_driver() {
    [[ "$INSPIRE_DRIVER" == skip ]] && { echo "INSPIRE: driver não gerenciado (INSPIRE_DRIVER=skip)."; return 0; }
    if [[ "$INSPIRE_DRIVER" == external ]]; then
        echo "INSPIRE: driver externo; health check passivo (${INSPIRE_DRIVER_TIMEOUT_S}s)..."
        inspire_health_check || inspire_fail "sem estado em rt/inspire_hand/state/{l,r} do driver externo."
        echo "INSPIRE: driver externo saudável."
        return 0
    fi
    mkdir -p "$INSPIRE_STATE_DIR"
    exec 7>"$inspire_lock"
    if ! flock -w 10 7; then
        echo "INSPIRE: lock $inspire_lock ocupado ($(fuser "$inspire_lock" 2>/dev/null || echo ?))." >&2
        exit 6
    fi
    local existing
    existing=$(inspire_find_drivers | tr '\n' ' ')
    if [[ -n "${existing// /}" ]]; then
        echo "INSPIRE: reaproveitando driver já em execução (PID ${existing% }); não será parado na saída."
        inspire_health_check || inspire_fail "driver existente (PID ${existing% }) sem estado em rt/inspire_hand/state/{l,r}; não mexi nele."
        exec 7>&-
        echo "INSPIRE: driver existente saudável."
        return 0
    fi
    inspire_check_ports || { exec 7>&-; exit 6; }
    trap inspire_on_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM HUP
    echo "INSPIRE: iniciando driver ($INSPIRE_LEFT_PORT / $INSPIRE_RIGHT_PORT, ${INSPIRE_BAUDRATE} baud, iface ${INSPIRE_DDS_IFACE:-auto}); log $inspire_log"
    echo "==== $(date -Is) start ====" >>"$inspire_log"
    (
        cd "$INSPIRE_SDK_DIR/example" 2>/dev/null || cd "$repo"
        exec setsid "${inspire_env[@]}" "$INSPIRE_PYTHON" -u -s "$inspire_wrapper" \
            --left-port "$INSPIRE_LEFT_PORT" --right-port "$INSPIRE_RIGHT_PORT" \
            --baudrate "$INSPIRE_BAUDRATE" --left-id "$INSPIRE_LEFT_ID" --right-id "$INSPIRE_RIGHT_ID" \
            --iface "$INSPIRE_DDS_IFACE" >>"$inspire_log" 2>&1 </dev/null 7>&- 8>&- 9>&-
    ) &
    inspire_started_pid=$!
    echo "$inspire_started_pid" >"$inspire_pid_file"
    echo "INSPIRE: driver PID $inspire_started_pid; health check passivo (${INSPIRE_DRIVER_TIMEOUT_S}s)..."
    inspire_health_check "$inspire_started_pid" \
        || inspire_fail "driver não publicou rt/inspire_hand/state/{l,r} em ${INSPIRE_DRIVER_TIMEOUT_S}s; abortando."
    exec 7>&-
    echo "INSPIRE: driver saudável (será parado ao sair)."
}

if [[ "${1:-}" == --inspire-preflight ]]; then
    existing=$(inspire_find_drivers | tr '\n' ' ')
    if [[ -n "${existing// /}" ]]; then
        echo "INSPIRE: driver já em execução (PID ${existing% }); seria reaproveitado."
        exit 0
    fi
    inspire_check_ports || exit 6
    echo "INSPIRE: portas OK ($INSPIRE_LEFT_PORT, $INSPIRE_RIGHT_PORT); nada foi iniciado."
    exit 0
elif [[ $# -gt 0 ]]; then
    echo "uso: $0 [--inspire-preflight]" >&2
    exit 2
fi

# Detection runs in its own session with a short timeout and never inherits
# the launcher lock (FD 9). Any failure (exception, timeout, import error) is a
# clear failure: no guessing.
detect_cameras() {
    local rc=0
    detected=$(cd "$repo" && timeout -k 2 "$TELEIMAGER_DETECT_TIMEOUT_S" setsid "$teleimager_python" -s -m teleop.utils.teleimager_head_only_server --detect 9>&- 8>&-) || rc=$?
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

# Session lock: one teleop per robot. FD 8 is inherited by the exec'd teleop and
# released when it exits; Teleimager and probes never inherit it (8>&-).
teleop_session_lock="${TELEIMAGER_STATE_DIR:-/home/unitree/.local/state/xr_teleoperate}/teleop.session.lock"
if [[ "${G1_LAUNCHER_SKIP_TELEOP:-0}" != 1 ]]; then
    mkdir -p "$(dirname "$teleop_session_lock")"
    exec 8>"$teleop_session_lock"
    if ! flock -n 8; then
        echo "outra sessão de teleop já está ativa (lock $teleop_session_lock: $(fuser "$teleop_session_lock" 2>/dev/null || echo ?)); não iniciando." >&2
        exit 5
    fi
    if pgrep -f 'python.*[t]eleop_hand_and_arm\.py' >/dev/null 2>&1; then
        echo "teleop_hand_and_arm.py já está em execução (outro checkout?); não iniciando." >&2
        exit 5
    fi
fi

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
    setsid "$teleimager_python" -s - "$teleimager_host" "$TELEIMAGER_CAMERA_MODE" "$XR_REALSENSE_PROFILE" 9>&- 8>&- <<'PY' &
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
            >>"$teleimager_log" 2>&1 < /dev/null 9>&- 8>&- &
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
# Inspire driver right before the teleop (after video is healthy), so a failed
# camera start never leaves a driver behind.
ensure_inspire_driver
cd "$repo/teleop"
[[ -n "${XR_TELEOP_VUER_IP:-}" ]] && echo "Quest: https://vuer.ai?ws=wss://${XR_TELEOP_VUER_IP}:${XR_NET_PORT}&grid=False"
echo "Mão: --ee ${G1_EE}, hand tracking (sem controles, sem locomoção pelo Quest); --motion=${G1_MOTION}"
# XR video plane: default auto = plane sized 1:1 to the D435i 69.4 deg HFOV
# (wider view, no crop/upscale). XR_VIDEO_PLANE_HEIGHT=3.0 XR_VIDEO_PLANE_DISTANCE=4.0
# restores this branch's previous plane.
XR_VIDEO_PLANE_HEIGHT=${XR_VIDEO_PLANE_HEIGHT:-auto}
echo "Plano de vídeo XR: altura ${XR_VIDEO_PLANE_HEIGHT} (XR_VIDEO_PLANE_HEIGHT=3.0 XR_VIDEO_PLANE_DISTANCE=4.0 = antigo dev); perfil RealSense ${XR_REALSENSE_PROFILE}"
video_plane_args=(--video-plane-height "$XR_VIDEO_PLANE_HEIGHT")
[[ -n "${XR_VIDEO_PLANE_DISTANCE:-}" ]] && video_plane_args+=(--video-plane-distance "$XR_VIDEO_PLANE_DISTANCE")
pose_web_start
teleop_cmd=("$teleimager_python" -s teleop_hand_and_arm.py
  --arm G1_29
  --ee "$G1_EE"
  --input-mode hand
  ${motion_args[@]+"${motion_args[@]}"}
  --camera-layout "$teleop_camera_layout"
  ${video_plane_args[@]+"${video_plane_args[@]}"})
if [[ -z "$inspire_started_pid" && -z "$pose_web_pid" ]]; then
    exec "${teleop_cmd[@]}"
fi
# We own the Inspire driver and/or the pose web: run the teleop as a child so
# the EXIT trap stops them after q / error / Ctrl+C (SIGINT reaches the
# foreground teleop; the setsid children do not get the terminal's SIGINT).
teleop_rc=0
"${teleop_cmd[@]}" || teleop_rc=$?
exit "$teleop_rc"
