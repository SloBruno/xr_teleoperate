#!/usr/bin/env bash
# G1D teach-and-replay launcher (arms + Dex3). Separate from run_g1d_quest_dex3.sh:
# no XR, no video, no IK. Keys r/s/p/q are read by Python from this terminal.
# Same DDS interface binding and runtime env as the G1D teleop launcher.
set -euo pipefail

for g1d_extra_arg in "$@"; do
    case "$g1d_extra_arg" in
        --network-interface|--network-interface=*)
            echo "G1D teach launcher: forwarded argument refused: $g1d_extra_arg (use G1D_DDS_INTERFACE)." >&2
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
export PYTHONPATH=$repo:/home/unitree/unitree_sdk2_python${PYTHONPATH:+:${PYTHONPATH}}
teach_python=${TELEIMAGER_PYTHON:-/home/unitree/.local/envs/xr-g1d/bin/python}

cmd=("$teach_python" -s teach_replay_g1d.py --network-interface "$g1d_dds_interface" "$@")
echo "G1D TEACH/REPLAY: robot SUPPORTED, area clear, R3/e-stop in reach. Keys: r=teach s=save p=replay q=quit."
echo "Optional browser panel: add --web [PORT] (default 8095); URL with token is printed below."
echo "command: ${cmd[*]}"
[[ "${G1D_TEACH_SKIP_RUN:-0}" == 1 ]] && exit 0
cd "$repo/teleop"
exec "${cmd[@]}"
