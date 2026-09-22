#!/usr/bin/env bash
# Operator-run launcher: this process waits for local terminal r/q.
set -euo pipefail

repo=/home/unitree/xr_teleoperate_slo
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/home/unitree/cyclonedds/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# The Dex3 boards publish on the robot's internal Ethernet bus. Explicitly pin
# Cyclone DDS here so Wi-Fi/Tailscale never becomes the discovery interface.
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="enP8p1s0"/></Interfaces></General></Domain></CycloneDDS>'
export PYTHONPATH=$repo:$repo/teleop/televuer/src:$repo/teleop/teleimager/src:$repo/teleop/robot_control/dex-retargeting/src:/home/unitree/unitree_sdk2_python${PYTHONPATH:+:${PYTHONPATH}}
export XR_TELEOP_CERT=/home/unitree/.config/xr_teleoperate/cert.pem
export XR_TELEOP_KEY=/home/unitree/.config/xr_teleoperate/key.pem

cd "$repo/teleop"
exec /home/unitree/miniconda3/envs/tv/bin/python -s teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex3 \
  --input-mode hand \
  --motion \
  --camera-layout vertical
