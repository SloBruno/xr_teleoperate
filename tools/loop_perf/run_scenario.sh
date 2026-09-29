#!/usr/bin/env bash
# INERT scenario runner for bench_loop_stages.py (no DDS participant, no motors).
set -u
here=/home/unitree/.local/state/xr_teleoperate/loop_perf
pose=/home/unitree/.local/state/xr_teleoperate/incidents/review_20260929_0317/pose-telemetry-20260928T191645.653523Z-148c02e36d684668b3612748e4dd666b.jsonl
cd /home/unitree/xr_teleoperate_slo/teleop
name=$1; shift
timeout 180 /home/unitree/miniconda3/bin/conda run -n tv env PYTHONDONTWRITEBYTECODE=1 \
  LD_LIBRARY_PATH=/home/unitree/cyclonedds/build/lib \
  PYTHONPATH=/home/unitree/unitree_sdk2_python:/home/unitree/xr_teleoperate_slo:/home/unitree/xr_teleoperate_slo/teleop/televuer/src:/home/unitree/xr_teleoperate_slo/teleop/teleimager/src \
  python "$here/bench_loop_stages.py" --pose "$pose" --out "$here/$name.json" --cycles 300 "$@" >"$here/$name.log" 2>&1
echo "$name rc=$?"
python3 - "$here/$name.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for k, v in d.items():
    if isinstance(v, dict) and "p50" in v:
        print(f"  {k:28s} p50={v['p50']:8.2f} p95={v['p95']:8.2f} max={v['max']:8.2f}")
    elif k != "args":
        print(f"  {k:28s} {v}")
EOF
