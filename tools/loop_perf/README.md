# Loop-performance probes (inert)

These probes reproduce and decompose the G1 teleoperation tracking loop **without launching teleop or creating DDS participants, publishers, subscribers, or motor commands**. They are diagnostic tooling only; none is imported by the production control path.

## Safety contract

- Run only through an already-authorized SSH session on the robot.
- `bench_loop_stages.py` reads existing ZMQ camera streams and a recorded pose JSONL. It does not start Teleimager, Vuer, teleop, DDS, or any actuator process.
- The optional background load performs local SDK serialization/deserialization, CRC, and Python data-copy work. It never opens a DDS transport.
- Pose telemetry is built and handed to `PoseTelemetryJsonlSink`; that sink is asynchronous. The timing loop performs no direct JSON/file I/O and its bounded sink may drop rather than wait.
- Do not run a scenario when the recorded pose file or both camera streams are unavailable. The probe deliberately makes no attempt to start them.

## Probes

| File | Purpose | Output |
| --- | --- | --- |
| `bench_loop_stages.py` | Replays recorded controller poses and records p50/p95/max per stage. | JSON: `img_get`, `stack`, `render_handoff`, `calib_targets`, `ik_solve`, `fk_gate`, `telemetry`, `cycle_work`, `period`, JPEG side cost, and load average. |
| `bench_dds_cpu.py` | Measures local-only Unitree SDK serialization/deserialization and CRC work. | JSON p50/p95 plus a GIL-ms/s estimate at production-rate assumptions. |
| `bench_ik_gil.py` | Tests whether `solve_ik` releases the Python GIL. | Idle counter rate vs. counter rate during IK. |
| `run_scenario.sh` | Calls the stage probe with the read-only deployment paths used in the incident investigation. | Human-readable summary plus its JSON result. |

## Reproducible scenarios

On the robot, after placing the probe files under `~/.local/state/xr_teleoperate/loop_perf/`:

```bash
# Baseline: recorded poses + existing cameras, no emulated in-process work.
bash run_scenario.sh noload

# Representative contention: local DDS work at 500 Hz, hand work at 100 Hz,
# and video stack/render on a separate thread.
bash run_scenario.sh dds_vt_h100 --dds-load --hand-state-hz 100 --video-thread

# Isolate the local-only DDS Python budget and the IK/GIL behavior.
python bench_dds_cpu.py
python bench_ik_gil.py /path/to/pose-telemetry.jsonl
```

The stage probe requires both existing camera streams before it starts. A failed frame precondition is a blocked benchmark, not a control-path result.

## Interpretation

Compare `cycle_work` and `period` first. A 30 Hz loop has a 33.3 ms period budget. Then attribute the excess using stage p50/p95. The DDS probe's GIL estimate is a CPU-budget indicator, not a latency estimate on its own: confirm it with the all-thread stage scenario and the one-component ablations before changing production cadence, queues, or priorities.
