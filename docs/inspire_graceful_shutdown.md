# Graceful shutdown (dev-inspire, Inspire DFQ/DFX)

Ported from the main line (`edfd901`, `teleop/utils/arm_graceful_shutdown.py`,
same semantics and constants) and wired through `teleop/utils/session_shutdown.py`.

Triggers: terminal `q`, Ctrl+C, SIGTERM/SIGHUP (converted to the same path),
or any exception in the main loop. Every step is bounded, an exception in one
step never skips the next, and repeating it is a no-op.

What happens, in order:

1. **No new IK target**: the loop exits; a `q` that lands during IK is not
   published (`if STOP: break` before `ctrl_dual_arm`). A late `r` is ignored.
2. **Inspire hand**: the command process gets a stop event, joined for 1 s,
   then terminate/kill (0.5 s each). **No open/close command is sent**: the
   hand stays at its last commanded position (operator preference: no
   automatic hand motion at session edges). After this nothing more is
   published on `rt/inspire_hand/ctrl/{l,r}`.
3. **G1_29 arms** (works for `q` before or after `r`, because on this branch
   the writer already publishes from start-up):
   - return to the all-zero pose with a smoothstep trajectory, peak
     ≤ 0.5 rad/s per joint (log `shutdown_return_started duration_s,
     max_joint_velocity, max_distance_rad`), feed-forward blended to gravity,
     wait ≤ 2 s for arrival (≤ 0.05 rad);
   - invalid/stale state or a return longer than 12 s skips the motion;
   - ramp `kNotUsedJoint0.q` (rt/arm_sdk weight) 1 → 0 over 2 s, confirm a
     published weight-0 frame (Unitree motion controller takes the arms back);
   - stop the writer thread (bounded join 1 s).
   Other arm profiles keep the legacy `ctrl_dual_arm_go_home`.
4. Keyboard/IPC listener stopped (bounded), image client closed.
5. **TeleVuer**: `close()` bounded to 1 s; the Vuer process and its own
   children are then terminated/killed so nothing is left orphaned.
6. Remaining multiprocessing children (except the logging listener) are reaped.
7. **Launcher** (`run_g1_quest_inspire.sh`): only after the teleop process
   exits does the EXIT trap stop the RS-485 driver it started (SIGINT, 8 s,
   SIGTERM, SIGKILL). The driver runs in its own session (`setsid`), so a
   terminal Ctrl+C reaches the teleop but not the driver. A driver that was
   already running is reused and never stopped. Teleimager is a persistent
   daemon and is left running (as before).

Worst case: ~1.5 s hand + ~17 s arm + ~2 s TeleVuer.

Not changed here (pending): arm writer and Inspire command process still
start publishing at initialisation, before `r` (weight 1.0; hand `angle_set`
1000 = open at ~200 Hz, which saturates the RS-485 link).
