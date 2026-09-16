# G1 Quest + Dex3 Teleoperation Design

## Goal
Extend the robot snapshot on the personal fork only with terminal-gated teleoperation, Quest joystick locomotion, Dex3 trigger closure, and pressure-based haptics.

## Approved behavior

- Publication target: personal GitHub fork only. Do not open or request anything from Unitree upstream.
- Keep the existing terminal launcher and its hand selector. Do not add a second UI.
- `r` and `q` in the terminal/IPC remain the only lifecycle controls. Quest buttons must not start or stop tracking.
- After terminal start, Quest left stick commands forward/lateral velocity and right stick commands yaw at the native `LocoClient` scale. Remove the 0.3 multiplier and do not issue `Damp()` from Quest controls.
- Retain Inspire support and its driver. Select the appropriate end effector at launch; do not delete Inspire code or service definitions.
- For Dex3-1, each corresponding Quest trigger proportionally closes all three digit groups: thumb, index, and middle.
- Haptic feedback goes to the controller on the same side as contact and is proportional to contact pressure. It must be bounded, rate-limited, and zero for stale/invalid sensor data.

## Architecture

`teleop_hand_and_arm.py` remains the lifecycle coordinator and control loop. Televuer decodes XR hand and controller data, then publishes a side-specific normalized trigger value and any supported haptic commands. `robot_hand_unitree.py` owns Dex3 joint target generation and DDS publication. Small pure helpers own joystick mapping, trigger-to-Dex3 mapping, and pressure-to-haptic mapping so they can be tested without hardware.

The existing launcher remains the deployment interface; it forwards `--ee` explicitly. The Inspire service is preserved and only selected when its profile is requested.

## Safety constraints

- No Quest controller event may mutate START or STOP.
- Trigger and pressure values must be finite and clamped.
- Haptics must fail closed: unsupported transport, absent pressure, stale sample, or errors result in no vibration.
- Hardware movement validation needs the operator to clear the area and approve a physical test after offline tests pass.

## Repository publication

The robot snapshot contains uncommitted parent and submodule changes. The parent branch cannot be reproducible until modified `televuer` and `teleimager` submodules are committed to personal-fork remotes and the parent records their commit pointers. Device-specific camera serial configuration and local backups are excluded from the portable feature commits.
