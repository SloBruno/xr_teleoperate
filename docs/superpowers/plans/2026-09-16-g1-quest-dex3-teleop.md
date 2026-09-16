# G1 Quest + Dex3 Teleoperation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add terminal-gated Quest locomotion, Dex3 trigger closure, and safe pressure haptics without removing Inspire support.

**Architecture:** Keep `teleop_hand_and_arm.py` as the loop coordinator; add pure mappers for joystick, trigger, and pressure behavior; extend Televuer only for hand-tracking controller analog/haptic transport. Preserve the existing launcher and explicit `--ee` selection.

**Tech Stack:** Python 3.10, Unitree SDK2 DDS, Vuer/Televuer, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-g1-quest-dex3-teleop-design.md`

## Global Constraints

- Work only on `SloBruno` remotes; no Unitree upstream PR, push, or remote change.
- `r`/`q` own lifecycle; Quest events never mutate START or STOP.
- Do not remove Inspire code or its service.
- Do not issue `Damp()` from Quest inputs.
- Dex3 triggers command thumb, index, and middle proportionally.
- Haptics fail closed when pressure/transport is invalid.

---

### Task 1: Publish a reproducible robot snapshot

**Files:**
- Modify: selected robot-source files already dirty in parent and submodules
- Exclude: `*.pre_mode_manager`, `g1_29_model_cache.pkl`, device-specific `cam_config_server.yaml`

- [ ] Audit every dirty parent/submodule path and classify portable source versus device-local state.
- [ ] Commit portable Televuer source to a personal Televuer fork; push that fork.
- [ ] Commit portable parent source and update the Televuer gitlink; push `feat/g1-quest-dex3-teleop` only to `SloBruno/xr_teleoperate`.
- [ ] Verify with a clean recursive clone of the personal branch.

### Task 2: Terminal lifecycle and joystick mapper

**Files:**
- Modify: `teleop/teleop_hand_and_arm.py`
- Create: `teleop/utils/quest_controls.py`
- Test: `tests/test_quest_controls.py`

**Interfaces:**
- Produces: `joystick_to_locomotion(left_xy, right_xy) -> tuple[float,float,float]`

- [ ] Write failing tests for zero, forward, lateral, yaw, and maximum normalized sticks.
- [ ] Run `pytest tests/test_quest_controls.py -q`; expect import failure.
- [ ] Implement direct mapping `(-left_y, -left_x, -right_x)` with finite-value clamping.
- [ ] Replace the `0.3` mapping and remove Quest-triggered `Damp()`; remove Quest-button lifecycle transitions.
- [ ] Re-run the focused tests and add a fake LocoClient test asserting `Move` receives the same tuple.

### Task 3: Dex3 trigger mapping

**Files:**
- Modify: `teleop/robot_control/robot_hand_unitree.py`, `teleop/teleop_hand_and_arm.py`, `teleop/televuer/src/televuer/televuer.py`, `teleop/televuer/src/televuer/tv_wrapper.py`
- Create: `teleop/utils/dex3_controls.py`
- Test: `tests/test_dex3_controls.py`

**Interfaces:**
- Produces: `trigger_to_dex3_targets(trigger: float, open_pose: ndarray, closed_pose: ndarray) -> ndarray`

- [ ] Write failing tests for released, midpoint, full trigger, NaN, and per-side isolation across all seven slots.
- [ ] Implement clamped interpolation for thumb0-2, index0-1, and middle0-1.
- [ ] Extend hand-tracking TeleData so left/right controller trigger analog values remain available with hands.
- [ ] In the Dex3 branch, overlay trigger targets onto the Dex3 command vector and preserve non-Dex3 controllers unchanged.
- [ ] Run focused tests with fake DDS publishers and confirm both side topics receive their side-specific command.

### Task 4: Pressure and haptic transport

**Files:**
- Modify: `teleop/robot_control/robot_hand_unitree.py`, `teleop/televuer/src/televuer/televuer.py`, `teleop/televuer/src/televuer/tv_wrapper.py`
- Create: `teleop/utils/haptics.py`
- Test: `tests/test_haptics.py`

**Interfaces:**
- Produces: `pressure_to_haptic(pressure: ndarray, maximum: float) -> float`

- [ ] Verify the actual Dex3 `HandState_` pressure fields and the installed Vuer haptic API before coding transport.
- [ ] Write failing tests: invalid/stale pressure gives 0; zero gives 0; calibrated max gives 1; monotonic finite pressures stay in `[0,1]`.
- [ ] Implement the pure pressure mapper with rate limiting and a contact deadband.
- [ ] Add per-side pressure extraction in the Dex3 state subscriber only after field names and channel mapping are verified.
- [ ] Add haptic emission through the verified Vuer API; unsupported sessions must no-op.
- [ ] Run focused tests with fake HandState and fake XR session.

### Task 5: Integration and deployment verification

**Files:**
- Modify: existing terminal launcher only if it fails to forward explicit `--ee`
- Test: all focused tests plus import/syntax checks

- [ ] Run `git diff --check` in parent and modified submodules.
- [ ] Run all added tests and `python -m py_compile` for modified Python modules.
- [ ] Confirm launcher retains Dex3/Inspire selection and never starts a lifecycle from Quest buttons.
- [ ] Commit focused changes and push only to the personal feature branch.
- [ ] Do not start robot services or move hardware; request physical-area approval before a live test.
