# Task 1 report — Publish a reproducible robot snapshot

Date: 2026-09-16
Parent branch: `feat/g1-quest-dex3-teleop`
Parent base/docs commit: `64cbcd7`

## Dirty-tree audit before staging

Initial parent status classified as follows:

- Portable source staged: `teleop/robot_control/robot_arm.py`,
  `teleop/robot_control/robot_hand_inspire.py`, and
  `teleop/teleop_hand_and_arm.py`.
- Portable Televuer snapshot: submodule advanced from `766de45` to
  `9ec7df248a25b48199a42cb4adf436f52cace2e4`; its commit contains only
  `src/televuer/televuer.py` and `src/televuer/tv_wrapper.py`.
- Reproducibility metadata staged: `.gitmodules` Televuer URL changed to
  `https://github.com/SloBruno/televuer.git`.
- Excluded parent-local state: `teleop/run_teleop.sh`,
  `teleop/robot_control/robot_arm.py.pre_mode_manager`,
  `teleop/teleop_hand_and_arm.py.pre_mode_manager`, and ignored
  `teleop/g1_29_model_cache.pkl`.
- Excluded submodule-local state: `teleop/televuer/src/televuer/televuer.py.pre_mode_manager`
  and `teleop/televuer/src/televuer/tv_wrapper.py.pre_mode_manager`.
- Excluded device-specific state: modified
  `teleop/teleimager/cam_config_server.yaml`; the Teleimager gitlink was not
  staged or changed.
- No keys, secrets, Dex retargeting changes, or other device-local files were
  staged. Dex retargeting remained at `d7753d38c9ff11f80bafea6cd168351fd3db9b0e`.

Commands/evidence:

```text
git diff --name-only
teleop/robot_control/robot_arm.py
teleop/robot_control/robot_hand_inspire.py
teleop/teleimager
teleop/teleop_hand_and_arm.py
teleop/televuer

git diff --cached --name-status
M .gitmodules
M teleop/robot_control/robot_arm.py
M teleop/robot_control/robot_hand_inspire.py
M teleop/teleop_hand_and_arm.py
M teleop/televuer

git diff --cached --check
exit 0; no output
```

The Televuer local `origin` was repointed from Unitree to the personal URL;
no Unitree remote URL remains in that submodule. The snapshot push reported
`Everything up-to-date` for `9ec7df2`.

## Commits and pushes

- Televuer personal fork: `9ec7df248a25b48199a42cb4adf436f52cace2e4`, branch
  `snapshot/g1-quest-dex3-teleop`, pushed to `SloBruno/televuer`.
- Parent: `3416c4470b8a006ef2640c46c5096d2752caf651`, message
  `Publish G1 Quest Dex3 teleoperation snapshot`, pushed to
  `SloBruno/xr_teleoperate` branch `feat/g1-quest-dex3-teleop`.
- Remote branch evidence:

```text
git ls-remote https://github.com/SloBruno/xr_teleoperate.git refs/heads/feat/g1-quest-dex3-teleop
3416c4470b8a006ef2640c46c5096d2752caf651  refs/heads/feat/g1-quest-dex3-teleop
```

## Verification

Source-only syntax check:

```text
python -m py_compile teleop/robot_control/robot_arm.py teleop/robot_control/robot_hand_inspire.py teleop/teleop_hand_and_arm.py
exit 0
```

A fresh local clone of the committed parent branch was made with
`--no-local --no-recurse-submodules`; only `teleop/televuer` was initialized.
The HTTPS clone from GitHub stalled before checkout in this environment, so
the pushed branch was separately verified with `git ls-remote`, and the
content verification used the fresh local clone of the exact pushed commit.

```text
clone_head=3416c4470b8a006ef2640c46c5096d2752caf651
televuer_head=9ec7df248a25b48199a42cb4adf436f52cace2e4
git diff --check: exit 0; no output
git status --short --branch: clean
teleimager_initialized=no
dex_retargeting_initialized=no
python -m py_compile [three parent source files]: exit 0
clone_verified=true
```

The clone registered Televuer from `https://github.com/SloBruno/televuer.git`
and did not initialize Teleimager or Dex retargeting.

## Concerns

- Device-local files remain dirty/untracked in the working tree by design and
  were not committed or pushed.
- No robot services, hardware, or runtime teleoperation were started.
- No full runtime test suite was run because this task is snapshot publication
  and runtime verification would require device/hardware state.

## Round 1 review fixes

- Published the initialized Teleimager snapshot to `SloBruno/teleimager` at
  `2aab15d9601865ab6bee334ae26839e0306b0770`.
- Published the initialized Dex retargeting snapshot to
  `SloBruno/dex-retargeting` at
  `d7753d38c9ff11f80bafea6cd168351fd3db9b0e`.
- Retargeted all parent submodule URLs in committed `.gitmodules`, local
  submodule config, and initialized submodule remotes to SloBruno URLs.
- Added guarded vertical camera image stacking and used it in both startup
  and main loops. Missing frame objects or `.bgr` values return `None`, while
  the existing head-only fallback remains unchanged.
- The requested `ledger.md` was not present in the task directory; the
  available `progress.md` was read as the SDD ledger.

### Camera RED output (before implementation)

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_______________ test_vertical_stack_skips_missing_camera_frames ________________

>   ???
E   AttributeError: module 'teleop.teleop_hand_and_arm' has no attribute 'stack_camera_images_vertical'. Did you mean: 'stack_camera_frames_vertical'?

tests/test_vertical_camera_frames.py:51: AttributeError
=========================== short test summary info ============================
FAILED tests/test_vertical_camera_frames.py::test_vertical_stack_skips_missing_camera_frames
1 failed in 0.16s
```

### Camera GREEN output (after minimal implementation)

```text
.                                                                        [100%]
1 passed in 0.15s
```
