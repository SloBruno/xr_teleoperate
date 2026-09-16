# Task 4: Pressure and haptic transport

## Evidence

- No prior Task 4 review/report file was present in this checkout or reachable git history.
- Local test-only `.venv-haptics` verification: `vuer==0.0.60`; `from vuer.schemas import MotionControllers` imports successfully.
- Verified constructor signature: `MotionControllers(key='motionControllers', eventTypes=('trigger', 'squeeze'), stream=True, left=None, right=None, **kwargs)`.
- Verified Vuer session API: `session.upsert` is an `@` proxy, so haptic emission uses `session.upsert @ MotionControllers(...)`.
- Existing established scene key is `motionControllers`.

## Implementation

- Dex3 DDS subscriber publishes side-local peak pressure with a `time.monotonic()` timestamp.
- The parent loop forwards both samples into shared Televuer values; the Vuer process maps each side independently through `PressureHapticMapper`.
- Only fresh finite positive mapped contact emits. Emissions are bounded to `[0, 1]`, rate-limited, deadbanded, duration-limited to 250 ms, and fail closed for absent sessions/errors.
- Left and right pulses use only their side-specific strength, duration, and unique `puseLeftHash`/`puseRightHash` kwargs.

## Tests

- RED: the new transport tests failed before implementation because the adapter had no mapper handoff API and no `MotionControllers` symbol; the prior focused tests remained green.
- GREEN: `PYTHONPATH=.:teleop/televuer/src pytest -q tests` → `32 passed`.
- `python -m py_compile` passed for all modified Python modules.
- `git diff --check` passed in the parent and Televuer submodule.
- An actual `.venv-haptics` Vuer 0.0.60 construction/emission check passed with the verified left-side fields present on the `MotionControllers` element.
- No robot hardware, DDS services, or external services were started.

## Round 2 safety fix evidence

- RED: `PYTHONPATH=.:teleop/televuer/src pytest -q tests/test_haptics.py::test_timestamped_zero_after_contact_suppresses_haptic_pulse` failed because a timestamped zero after a max-pressure pulse returned `True` and emitted a second pulse.
- Fix: zero/non-contact mapper outputs now reset state and return zero before rate limiting; invalid timestamp conversion also resets the side mapper. Positive finite contact remains rate-limited.
- GREEN: the focused regression test passed, followed by `PYTHONPATH=.:teleop/televuer/src pytest -q tests` → `33 passed`.
