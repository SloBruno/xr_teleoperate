import threading
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from teleop.utils.arm_tracking_orchestration import run_arm_tracking_cycle


class FakeArm:
    def __init__(self):
        self.measured_q = np.arange(14, dtype=float)
        self.commands = []
        self.deactivated = False

    def get_current_dual_arm_q(self):
        return self.measured_q

    def ctrl_dual_arm(self, q, tau):
        self.commands.append((np.asarray(q).copy(), np.asarray(tau).copy()))

    def deactivate(self):
        self.deactivated = True


class FakeIK:
    def __init__(self):
        self.calls = []

    def solve_ik(self, left, right, current_q, current_dq):
        self.calls.append((left, right))
        return np.full(14, 9.0), np.full(14, 3.0)


class FakeCalibrator:
    calibrated = True

    def __init__(self, target):
        self.target = target
        self.calls = []

    def targets(self, poses, sample_timestamp, now=None):
        self.calls.append((poses, sample_timestamp))
        return self.target


def poses():
    left = np.eye(4)
    right = np.eye(4)
    left[0, 3] = 1.0
    right[0, 3] = -1.0
    return left, right


def run(**overrides):
    arm = overrides.pop("arm", FakeArm())
    ik = overrides.pop("ik", FakeIK())
    result = run_arm_tracking_cycle(
        arm_ctrl=arm,
        arm_ik=ik,
        calibrator=overrides.pop("calibrator", FakeCalibrator(poses())),
        controller_poses=poses(),
        sample_timestamp=overrides.pop("sample_timestamp", 10.1),
        now=overrides.pop("now", 10.1),
        current_q=arm.measured_q,
        current_dq=np.zeros(14),
        first_target=overrides.pop("first_target", None),
        lifecycle_lock=threading.Lock(),
        is_started=overrides.pop("is_started", lambda: True),
        is_stopped=overrides.pop("is_stopped", lambda: False),
        **overrides,
    )
    return result, arm, ik


def test_calibrated_first_target_solves_then_publishes():
    first = poses()
    result, arm, ik = run(first_target=first)
    assert result.target_accepted
    assert len(ik.calls) == 1
    np.testing.assert_allclose(arm.commands[0][0], 9.0)


def test_stop_race_holds_measured_pose_and_deactivates_without_publishing_ik():
    result, arm, ik = run(is_stopped=lambda: True)
    assert not result.published
    assert not ik.calls
    np.testing.assert_allclose(arm.commands[0][0], arm.measured_q)
    assert arm.deactivated


def test_rejected_workspace_target_holds_without_ik():
    result, arm, ik = run(calibrator=FakeCalibrator(None))
    assert not result.target_accepted
    assert not ik.calls
    np.testing.assert_allclose(arm.commands[0][0], arm.measured_q)


def test_stale_and_invalid_targets_hold_without_ik():
    stale, stale_arm, stale_ik = run(sample_timestamp=1.0, now=10.1)
    assert not stale.target_accepted
    assert not stale_ik.calls
    invalid, invalid_arm, invalid_ik = run(calibrator=SimpleNamespace(calibrated=False))
    assert not invalid.target_accepted
    assert not invalid_ik.calls
