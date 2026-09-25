import threading
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from teleop.utils.arm_command_gate import publish_arm_command, publish_if_authorized


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


def test_rejected_target_publishes_measured_zero_hold_not_new_ik_q():
    arm = FakeArm()
    ik_q = np.full(14, 99.0)
    assert not publish_if_authorized(
        arm, ik_q, np.ones(14), target_accepted=False, sample_fresh=True,
        lifecycle_lock=threading.Lock(), is_started=lambda: True, is_stopped=lambda: False,
    )
    assert np.array_equal(arm.commands[0][0], arm.measured_q)
    assert np.array_equal(arm.commands[0][1], np.zeros(14))
    assert not np.array_equal(arm.commands[0][0], ik_q)


def test_stop_wins_final_check_and_deactivates_without_ik_q():
    arm = FakeArm()
    ik_q = np.full(14, 99.0)
    assert not publish_if_authorized(
        arm, ik_q, np.ones(14), target_accepted=True, sample_fresh=True,
        lifecycle_lock=threading.Lock(), is_started=lambda: False, is_stopped=lambda: True,
    )
    assert np.array_equal(arm.commands[0][0], arm.measured_q)
    assert np.array_equal(arm.commands[0][1], np.zeros(14))
    assert arm.deactivated


def test_authorized_target_is_the_only_new_q_published():
    arm = FakeArm()
    ik_q = np.full(14, 99.0)
    assert publish_if_authorized(
        arm, ik_q, np.ones(14), target_accepted=True, sample_fresh=True,
        lifecycle_lock=threading.Lock(), is_started=lambda: True, is_stopped=lambda: False,
    )
    assert np.array_equal(arm.commands[0][0], ik_q)


def test_rejected_cycles_keep_one_frozen_hold_reference():
    arm = FakeArm()
    lock = threading.Lock()

    first = publish_arm_command(
        arm, np.full(14, 99.0), np.ones(14), target_accepted=False,
        sample_fresh=True, lifecycle_lock=lock,
        is_started=lambda: True, is_stopped=lambda: False,
    )
    arm.measured_q = arm.measured_q - 0.2
    second = publish_arm_command(
        arm, np.full(14, 99.0), np.ones(14), target_accepted=False,
        sample_fresh=True, lifecycle_lock=lock,
        is_started=lambda: True, is_stopped=lambda: False,
    )

    np.testing.assert_array_equal(second.selected_q, first.selected_q)
    np.testing.assert_array_equal(arm.commands[-1][0], first.selected_q)


def test_valid_command_releases_frozen_hold_for_next_rejection():
    arm = FakeArm()
    lock = threading.Lock()
    publish_arm_command(
        arm, np.zeros(14), np.zeros(14), target_accepted=False,
        sample_fresh=True, lifecycle_lock=lock,
        is_started=lambda: True, is_stopped=lambda: False,
    )
    arm.measured_q = arm.measured_q + 0.5
    publish_arm_command(
        arm, np.ones(14), np.ones(14), target_accepted=True,
        sample_fresh=True, lifecycle_lock=lock,
        is_started=lambda: True, is_stopped=lambda: False,
    )
    next_hold = publish_arm_command(
        arm, np.zeros(14), np.zeros(14), target_accepted=False,
        sample_fresh=True, lifecycle_lock=lock,
        is_started=lambda: True, is_stopped=lambda: False,
    )
    np.testing.assert_array_equal(next_hold.selected_q, arm.measured_q)


def test_nonfinite_ik_output_never_reaches_arm_controller():
    arm = FakeArm()
    decision = publish_arm_command(
        arm, np.full(14, np.nan), np.full(14, np.inf),
        target_accepted=True, sample_fresh=True,
        lifecycle_lock=threading.Lock(),
        is_started=lambda: True, is_stopped=lambda: False,
    )

    assert decision.hold
    assert not decision.published
    assert np.isfinite(arm.commands[-1][0]).all()
    assert np.isfinite(arm.commands[-1][1]).all()
