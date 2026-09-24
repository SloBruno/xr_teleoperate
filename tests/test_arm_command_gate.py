import threading
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from teleop.utils.arm_command_gate import publish_if_authorized


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
