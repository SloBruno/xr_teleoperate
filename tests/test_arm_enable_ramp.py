"""Arm enable ramp (blend_ratio 0 -> 1 between held pose and IK command)."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from teleop.utils.arm_enable_ramp import ArmEnableRamp  # noqa: E402


HOLD = np.zeros(14)
IK = np.full(14, 1.0)
TAU = np.full(14, 2.0)


def started(duration=1.0, t0=100.0):
    ramp = ArmEnableRamp(duration_s=duration)
    assert ramp.begin(t0, HOLD)
    return ramp


def test_first_command_equals_held_pose_and_ramp_is_monotonic_to_one():
    ramp = started()
    q, tau, alpha = ramp.apply(IK, TAU, 100.0)
    np.testing.assert_allclose(q, HOLD)
    np.testing.assert_allclose(tau, 0.0)
    assert alpha == 0.0
    previous = -1.0
    for t in np.linspace(100.0, 101.0, 21):
        q, tau, alpha = ramp.apply(IK, TAU, float(t))
        assert alpha >= previous
        np.testing.assert_allclose(q, alpha * IK)
        np.testing.assert_allclose(tau, alpha * TAU)
        previous = alpha
    assert alpha == 1.0
    assert ramp.complete


def test_midpoint_blend_and_after_duration_passthrough():
    ramp = started()
    q, _, alpha = ramp.apply(IK, TAU, 100.5)
    assert alpha == pytest.approx(0.5)  # smoothstep(0.5) == 0.5
    np.testing.assert_allclose(q, 0.5 * IK)
    q, tau, alpha = ramp.apply(IK * 3, TAU, 105.0)
    np.testing.assert_allclose(q, IK * 3)
    np.testing.assert_allclose(tau, TAU)
    assert alpha == 1.0


def test_smoothstep_has_zero_slope_at_ends():
    ramp = started()
    _, _, a_small = ramp.apply(IK, TAU, 100.01)
    assert a_small < 0.01 * 0.5  # slower than linear near 0


def test_repeated_begin_is_idempotent_and_does_not_restart():
    ramp = started()
    ramp.apply(IK, TAU, 100.6)
    assert ramp.begin(100.6, np.full(14, 5.0)) is False
    _, _, alpha = ramp.apply(IK, TAU, 100.6)
    assert alpha > 0.5
    q, _, _ = ramp.apply(IK, TAU, 101.5)
    np.testing.assert_allclose(q, IK)


def test_interrupt_stops_ramp_immediately_and_permanently():
    ramp = started()
    ramp.apply(IK, TAU, 100.3)
    ramp.interrupt()
    assert ramp.interrupted
    assert ramp.apply(IK, TAU, 100.4) is None
    assert ramp.begin(100.5, HOLD) is False
    assert ramp.apply(IK, TAU, 102.0) is None


def test_apply_before_begin_fails_closed():
    ramp = ArmEnableRamp(duration_s=1.0)
    assert ramp.apply(IK, TAU, 1.0) is None


@pytest.mark.parametrize("bad_now", [float("nan"), float("inf"), None, "x"])
def test_invalid_time_does_not_advance_alpha(bad_now):
    ramp = started()
    _, _, a1 = ramp.apply(IK, TAU, 100.2)
    _, _, a2 = ramp.apply(IK, TAU, bad_now)
    assert a2 == a1


def test_backwards_time_does_not_decrease_alpha():
    ramp = started()
    _, _, a1 = ramp.apply(IK, TAU, 100.6)
    _, _, a2 = ramp.apply(IK, TAU, 100.1)
    assert a2 == a1


@pytest.mark.parametrize("bad", [np.full(14, np.nan), np.full(13, 1.0), np.full((2, 7), 1.0)])
def test_invalid_command_fails_closed(bad):
    ramp = started()
    assert ramp.apply(bad, TAU, 100.5) is None
    assert ramp.apply(IK, np.full(14, np.inf), 100.5) is None


@pytest.mark.parametrize("bad_hold", [np.full(14, np.nan), np.zeros(3), None])
def test_invalid_hold_refuses_begin(bad_hold):
    ramp = ArmEnableRamp(duration_s=1.0)
    assert ramp.begin(1.0, bad_hold) is False
    assert not ramp.started


@pytest.mark.parametrize("bad_time", [float("nan"), None])
def test_invalid_begin_time_refuses_begin(bad_time):
    ramp = ArmEnableRamp(duration_s=1.0)
    assert ramp.begin(bad_time, HOLD) is False


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), 10.0])
def test_duration_validation(duration):
    with pytest.raises(ValueError):
        ArmEnableRamp(duration_s=duration)


def test_hold_is_copied():
    hold = np.zeros(14)
    ramp = ArmEnableRamp(duration_s=1.0)
    ramp.begin(100.0, hold)
    hold[:] = 7.0
    q, _, _ = ramp.apply(IK, TAU, 100.0)
    np.testing.assert_allclose(q, 0.0)
