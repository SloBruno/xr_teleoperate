"""Three-band Cartesian rate limiter (ported from IsaacTeleop EePoseRateLimiter)."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from teleop.utils.ee_rate_limiter import (  # noqa: E402
    BAND_CLAMP,
    BAND_FIRST,
    BAND_PASS,
    BAND_REACCEPT,
    BAND_REJECT_HOLD,
    DualEePoseRateLimiter,
    EeRateLimiterConfig,
)


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    m = np.eye(4)
    m[:3, :3] = rot_z(yaw)
    m[:3, 3] = [x, y, z]
    return m


def pair(lx=0.25, ly=0.15, lyaw=0.0, rx=0.25, ry=-0.15, ryaw=0.0):
    return pose(lx, ly, 0.1, lyaw), pose(rx, ry, 0.1, ryaw)


def rot_angle(a, b):
    r = a[:3, :3].T @ b[:3, :3]
    return math.acos(float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))


CFG = EeRateLimiterConfig(
    max_linear_velocity=0.5,
    max_angular_velocity=3.0,
    nominal_dt=0.05,
    min_dt=0.001,
    max_dt=0.15,
    reject_linear_velocity=3.0,
    reject_angular_velocity=20.0,
    max_consecutive_rejections=3,
)


def latched(t0=10.0, cfg=CFG):
    limiter = DualEePoseRateLimiter(cfg)
    result = limiter.limit(pair(), t0)
    assert result.bands == (BAND_FIRST, BAND_FIRST)
    limiter.commit()
    return limiter


def step(limiter, target, now):
    result = limiter.limit(target, now)
    assert result is not None
    limiter.commit()
    return result


def test_first_frame_passes_through_exactly():
    limiter = DualEePoseRateLimiter(CFG)
    target = pair(lyaw=0.3)
    result = limiter.limit(target, 5.0)
    np.testing.assert_allclose(result.targets[0], target[0])
    np.testing.assert_allclose(result.targets[1], target[1])


def test_band_a_small_step_passes_without_lag():
    limiter = latched()
    target = pair(lx=0.27, lyaw=0.05)  # 2 cm, 0.05 rad in 50 ms (< 2.5 cm, 0.15 rad)
    result = step(limiter, target, 10.05)
    assert result.bands == (BAND_PASS, BAND_PASS)
    np.testing.assert_allclose(result.targets[0], target[0], atol=1e-12)
    np.testing.assert_allclose(result.targets[1], target[1], atol=1e-12)


def test_band_b_linear_step_clamped_to_velocity_times_real_dt():
    limiter = latched()
    target = pair(lx=0.45)  # 20 cm in 80 ms -> 2.5 m/s (< reject 3 m/s)
    result = step(limiter, target, 10.08)
    assert result.bands[0] == BAND_CLAMP
    moved = result.targets[0][:3, 3] - pair()[0][:3, 3]
    assert np.linalg.norm(moved) == pytest.approx(0.5 * 0.08, rel=1e-9)
    # Straight line toward the target; the right side is untouched.
    assert moved[0] > 0 and abs(moved[1]) < 1e-12
    assert result.bands[1] == BAND_PASS


def test_band_b_converges_to_persistent_target_at_bounded_speed():
    limiter = latched()
    target = pair(lx=0.38)  # 13 cm in 50 ms = 2.6 m/s: clamp band, not reject
    now = 10.0
    for _ in range(40):
        now += 0.05
        result = step(limiter, target, now)
    np.testing.assert_allclose(result.targets[0][:3, 3], target[0][:3, 3], atol=1e-12)
    assert result.bands[0] == BAND_PASS


def test_band_b_rotation_clamped_along_geodesic():
    limiter = latched()
    target = pair(lyaw=0.6)  # 0.6 rad in 50 ms = 12 rad/s (> 3, < reject 20)
    result = step(limiter, target, 10.05)
    assert result.bands[0] == BAND_CLAMP
    assert rot_angle(pair()[0], result.targets[0]) == pytest.approx(3.0 * 0.05, rel=1e-6)
    # Moves along the same axis (pure yaw) and stays a proper rotation.
    r = result.targets[0][:3, :3]
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-9)
    assert np.linalg.det(r) == pytest.approx(1.0)
    np.testing.assert_allclose(r, rot_z(0.15), atol=1e-9)


def test_rotation_near_pi_is_clamped_without_nan():
    target = pair(lyaw=math.pi - 1e-7)
    # Reject tier off for this check.
    limiter = DualEePoseRateLimiter(EeRateLimiterConfig(max_linear_velocity=0.5, max_angular_velocity=3.0))
    limiter.limit(pair(), 1.0)
    limiter.commit()
    result = limiter.limit(target, 1.05)
    assert np.all(np.isfinite(result.targets[0]))
    assert rot_angle(pair()[0], result.targets[0]) == pytest.approx(0.15, rel=1e-5)


def test_band_c_large_jump_holds_then_reaccepts_after_n_frames_clamped():
    limiter = latched()
    far = pair(lx=0.60)  # 35 cm in 50 ms = 7 m/s > reject 3 m/s
    now = 10.0
    held = []
    for _ in range(CFG.max_consecutive_rejections):
        now += 0.05
        result = step(limiter, far, now)
        held.append(result.bands[0])
        np.testing.assert_allclose(result.targets[0], pair()[0])
    assert held == [BAND_REJECT_HOLD] * CFG.max_consecutive_rejections
    assert limiter.max_hold_run == CFG.max_consecutive_rejections
    now += 0.05
    result = step(limiter, far, now)
    assert result.bands[0] == BAND_REACCEPT
    # Re-accepted target is approached, still velocity clamped.
    moved = np.linalg.norm(result.targets[0][:3, 3] - pair()[0][:3, 3])
    assert moved == pytest.approx(0.5 * 0.05, rel=1e-9)
    # Next frames: input no longer anomalous relative to the accepted input.
    now += 0.05
    result = step(limiter, far, now)
    assert result.bands[0] == BAND_CLAMP


def test_band_c_single_glitch_is_ignored_and_tracking_resumes():
    limiter = latched()
    step(limiter, pair(lx=0.9), 10.05)  # glitch
    result = step(limiter, pair(lx=0.26), 10.10)
    assert result.bands[0] == BAND_PASS
    np.testing.assert_allclose(result.targets[0][:3, 3], pair(lx=0.26)[0][:3, 3])


def test_rotation_jump_triggers_reject_tier():
    limiter = latched()
    result = step(limiter, pair(lyaw=1.5), 10.05)  # 30 rad/s > 20
    assert result.bands[0] == BAND_REJECT_HOLD


@pytest.mark.parametrize("bad_now", [float("nan"), float("inf"), -float("inf"), None, "x"])
def test_invalid_timestamp_uses_nominal_dt(bad_now):
    limiter = latched()
    result = step(limiter, pair(lx=0.33), bad_now)
    assert result.dt == pytest.approx(CFG.nominal_dt)
    moved = np.linalg.norm(result.targets[0][:3, 3] - pair()[0][:3, 3])
    assert moved == pytest.approx(0.5 * CFG.nominal_dt, rel=1e-9)


def test_backwards_or_duplicate_timestamp_uses_nominal_dt():
    limiter = latched(t0=10.0)
    assert step(limiter, pair(lx=0.26), 9.0).dt == pytest.approx(CFG.nominal_dt)
    assert step(limiter, pair(lx=0.27), 9.0).dt == pytest.approx(CFG.nominal_dt)


def test_huge_gap_is_clamped_to_max_dt():
    limiter = latched()
    result = step(limiter, pair(lx=0.40), 10.0 + 5.0)
    assert result.dt == pytest.approx(CFG.max_dt)
    moved = np.linalg.norm(result.targets[0][:3, 3] - pair()[0][:3, 3])
    assert moved == pytest.approx(0.5 * CFG.max_dt, rel=1e-9)
    # A big gap does not make the step look anomalous (0.15 m / 0.15 s = 1 m/s).
    assert result.bands[0] == BAND_CLAMP


def test_tiny_dt_is_clamped_to_min_dt():
    limiter = latched()
    result = step(limiter, pair(lx=0.26), 10.0 + 1e-7)
    assert result.dt == pytest.approx(CFG.min_dt)


@pytest.mark.parametrize("bad", [
    lambda t: (np.full((4, 4), np.nan), t[1]),
    lambda t: (t[0], np.where(np.eye(4) > 0, np.inf, 0.0)),
    lambda t: (t[0][:3], t[1]),
    lambda t: (t[0],),
    lambda t: None,
    lambda t: (np.diag([2.0, 1.0, 1.0, 1.0]), t[1]),  # non-rigid
])
def test_invalid_target_fails_closed_and_does_not_advance(bad):
    limiter = latched()
    assert limiter.limit(bad(pair()), 10.05) is None
    # State untouched: the next good frame is judged against the latched pose.
    result = step(limiter, pair(lx=0.26), 10.10)
    assert result.bands[0] == BAND_PASS


def test_uncommitted_frame_does_not_move_the_reference():
    """The reference is the last command actually emitted (committed)."""
    limiter = latched()
    limiter.limit(pair(lx=0.35), 10.05)  # IK/gate rejected: no commit
    result = step(limiter, pair(lx=0.35), 10.10)
    moved = np.linalg.norm(result.targets[0][:3, 3] - pair()[0][:3, 3])
    # dt spans from the last commit (0.10 s), step from the latched pose.
    assert moved == pytest.approx(0.5 * 0.10, rel=1e-9)


def test_reset_relatches_next_frame():
    limiter = latched()
    limiter.reset()
    result = limiter.limit(pair(lx=0.6), 10.05)
    assert result.bands == (BAND_FIRST, BAND_FIRST)


def test_outputs_are_copies():
    limiter = latched()
    target = pair(lx=0.26)
    result = step(limiter, target, 10.05)
    target[0][0, 3] = 99.0
    result.targets[0][0, 3] = 42.0
    again = step(limiter, pair(lx=0.26), 10.10)
    assert again.targets[0][0, 3] == pytest.approx(0.26)


@pytest.mark.parametrize("kwargs", [
    dict(max_linear_velocity=0.0),
    dict(max_angular_velocity=float("nan")),
    dict(min_dt=0.2, nominal_dt=0.1, max_dt=0.15),
    dict(reject_linear_velocity=0.1),
    dict(reject_angular_velocity=1.0),
    dict(max_consecutive_rejections=0),
])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        EeRateLimiterConfig(**kwargs)


def test_emitted_step_never_exceeds_limit_on_random_walk():
    rng = np.random.default_rng(3)
    limiter = latched()
    last = pair()
    now = 10.0
    for _ in range(300):
        dt = float(rng.uniform(0.02, 0.2))
        now += dt
        yaw = float(rng.normal(0, 0.8))
        target = pair(lx=0.25 + float(rng.normal(0, 0.1)), lyaw=yaw)
        result = step(limiter, target, now)
        bound_dt = min(max(dt, CFG.min_dt), CFG.max_dt)
        assert np.linalg.norm(result.targets[0][:3, 3] - last[0][:3, 3]) <= 0.5 * bound_dt + 1e-9
        assert rot_angle(last[0], result.targets[0]) <= 3.0 * bound_dt + 1e-6
        last = result.targets
