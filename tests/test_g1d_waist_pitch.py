"""G1-D waist pitch from the Quest controller (teleop/utils/g1d_waist_pitch.py).

Executable tests against the REAL vendored controller (pitch_ct.
ComputedTorqueController + body_model/Pinocchio + identified ct_params.json)
and, for closed-loop behaviour, the vendored whole-body pitch plant
(ct_sim/body_plant.py) driven by a deterministic simulated clock. The DDS bus
is a fake; nothing here touches DDS or hardware.
"""
import hashlib
import json
import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("pinocchio")

from teleop.utils import g1d_waist_pitch as wp  # noqa: E402

STACK = wp.load_pitch_stack()
BODY = wp.build_body_model(STACK)
ARM_SLOTS = STACK.body_state.ARM_SLOTS

NEUTRAL_ARMS = {n: 0.0 for n in ARM_SLOTS}
# arms extended straight forward relative to the torso
EXTENDED_ARMS = dict(NEUTRAL_ARMS, left_shoulder_pitch_joint=-1.57, right_shoulder_pitch_joint=-1.57)
# worst-case pose found by the cap search at q=0.30 (arm reaching forward, horizontal in the world)
_HEAVY_LEFT = (-1.86, 0.25, 1.51, 1.37, 0.21, 0.08, 0.03)
_MIRROR = (1, -1, -1, 1, -1, 1, -1)
_JOINTS = ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint",
           "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint")
HEAVY_ARMS = dict(NEUTRAL_ARMS)
for _j, _v, _m in zip(_JOINTS, _HEAVY_LEFT, _MIRROR):
    HEAVY_ARMS["left_" + _j] = _v
    HEAVY_ARMS["right_" + _j] = _v * _m


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------
class FakeArmSource:
    """Mimics G1D_ArmController.lowstate_buffer.GetSnapshot() (slots 0..28)."""

    def __init__(self, arms=None, clock=None):
        self.arms = dict(arms or NEUTRAL_ARMS)
        self.clock = clock
        self.stale = False
        self.stamp = None

    def snapshot(self):
        if self.stale:
            return (self._states(), self.stamp)
        self.stamp = self.clock()
        return (self._states(), self.stamp)

    def _states(self):
        ms = [SimpleNamespace(q=0.0, dq=0.0) for _ in range(29)]
        for n, i in ARM_SLOTS.items():
            ms[i].q = self.arms.get(n, 0.0)
        return ms


class SimClock:
    def __init__(self, t0=1000.0):
        self.t = t0

    def __call__(self):
        return self.t


class PlantBus:
    """Fake rt/waistpitch{cmd,state} bus around the vendored whole-body plant."""

    def __init__(self, clock, q0=0.0, arm_source=None, tau_limit=85.0, foreign=None, create_ok=True):
        sys.path.insert(0, str(wp.VENDOR_DIR / "ct_sim"))
        from body_plant import BodyPlant  # vendored, test-only plant
        self.clock = clock
        self.arm_source = arm_source
        self.plant = BodyPlant(BODY, q0=q0, tau_limit=tau_limit,
                               arm_traj=lambda t: (dict(self.arm_source.arms) if self.arm_source else {}, {}, {}))
        self.published = []
        self.writer_created = False
        self.create_calls = 0
        self.create_ok = create_ok
        self.foreign = list(foreign or [])
        self.state_stale = False
        self.temperature = (35, 36)
        self._last_t = clock()
        self._stamp = clock()

    def advance(self):
        now = self.clock()
        dt = max(0.0, now - self._last_t)
        if dt > 0:
            self.plant.step(dt)
        self._last_t = now
        if not self.state_stale:
            self._stamp = now

    def read_state(self):
        st = self.plant.state()
        st.temperature = self.temperature
        return st, self._stamp

    def foreign_writers(self):
        return list(self.foreign)

    def create_cmd_writer(self):
        self.create_calls += 1
        self.writer_created = True
        return self.create_ok

    def publish_cmd(self, writer_id, cmd, now=None):
        assert self.writer_created, "published before the writer was created"
        self.published.append((self.clock(), cmd))
        self.plant.cmd = cmd

    def close(self):
        pass


def make(q0=0.0, arms=None, **bus_kw):
    clock = SimClock()
    arm_source = FakeArmSource(arms, clock)
    bus = PlantBus(clock, q0=q0, arm_source=arm_source, **bus_kw)
    ctrl = wp.build_teleop_controller(STACK, BODY)
    sup = wp.WaistPitchSupervisor(ctrl, bus, wp.make_arm_filter(STACK, BODY), arm_source.snapshot,
                                  clock=clock, log=lambda *a, **k: None, main_alive=lambda: True)
    return SimpleNamespace(clock=clock, bus=bus, arms=arm_source, ctrl=ctrl, sup=sup)


def run(env, seconds, stick=None, failsafe=False, submit=True):
    """Advance the simulated world at the supervisor's own rate.
    stick=None -> no new Quest samples (they age); otherwise submit each cycle."""
    n = int(round(seconds / wp.CTRL_DT))
    for _ in range(n):
        env.clock.t += wp.CTRL_DT
        env.bus.advance()
        if submit and stick is not None:
            env.sup.submit(stick, failsafe, env.clock())
        env.sup.step()


def arm_now(env):
    env.sup.request_arm(start_thread=False)
    run(env, 0.1, stick=(0.0, 0.0))
    assert env.sup.state == wp.ACTIVE, env.sup.last_reason


# --------------------------------------------------------------------------
# mapping, deadzone, stale input
# --------------------------------------------------------------------------
def test_mapping_uses_right_stick_horizontal_axis_with_rate_cap():
    now = 50.0
    assert wp.pitch_rate_from_stick((0.0, 0.0), now, now) == 0.0
    assert wp.pitch_rate_from_stick((1.0, 0.0), now, now) == pytest.approx(wp.PITCH_RATE_MAX)
    assert wp.pitch_rate_from_stick((-1.0, 0.0), now, now) == pytest.approx(-wp.PITCH_RATE_MAX)
    half = wp.pitch_rate_from_stick((0.5, 0.0), now, now)
    assert 0.0 < half < wp.PITCH_RATE_MAX
    # vertical axis (lift column in later releases) never moves the pitch
    assert wp.pitch_rate_from_stick((0.0, 1.0), now, now) == 0.0
    assert wp.pitch_rate_from_stick((0.0, -1.0), now, now) == 0.0
    # out-of-range values are clamped to the cap
    assert wp.pitch_rate_from_stick((7.0, 0.0), now, now) == pytest.approx(wp.PITCH_RATE_MAX)
    assert wp.PITCH_RATE_MAX <= 0.10


def test_deadzone_and_invalid_values_map_to_zero():
    now = 50.0
    for x in (0.05, -0.05, 0.12, -0.12):
        assert wp.pitch_rate_from_stick((x, 0.0), now, now) == 0.0
    for bad in ((math.nan, 0.0), (math.inf, 0.0), (-math.inf, 0.0), None, (1.0,), "ab", object(), ("x", 0)):
        assert wp.pitch_rate_from_stick(bad, now, now) == 0.0


def test_stale_or_invalid_sample_timestamp_holds():
    now = 50.0
    assert wp.pitch_rate_from_stick((1.0, 0.0), now - 0.24, now) > 0.0
    assert wp.pitch_rate_from_stick((1.0, 0.0), now - 0.26, now) == 0.0
    assert wp.pitch_rate_from_stick((1.0, 0.0), 0.0, now) == 0.0
    assert wp.pitch_rate_from_stick((1.0, 0.0), math.nan, now) == 0.0
    assert wp.pitch_rate_from_stick((1.0, 0.0), now + 1.0, now) == 0.0   # future sample
    assert wp.STICK_STALE_S == 0.25


# --------------------------------------------------------------------------
# limits: hard cap, neutral, speed, worst-case return proof
# --------------------------------------------------------------------------
def test_controller_limits_neutral_and_speed():
    ctrl = wp.build_teleop_controller(STACK, BODY)
    assert wp.PITCH_Q_MAX <= 0.80
    assert ctrl.p.q_hi == wp.PITCH_Q_MAX
    assert ctrl.p.home_q == 0.0 == wp.PITCH_HOME_Q
    assert ctrl.p.q_lo == 0.0
    assert ctrl.p.v_max <= wp.PITCH_RATE_MAX
    assert ctrl.model_budget == wp.MODEL_BUDGET_NM
    assert isinstance(ctrl, STACK.pitch_ct.ComputedTorqueController)


def _worst_case_lift(q, payload_per_hand):
    """Max predicted lift-from-hold torque over a dense arm-pose search at q."""
    body = STACK.body_model.BodyModel(payload_per_hand=payload_per_hand, scale=BODY.scale, offset=BODY.offset)
    fr = wp.load_ct_friction(wp.CT_PARAMS_FILE)
    m = body.model
    names = ["shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint",
             "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint"]
    import numpy as np
    lo = np.array([m.lowerPositionLimit[body.arm_idx["left_" + n][0]] for n in names])
    hi = np.array([m.upperPositionLimit[body.arm_idx["left_" + n][0]] for n in names])
    mirror = np.array([1, -1, -1, 1, -1, 1, -1])
    rlo = np.array([m.lowerPositionLimit[body.arm_idx["right_" + n][0]] for n in names])
    rhi = np.array([m.upperPositionLimit[body.arm_idx["right_" + n][0]] for n in names])

    def need(x):
        xr = np.clip(x * mirror, rlo, rhi)
        arms = {"left_" + n: float(x[i]) for i, n in enumerate(names)}
        arms.update({"right_" + n: float(xr[i]) for i, n in enumerate(names)})
        g = body.gravity(q, arms)
        gb = (g - body.offset) / body.scale if fr[3] == "raw" else g
        return STACK.pitch_ct.lift_breakaway_nm(g, fr, gb)

    rng = np.random.default_rng(1)
    best, bx = -1.0, None
    for _ in range(1500):
        x = lo + rng.random(7) * (hi - lo)
        n = need(x)
        if n > best:
            best, bx = n, x
    step = (hi - lo) / 8
    for _ in range(40):
        improved = False
        for i in range(7):
            for s in (-1, 1):
                x = bx.copy()
                x[i] = np.clip(x[i] + s * step[i], lo[i], hi[i])
                n = need(x)
                if n > best + 1e-9:
                    best, bx, improved = n, x, True
        if not improved:
            step = step / 2
    return best


def test_hard_cap_keeps_worst_case_return_within_budget_for_any_arm_pose():
    """ct_params.json + body_model + breakaway law (2.14|g| + 3.9):
    any arm pose inside the URDF limits, incl. extended forward, Dex3 hands
    (in the URDF) plus a 0.25 kg object per hand, returns with <= 85 N.m."""
    assert _worst_case_lift(wp.PITCH_Q_MAX, 0.0) <= wp.WORST_CASE_BUDGET_NM - 5.0
    assert _worst_case_lift(wp.PITCH_Q_MAX, 0.25) <= wp.WORST_CASE_BUDGET_NM
    # and the previously proposed 0.80 rad would NOT be safe for every pose
    assert _worst_case_lift(0.80, 0.0) > 110.0


# --------------------------------------------------------------------------
# lifecycle: nothing before arming, single owner
# --------------------------------------------------------------------------
def test_nothing_published_or_created_before_arming_even_with_inputs():
    env = make()
    run(env, 2.0, stick=(1.0, 0.0), failsafe=True)
    assert env.bus.published == []
    assert env.bus.create_calls == 0
    assert env.sup.state == wp.IDLE
    assert not env.sup.failsafe_latched   # pre-arm presses are ignored


def test_arming_creates_single_writer_and_holds_measured_pose():
    env = make(q0=0.0)
    arm_now(env)
    assert env.bus.create_calls == 1
    assert env.bus.published
    first = env.bus.published[0][1]
    assert first.mode == 1 and abs(first.q - 0.0) < 1e-6
    run(env, 1.0, stick=(0.0, 0.0))
    assert env.bus.create_calls == 1
    assert abs(env.bus.plant.q) < 0.005


def test_foreign_writer_at_arm_refuses_without_creating_or_publishing():
    env = make(foreign=["other-guid"])
    env.sup.request_arm(start_thread=False)
    run(env, 1.0, stick=(1.0, 0.0))
    assert env.sup.state == wp.REFUSED
    assert env.bus.create_calls == 0
    assert env.bus.published == []


def test_unconfirmed_self_guid_refuses_without_publishing():
    env = make(create_ok=False)
    env.sup.request_arm(start_thread=False)
    run(env, 1.0, stick=(1.0, 0.0))
    assert env.sup.state == wp.REFUSED
    assert env.bus.published == []


def test_stale_pitch_state_at_arm_refuses_after_timeout_without_publishing():
    env = make()
    env.bus.state_stale = True
    env.clock.t += 1.0
    env.sup.request_arm(start_thread=False)
    run(env, wp.ARM_TIMEOUT_S + 0.5, stick=(0.0, 0.0))
    assert env.sup.state == wp.REFUSED
    assert env.bus.published == []
    assert env.bus.create_calls == 0


def test_stale_arm_state_at_arm_refuses_without_publishing():
    env = make()
    env.arms.stale = True
    env.arms.stamp = env.clock() - 5.0
    env.sup.request_arm(start_thread=False)
    run(env, wp.ARM_TIMEOUT_S + 0.5, stick=(0.0, 0.0))
    assert env.sup.state == wp.REFUSED
    assert env.bus.published == []


def test_stop_before_arming_completes_never_creates_a_writer():
    env = make()
    env.bus.state_stale = True
    env.clock.t += 1.0
    env.sup.request_arm(start_thread=False)
    run(env, 0.2, stick=(0.0, 0.0))
    env.sup.request_home_and_release("teleop stop")
    run(env, 0.2, stick=(0.0, 0.0))
    assert env.sup.finished and env.sup.state == wp.DONE
    assert env.bus.create_calls == 0 and env.bus.published == []


# --------------------------------------------------------------------------
# joystick -> rate -> jog API
# --------------------------------------------------------------------------
def test_full_stick_moves_forward_at_most_at_rate_cap_and_release_holds():
    env = make()
    arm_now(env)
    q0 = env.bus.plant.q
    run(env, 1.5, stick=(1.0, 0.0))
    q1 = env.bus.plant.q
    assert q1 > q0 + 0.05
    assert q1 - q0 <= wp.PITCH_RATE_MAX * 1.5 + 0.01
    assert env.bus.plant.max_abs_dq <= wp.PITCH_RATE_MAX * 1.3
    # release: holds (no further motion, still publishing)
    run(env, 0.3, stick=(0.0, 0.0))
    q_rel = env.bus.plant.q
    n_pub = len(env.bus.published)
    run(env, 2.0, stick=(0.0, 0.0))
    assert abs(env.bus.plant.q - q_rel) < 0.01
    assert len(env.bus.published) > n_pub
    assert env.sup.state == wp.ACTIVE


def test_stick_left_returns_toward_neutral_and_not_below_zero():
    env = make()
    arm_now(env)
    run(env, 1.5, stick=(1.0, 0.0))
    assert env.bus.plant.q > 0.08
    run(env, 4.0, stick=(-1.0, 0.0))
    assert env.bus.plant.q < 0.02
    assert env.ctrl.target >= 0.0
    assert env.bus.plant.q > -0.01


def test_stale_quest_input_holds_without_motion():
    env = make()
    arm_now(env)
    run(env, 0.8, stick=(1.0, 0.0))
    target_at_loss = env.ctrl.target
    run(env, 3.0, stick=None)               # no new samples: they age past 0.25 s
    q_after = env.bus.plant.q
    run(env, 2.0, stick=None)
    assert env.ctrl.target <= target_at_loss + 1e-9
    assert abs(env.bus.plant.q - q_after) < 0.005
    assert env.sup.state == wp.ACTIVE


def test_hard_cap_is_never_exceeded_with_stick_held_forward():
    env = make()
    arm_now(env)
    run(env, 8.0, stick=(1.0, 0.0))
    assert env.ctrl.target <= wp.PITCH_Q_MAX + 1e-9
    assert env.bus.plant.q <= wp.PITCH_Q_MAX + 0.02
    assert all(c.q <= wp.PITCH_Q_MAX + 0.03 for _, c in env.bus.published)


def test_model_admission_refuses_forward_when_return_exceeds_budget_with_measured_arms():
    env = make(arms=HEAVY_ARMS)
    arm_now(env)
    run(env, 8.0, stick=(1.0, 0.0))
    # with the arms reaching forward the CT admission stops short of the cap
    assert env.ctrl.target < wp.PITCH_Q_MAX - 0.01
    assert env.ctrl.predicted_lift(env.ctrl.target) <= wp.MODEL_BUDGET_NM + 1e-6
    assert env.sup.admission_refusals > 0


def test_arms_extended_forward_reach_the_cap_and_still_return():
    env = make(arms=EXTENDED_ARMS)
    arm_now(env)
    run(env, 6.0, stick=(1.0, 0.0))
    assert env.bus.plant.q > wp.PITCH_Q_MAX - 0.03
    env.arms.arms = dict(HEAVY_ARMS)            # operator then reaches further while leaned
    env.sup.request_home_and_release("teleop stop")
    run(env, 15.0, stick=None)
    _assert_released_only_at_neutral(env)


def test_forward_refused_when_arm_state_goes_stale():
    env = make()
    arm_now(env)
    env.arms.stale = True
    run(env, 1.0, stick=(1.0, 0.0))
    assert env.ctrl.target < 0.01


# --------------------------------------------------------------------------
# fail-safe button -> neutral (q = 0.0) -> release
# --------------------------------------------------------------------------
def _assert_released_only_at_neutral(env):
    assert env.sup.state == wp.RELEASED
    assert env.ctrl.released
    last_t, last = env.bus.published[-1]
    assert abs(env.bus.plant.q - wp.PITCH_HOME_Q) <= 0.012
    assert abs(last.q - wp.PITCH_HOME_Q) <= 0.012
    n = len(env.bus.published)
    run(env, 1.0, stick=(1.0, 0.0))
    assert len(env.bus.published) == n     # nothing after release


def test_failsafe_button_homes_to_zero_and_releases_only_at_neutral():
    env = make()
    arm_now(env)
    run(env, 3.0, stick=(1.0, 0.0))
    assert env.bus.plant.q > 0.2
    env.sup.submit((1.0, 0.0), True, env.clock())   # left Y pressed
    assert env.sup.failsafe_latched
    run(env, 12.0, stick=(1.0, 0.0))                # stick ignored while homing
    _assert_released_only_at_neutral(env)
    assert env.bus.plant.max_abs_dq <= 0.15


def test_failsafe_target_is_exact_zero_even_when_armed_away_from_zero():
    env = make(q0=0.2)
    arm_now(env)
    assert abs(env.ctrl.q_ref - 0.2) < 0.01        # armed holding measured 0.2
    env.sup.submit((0.0, 0.0), True, env.clock())
    run(env, 0.1, stick=(0.0, 0.0))
    assert env.ctrl.target == 0.0
    assert env.ctrl.homing_then_release
    run(env, 10.0, stick=(0.0, 0.0))
    _assert_released_only_at_neutral(env)


def test_failsafe_latch_needs_a_fresh_sample_and_survives_release_of_the_button():
    env = make()
    arm_now(env)
    run(env, 2.0, stick=(1.0, 0.0))
    env.sup.submit((0.0, 0.0), True, env.clock() - 5.0)   # stale (late) sample: ignored
    assert not env.sup.failsafe_latched
    env.sup.submit((0.0, 0.0), True, env.clock())          # held Y on a fresh sample
    assert env.sup.failsafe_latched
    env.sup.submit((1.0, 0.0), False, env.clock())         # button released: stays latched
    assert env.sup.failsafe_latched
    run(env, 12.0, stick=(1.0, 0.0))
    _assert_released_only_at_neutral(env)


def test_failsafe_works_when_quest_input_then_goes_stale_and_controller_locked_out():
    env = make()
    arm_now(env)
    run(env, 2.0, stick=(1.0, 0.0))
    env.sup.submit((1.0, 0.0), True, env.clock())
    env.ctrl.locked_out = True
    run(env, 12.0, stick=None)                              # Quest input stale from here on
    _assert_released_only_at_neutral(env)


def test_failsafe_during_arming_cancels_without_publishing():
    env = make()
    env.bus.state_stale = True
    env.clock.t += 1.0
    env.sup.request_arm(start_thread=False)
    run(env, 0.2, stick=(0.0, 0.0))
    assert env.sup.state == wp.ARMING
    env.sup.submit((0.0, 0.0), True, env.clock())
    env.bus.state_stale = False
    run(env, 0.5, stick=(0.0, 0.0))
    assert env.sup.state == wp.DONE
    assert env.bus.published == [] and env.bus.create_calls == 0


def test_failsafe_with_stale_pitch_state_holds_then_homes_when_fresh():
    env = make()
    arm_now(env)
    run(env, 2.5, stick=(1.0, 0.0))
    q_lean = env.bus.plant.q
    env.bus.state_stale = True
    env.sup.submit((0.0, 0.0), True, env.clock())
    n0 = len(env.bus.published)
    run(env, 2.0, stick=None)
    assert env.sup.state == wp.HOMING
    assert len(env.bus.published) > n0          # keeps publishing (holds)
    assert abs(env.bus.plant.q - q_lean) < 0.02  # never released while stale
    env.bus.state_stale = False
    run(env, 12.0, stick=None)
    _assert_released_only_at_neutral(env)


def test_never_releases_away_from_neutral_when_return_stalls():
    env = make(tau_limit=25.0)                    # too weak to lift back
    arm_now(env)
    env.bus.plant.tau_limit = 85.0
    run(env, 3.0, stick=(1.0, 0.0))
    assert env.bus.plant.q > 0.2
    env.bus.plant.tau_limit = 25.0
    env.sup.request_home_and_release("teleop stop")
    run(env, 15.0, stick=None)
    assert env.sup.state == wp.HOMING
    assert not env.ctrl.released
    assert env.bus.published[-1][0] == pytest.approx(env.clock())   # still publishing now
    assert env.bus.plant.q > 0.15


def test_teleop_stop_homes_then_releases():
    env = make()
    arm_now(env)
    run(env, 2.5, stick=(1.0, 0.0))
    env.sup.request_home_and_release("teleop stop")
    run(env, 12.0, stick=(1.0, 0.0))
    _assert_released_only_at_neutral(env)
    assert env.sup.finished


def test_foreign_writer_mid_session_homes_instead_of_dropping():
    env = make()
    arm_now(env)
    run(env, 2.5, stick=(1.0, 0.0))
    env.bus.foreign.append("intruder")
    run(env, 1.0, stick=(1.0, 0.0))
    assert env.sup.state in (wp.HOMING, wp.RELEASED)
    assert not (env.sup.state == wp.RELEASED and abs(env.bus.plant.q) > 0.012)
    run(env, 12.0, stick=None)
    _assert_released_only_at_neutral(env)


def test_controller_exception_keeps_publishing_last_command():
    env = make()
    arm_now(env)
    run(env, 1.0, stick=(1.0, 0.0))
    real_tick = env.ctrl.tick

    def boom(*a, **k):
        raise RuntimeError("model blew up")
    env.ctrl.tick = boom
    n0 = len(env.bus.published)
    last = env.bus.published[-1][1]
    run(env, 0.5, stick=(0.0, 0.0))
    assert len(env.bus.published) >= n0 + 20
    assert env.bus.published[-1][1] == last
    env.ctrl.tick = real_tick
    env.sup.request_home_and_release("teleop stop")
    run(env, 12.0, stick=None)
    _assert_released_only_at_neutral(env)


def test_main_thread_death_triggers_home_and_release():
    env = make()
    alive = {"v": True}
    env.sup._main_alive = lambda: alive["v"]
    arm_now(env)
    run(env, 2.0, stick=(1.0, 0.0))
    alive["v"] = False
    run(env, 12.0, stick=None)
    _assert_released_only_at_neutral(env)


def test_thermal_home_temperature_drives_home_and_release():
    env = make()
    arm_now(env)
    run(env, 2.0, stick=(1.0, 0.0))
    env.bus.temperature = (40, 79)
    run(env, 12.0, stick=(1.0, 0.0))
    _assert_released_only_at_neutral(env)


def test_hot_motor_refuses_forward_but_allows_return():
    env = make()
    arm_now(env)
    run(env, 1.5, stick=(1.0, 0.0))
    env.bus.temperature = (40, 72)
    t0 = env.ctrl.target
    run(env, 1.0, stick=(1.0, 0.0))
    assert env.ctrl.target <= t0 + 1e-9
    run(env, 3.0, stick=(-1.0, 0.0))
    assert env.bus.plant.q < 0.05


# --------------------------------------------------------------------------
# non-blocking main-loop interface and thread lifecycle
# --------------------------------------------------------------------------
def test_submit_is_constant_time_and_never_touches_the_bus():
    env = make()
    calls = []
    env.bus.publish_cmd = lambda *a, **k: calls.append(a)
    env.bus.read_state = lambda: calls.append("read") or (None, None)
    t0 = time.perf_counter()
    for _ in range(1000):
        env.sup.submit((1.0, 0.0), False, 1.0)
    assert time.perf_counter() - t0 < 0.05
    assert calls == []


def test_real_thread_arms_and_finishes_after_stop_with_wall_clock():
    clock = time.monotonic
    arm_source = FakeArmSource(None, clock)
    bus = PlantBus(clock, q0=0.0, arm_source=arm_source)
    stop_plant = threading.Event()

    def plant_loop():
        while not stop_plant.is_set():
            bus.advance()
            time.sleep(0.002)
    th = threading.Thread(target=plant_loop, daemon=True)
    th.start()
    try:
        ctrl = wp.build_teleop_controller(STACK, BODY)
        sup = wp.WaistPitchSupervisor(ctrl, bus, wp.make_arm_filter(STACK, BODY), arm_source.snapshot,
                                      clock=clock, log=lambda *a, **k: None)
        sup.request_arm()
        deadline = time.monotonic() + 3.0
        while sup.state != wp.ACTIVE and time.monotonic() < deadline:
            sup.submit((0.0, 0.0), False, time.monotonic())
            time.sleep(0.01)
        assert sup.state == wp.ACTIVE, sup.last_reason
        assert not sup._thread.daemon        # interpreter waits for the pitch to get home
        result = wp.finish_waist_pitch(sup, log=lambda *a, **k: None, max_wait_s=10.0)
        assert result == wp.RELEASED
        sup._thread.join(2.0)
        assert not sup._thread.is_alive()
    finally:
        stop_plant.set()


def test_finish_requires_two_interrupts_to_abandon():
    env = make()
    arm_now(env)
    sleeps = {"n": 0}

    def sleeper(_):
        sleeps["n"] += 1
        if sleeps["n"] in (1, 2):
            raise KeyboardInterrupt
    logs = []
    result = wp.finish_waist_pitch(env.sup, log=lambda *a, **k: logs.append(a), sleep=sleeper)
    assert result == "forced"
    assert env.sup.abandoned
    assert any("CAIR" in str(m) for m in logs)


def test_finish_absent_supervisor_is_noop():
    assert wp.finish_waist_pitch(None, log=lambda *a, **k: None) == "absent"


def test_failsafe_button_is_left_y():
    assert wp.FAILSAFE_FIELD == "left_ctrl_bButton"
    td = SimpleNamespace(right_ctrl_thumbstickValue=(1.0, 0.0), left_ctrl_bButton=True,
                         controller_sample_timestamp=5.0)
    xy, pressed, ts = wp.extract_quest_inputs(td)
    assert tuple(xy) == (1.0, 0.0) and pressed is True and ts == 5.0
    td2 = SimpleNamespace()
    xy, pressed, ts = wp.extract_quest_inputs(td2)
    assert pressed is False and ts == 0.0


# --------------------------------------------------------------------------
# vendored code integrity
# --------------------------------------------------------------------------
def test_vendored_files_match_manifest():
    manifest = json.loads((wp.VENDOR_DIR / "VENDORED.json").read_text())
    assert manifest["source_commit"] == "785a29d"
    for rel, sha in manifest["sha256"].items():
        assert hashlib.sha256((wp.VENDOR_DIR / rel).read_bytes()).hexdigest() == sha, rel
    wp.verify_vendored()   # runtime check used before construction


def test_stack_is_loaded_from_pinned_vendor_dir():
    assert Path(STACK.pitch_controller.__file__).resolve().parent == wp.VENDOR_DIR.resolve()
    assert Path(STACK.pitch_ct.__file__).resolve().parent == wp.VENDOR_DIR.resolve()
