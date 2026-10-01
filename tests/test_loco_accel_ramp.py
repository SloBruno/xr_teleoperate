"""Acceleration ramp + pulse debounce for walking; safety zero stays immediate."""
import math
from teleop.utils import quest_controls as qc


class Rec:
    def __init__(self):
        self.events = []

    def Move(self, *c):
        self.events.append(("move", c))

    def StopMove(self, reason="", **kw):
        self.events.append(("stop", reason))


DT = 0.1


def run(ramp, rec, seq, t0=100.0, dt=DT, fresh=True, enabled=True, caps=(0.3, 0.3)):
    out = []
    t = t0
    for left, right in seq:
        out.append(qc.dispatch_joystick_locomotion(
            rec, enabled, fresh, left, right, walk_cap=caps[0], turn_cap=caps[1], ramp=ramp, now=t))
        t += dt
    return out


FWD = ((0.0, -1.0), (0.0, 0.0))
IDLE = ((0.0, 0.0), (0.0, 0.0))


def test_defaults_cap_is_0_3_hard_limits_unchanged():
    assert qc.MIN_OPERATOR_WALK_SPEED_MPS == 0.3
    assert qc.MIN_OPERATOR_TURN_RATE_RADPS == 0.3
    assert qc.MAX_WALK_SPEED_CAP_MPS == 0.6 and qc.MAX_TURN_RATE_CAP_RADPS == 1.0
    assert qc.resolve_speed_caps(None, None, {}) == (0.3, 0.3)
    assert qc.LOCO_LINEAR_ACCEL_MPS2 == 0.5 and qc.LOCO_YAW_ACCEL_RADPS2 == 1.0
    assert qc.LOCO_DEBOUNCE_MIN_CYCLES >= 3 and qc.LOCO_DEBOUNCE_MIN_S >= 0.08


def test_one_cycle_pulse_emits_nothing():
    r, ramp = Rec(), qc.LocomotionRamp()
    out = run(ramp, r, [FWD, IDLE, IDLE, IDLE])
    assert all(o == (0.0, 0.0, 0.0) for o in out)
    assert not [e for e in r.events if e[0] == "stop"]  # never moved -> no stop spam


def test_two_cycle_pulse_still_emits_nothing():
    r, ramp = Rec(), qc.LocomotionRamp()
    out = run(ramp, r, [FWD, FWD, IDLE, IDLE])
    assert all(o == (0.0, 0.0, 0.0) for o in out)


def test_sustained_push_ramps_without_step_and_reaches_cap():
    r, ramp = Rec(), qc.LocomotionRamp()
    out = run(ramp, r, [FWD] * 15)
    vx = [o[0] for o in out]
    assert vx[0] == vx[1] == 0.0                # debounce window
    assert max(vx) == 0.3 and vx[-1] == 0.3
    steps = [b - a for a, b in zip(vx, vx[1:])]
    assert max(steps) <= qc.LOCO_LINEAR_ACCEL_MPS2 * DT + 1e-9   # no step
    assert vx == sorted(vx)


def test_release_decelerates_by_same_ramp_then_one_stop():
    r, ramp = Rec(), qc.LocomotionRamp()
    out = run(ramp, r, [FWD] * 15 + [IDLE] * 10)
    vx = [o[0] for o in out[15:]]
    steps = [a - b for a, b in zip([0.3] + vx, vx)]
    assert max(steps) <= qc.LOCO_LINEAR_ACCEL_MPS2 * DT + 1e-9
    assert vx[0] > 0.0 and vx[-1] == 0.0
    stops = [e for e in r.events if e[0] == "stop"]
    assert stops == [("stop", "release")]
    # stop is issued after the command reached zero
    last_move = [i for i, e in enumerate(r.events) if e[0] == "move"][-1]
    assert r.events.index(("stop", "release")) > last_move - 5


def test_yaw_uses_its_own_ramp():
    r, ramp = Rec(), qc.LocomotionRamp()
    out = run(ramp, r, [((0, 0), (-1.0, 0.0))] * 10, caps=(0.3, 0.3))
    yaw = [o[2] for o in out]
    assert max(yaw) == 0.3
    assert max(b - a for a, b in zip(yaw, yaw[1:])) <= qc.LOCO_YAW_ACCEL_RADPS2 * DT + 1e-9


def test_stale_goes_zero_immediately_with_stop_and_resets_ramp():
    r, ramp = Rec(), qc.LocomotionRamp()
    run(ramp, r, [FWD] * 15)
    out = run(ramp, r, [FWD], t0=110.0, fresh=False)
    assert out == [(0.0, 0.0, 0.0)]
    assert r.events[-2:] == [("move", (0.0, 0.0, 0.0)), ("stop", "stale")]
    assert ramp.last["reason"] == "safety_zero"
    # next push must debounce + ramp from zero again
    out = run(ramp, r, [FWD] * 3, t0=110.2)
    assert out[0] == out[1] == (0.0, 0.0, 0.0)


def test_motion_disabled_goes_zero_immediately_and_resets():
    r, ramp = Rec(), qc.LocomotionRamp()
    run(ramp, r, [FWD] * 15)
    assert run(ramp, r, [FWD], t0=110.0, enabled=False) == [(0.0, 0.0, 0.0)]
    assert ("stop", "motion_disabled") in r.events
    assert ramp.last["reason"] == "safety_zero"
    assert ramp.output == (0.0, 0.0, 0.0)


def test_safety_zero_is_not_delayed_by_ramp_even_at_cap():
    r, ramp = Rec(), qc.LocomotionRamp()
    run(ramp, r, [FWD] * 15)
    n = len(r.events)
    run(ramp, r, [FWD], t0=111.0, fresh=False)
    assert r.events[n] == ("move", (0.0, 0.0, 0.0))


def test_push_during_decel_does_not_debounce():
    r, ramp = Rec(), qc.LocomotionRamp()
    out = run(ramp, r, [FWD] * 15 + [IDLE] * 2 + [FWD])
    assert out[-1][0] > out[-2][0]


def test_hard_cap_is_never_exceeded_by_ramp():
    r, ramp = Rec(), qc.LocomotionRamp()
    out = run(ramp, r, [((-1.0, -1.0), (-1.0, 0.0))] * 40, caps=(0.6, 1.0))
    assert max(abs(o[0]) for o in out) <= 0.6 and max(abs(o[1]) for o in out) <= 0.6
    assert max(abs(o[2]) for o in out) <= 1.0


def test_large_dt_gap_is_clamped():
    r, ramp = Rec(), qc.LocomotionRamp()
    run(ramp, r, [FWD] * 4)                      # output 0.1 after 4 cycles
    before = ramp.output[0]
    out = qc.dispatch_joystick_locomotion(r, True, True, *FWD, walk_cap=0.6, ramp=ramp, now=500.0)
    assert out[0] - before <= qc.LOCO_LINEAR_ACCEL_MPS2 * qc.LOCO_RAMP_MAX_DT_S + 1e-9


def test_telemetry_records_raw_ramped_and_reason():
    r, ramp = Rec(), qc.LocomotionRamp()
    run(ramp, r, [FWD])
    assert ramp.last == {"raw_command": [0.3, 0.0, 0.0], "ramp_command": [0.0, 0.0, 0.0], "reason": "debounced"}
    run(ramp, r, [FWD] * 3, t0=100.1)
    assert ramp.last["reason"] == "ramped"
    run(ramp, r, [FWD] * 10, t0=100.4)
    assert ramp.last["reason"] == "steady"
    run(ramp, r, [FWD], t0=120.0, fresh=False)
    assert ramp.last["reason"] == "safety_zero" and ramp.last["raw_command"] == [0.3, 0.0, 0.0]  # post-curve intent kept for diagnosis


def test_no_ramp_arg_keeps_legacy_passthrough():
    r = Rec()
    out = qc.dispatch_joystick_locomotion(r, True, True, *FWD, walk_cap=0.3)
    assert out == (0.3, 0.0, 0.0)


def test_wiring_in_teleop_loop_and_launcher():
    src = open("teleop/teleop_hand_and_arm.py").read()
    assert "LocomotionRamp()" in src and "ramp=loco_ramp" in src
    assert "ramp_reason" in src and "raw_command" in src
    sh = open("teleop/run_g1_quest_dex3.sh").read()
    assert "walk_speed_cap=${G1_WALK_SPEED_CAP:-0.3}" in sh
