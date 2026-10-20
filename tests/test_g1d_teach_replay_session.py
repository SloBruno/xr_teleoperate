"""Teach/replay session: lifecycle, safety gates and command shaping (fake IO)."""
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from teleop.utils import teach_replay as tr  # noqa: E402

DT = 1.0 / 200.0
LOWER = np.full(14, -2.0)
UPPER = np.full(14, 2.0)
EFFORT = np.array([25.0] * 5 + [5.0] * 2 + [25.0] * 5 + [5.0] * 2)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def gravity(q):
    q = np.asarray(q, dtype=float)
    return 0.5 * np.sin(q) + np.linspace(-1.0, 1.0, 14)


class FakeIO:
    def __init__(self, clock, sides=("left", "right")):
        self.clock = clock
        self.arm_q = np.linspace(-0.3, 0.3, 14)
        self.hand_q = {"left": np.array([0.0, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1]),
                       "right": np.array([0.0, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1])}
        self.arm_frozen_at = None
        self.hand_frozen_at = {}
        self.hand_present = {side: side in sides for side in ("left", "right")}
        self.arm_present = True
        self.activations = 0
        self.released = 0
        self.writes = []
        # Ideal position tracking: the next measured q equals the last
        # commanded q (only while the state is fresh).
        self.follow = True

    def arm_state(self):
        if not self.arm_present:
            return None
        ts = self.clock() if self.arm_frozen_at is None else self.arm_frozen_at
        return tr.ArmState(q=self.arm_q.copy(), dq=np.zeros(14), timestamp=ts)

    def hand_state(self, side):
        if not self.hand_present[side]:
            return None
        ts = self.hand_frozen_at.get(side, self.clock())
        q = self.hand_q[side].copy()
        return tr.HandState(q=q, dq=np.zeros(7), timestamp=ts,
                            protection={"timestamp": ts, "q": list(q), "dq": [0.0] * 7})

    def activate(self, hand_sides):
        self.activations += 1
        self.active_sides = tuple(hand_sides)

    def write_arm(self, cmd):
        self.writes.append(("arm", None, cmd, threading.get_ident()))
        if self.follow and self.arm_frozen_at is None and np.all(cmd.kp > 0):
            self.arm_q = np.asarray(cmd.q, dtype=float).copy()

    def write_hand(self, side, cmd):
        self.writes.append(("hand", side, cmd, threading.get_ident()))
        if self.follow and side not in self.hand_frozen_at and np.all(cmd.kp > 0):
            self.hand_q[side] = np.asarray(cmd.q, dtype=float).copy()

    def release(self):
        self.released += 1

    def arm_writes(self):
        return [w[2] for w in self.writes if w[0] == "arm"]

    def hand_writes(self, side):
        return [w[2] for w in self.writes if w[0] == "hand" and w[1] == side]


class FakeWriter:
    instances = []

    def __init__(self, directory, header, **_kw):
        self.directory = directory
        self.header = header
        self.samples = []
        self.close_status = None
        self.result = None
        FakeWriter.instances.append(self)

    def emit(self, sample):
        self.samples.append(sample)
        return True

    def request_close(self, status="saved"):
        self.close_status = status
        self.result = tr.RecordingResult(path=Path(self.directory) / f"x.{status}.jsonl",
                                         samples=len(self.samples), dropped=0, status=status, error=None)

    def wait(self, timeout=None):
        return self.result


def make(tmp_path, sides=("left", "right"), io_sides=None, gravity_fn=gravity, **cfg):
    clock = Clock()
    io = FakeIO(clock, sides if io_sides is None else io_sides)
    config = tr.TeachReplayConfig(hand_sides=tuple(sides), **cfg)
    session = tr.TeachReplaySession(
        io, gravity_fn=gravity_fn, arm_lower=LOWER, arm_upper=UPPER, arm_effort=EFFORT,
        config=config, teach_dir=tmp_path, clock=clock, writer_factory=FakeWriter)
    return session, io, clock


def run(session, clock, seconds, on_tick=None):
    steps = int(round(seconds / DT))
    for i in range(steps):
        clock.t += DT
        if on_tick is not None:
            on_tick(i)
        session.tick()


def write_recording(path, t, arm, hands):
    sides = tuple(hands)
    header = tr.build_header(hand_sides=sides, rate_hz=50.0)
    writer = tr.RecordingWriter(path.parent, header, name=path.stem)
    for i, ti in enumerate(t):
        sample = {"t": float(ti), "t_mono": float(ti), "arm_q": list(arm[i]), "arm_dq": [0.0] * 14}
        for side in ("left", "right"):
            sample[f"{side}_hand_q"] = list(hands[side][i]) if side in hands else None
            sample[f"{side}_hand_dq"] = [0.0] * 7 if side in hands else None
        writer.emit(sample)
    return writer.close().path


# ---------------------------------------------------------------- pre-start
def test_no_activation_or_write_before_r(tmp_path):
    session, io, clock = make(tmp_path)
    run(session, clock, 0.5)
    assert io.activations == 0 and io.writes == []
    assert session.mode == tr.Mode.IDLE


@pytest.mark.parametrize("break_state", ["arm_stale", "arm_missing", "left_missing", "right_stale"])
def test_teach_refused_on_missing_or_stale_state_without_activation(tmp_path, break_state):
    session, io, clock = make(tmp_path)
    if break_state == "arm_stale":
        io.arm_frozen_at = clock.t - 1.0
    elif break_state == "arm_missing":
        io.arm_present = False
    elif break_state == "left_missing":
        io.hand_present["left"] = False
    else:
        io.hand_frozen_at["right"] = clock.t - 1.0
    session.request_teach()
    run(session, clock, 0.1)
    assert session.mode == tr.Mode.IDLE
    assert io.activations == 0 and io.writes == []
    assert "refus" in session.last_message.lower()


def test_single_hand_opt_in_starts_without_the_absent_hand_and_never_commands_it(tmp_path):
    session, io, clock = make(tmp_path, sides=("left",), io_sides=("left",))
    session.request_teach()
    run(session, clock, 0.2)
    assert session.mode == tr.Mode.TEACH
    assert io.active_sides == ("left",)
    assert io.hand_writes("right") == [] and io.hand_writes("left")


def test_arms_only_opt_in(tmp_path):
    session, io, clock = make(tmp_path, sides=(), io_sides=())
    session.request_teach()
    run(session, clock, 0.1)
    assert session.mode == tr.Mode.TEACH
    assert all(w[0] == "arm" for w in io.writes)


def test_teach_fails_closed_when_gravity_model_unavailable(tmp_path):
    def broken(q):
        g = gravity(q)
        g[3] = np.nan
        return g

    session, io, clock = make(tmp_path, gravity_fn=broken)
    session.request_teach()
    run(session, clock, 0.1)
    assert session.mode == tr.Mode.IDLE
    assert io.activations == 0 and io.writes == []
    assert "gravity" in session.last_message.lower()


# ---------------------------------------------------------------- teach
def test_teach_command_is_seeded_low_gain_with_gravity_ff_at_measured_q(tmp_path):
    session, io, clock = make(tmp_path)
    q0 = io.arm_q.copy()
    session.request_teach()
    clock.t += DT
    session.tick()
    assert io.activations == 1
    first = io.arm_writes()[0]
    np.testing.assert_allclose(first.q, q0)            # seeded: no jump
    np.testing.assert_allclose(first.dq, 0.0)
    assert np.all(first.kp <= tr.TeachReplayConfig().teach_arm_kp_max)
    assert np.all(first.kd > 0) and np.all(first.kd <= 1.5)
    g0 = gravity(q0)
    assert np.all(np.abs(first.tau) <= np.abs(g0) + 1e-9)   # gravity ramps in

    # Operator moves the arm by hand: command follows measured q each cycle,
    # full gravity ff evaluated at the measured q once the ramp is over.
    def move(i):
        io.arm_q = q0 + 0.002 * (i + 1)
    run(session, clock, 1.0, on_tick=move)
    last = io.arm_writes()[-1]
    np.testing.assert_allclose(last.q, io.arm_q)
    np.testing.assert_allclose(last.tau, gravity(io.arm_q))

    hand = io.hand_writes("left")[-1]
    np.testing.assert_allclose(hand.kp, 0.0)
    assert np.all(hand.kd > 0) and np.all(hand.kd <= 0.2)
    np.testing.assert_allclose(hand.q, io.hand_q["left"])


def test_gravity_ff_is_clipped_to_a_fraction_of_the_model_effort(tmp_path):
    session, io, clock = make(tmp_path, gravity_fn=lambda q: np.full(14, 100.0))
    session.request_teach()
    run(session, clock, 1.0)
    tau = io.arm_writes()[-1].tau
    cfg = tr.TeachReplayConfig()
    np.testing.assert_allclose(tau, cfg.gravity_effort_fraction * EFFORT)


def test_recording_rate_and_content_at_50hz(tmp_path):
    FakeWriter.instances.clear()
    session, io, clock = make(tmp_path)
    session.request_teach()

    def move(i):
        io.arm_q = io.arm_q + 0.001
    run(session, clock, 2.0, on_tick=move)
    writer = FakeWriter.instances[-1]
    assert writer.header["rate_hz"] == 50.0
    assert writer.header["hand_sides"] == ["left", "right"]
    n = len(writer.samples)
    assert 98 <= n <= 102
    t = np.array([s["t"] for s in writer.samples])
    dt = np.diff(t)
    assert np.all(dt > 0)
    np.testing.assert_allclose(np.median(dt), 0.02, atol=DT)
    assert np.max(dt) <= 0.02 + DT + 1e-9
    last = writer.samples[-1]
    np.testing.assert_allclose(last["arm_q"], io.arm_q, atol=0.005)
    assert len(last["left_hand_q"]) == 7 and len(last["arm_dq"]) == 14


def test_s_saves_recording_and_holds_measured_pose_with_tracking_gains(tmp_path):
    FakeWriter.instances.clear()
    session, io, clock = make(tmp_path)
    session.request_teach()
    run(session, clock, 0.5)
    q_at_save = io.arm_q.copy()
    session.request_save()
    run(session, clock, 0.05)
    assert FakeWriter.instances[-1].close_status == "saved"
    assert session.mode == tr.Mode.HOLD
    assert "saved" in session.last_message.lower()
    io.arm_q = q_at_save + 0.05      # sag must not ratchet into the reference
    run(session, clock, 0.2)
    hold = io.arm_writes()[-1]
    np.testing.assert_allclose(hold.q, q_at_save)
    np.testing.assert_allclose(hold.kp, tr.TeachReplayConfig().track_arm_kp)
    np.testing.assert_allclose(hold.tau, gravity(q_at_save))


def test_stale_state_during_teach_aborts_recording_and_holds(tmp_path):
    FakeWriter.instances.clear()
    session, io, clock = make(tmp_path)
    session.request_teach()
    run(session, clock, 0.3)
    q_last = io.arm_q.copy()
    io.arm_frozen_at = clock.t
    run(session, clock, 0.5)
    assert session.mode == tr.Mode.HOLD
    assert FakeWriter.instances[-1].close_status == "aborted_stale"
    tail = io.arm_writes()[-20:]
    for cmd in tail:
        np.testing.assert_allclose(cmd.q, q_last)


# ---------------------------------------------------------------- replay
def _replay_setup(tmp_path, arm_offset=1.0, finger_offset=1.5, n=50, arm_fn=None):
    session, io, clock = make(tmp_path)
    t = np.arange(n) * 0.02
    arm = np.tile(io.arm_q + arm_offset * (np.arange(14) == 0), (n, 1))
    if arm_fn is not None:
        arm = arm_fn(t, arm)
    left = np.tile(io.hand_q["left"], (n, 1))
    left[:, 0] = 0.0
    left[:, 3] = -finger_offset if finger_offset else left[:, 3]
    right = np.tile(io.hand_q["right"], (n, 1))
    right[:, 3] = io.hand_q["right"][3] + 0.3
    path = write_recording(tmp_path / "20261019T100000_000000Z.jsonl", t, arm, {"left": left, "right": right})
    return session, io, clock, path, t, arm, left, right


def _play_rows(session, io, clock):
    while session.mode != tr.Mode.REPLAY_PLAY:
        clock.t += DT
        session.tick()
        assert clock.t < 1100.0
    rows = []
    while session.mode == tr.Mode.REPLAY_PLAY:
        clock.t += DT
        session.tick()
        rows.append(io.arm_writes()[-1].q.copy())
        io.arm_q = rows[-1].copy()          # perfect tracking: isolate the planner
        assert clock.t < 1100.0
    return np.array(rows)


def test_fast_teach_is_replayed_slower_along_the_same_path_within_the_velocity_cap(tmp_path):
    """A hand-taught motion faster than the cap is time-scaled, not step-clipped."""
    def fast(t, arm):
        arm = arm.copy()
        arm[:, 0] = arm[0, 0] + 0.6 * np.sin(2 * np.pi * 1.0 * t)   # peak 3.8 rad/s
        return arm

    session, io, clock, path, t, arm, left, right = _replay_setup(
        tmp_path, arm_offset=0.0, finger_offset=0.0, n=100, arm_fn=fast)
    io.arm_q = arm[0].copy()
    io.hand_q["left"] = left[0].copy()
    io.hand_q["right"] = right[0].copy()
    session.request_replay(str(path))
    q = _play_rows(session, io, clock)
    vmax = session.config.replay_arm_vmax
    v = np.max(np.abs(np.diff(q, axis=0)), axis=1) / DT
    assert np.max(v) <= vmax + 1e-6
    assert vmax <= 1.0                                 # conservative default
    # same path: the full +/-0.6 rad excursion is reproduced, not shaved off
    assert np.max(q[:, 0]) - arm[0, 0] >= 0.6 - 0.02
    assert arm[0, 0] - np.min(q[:, 0]) >= 0.6 - 0.02
    # playback stretched to about total travel / vmax (2 periods x 4 x 0.6 rad)
    assert len(q) * DT >= (2 * 4 * 0.6 / vmax) * 0.9
    assert session.mode == tr.Mode.HOLD


def test_slow_teach_is_replayed_at_recorded_speed(tmp_path):
    def slow(t, arm):
        arm = arm.copy()
        arm[:, 0] = arm[0, 0] + 0.2 * t                          # 0.2 rad/s
        return arm

    session, io, clock, path, t, arm, left, right = _replay_setup(
        tmp_path, arm_offset=0.0, finger_offset=0.0, n=100, arm_fn=slow)
    io.arm_q = arm[0].copy()
    io.hand_q["left"] = left[0].copy()
    io.hand_q["right"] = right[0].copy()
    session.request_replay(str(path))
    q = _play_rows(session, io, clock)
    assert len(q) * DT <= t[-1] + 4 * DT


def test_replay_ramp_in_respects_joint_velocity_caps_and_reaches_first_sample(tmp_path):
    session, io, clock, path, t, arm, left, right = _replay_setup(tmp_path)
    session.request_replay(None)
    clock.t += DT
    session.tick()
    assert session.mode == tr.Mode.REPLAY_RAMP
    first = io.arm_writes()[0]
    np.testing.assert_allclose(first.q, io.arm_q, atol=1e-9)      # starts at measured
    np.testing.assert_allclose(first.kp, tr.TeachReplayConfig().track_arm_kp)
    times, arm_cmds, left_cmds = [clock.t], [first.q], [io.hand_writes("left")[0].q]

    while session.mode == tr.Mode.REPLAY_RAMP:
        clock.t += DT
        session.tick()
        times.append(clock.t)
        arm_cmds.append(io.arm_writes()[-1].q)
        left_cmds.append(io.hand_writes("left")[-1].q)
        assert clock.t < 1100.0
    arm_cmds, left_cmds, times = np.array(arm_cmds), np.array(left_cmds), np.array(times)
    arm_v = np.max(np.abs(np.diff(arm_cmds, axis=0)), axis=1) / DT
    hand_v = np.max(np.abs(np.diff(left_cmds, axis=0)), axis=1) / DT
    assert np.max(arm_v) <= 0.5 + 1e-6
    assert np.max(hand_v) <= 2.0 + 1e-6
    np.testing.assert_allclose(arm_cmds[-1], arm[0], atol=1e-6)
    np.testing.assert_allclose(left_cmds[-1], left[0], atol=1e-6)
    assert times[-1] - times[0] >= 1.5 * 1.0 / 0.5 - 2 * DT        # smoothstep peak factor


def test_replay_repeat_count_reramps_safely_between_cycles_and_reports_iteration(tmp_path):
    session, io, clock, path, t, arm, left, right = _replay_setup(tmp_path, n=4, arm_offset=0.0)
    io.arm_q = arm[0].copy(); io.hand_q["left"] = left[0].copy(); io.hand_q["right"] = right[0].copy()
    assert session.request_replay(str(path), repeat=2)
    ramp_count = 0; previous = None
    for _ in range(2000):
        clock.t += DT; session.tick()
        if session.mode == tr.Mode.REPLAY_RAMP:
            if previous != tr.Mode.REPLAY_RAMP:
                ramp_count += 1
            previous = session.mode
        else:
            previous = session.mode
        if session.mode == tr.Mode.REPLAY_PLAY:
            io.arm_q = io.arm_writes()[-1].q.copy()
            io.hand_q["left"] = io.hand_writes("left")[-1].q.copy()
            io.hand_q["right"] = io.hand_writes("right")[-1].q.copy()
        if session.mode == tr.Mode.HOLD:
            break
    assert session.mode == tr.Mode.HOLD
    assert ramp_count == 2
    assert session.status_view()["replay"] is None
    assert any("2/2" in message for message in session.messages)


def test_latched_hand_protection_remains_disabled_across_loop_ramp(tmp_path):
    session, io, clock, path, *_ = _replay_setup(tmp_path, n=4, arm_offset=0.0)
    io.arm_q = io.arm_q.copy()
    assert session.request_replay(str(path), repeat=2)
    session._hand_disabled["left"] = np.array([True, False, False, False, False, False, False])
    for _ in range(2500):
        clock.t += DT; session.tick()
        if session.mode == tr.Mode.HOLD:
            break
        if session.mode == tr.Mode.REPLAY_RAMP and session._replay_iteration >= 2:
            assert np.all(session.io.hand_writes("left")[-1].kp[[0]] == 0.0)
            break


def test_continuous_replay_never_enters_terminal_hold_and_reports_continuous(tmp_path):
    session, io, clock, path, t, arm, left, right = _replay_setup(tmp_path, n=4, arm_offset=0.0)
    io.arm_q = arm[0].copy(); io.hand_q["left"] = left[0].copy(); io.hand_q["right"] = right[0].copy()
    assert session.request_replay(str(path), continuous=True)
    saw_second = False
    for _ in range(2500):
        clock.t += DT; session.tick()
        if getattr(session, "_replay_iteration", 1) >= 2:
            saw_second = True
            break
        if session.mode == tr.Mode.REPLAY_PLAY:
            io.arm_q = io.arm_writes()[-1].q.copy()
            io.hand_q["left"] = io.hand_writes("left")[-1].q.copy()
            io.hand_q["right"] = io.hand_writes("right")[-1].q.copy()
    assert saw_second and session.mode in (tr.Mode.REPLAY_RAMP, tr.Mode.REPLAY_PLAY)
    assert session.status_view()["replay"]["total"] is None


def test_replay_refused_when_ramp_would_be_too_long(tmp_path):
    session, io, clock, path, *_ = _replay_setup(tmp_path, arm_offset=1.0)
    session.config.ramp_max_s = 1.0
    session.request_replay(None)
    run(session, clock, 0.05)
    assert session.mode == tr.Mode.IDLE and io.activations == 0


def test_replay_playback_interpolates_and_clamps_step_and_limits(tmp_path):
    def shape(t, arm):
        arm = arm.copy()
        arm[:, 1] = arm[0, 1] + 0.5 * t            # smooth ramp, 0.5 rad/s
        arm[25:, 2] = arm[0, 2] + 1.0              # 1 rad step in one sample
        arm[:, 3] = 1.99                            # inside the validated session limit
        return arm

    session, io, clock, path, t, arm, left, right = _replay_setup(
        tmp_path, arm_offset=0.0, finger_offset=0.0, arm_fn=shape)
    io.arm_q = arm[0].copy()
    io.arm_q[3] = UPPER[3] - session.config.limit_margin
    io.hand_q["left"] = left[0].copy()
    io.hand_q["right"] = right[0].copy()
    session.request_replay(str(path))
    while session.mode != tr.Mode.REPLAY_PLAY:
        clock.t += DT
        session.tick()
        assert clock.t < 1100.0
    play_t0 = session.play_start_time
    rows = []
    while session.mode == tr.Mode.REPLAY_PLAY:
        clock.t += DT
        session.tick()
        rows.append((clock.t - play_t0, io.arm_writes()[-1].q.copy()))
    ts = np.array([r[0] for r in rows])
    q = np.array([r[1] for r in rows])
    warped = tr.time_warp(t, np.clip(arm, LOWER + session.config.limit_margin,
                                     UPPER - session.config.limit_margin),
                          {"left": left, "right": right},
                          session.config.replay_arm_vmax, session.config.replay_hand_vmax)
    inside = ts < warped[-1]
    np.testing.assert_allclose(q[inside, 1], np.interp(ts[inside], warped, arm[:, 1]), atol=1e-6)
    step = np.max(np.abs(np.diff(q, axis=0)), axis=0)
    assert step[2] <= session.config.replay_arm_vmax * DT + 1e-9
    assert np.all(q[:, 3] <= UPPER[3] - session.config.limit_margin + 1e-12)
    # end of recording: hold final pose, never go home
    assert session.mode == tr.Mode.HOLD
    final = io.arm_writes()[-1].q.copy()
    run(session, clock, 0.3)
    np.testing.assert_allclose(io.arm_writes()[-1].q, final)
    np.testing.assert_allclose(io.arm_writes()[-1].tau, gravity(final))


def test_stale_state_during_replay_holds_instead_of_streaming(tmp_path):
    def shape(t, arm):
        arm = arm.copy()
        arm[:, 0] = arm[0, 0] + 0.5 * t
        return arm

    session, io, clock, path, t, arm, left, right = _replay_setup(
        tmp_path, arm_offset=0.0, finger_offset=0.0, arm_fn=shape)
    io.arm_q = arm[0].copy()
    session.request_replay(None)
    while session.mode != tr.Mode.REPLAY_PLAY:
        clock.t += DT
        session.tick()
    run(session, clock, 0.2)
    last_fresh_q = io.arm_q.copy()
    io.arm_frozen_at = clock.t
    n_before = len(io.arm_writes())
    run(session, clock, 0.5)
    assert session.mode == tr.Mode.HOLD
    assert "stale" in session.last_message.lower()
    tail = io.arm_writes()[-40:]
    hold_q = tail[-1].q
    for cmd in tail:
        np.testing.assert_allclose(cmd.q, hold_q)
        np.testing.assert_allclose(cmd.tau, gravity(hold_q))
    # Hold = last validated command (continuity), never beyond what could be
    # streamed inside the freshness window, and streaming really stopped.
    window = session.config.arm_state_max_age + 2 * DT
    assert np.max(np.abs(hold_q - last_fresh_q)) <= session.config.replay_arm_vmax * window + 1e-9
    streamed = [c.q for c in io.arm_writes()[n_before:]]
    distinct = {tuple(np.round(q, 12)) for q in streamed}
    assert len(distinct) <= int(window / DT) + 1


def test_stale_hand_during_replay_holds_and_retains_last_hand_command(tmp_path):
    session, io, clock, path, *_ = _replay_setup(tmp_path, arm_offset=0.0, finger_offset=0.0)
    session.request_replay(None)
    run(session, clock, 0.3)
    io.hand_frozen_at["right"] = clock.t
    io.follow = False
    n_before = len(io.hand_writes("right"))
    run(session, clock, 0.9)
    assert session.mode == tr.Mode.HOLD
    assert any("stale" in m.lower() for m in session.messages)
    # Hold retains the last validated hand command (never the open pose):
    # the tail is constant and equals the last command streamed while fresh.
    cmds = io.hand_writes("right")
    tail = cmds[-20:]
    for cmd in tail:
        np.testing.assert_allclose(cmd.q, tail[-1].q)
    first_hold = next(i for i in range(len(cmds)) if all(np.allclose(c.q, tail[-1].q) for c in cmds[i:]))
    assert first_hold > n_before
    # the held value is exactly a command already streamed (no recomputation)
    assert np.allclose(cmds[first_hold].q, tail[-1].q) and not np.allclose(cmds[first_hold - 1].q, tail[-1].q)
    assert not np.allclose(tail[-1].q, 0.0)
    window = session.config.hand_state_max_age + 2 * DT
    assert (first_hold - n_before) <= window / DT + 1


def test_replay_refused_for_stale_state_or_invalid_file(tmp_path):
    session, io, clock, path, *_ = _replay_setup(tmp_path)
    io.arm_frozen_at = clock.t - 1.0
    session.request_replay(None)
    run(session, clock, 0.05)
    assert session.mode == tr.Mode.IDLE and io.activations == 0
    io.arm_frozen_at = None
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    session.request_replay(str(bad))
    run(session, clock, 0.05)
    assert session.mode == tr.Mode.IDLE and io.activations == 0


def test_request_replay_validates_with_session_limits(tmp_path, monkeypatch):
    session, io, clock, path, *_ = _replay_setup(tmp_path)
    seen = {}
    real = tr.load_recording

    def spy(recording_path, **kwargs):
        seen.update(kwargs)
        return real(recording_path, **kwargs)

    monkeypatch.setattr(tr, "load_recording", spy)
    assert session.request_replay(path.name)
    np.testing.assert_array_equal(seen["arm_lower"], session.arm_lower)
    np.testing.assert_array_equal(seen["arm_upper"], session.arm_upper)
    assert seen["hand_limits"] is tr.DEX3_LIMITS


# ---------------------------------------------------------------- lifecycle
def test_q_before_r_exits_without_any_output(tmp_path):
    session, io, clock = make(tmp_path)
    session.request_quit()
    run(session, clock, 0.05)
    assert session.done and io.activations == 0 and io.writes == []


def test_q_wins_over_pending_p_and_r_and_later_requests_are_refused(tmp_path):
    session, io, clock, path, *_ = _replay_setup(tmp_path)
    session.request_replay(None)
    session.request_teach()
    session.request_quit()
    run(session, clock, 0.05)
    assert session.done and io.activations == 0 and io.writes == []
    assert session.request_teach() is False
    assert session.request_replay(None) is False


def test_q_during_replay_holds_measured_with_gravity_then_releases(tmp_path):
    session, io, clock, path, *_ = _replay_setup(tmp_path)
    session.request_replay(None)
    run(session, clock, 0.4)
    assert session.mode == tr.Mode.REPLAY_RAMP
    q_now = io.arm_q.copy()
    session.request_quit()
    n_before = len(io.arm_writes())
    run(session, clock, session.config.quit_hold_s + 0.2)
    after = io.arm_writes()[n_before:]
    assert after, "quit must publish a hold before release"
    for cmd in after:
        np.testing.assert_allclose(cmd.q, q_now)
        np.testing.assert_allclose(cmd.tau, gravity(q_now))
    assert session.done and io.released == 1
    count = len(io.writes)
    run(session, clock, 0.2)
    assert len(io.writes) == count


def test_q_during_teach_keeps_recording_as_aborted(tmp_path):
    FakeWriter.instances.clear()
    session, io, clock = make(tmp_path)
    session.request_teach()
    run(session, clock, 0.3)
    session.request_quit()
    run(session, clock, session.config.quit_hold_s + 0.2)
    assert FakeWriter.instances[-1].close_status == "aborted_quit"
    assert session.done


def test_all_writes_come_from_the_ticking_thread(tmp_path):
    session, io, clock, path, *_ = _replay_setup(tmp_path)
    results = []

    def keys():
        results.append(session.request_teach())
        results.append(session.request_save())
        results.append(session.request_replay(None))

    session.request_teach()
    run(session, clock, 0.2)
    thread = threading.Thread(target=keys)
    thread.start(); thread.join()
    run(session, clock, 0.3)
    assert {w[3] for w in io.writes} == {threading.get_ident()}


def test_replay_and_teach_requests_are_refused_while_replaying(tmp_path):
    session, io, clock, path, *_ = _replay_setup(tmp_path)
    session.request_replay(None)
    run(session, clock, 0.1)
    assert session.request_teach() is False
    assert session.request_replay(None) is False
    assert session.request_save() is False


def test_replay_applies_existing_dex3_protection_and_hold_keeps_faulted_joint_disabled(tmp_path):
    session, io, clock, path, *_ = _replay_setup(tmp_path, arm_offset=0.0, finger_offset=0.0)
    original = io.hand_state

    def faulty(side):
        state = original(side)
        if state is None or side != "left":
            return state
        prot = dict(state.protection)
        prot["mode"] = [1] * 7
        prot["motorstate"] = [0, 0, 512, 0, 0, 0, 0]     # Thumb2 fault
        return tr.HandState(q=state.q, dq=state.dq, timestamp=state.timestamp, protection=prot)

    io.hand_state = faulty
    session.request_replay(None)
    run(session, clock, 0.3)
    cmd = io.hand_writes("left")[-1]
    assert cmd.kp[2] == 0.0 and cmd.kd[2] == 0.0
    assert np.all(np.delete(cmd.kp, 2) == session.config.track_hand_kp)
    session.request_quit()
    run(session, clock, 0.1)
    assert io.hand_writes("left")[-1].kp[2] == 0.0


def test_large_tracking_error_during_replay_holds_at_measured_pose(tmp_path):
    session, io, clock, path, *_ = _replay_setup(tmp_path, arm_offset=1.0)
    session.request_replay(None)
    run(session, clock, 0.5)
    io.follow = False
    blocked = io.arm_q.copy()
    run(session, clock, 2.0)
    assert session.mode == tr.Mode.HOLD
    assert "tracking error" in session.last_message
    np.testing.assert_allclose(io.arm_writes()[-1].q, blocked)


def test_state_stamped_slightly_after_tick_reference_is_not_stale(tmp_path):
    session, io, clock = make(tmp_path)
    io.arm_frozen_at = clock.t + 0.003          # callback ran after `now` was taken
    io.hand_frozen_at["left"] = clock.t + 0.003
    session.request_teach()
    session.tick()
    assert session.mode == tr.Mode.TEACH
    io.arm_frozen_at = clock.t + 0.5            # genuinely future timestamp
    session.tick()
    assert session.mode == tr.Mode.HOLD


def test_unexpected_write_failure_turns_into_quit_and_release(tmp_path):
    session, io, clock = make(tmp_path)
    session.request_teach()
    run(session, clock, 0.1)
    real = io.write_hand

    def broken(side, cmd):
        raise OSError("dds write failed")

    io.write_hand = broken
    run(session, clock, session.config.quit_hold_s + 0.5)
    assert session.done and io.released == 1
    assert "error" in " ".join(session.messages).lower()
    del real


def test_end_to_end_teach_save_then_replay_the_saved_file(tmp_path):
    clock = Clock()
    io = FakeIO(clock)
    session = tr.TeachReplaySession(io, gravity_fn=gravity, arm_lower=LOWER, arm_upper=UPPER,
                                    arm_effort=EFFORT, teach_dir=tmp_path, clock=clock)
    q0 = io.arm_q.copy()
    session.request_teach()

    def move(i):
        io.arm_q = q0 + 0.2 * np.sin(np.pi * i * DT)      # operator moves the arm
    run(session, clock, 1.0, on_tick=move)
    taught_end = io.arm_q.copy()
    session.request_save()
    run(session, clock, 0.1)
    results = session.wait_for_recordings(timeout=5.0)
    assert results[-1].status == "saved" and results[-1].error is None
    rec = tr.load_recording(results[-1].path)
    assert 48 <= len(rec.t) <= 52
    np.testing.assert_allclose(rec.arm_q[0], q0, atol=1e-9)

    assert session.request_replay(None) is True
    while session.mode != tr.Mode.HOLD or session.play_start_time is None:
        clock.t += DT
        session.tick()
        assert clock.t < 1100.0
    np.testing.assert_allclose(io.arm_writes()[-1].q, rec.arm_q[-1], atol=1e-9)
    np.testing.assert_allclose(rec.arm_q[-1], taught_end, atol=0.01)
    session.request_quit()
    run(session, clock, session.config.quit_hold_s + 0.1)
    assert session.done and io.released == 1


def test_dex3_freshness_matches_the_existing_protector_predicate():
    from teleop.utils import dex3_protection
    assert tr.TeachReplayConfig().hand_state_max_age == dex3_protection.STATE_STALE_S


@pytest.mark.parametrize("effort", [np.full(14, np.nan), np.zeros(14), np.full(13, 25.0)])
def test_session_refuses_invalid_effort_limits(tmp_path, effort):
    with pytest.raises(ValueError):
        tr.TeachReplaySession(FakeIO(Clock()), gravity_fn=gravity, arm_lower=LOWER, arm_upper=UPPER,
                              arm_effort=effort, teach_dir=tmp_path)


def test_session_refuses_invalid_joint_limits(tmp_path):
    with pytest.raises(ValueError):
        tr.TeachReplaySession(FakeIO(Clock()), gravity_fn=gravity, arm_lower=UPPER, arm_upper=LOWER,
                              arm_effort=EFFORT, teach_dir=tmp_path)
