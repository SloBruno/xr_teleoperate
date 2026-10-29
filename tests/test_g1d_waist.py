"""Tests for teleop.utils.g1d_waist: pure mapping + fail-closed controller.

Uses local fakes (not an import of /home/bruno/g1d_pitch_offline, a separate,
isolated effort) that implement the WaistPitchBackend contract documented in
g1d_waist.py, so these tests exercise the integration surface a real backend
will plug into, without duplicating or depending on that other work.
"""
import math
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teleop.utils import g1d_waist as w


# --------------------------------------------------------------------- pure mapping

def test_map_pitch_intent_deadzone_and_shape():
    assert w.map_pitch_intent((0.0, 0.5)) == 0.0
    assert w.map_pitch_intent((0.05, 0.9)) == 0.0  # inside deadzone
    assert w.map_pitch_intent((1.0, 0.0)) == pytest.approx(1.0)
    assert w.map_pitch_intent((-1.0, 0.0)) == pytest.approx(-1.0)


def test_map_pitch_intent_ignores_vertical_axis():
    # Vertical axis still drives the AGV column elsewhere; must not leak in.
    a = w.map_pitch_intent((0.5, -1.0))
    b = w.map_pitch_intent((0.5, 1.0))
    assert a == b


def test_map_pitch_intent_malformed_input_is_zero():
    assert w.map_pitch_intent(None) == 0.0
    assert w.map_pitch_intent((float("nan"), 0.0)) == 0.0
    assert w.map_pitch_intent("xy") == 0.0
    assert w.map_pitch_intent((1,)) == 0.0


def _rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    pose = [[c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
    return pose


def test_head_yaw_rad_identity_is_zero():
    assert w.head_yaw_rad(_rot_z(0.0)) == pytest.approx(0.0, abs=1e-9)


def test_head_yaw_rad_quarter_turn():
    assert w.head_yaw_rad(_rot_z(math.pi / 2)) == pytest.approx(math.pi / 2, abs=1e-9)
    assert w.head_yaw_rad(_rot_z(-math.pi / 2)) == pytest.approx(-math.pi / 2, abs=1e-9)


def test_head_yaw_rad_ignores_pitch_roll_tilt_in_other_axes():
    # x-axis (column 0) still has a well-defined XY projection even if the
    # matrix also encodes some pitch/roll; only the projected heading counts.
    pose = _rot_z(0.3)
    pose[2][0] = 0.4  # tilt the x-axis out of plane a bit
    yaw = w.head_yaw_rad(pose)
    assert yaw is not None
    assert math.isfinite(yaw)


def test_head_yaw_rad_degenerate_or_nonfinite_is_none():
    assert w.head_yaw_rad([[0.0, 0, 0, 0], [0.0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) is None
    assert w.head_yaw_rad([[float("nan"), 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) is None
    assert w.head_yaw_rad(None) is None
    assert w.head_yaw_rad([[1]]) is None


def test_yaw_delta_rad_wraps_shortest_path():
    assert w.yaw_delta_rad(0.0, 0.1) == pytest.approx(0.1)
    assert w.yaw_delta_rad(3.0, -3.0) == pytest.approx(2 * math.pi - 6.0)
    assert w.yaw_delta_rad(None, 0.1) is None
    assert w.yaw_delta_rad(0.0, None) is None
    assert w.yaw_delta_rad(float("nan"), 0.0) is None


# --------------------------------------------------------------------- make_waist_pitch_backend

def test_make_waist_pitch_backend_always_refuses_today():
    with pytest.raises(w.WaistPitchBackendUnavailable):
        w.make_waist_pitch_backend()


# --------------------------------------------------------------------- G1dWaistController, no backend

def test_controller_without_backend_never_arms_and_reports_why():
    ctrl = w.G1dWaistController(armed_fn=lambda: True)
    snap = ctrl.submit((1.0, 0.0), _rot_z(0.0), time.monotonic())
    assert snap["pitch_backend_present"] is False
    assert snap["pitch_reason"] == "no_backend"
    assert snap["pitch_armed"] is False
    # Intent mapping still runs (useful telemetry even though nothing is sent).
    assert snap["pitch_intent"] == pytest.approx(1.0)


def test_controller_without_backend_tracks_yaw_delta_from_arm_time():
    now = [time.monotonic()]
    armed = [False]
    ctrl = w.G1dWaistController(armed_fn=lambda: armed[0])
    # Not armed yet: no yaw reference, no intent.
    snap = ctrl.submit((0.0, 0.0), _rot_z(0.2), now[0])
    assert snap["yaw_intent"] is None
    assert snap["yaw_reference"] is None
    # Arm with the head already turned 0.2 rad: that becomes the reference,
    # so yaw_intent right at arming is 0, not 0.2.
    armed[0] = True
    snap = ctrl.submit((0.0, 0.0), _rot_z(0.2), now[0])
    assert snap["yaw_reference"] == pytest.approx(0.2)
    assert snap["yaw_intent"] == pytest.approx(0.0, abs=1e-9)
    # Turn further: delta tracks relative to the reference.
    snap = ctrl.submit((0.0, 0.0), _rot_z(0.5), now[0])
    assert snap["yaw_intent"] == pytest.approx(0.3, abs=1e-6)
    # Release: reference clears, future re-arm picks a fresh one.
    armed[0] = False
    snap = ctrl.submit((0.0, 0.0), _rot_z(0.5), now[0])
    assert snap["yaw_reference"] is None
    assert snap["yaw_intent"] is None


def test_controller_zeroes_intents_when_not_armed():
    ctrl = w.G1dWaistController(armed_fn=lambda: False)
    snap = ctrl.submit((1.0, 0.0), _rot_z(1.0), time.monotonic())
    assert snap["pitch_intent"] == 0.0
    assert snap["yaw_intent"] is None


def test_controller_zeroes_intents_when_controller_sample_stale():
    ctrl = w.G1dWaistController(armed_fn=lambda: True)
    stale_timestamp = time.monotonic() - 10.0
    snap = ctrl.submit((1.0, 0.0), _rot_z(1.0), stale_timestamp)
    assert snap["pitch_intent"] == 0.0
    assert snap["yaw_intent"] is None


def test_controller_malformed_sample_timestamp_never_raises():
    ctrl = w.G1dWaistController(armed_fn=lambda: True)
    snap = ctrl.submit((1.0, 0.0), _rot_z(0.0), float("nan"))
    assert snap["pitch_intent"] == 0.0


def test_controller_shutdown_is_idempotent_and_never_raises():
    ctrl = w.G1dWaistController(armed_fn=lambda: True)
    ctrl.submit((1.0, 0.0), _rot_z(0.0), time.monotonic())
    ctrl.shutdown("test")
    ctrl.shutdown("test_again")  # must not raise
    snap = ctrl.snapshot()
    assert snap["closed"] is True


def test_controller_armed_fn_exception_is_treated_as_not_armed():
    def boom():
        raise RuntimeError("armed_fn exploded")
    ctrl = w.G1dWaistController(armed_fn=boom)
    snap = ctrl.submit((1.0, 0.0), _rot_z(0.0), time.monotonic())
    assert snap["pitch_intent"] == 0.0
    assert snap["yaw_intent"] is None


# --------------------------------------------------------------------- G1dWaistController, fake backend
# A local fake implementing the WaistPitchBackend contract from the module
# docstring -- this is NOT the other effort's offline tool; it is a minimal
# stand-in to exercise the integration contract end to end.

class _Status:
    def __init__(self, ok, q_cmd=None, reason=None):
        self.ok = ok
        self.q_cmd = q_cmd
        self.reason = reason


class FakePitchBackend:
    def __init__(self, initial_q=0.0, raise_on_arm=None):
        self.armed = False
        self.q = initial_q
        self.arm_calls = 0
        self.step_calls = []
        self.tick_calls = 0
        self.stop_calls = []
        self._raise_on_arm = raise_on_arm

    def arm(self, now):
        self.arm_calls += 1
        if self._raise_on_arm:
            raise self._raise_on_arm
        self.armed = True
        return self.q

    def request_step(self, direction, now):
        self.step_calls.append(direction)
        self.q += direction * 0.02
        return _Status(True, q_cmd=self.q)

    def tick(self, now):
        self.tick_calls += 1
        if not self.armed:
            return None
        return {"q": self.q}

    def stop(self, reason="operator_stop"):
        self.stop_calls.append(reason)
        self.armed = False

    def status(self):
        return {"q": self.q, "armed": self.armed}


def test_controller_with_backend_arms_once_and_steps_while_armed():
    backend = FakePitchBackend()
    ctrl = w.G1dWaistController(armed_fn=lambda: True, pitch_backend=backend)
    now = time.monotonic()
    ctrl.submit((1.0, 0.0), _rot_z(0.0), now)
    ctrl.submit((1.0, 0.0), _rot_z(0.0), now)
    assert backend.arm_calls == 1  # arm() only called once while already armed
    assert backend.step_calls == [1, 1]
    snap = ctrl.snapshot()
    assert snap["pitch_backend_present"] is True
    assert snap["pitch_armed"] is True
    assert snap["pitch_reason"] == "active"


def test_controller_with_backend_never_steps_when_intent_inside_deadzone():
    backend = FakePitchBackend()
    ctrl = w.G1dWaistController(armed_fn=lambda: True, pitch_backend=backend)
    ctrl.submit((0.02, 0.0), _rot_z(0.0), time.monotonic())
    assert backend.step_calls == []
    assert backend.arm_calls == 1  # still arms (state tracking), just doesn't step


def test_controller_with_backend_stops_on_release():
    backend = FakePitchBackend()
    armed = [True]
    ctrl = w.G1dWaistController(armed_fn=lambda: armed[0], pitch_backend=backend)
    ctrl.submit((1.0, 0.0), _rot_z(0.0), time.monotonic())
    assert backend.armed is True
    armed[0] = False
    ctrl.submit((1.0, 0.0), _rot_z(0.0), time.monotonic())
    assert backend.stop_calls == ["released"]
    snap = ctrl.snapshot()
    assert snap["pitch_reason"] == "not_armed"


def test_controller_with_backend_stops_on_stale_controller_sample():
    backend = FakePitchBackend()
    ctrl = w.G1dWaistController(armed_fn=lambda: True, pitch_backend=backend)
    now = time.monotonic()
    ctrl.submit((1.0, 0.0), _rot_z(0.0), now)
    assert backend.armed is True
    ctrl.submit((1.0, 0.0), _rot_z(0.0), now - 10.0)
    assert backend.stop_calls == ["released"]


def test_controller_with_backend_arm_error_is_caught_and_reported():
    backend = FakePitchBackend(raise_on_arm=RuntimeError("no fresh state"))
    ctrl = w.G1dWaistController(armed_fn=lambda: True, pitch_backend=backend)
    snap = ctrl.submit((1.0, 0.0), _rot_z(0.0), time.monotonic())
    assert snap["pitch_reason"] == "backend_error"
    assert snap["pitch_armed"] is False


def test_controller_shutdown_stops_armed_backend():
    backend = FakePitchBackend()
    ctrl = w.G1dWaistController(armed_fn=lambda: True, pitch_backend=backend)
    ctrl.submit((1.0, 0.0), _rot_z(0.0), time.monotonic())
    assert backend.armed is True
    ctrl.shutdown("shutdown")
    assert backend.stop_calls == ["shutdown"]


def test_controller_shutdown_does_not_stop_already_unarmed_backend_twice():
    backend = FakePitchBackend()
    ctrl = w.G1dWaistController(armed_fn=lambda: False, pitch_backend=backend)
    ctrl.submit((1.0, 0.0), _rot_z(0.0), time.monotonic())
    ctrl.shutdown("shutdown")
    assert backend.stop_calls == []  # never armed, nothing to stop


def test_controller_with_backend_tick_exception_is_caught_and_stops():
    class ExplodingBackend(FakePitchBackend):
        def tick(self, now):
            super().tick(now)
            raise RuntimeError("DDS write failed")

    backend = ExplodingBackend()
    ctrl = w.G1dWaistController(armed_fn=lambda: True, pitch_backend=backend)
    snap = ctrl.submit((1.0, 0.0), _rot_z(0.0), time.monotonic())
    assert snap["pitch_reason"] == "backend_error"
    assert backend.stop_calls == ["exception"]
