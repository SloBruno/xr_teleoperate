"""G1-D waist pitch (torso lean) + waist yaw (torso rotation) Quest mapping.

Opt-in, launcher-controlled, `--g1d-waist` (`G1D_WAIST=1`). Adds two new Quest
inputs on top of the existing G1D arms+Dex3+AGV mapping, reusing the right
stick that already drives the AGV lift column:

  * right stick HORIZONTAL axis -> waist PITCH (torso lean) intent.
  * right stick VERTICAL axis   -> unchanged: AGV lift column (g1d_agv.py).
  * headset YAW (head rotation about the robot's vertical/Z axis) -> waist
    YAW (torso rotation) intent, as a delta from the yaw measured at the
    moment tracking armed (so putting on the headset facing any direction
    does not snap the torso anywhere; only turning the head *after* `r`
    commands a turn).

THIS MODULE NEVER TALKS TO DDS AND NEVER MOVES THE ROBOT. It only computes
intents (pure functions) and maintains fail-closed telemetry through
``G1dWaistController``. Real actuation is delegated to an injected backend
object; with no backend injected (the only state this repo ships), the
controller reports why it is inert and sends nothing, ever.

-------------------------------------------------------------------------
PITCH: wire contract (confirmed, NOT a safe-to-use adapter)
-------------------------------------------------------------------------
``rt/waistpitchcmd`` (MotorCmd_) / ``rt/waistpitchstate`` (MotorState_), DDS
domain 0 -- see ``/home/unitree/g1d_sdk2_demo/example/g1/g1d/g1d_waist_example.cpp``
on the robot (grep-confirmed 2026-10-29, read-only, nothing published). This
topic is NOT part of ``rt/lowcmd``, so it cannot conflict with the G1D arm
writer (``robot_arm.py``'s ``G1D_ArmController``, which must keep
``WaistYaw``/``WaistRoll``/``WaistPitch`` passive in the body LowCmd array --
see ``tests/test_g1d_arms_without_waist.py``). Same non-conflict reasoning
as ``g1d_agv.py``'s AGV/column RPC service.

A separate, isolated engineering effort (``/home/bruno/g1d_pitch_offline/``,
NOT this repo, NOT duplicated here) is building and testing the actual
safety-gated session for this topic (``control_session.PitchControlSession``):
arm-time fresh-state + no-concurrent-publisher gate, bounded step/ramp,
tracking-error/staleness/session-timeout stop conditions, fail-closed once
stopped. As of this writing that tool's ``--real`` mode is a deliberate stub
(prints the blockers below and exits 2); no real transport exists anywhere.
This module's job is ONLY the Quest-side mapping and the integration point
that a real backend plugs into once that work lands -- see
``WaistPitchBackend`` below for the exact contract expected here.

Blockers before any real pitch backend may be constructed (copied from that
tool's README so both efforts share one source of truth -- update both if
either changes):
  1. Vendor confirmation: DDS q zero/sign vs. the reported geometric range
     (-2.5..+135 deg local URDF Y axis), approved kp/kd, `mode`/`motorstate`
     semantics, command-loss behavior (hold last? ramp to safe pose? fault?),
     topic QoS/rate.
  2. A real Cyclone DDS adapter (domain 0) for ``rt/waistpitchcmd`` /
     ``rt/waistpitchstate``, reviewed and tested on its own before any use.
  3. Real concurrent-publisher detection via passive Cyclone DDS discovery
     on domain 0 for ``rt/waistpitchcmd`` (not just the fake bus used for
     the session's own unit tests).
  4. Defined mechanical support, reachable E-stop, exclusive control
     authority, and an explicit stop procedure for lost feedback / motor
     fault / timeout / dead process / lost network -- validated on real
     hardware, not inferred from simulation.
  5. Explicit operator approval, clear area, robot mechanically supported,
     E-stop reachable, a human watching the joint for the entire first test.

-------------------------------------------------------------------------
YAW: passive investigation result (2026-10-29) -- NO SAFE PATH FOUND YET
-------------------------------------------------------------------------
Searched (read-only, nothing published, nothing subscribed) the robot's SDK
trees for a waist-yaw-only analogue of ``rt/waistpitchcmd``:
  * ``grep -rli waist`` across ``g1d_sdk2_demo`` / ``unitree_sdk2`` examples:
    only ``g1d_waist_example.cpp`` (the pitch topic) mentions a dedicated
    waist topic.
  * ``grep -rli waistyaw``: the only hits are the ``WaistYaw = 12`` entry in
    ``unitree/dds_wrapper/robots/g1/defines.h``'s ``JointIndex`` enum (shared
    with ``WaistRoll``/``WaistPitch``) and this repo's own G1D arms-only
    tests/source referencing that same enum. There is no
    ``rt/waistyawcmd``/``rt/waistyawstate`` topic anywhere in the trees
    searched.
  * ``grep -rli "rt/waist"``: only the pitch topic literal appears.

Conclusion: unlike pitch, waist yaw has **no isolated, G1D-safe DDS
interface**. The only existing path to actuate ``WaistYaw`` is index 12 of
the normal G1 body ``rt/lowcmd`` array -- the exact array this repo
deliberately keeps 100% passive for G1D (``command_waist = False``,
``kp=kd=tau=0`` on all three waist slots; see
``tests/test_g1d_arms_without_waist.py``), because G1D is validated
arms-only and writing any non-zero gain into that shared array risks
whole-body balance/control semantics that have not been validated for this
platform. Enabling real waist yaw would mean deliberately breaking that
invariant, which is a much bigger safety decision than "wire up a backend"
and is explicitly out of scope here.

This module therefore implements the YAW *mapping* (head yaw -> intent) so
it is ready to integrate, but HARD-DISABLES any real yaw backend: there is
currently no constructor path in ``G1dWaistController`` that accepts one.
Before that can change, someone needs to get Unitree to confirm/provide an
isolated waist-yaw interface analogous to ``rt/waistpitchcmd`` (or an
explicit, reviewed decision to command ``rt/lowcmd`` index 12 for G1D, with
its own safety case and test coverage, separate from this change).
"""
import math
import threading
import time

from teleop.utils.quest_controls import _shape_stick_value
from teleop.utils.quest_safety import controller_sample_is_fresh

WAIST_PITCH_CMD_TOPIC = "rt/waistpitchcmd"
WAIST_PITCH_STATE_TOPIC = "rt/waistpitchstate"
WAIST_DDS_DOMAIN = 0
SEND_HZ = 20.0  # parity with g1d_agv's sender cadence; unused until a real backend exists


def _finite_or_zero(value):
    try:
        value = float(value)
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _clamp(value, limit):
    value = _finite_or_zero(value)
    value = max(-limit, min(limit, value))
    return 0.0 if value == 0 else value


def _axis(value):
    try:
        return _shape_stick_value(value)
    except Exception:
        return 0.0


def _pair(xy):
    try:
        if isinstance(xy, (str, bytes)) or len(xy) != 2:
            return None
        return xy[0], xy[1]
    except Exception:
        return None


def _raw_xy(xy):
    pair = _pair(xy)
    if pair is None:
        return None
    out = []
    for v in pair:
        try:
            v = float(v)
            out.append(v if math.isfinite(v) else None)
        except Exception:
            out.append(None)
    return out


def map_pitch_intent(right_xy):
    """Right stick HORIZONTAL axis -> normalized pitch intent in [-1, 1].

    Deadzone + cubic shaping from quest_controls (identical curve to every
    other Quest stick mapping in this repo). Non-finite/malformed input is 0.

    Sign is intentionally left as raw shaped stick-x here (no flip baked in):
    the pitch DDS sign vs. the vendor-reported geometric range is NOT
    confirmed (see module docstring), so this function must not be trusted
    to pick the "forward = lean forward" direction. Whoever wires a real
    backend must verify this physically on the bench, exactly like the AGV
    column/base sign was verified, and may need to negate this value.
    """
    pair = _pair(right_xy)
    rx = _axis(pair[0]) if pair else 0.0
    return _clamp(rx, 1.0)


def head_yaw_rad(head_pose):
    """Robot-frame head yaw (rad) from a (4, 4) SE(3) head_pose matrix.

    Uses the exact x-axis/XY-plane projection already used by
    ``televuer.tv_wrapper.get_Brobot_world_head_yaw_rot`` for arm
    referencing (Robot Convention: z up, y left, x front), so this stays
    consistent with how the rest of the pipeline already treats head yaw.
    Returns None (never guesses) for a degenerate/non-finite pose.
    """
    try:
        x0 = float(head_pose[0][0])
        x1 = float(head_pose[1][0])
    except Exception:
        return None
    if not (math.isfinite(x0) and math.isfinite(x1)):
        return None
    norm = math.hypot(x0, x1)
    if norm < 1e-6:
        return None
    return math.atan2(x1 / norm, x0 / norm)


def yaw_delta_rad(reference_yaw, current_yaw):
    """Shortest signed delta reference->current in (-pi, pi]; None if either is None."""
    if reference_yaw is None or current_yaw is None:
        return None
    if not (math.isfinite(reference_yaw) and math.isfinite(current_yaw)):
        return None
    delta = current_yaw - reference_yaw
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta <= -math.pi:
        delta += 2.0 * math.pi
    return delta


class WaistPitchBackendUnavailable(RuntimeError):
    """Raised by make_waist_pitch_backend(): no real DDS adapter exists yet."""


def make_waist_pitch_backend(*_args, **_kwargs):
    """Production factory: always fails closed today.

    There is no real ``rt/waistpitchcmd`` adapter in this repository (see
    module docstring for the exact blockers). ``--g1d-waist`` refuses to
    start rather than silently running with no effect -- see
    ``teleop_hand_and_arm.py``'s argument validation.

    When the real adapter exists (outside this change), wire it in here:
    it must expose the ``WaistPitchBackend`` contract below, which mirrors
    ``g1d_pitch_offline/control_session.PitchControlSession`` (same
    arm/stop semantics, same fail-closed-forever-after-stop lifecycle) via
    a thin production wrapper that owns the real bus/subscriber internally
    so ``tick(now)`` needs no externally supplied state:

        class WaistPitchBackend(Protocol):
            armed: bool
            def arm(self, now: float) -> float: ...
                # returns measured q0, or raises ArmError (fail closed)
            def request_step(self, direction: int, now: float) -> "Status": ...
                # direction +1/-1; Status.ok False without moving on refusal
            def tick(self, now: float):
                # -> command object or None; None means stopped/refused.
                # Must apply every safety gate PitchControlSession already
                # has: state freshness, tracking error, session timeout,
                # concurrent-publisher detection -- all before publishing.
            def stop(self, reason: str = "operator_stop") -> None: ...
            def status(self) -> dict: ...  # for telemetry/snapshot()
    """
    raise WaistPitchBackendUnavailable(
        "No real rt/waistpitchcmd backend is implemented in this repo yet. "
        "See teleop/utils/g1d_waist.py module docstring and "
        "/home/bruno/g1d_pitch_offline/README.md for the exact blockers "
        "(vendor DDS zero/sign + gain confirmation, a reviewed real DDS "
        "adapter, real concurrent-publisher discovery, mechanical/E-stop "
        "validation, explicit operator sign-off) before constructing one."
    )


class G1dWaistController:
    """Quest-side mapping + fail-closed lifecycle for waist pitch/yaw.

    With no pitch backend (the only state this repo ships: ``pitch_backend``
    stays ``None``), this computes and exposes pitch/yaw intent telemetry
    every cycle but NEVER arms, NEVER steps, NEVER sends anything. Yaw has
    no backend parameter at all: see module docstring for why that is a
    hard block, not a "not wired up yet".
    """

    def __init__(self, armed_fn, pitch_backend=None, warn=None, clock=time.monotonic):
        self._armed_fn = armed_fn
        self.pitch_backend = pitch_backend
        self._warn = warn or (lambda message: None)
        self._clock = clock
        self._lock = threading.Lock()
        self._closed = False
        self._was_armed = False
        self._yaw_reference = None
        self._raw_right = None
        self._raw_head_yaw = None
        self._pitch_intent = 0.0
        self._yaw_intent = None
        self._yaw_delta = None
        self._pitch_reason = "no_backend" if pitch_backend is None else "inactive"
        self._yaw_reason = "no_interface"  # always true today; see docstring
        self._pitch_armed = False
        self._pitch_last_status = None
        self._warned_no_backend = False

    def _armed(self):
        try:
            return bool(self._armed_fn())
        except Exception:
            return False

    def submit(self, right_xy, head_pose, sample_timestamp):
        """Pure in-memory update: compute intents, drive the pitch backend
        (if any) through its own arm/step/tick lifecycle. Never raises.
        Returns a telemetry snapshot (same shape as ``snapshot()``).
        """
        now = self._clock()
        raw_right = _raw_xy(right_xy)
        current_yaw = head_yaw_rad(head_pose)
        armed = self._armed() and not self._closed
        fresh = controller_sample_is_fresh(_finite_or_zero(sample_timestamp), now)

        with self._lock:
            self._raw_right = raw_right
            self._raw_head_yaw = current_yaw
            if armed and not self._was_armed:
                # Rising edge of tracking: zero the yaw reference right now
                # so turning the head *before* this moment never commands a
                # jump, only turning it *after* does.
                self._yaw_reference = current_yaw
            if not armed:
                self._yaw_reference = None
            self._was_armed = armed

            if armed and fresh:
                self._pitch_intent = map_pitch_intent(right_xy)
                self._yaw_delta = yaw_delta_rad(self._yaw_reference, current_yaw)
                self._yaw_intent = self._yaw_delta
            else:
                self._pitch_intent = 0.0
                self._yaw_delta = None
                self._yaw_intent = None

        self._drive_pitch_backend(armed, fresh, now)
        return self.snapshot(now)

    def _drive_pitch_backend(self, armed, fresh, now):
        backend = self.pitch_backend
        if backend is None:
            with self._lock:
                self._pitch_reason = "no_backend"
            if not self._warned_no_backend:
                self._warned_no_backend = True
                self._safe_warn(
                    "[waist] pitch mapping is ACTIVE (telemetry only): no real "
                    "rt/waistpitchcmd backend is wired in, so nothing is ever "
                    "sent to the robot."
                )
            return
        try:
            if not armed or not fresh or self._closed:
                if getattr(backend, "armed", False):
                    backend.stop("released")
                reason = "not_armed" if not armed else ("controller_stale" if not fresh else "closed")
                with self._lock:
                    self._pitch_reason = reason
                return
            if not getattr(backend, "armed", False):
                backend.arm(now)
            with self._lock:
                intent = self._pitch_intent
            if abs(intent) > 1e-6:
                backend.request_step(1 if intent > 0 else -1, now)
            status = backend.tick(now)
            with self._lock:
                self._pitch_last_status = status
                self._pitch_armed = bool(getattr(backend, "armed", False))
                self._pitch_reason = "active" if status is not None else "stopped"
        except Exception as error:
            self._safe_warn(f"[waist] pitch backend raised {type(error).__name__}: {error}; treating as stopped")
            with self._lock:
                self._pitch_reason = "backend_error"
            try:
                backend.stop("exception")
            except Exception:
                pass

    def _safe_warn(self, message):
        try:
            self._warn(message)
        except Exception:
            pass

    def shutdown(self, reason="shutdown"):
        """Never raises. Stops the pitch backend if one exists; always inert for yaw."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        backend = self.pitch_backend
        if backend is not None:
            try:
                if getattr(backend, "armed", False):
                    backend.stop(reason)
            except Exception as error:
                self._safe_warn(f"[waist] pitch backend stop() raised {type(error).__name__}: {error}")

    def snapshot(self, now=None):
        with self._lock:
            return {
                "enabled": True,
                "pitch_backend_present": self.pitch_backend is not None,
                "raw_right_xy": self._raw_right,
                "raw_head_yaw": self._raw_head_yaw,
                "yaw_reference": self._yaw_reference,
                "pitch_intent": self._pitch_intent,
                "yaw_intent": self._yaw_intent,
                "pitch_reason": self._pitch_reason,
                "pitch_armed": self._pitch_armed,
                "yaw_reason": self._yaw_reason,
                "closed": self._closed,
            }
