"""G1-D waist pitch (torso bend forward/back) from the Quest controller.

Opt-in, launcher-controlled: ``G1D_WAIST_PITCH=1`` -> ``--g1d-waist-pitch``.

Quest mapping (G1D profile)
  * RIGHT stick HORIZONTAL axis -> waist pitch RATE (right = bend forward,
    left = come back). Deadzone 0.12 + cubic shaping (quest_controls), rate
    <= PITCH_RATE_MAX = 0.10 rad/s. Releasing the stick HOLDS. Stale controller
    sample (> 0.25 s) -> rate 0 (hold, no motion). Hold = the vendored
    controller's stop(): reference re-anchored on the MEASURED q (removes the
    profiled lead, never a new forward target). The right stick VERTICAL
    axis stays the lift column (G1D_AGV=1) and the left stick the base; neither
    is read here.
  * LEFT Y (``left_ctrl_bButton``) = FAIL-SAFE. Held on a FRESH controller
    sample it latches (level/freshness based, not a single edge: offline
    button events can arrive late). Latched -> the pitch goes back to NEUTRAL
    q = 0.0 rad (exactly, not the armed pose) at the controller's profiled
    speed, holds there and only then stops publishing (release). Works with
    stale stick input and with the controller locked out; never stops arms,
    hands, AGV or the teleop. The latch is one-way for the session.
  * Terminal q / right B (teleop stop), Ctrl-C, exceptions, main-thread death,
    a foreign rt/waistpitchcmd writer (checked every cycle), 0.5 s of failed
    writes, thermal (78 C) and the 30 min session cap all do the same: home to
    0.0, then release (operator decision: teleop stop drives the PITCH to
    q=0.0; arms/hands keep their declared stable pose, no automatic home). The torso is never
    released away from neutral (gravity drops it); a stalled return HOLDS.

Control is NOT re-implemented here. It reuses, vendored byte-identical under
``teleop/third_party/g1d_pitch`` (manifest VENDORED.json, source
g1d_pitch_offline 785a29d, sha256-checked at runtime), the robot-proven
``pitch_ct.ComputedTorqueController`` (whole-body URDF gravity from the
MEASURED arms, identified ct_params.json, friction/breakaway feed-forward,
lead governor, HOLD-on-fault, thermal guards 70/78 C) and ``g1d_dds_real.RealBus``
(rt/waistpitchcmd writer + rt/waistpitchstate reader, self-GUID discovery,
foreign-writer detection). Vendoring (not importing ~/g1d_pitch_offline) keeps
every release immutable and self-contained: that directory is a mutable
working copy (not even a git checkout on the robot).

Joystick -> controller: the stick rate is integrated into small target steps
through the controller's own jog API (``jog`` with a per-call step), so the
envelope, hard cap, model admission (predicted return torque with the
measured arms <= MODEL_BUDGET_NM) and thermal refusal all apply. No q is ever
injected directly.

Limit (q_hi) = PITCH_Q_MAX = 0.30 rad. Worst case over ANY arm pose inside
the URDF limits (incl. arms extended forward), Dex3 hands (in the URDF) and
the measured lifting breakaway law |g| + 1.14|g| + 3.9 with ct_params.json:
76.8 N.m (83.0 N.m with an extra 0.25 kg object per hand) <= 85 N.m. At 0.80
rad the same worst case is ~133 N.m, so 0.80 is NOT safe for every pose.
tests/test_g1d_waist_pitch.py recomputes this.

Lifecycle: before arming nothing but passive subscribers exist (state reader +
DDS discovery). The command writer is created and the first command
published only after terminal r / right A started tracking, from the pitch
thread, after: fresh pitch state, fresh measured arms, no foreign writer, own
writer GUID confirmed. Single owner: a foreign writer refuses arming; one
appearing later sends the pitch home (never a drop). The pitch runs in its own
non-daemon 50 Hz thread; the teleop loop only posts an O(1) input sample.
"""
import hashlib
import importlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from teleop.utils.quest_controls import _shape_stick_value
from teleop.utils.quest_safety import CONTROLLER_SAMPLE_FRESHNESS_LIMIT_S

VENDOR_DIR = Path(__file__).resolve().parents[1] / "third_party" / "g1d_pitch"
CT_PARAMS_FILE = VENDOR_DIR / "ct_params.json"
URDF_FILE = VENDOR_DIR / "g1_d.urdf"

PITCH_Q_MAX = 0.30            # rad, hard cap (see module docstring)
PITCH_Q_ABSOLUTE_CEILING = 0.80
PITCH_HOME_Q = 0.0            # neutral = upright, exactly 0.0 rad
PITCH_RATE_MAX = 0.10         # rad/s, stick rate and reference speed
STICK_STALE_S = CONTROLLER_SAMPLE_FRESHNESS_LIMIT_S   # 0.25 s
STICK_FUTURE_SKEW_S = 0.05
FAILSAFE_FIELD = "left_ctrl_bButton"   # left Y
PITCH_STICK_FIELD = "right_ctrl_thumbstickValue"
PITCH_STICK_AXIS = 0          # horizontal
MODEL_BUDGET_NM = 74.0        # admission: predicted return with measured arms
WORST_CASE_BUDGET_NM = 85.0   # cap proof: any arm pose
CTRL_DT = 0.02                # 50 Hz pitch thread
TARGET_LEAD_MAX = 0.03        # target may lead measured q by this (jog direction)
MIN_JOG_STEP = 0.001          # rad: integrate the stick until this much
HOME_RETRY_S = 1.0            # re-issue home after a fault-hold while homing
ARM_TIMEOUT_S = 3.0
ARM_Q_TOL = 0.01              # rad: arming accepts q in [-tol, cap] (tol only on the
                              # gravity-safe side behind upright; vendor min -0.0436). The
                              # controller clips its first target into [0, cap], so this
                              # bounds any arming-time correction to <= tol, never above cap.
PUBLISH_FAIL_HOME_N = 25      # consecutive failed writes (0.5 s) -> end the pitch at neutral
STATE_MAX_AGE_S = 0.1
FOREIGN_CHECK_EVERY = 1       # every cycle (in-memory registry lookup); the residual
                              # window is one cycle, the arming re-check is right before
                              # the first publish
WRITER_ID = "g1d_teleop_pitch"
REFUSAL_LOG_PERIOD_S = 1.0

IDLE, ARMING, ACTIVE, HOMING, RELEASED, REFUSED, DONE = (
    "idle", "arming", "active", "homing", "released", "refused", "done")
FINISHED_STATES = (RELEASED, REFUSED, DONE)


# ---------------------------------------------------------------------------
# pure mapping
# ---------------------------------------------------------------------------
def _finite(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def stick_sample_is_fresh(sample_ts, now, stale_s=STICK_STALE_S):
    ts, now = _finite(sample_ts), _finite(now)
    if ts is None or now is None or ts <= 0.0:
        return False
    age = now - ts
    return -STICK_FUTURE_SKEW_S <= age <= stale_s


def pitch_rate_from_stick(right_xy, sample_ts, now, stale_s=STICK_STALE_S, rate_max=PITCH_RATE_MAX):
    """Right stick horizontal -> pitch rate (rad/s, + = bend forward).
    0.0 for stale/invalid samples, inside the deadzone, or malformed input."""
    if not stick_sample_is_fresh(sample_ts, now, stale_s):
        return 0.0
    try:
        if isinstance(right_xy, (str, bytes)) or len(right_xy) != 2:
            return 0.0
        x = _finite(right_xy[PITCH_STICK_AXIS])
    except Exception:
        return 0.0
    if x is None:
        return 0.0
    try:
        shaped = _shape_stick_value(x)
    except Exception:
        return 0.0
    rate = max(-1.0, min(1.0, shaped)) * float(rate_max)
    return 0.0 if rate == 0 else rate


def extract_quest_inputs(tele_data):
    """(right stick xy, left Y pressed, controller sample timestamp)."""
    xy = getattr(tele_data, PITCH_STICK_FIELD, None)
    pressed = bool(getattr(tele_data, FAILSAFE_FIELD, False))
    ts = _finite(getattr(tele_data, "controller_sample_timestamp", 0.0)) or 0.0
    return xy, pressed, ts


# ---------------------------------------------------------------------------
# vendored stack
# ---------------------------------------------------------------------------
_VENDORED_MODULES = ("g1d_dds_fake", "gravity_model", "pitch_controller", "pitch_ct",
                     "body_state", "g1d_dds_real", "body_model")


def verify_vendored(vendor_dir=VENDOR_DIR):
    manifest = json.loads((Path(vendor_dir) / "VENDORED.json").read_text(encoding="utf-8"))
    for rel, sha in manifest["sha256"].items():
        got = hashlib.sha256((Path(vendor_dir) / rel).read_bytes()).hexdigest()
        if got != sha:
            raise RuntimeError(f"vendored waist-pitch file modified: {rel} ({got[:12]} != {sha[:12]})")
    return manifest


def load_pitch_stack(vendor_dir=VENDOR_DIR):
    """Import the vendored modules from the pinned release directory only."""
    vendor_dir = str(Path(vendor_dir).resolve())
    for name in _VENDORED_MODULES:
        mod = sys.modules.get(name)
        if mod is not None and str(Path(getattr(mod, "__file__", "") or "").resolve().parent) != vendor_dir:
            raise RuntimeError(f"module {name!r} already imported from {mod.__file__}; refusing to mix")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    os.environ.setdefault("G1D_URDF", str(Path(vendor_dir) / "g1_d.urdf"))
    mods = {name: importlib.import_module(name) for name in _VENDORED_MODULES}
    for name, mod in mods.items():
        if str(Path(mod.__file__).resolve().parent) != vendor_dir:
            raise RuntimeError(f"vendored module {name} resolved outside {vendor_dir}")
    return SimpleNamespace(**mods)


def load_ct_friction(path=CT_PARAMS_FILE):
    cp = json.loads(Path(path).read_text(encoding="utf-8"))
    return (float(cp["friction_lower"][0]), float(cp["friction_lower"][1]),
            float(cp["friction_lift"][1]), cp.get("mu_basis", "real"))


def build_body_model(stack, ct_params=CT_PARAMS_FILE):
    cp = json.loads(Path(ct_params).read_text(encoding="utf-8"))
    return stack.body_model.BodyModel(scale=float(cp["scale"]), offset=float(cp["offset"]),
                                      urdf=str(URDF_FILE))


def make_arm_filter(stack, body):
    m = body.model
    limits = {n: (float(m.lowerPositionLimit[body.arm_idx[n][0]]),
                  float(m.upperPositionLimit[body.arm_idx[n][0]]))
              for n in stack.body_model.ARM_JOINTS}
    return stack.body_state.ArmStateFilter(limits=limits, max_age=STATE_MAX_AGE_S)


def build_teleop_controller(stack, body, ct_params=CT_PARAMS_FILE):
    """ComputedTorqueController with the teleop limits; jog_step adds a
    per-call step to the controller's own jog API (no direct q injection)."""
    pc = stack.pitch_controller
    CT = stack.pitch_ct.ComputedTorqueController

    class TeleopPitchController(CT):
        def jog_step(self, direction, step, now, temperature=None):
            step = _finite(step)
            if step is None or not (0.0 < step <= 0.05):
                return False, "passo inválido"
            old = self.p.step
            self.p.step = step
            try:
                return self.jog(direction, now, temperature)
            finally:
                self.p.step = old

    if not (0.0 < PITCH_Q_MAX <= PITCH_Q_ABSOLUTE_CEILING):
        raise ValueError("PITCH_Q_MAX outside (0, 0.80]")
    params = pc.Params(q_lo=0.0, q_hi=PITCH_Q_MAX, home_q=PITCH_HOME_Q, v_max=PITCH_RATE_MAX,
                       verified_q=pc.VERIFIED_SEED_Q)
    return TeleopPitchController(body, params, friction=load_ct_friction(ct_params),
                                 model_budget=MODEL_BUDGET_NM)


class LowStateArmSource:
    """Measured arms from the teleop's existing rt/lowstate reader
    (G1D_ArmController.lowstate_buffer): no second subscriber."""

    def __init__(self, arm_ctrl):
        self._buffer = arm_ctrl.lowstate_buffer

    def snapshot(self):
        lowstate, stamp = self._buffer.GetSnapshot()
        if lowstate is None:
            return None, None
        return lowstate.motor_state, stamp


class RealPitchBus:
    """Teleop adapter over the vendored RealBus (ChannelFactory is already
    initialised by the teleop; RealBus's second Initialize is a no-op)."""

    def __init__(self, stack, domain_id=0):
        self.bus = stack.g1d_dds_real.RealBus(domain_id=domain_id, iface=None)

    def read_state(self):
        return self.bus.read_state()

    def foreign_writers(self):
        return self.bus.cmd_registry.other_writers(WRITER_ID, time.monotonic())

    def create_cmd_writer(self):
        return self.bus.create_cmd_writer(writer_id=WRITER_ID)

    def publish_cmd(self, writer_id, cmd, now=None):
        self.bus.publish_cmd(writer_id, cmd, now)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# supervisor
# ---------------------------------------------------------------------------
class WaistPitchSupervisor:
    def __init__(self, controller, bus, arm_filter, arm_snapshot, clock=time.monotonic,
                 log=None, main_alive=None, jsonl_path=None):
        self.ctrl = controller
        self.bus = bus
        self.arm_filter = arm_filter
        self._arm_snapshot = arm_snapshot
        self._clock = clock
        self._log = log or (lambda level, msg: None)
        self._main_alive = main_alive or (lambda: threading.main_thread().is_alive())
        self._lock = threading.Lock()
        self._input = (None, 0.0)
        self._arm_requested = False
        self._home_reason = None
        self.failsafe_latched = False
        self.state = IDLE
        self.last_reason = None
        self._arm_t0 = None
        self._writer_created = False
        self._last_cmd = None
        self._last_t = None
        self._accum = 0.0
        self._moving = False
        self._last_home_cmd_t = None
        self._last_arm_stamp = None
        self._arm_sample_bad = False
        self._bus_closed = False
        self._n = 0
        self._tick_errors = 0
        self.publish_failures = 0
        self.admission_refusals = 0
        self._last_refusal_log = -1e9
        self._thread = None
        self._jsonl = None
        self._jsonl_path = jsonl_path
        self._jsonl_lines = 0

    # ---- teleop-loop side (O(1), never touches the bus) -------------------
    def submit(self, right_xy, failsafe_pressed, sample_ts):
        now = self._clock()
        try:
            xy = (float(right_xy[0]), float(right_xy[1]))
        except Exception:
            xy = None
        ts = _finite(sample_ts) or 0.0
        with self._lock:
            self._input = (xy, ts)
            if (failsafe_pressed and not self.failsafe_latched and self.state in (ARMING, ACTIVE, HOMING)
                    and stick_sample_is_fresh(ts, now)):
                self.failsafe_latched = True

    def request_arm(self, start_thread=True):
        with self._lock:
            if self.state != IDLE or self._arm_requested:
                return False
            self._arm_requested = True
        if start_thread:
            self._thread = threading.Thread(target=self._run, name="g1d-waist-pitch", daemon=False)
            self._thread.start()
        return True

    def request_home_and_release(self, reason):
        with self._lock:
            if self._home_reason is None:
                self._home_reason = str(reason)

    @property
    def finished(self):
        return self.state in FINISHED_STATES

    # ---- helpers ------------------------------------------------------------
    def _say(self, level, msg):
        try:
            self._log(level, "[waist-pitch] " + msg)
        except Exception:
            pass

    def _fresh_state(self, now):
        st, stamp = self.bus.read_state()
        s = _finite(stamp)
        if st is None or s is None:
            return st, stamp, False
        ok = -STICK_FUTURE_SKEW_S <= now - s <= STATE_MAX_AGE_S
        return st, stamp, ok

    def _update_arms(self, now):
        try:
            states, stamp = self._arm_snapshot()
        except Exception:
            states, stamp = None, None
        s = _finite(stamp)
        if states is not None and s is not None and s != self._last_arm_stamp:
            self._last_arm_stamp = s
            self._arm_sample_bad = not self.arm_filter.ingest(states, s)
        if self._arm_sample_bad:
            q, dq, ddq = self.arm_filter.get()
            return q, dq, ddq, False
        # Re-anchor: a sample stamped after this cycle's `now` (subscriber
        # thread) is fresh, not "negative age".
        stamp_now = self.arm_filter.stamp
        if stamp_now is not None and stamp_now > now and stamp_now - now <= STICK_FUTURE_SKEW_S:
            now = stamp_now
        fresh = self.arm_filter.fresh(now)
        q, dq, ddq = self.arm_filter.get()
        return q, dq, ddq, fresh and q is not None

    def _finish(self, state, reason, level="warning"):
        self.state = state
        self.last_reason = reason
        self._say(level, f"{state.upper()}: {reason}")
        self._close_jsonl()
        self._close_bus()

    def _close_bus(self):
        # only in a finished state: released at neutral, or never published
        if self._bus_closed or self.state not in FINISHED_STATES:
            return
        self._bus_closed = True
        try:
            self.bus.close()
        except Exception as error:
            self._say("warning", f"falha ao fechar o barramento do pitch: {error!r}")

    # ---- one control cycle --------------------------------------------------
    def step(self, now=None):
        now = self._clock() if now is None else now
        if self.finished:
            return None
        with self._lock:
            xy, ts = self._input
            arm_req = self._arm_requested
            home_reason = self._home_reason
            failsafe = self.failsafe_latched
        if home_reason is None and not self._main_alive():
            self.request_home_and_release("main thread ended")
            home_reason = "main thread ended"

        if self.state == IDLE:
            if home_reason is not None:
                self._finish(DONE, f"never armed ({home_reason})", "info")
                return None
            if not arm_req:
                return None
            self.state = ARMING
            self._arm_t0 = now
            self._say("info", "armando (aguardando estado fresco, braços, sem outro publisher)")
        if self.state == ARMING:
            if home_reason is not None or failsafe:
                self._finish(DONE, f"arming cancelled ({home_reason or 'fail-safe'}); nothing published", "info")
                return None
            if not self._try_arm(now):
                return None
        return self._cycle(now, xy, ts, home_reason, failsafe)

    def _try_arm(self, now):
        if now - self._arm_t0 > ARM_TIMEOUT_S:
            self._finish(REFUSED, self.last_reason or "timeout ao armar")
            return False
        st, stamp, ok = self._fresh_state(now)
        if not ok:
            self.last_reason = "sem estado fresco de rt/waistpitchstate"
            return False
        aq, adq, addq, arms_fresh = self._update_arms(now)
        if not arms_fresh:
            self.last_reason = "sem estado fresco dos braços (rt/lowstate)"
            return False
        q0 = _finite(getattr(st, "q", None))
        if q0 is None or not (-ARM_Q_TOL <= q0 <= PITCH_Q_MAX):
            self._finish(REFUSED, f"q medido {q0} fora de [{-ARM_Q_TOL:.2f}, {PITCH_Q_MAX:.2f}] rad: traga o tronco ao neutro "
                                  "com a ferramenta do pitch antes de usar o teleop")
            return False
        if not self._writer_created:
            try:
                foreign = self.bus.foreign_writers()
            except Exception as error:
                self._finish(REFUSED, f"descoberta DDS falhou ({error!r})")
                return False
            if foreign:
                self._finish(REFUSED, f"outro publisher em rt/waistpitchcmd: {foreign!r}")
                return False
            self._writer_created = True
            if not self.bus.create_cmd_writer():
                self._finish(REFUSED, "GUID do nosso publisher não confirmado")
                return False
            try:
                foreign = self.bus.foreign_writers()
            except Exception as error:
                self._finish(REFUSED, f"descoberta DDS falhou ({error!r})")
                return False
            if foreign:
                self._finish(REFUSED, f"outro publisher em rt/waistpitchcmd: {foreign!r}")
                return False
            now = self._clock()
            st, stamp, ok = self._fresh_state(now)
            if not ok:
                self.last_reason = "estado ficou velho ao criar o publisher"
                return False
            aq, adq, addq, arms_fresh = self._update_arms(now)
            if not arms_fresh:
                return False
        self.ctrl.set_body_state(aq, adq, addq, True)
        try:
            self.ctrl.arm(now, st, stamp)
        except RuntimeError as error:
            self.last_reason = str(error)
            return False
        self.state = ACTIVE
        self._last_t = now
        self._open_jsonl()
        self._say("warning", f"ATIVO: segurando q={float(st.q):+.3f}. Stick direito horizontal = inclinar "
                             f"(máx {PITCH_Q_MAX:.2f} rad, {PITCH_RATE_MAX:.2f} rad/s); Y esquerdo = voltar ao "
                             f"neutro e soltar.")
        return True

    def _cycle(self, now, xy, ts, home_reason, failsafe):
        ctrl = self.ctrl
        dt = 0.0 if self._last_t is None else max(0.0, min(now - self._last_t, 0.05))
        self._last_t = now
        self._n += 1

        # foreign writer appeared: go home, never drop the torso
        if self._n % FOREIGN_CHECK_EVERY == 0 and self.state == ACTIVE:
            try:
                foreign = self.bus.foreign_writers()
            except Exception as error:
                foreign = [f"descoberta DDS falhou: {error!r}"]
            if foreign:
                self.request_home_and_release(f"outro publisher em rt/waistpitchcmd {foreign!r}")
                home_reason = home_reason or "foreign writer"

        st, stamp, state_fresh = self._fresh_state(now)
        aq, adq, addq, arms_fresh = self._update_arms(now)
        if arms_fresh:
            ctrl.set_body_state(aq, adq, addq, True)
        else:
            ctrl.set_body_state(None, fresh=False)
        temp = None
        if st is not None:
            try:
                temp = [float(v) for v in getattr(st, "temperature", ())] or None
            except Exception:
                temp = None

        if self.state == ACTIVE and ctrl.homing_then_release and home_reason is None and not failsafe:
            # the controller started its own home+release (thermal 78 C,
            # session cap): adopt it so stick input can never cancel it
            home_reason = "controlador iniciou volta ao neutro (térmico/tempo de sessão)"
            self.request_home_and_release(home_reason)
        if self.state == ACTIVE and (failsafe or home_reason is not None):
            why = "FAIL-SAFE (Y esquerdo)" if failsafe else home_reason
            self.state = HOMING
            self.last_reason = why
            self._accum, self._moving = 0.0, False
            if not ctrl.homing_then_release:
                ctrl.safe_shutdown(now)
            self._last_home_cmd_t = now
            self._say("warning", f"{why}: voltando a q={PITCH_HOME_Q:.1f} e soltando só no neutro.")
        elif self.state == HOMING:
            if (not ctrl.released and not ctrl.move_pending and
                    (self._last_home_cmd_t is None or now - self._last_home_cmd_t >= HOME_RETRY_S)):
                ctrl.safe_shutdown(now)          # clears lockout, re-targets 0.0
                self._last_home_cmd_t = now
        elif self.state == ACTIVE:
            self._apply_stick(now, xy, ts, st if state_fresh else None, temp, dt)

        try:
            cmd = ctrl.tick(now, st, stamp, temp)
            self._tick_errors = 0
        except Exception as error:
            self._tick_errors += 1
            if self._tick_errors in (1, 50) or self._tick_errors % 500 == 0:
                self._say("error", f"erro no controlador ({error!r}); mantendo o último comando (segura).")
            cmd = self._last_cmd
            if cmd is None:
                return None
            if self.state == ACTIVE:
                self.request_home_and_release(f"erro no controlador: {error!r}")
        else:
            if cmd is None:
                if ctrl.released:
                    self._finish(RELEASED, f"solto no neutro (q={float(st.q):+.4f}); {ctrl.release_reason}. "
                                           "Braços/teleop continuam. Reinicie o teleop para usar o pitch de novo.")
                return None
        self._last_cmd = cmd
        try:
            self.bus.publish_cmd(WRITER_ID, cmd, now)
            self.publish_failures = 0
        except Exception as error:
            self._publish_failed(error)
        for ev in ctrl.drain_events():
            if ev.kind != "verified":
                self._say("info", f"[{ev.kind}] {ev.msg}")
        self._write_jsonl(now, st, cmd, temp, xy, ts, arms_fresh)
        return cmd

    def _publish_failed(self, error):
        self.publish_failures += 1
        n = self.publish_failures
        if n == 1 or n == PUBLISH_FAIL_HOME_N or n % 2500 == 0:
            self._say("error", f"ALARME: falha ao publicar rt/waistpitchcmd ({n}x seguidas): {error!r}. "
                               "Se o tronco não responder, segure-o / use a parada de emergência.")
        if n >= PUBLISH_FAIL_HOME_N and self.state == ACTIVE:
            self.request_home_and_release(f"falha de publicação ({n}x seguidas)")

    def _apply_stick(self, now, xy, ts, st, temp, dt):
        ctrl = self.ctrl
        rate = pitch_rate_from_stick(xy, ts, now)
        if rate == 0.0:
            self._accum = 0.0
            if self._moving:
                self._moving = False
                ctrl.stop(now)                  # release / stale -> hold at measured q
            return
        if self._moving and (self._accum * rate) < 0:
            self._accum = 0.0
        self._moving = True
        self._accum += rate * dt
        if abs(self._accum) < MIN_JOG_STEP:
            return
        direction = 1 if self._accum > 0 else -1
        want = abs(self._accum)
        self._accum = 0.0
        q_meas = ctrl.last_good_q
        if st is None or q_meas is None:
            return
        lead = direction * (ctrl.target - q_meas)
        room = TARGET_LEAD_MAX - lead
        if room <= 1e-6:
            return
        ok, why = ctrl.jog_step(direction, min(want, room, 0.05), now, temp)
        if not ok:
            self.admission_refusals += 1
            if now - self._last_refusal_log >= REFUSAL_LOG_PERIOD_S:
                self._last_refusal_log = now
                self._say("info", f"avanço recusado: {why}")

    # ---- thread -------------------------------------------------------------
    def _guarded_step(self):
        try:
            return self.step()
        except BaseException as error:   # never die silently while leaned (incl. SystemExit/
            # KeyboardInterrupt raised inside this thread): request home, keep holding
            self._say("error", f"erro inesperado no ciclo: {error!r}")
            self.request_home_and_release(f"erro no ciclo do pitch: {error!r}")
            if self._last_cmd is not None and self.state in (ACTIVE, HOMING):
                try:
                    self.bus.publish_cmd(WRITER_ID, self._last_cmd, self._clock())
                except Exception:
                    pass
            return self._last_cmd

    def _run(self):
        next_t = self._clock()
        while not self.finished:
            self._guarded_step()
            next_t += CTRL_DT
            delay = next_t - self._clock()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = self._clock()
        self._close_jsonl()
        self._close_bus()

    # ---- telemetry (pitch thread only) --------------------------------------
    def _open_jsonl(self):
        if not self._jsonl_path:
            return
        try:
            Path(self._jsonl_path).parent.mkdir(parents=True, exist_ok=True)
            self._jsonl = open(self._jsonl_path, "a", buffering=1 << 16)
        except Exception as error:
            self._jsonl = None
            self._say("warning", f"log JSONL indisponível ({error!r}); seguindo sem log.")

    def _write_jsonl(self, now, st, cmd, temp, xy, ts, arms_fresh):
        if self._jsonl is None:
            return
        c = self.ctrl
        try:
            self._jsonl.write(json.dumps({
                "t": now, "wall": time.time(), "state": self.state,
                "q": getattr(st, "q", None), "dq": getattr(st, "dq", None),
                "tau_est": getattr(st, "tau_est", None), "temperature": temp,
                "q_ref": c.q_ref, "v_ref": c.v_ref, "target": c.target,
                "cmd_q": cmd.q, "cmd_dq": cmd.dq, "cmd_tau": cmd.tau, "kp": cmd.kp, "kd": cmd.kd,
                "stick_x": xy[0] if xy else None, "stick_ts": ts, "failsafe": self.failsafe_latched,
                "arms_fresh": bool(arms_fresh), "g_body": c.last_g,
                "predicted_lift": c.predicted_lift(st.q) if st is not None else None,
                "locked_out": c.locked_out, "move_pending": c.move_pending,
                "homing": c.homing_then_release, "assist_nm": c._assist_torque(),
            }) + "\n")
            self._jsonl_lines += 1
            if self._jsonl_lines % 50 == 0:
                self._jsonl.flush()
        except Exception as error:
            self._say("warning", f"log JSONL desativado ({error!r}).")
            self._close_jsonl()

    def _close_jsonl(self):
        f, self._jsonl = self._jsonl, None
        if f is not None:
            try:
                f.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# teleop integration helpers
# ---------------------------------------------------------------------------
def create_waist_pitch_supervisor(arm_ctrl, log, jsonl_path=None, wait_state_s=2.0):
    """Pre-arm construction: passive subscribers only. Raises RuntimeError
    (fail closed, before any motion) if anything required is missing."""
    verify_vendored()
    stack = load_pitch_stack()
    body = build_body_model(stack)
    ctrl = build_teleop_controller(stack, body)
    bus = RealPitchBus(stack)
    deadline = time.monotonic() + wait_state_s
    while bus.read_state()[0] is None:
        if time.monotonic() > deadline:
            bus.close()
            raise RuntimeError("--g1d-waist-pitch: no rt/waistpitchstate samples")
        time.sleep(0.05)
    foreign = bus.foreign_writers()
    if foreign:
        bus.close()
        raise RuntimeError(f"--g1d-waist-pitch: another rt/waistpitchcmd writer exists: {foreign!r}")
    source = LowStateArmSource(arm_ctrl)
    return WaistPitchSupervisor(ctrl, bus, make_arm_filter(stack, body), source.snapshot,
                                log=log, jsonl_path=jsonl_path)


def finish_waist_pitch(sup, log, sleep=time.sleep, poll_s=0.1, report_every_s=2.0, max_wait_s=None):
    """Teleop shutdown: home to 0.0, hold, release; block until done.
    There is deliberately NO software override that stops publishing away
    from neutral: Ctrl-C is acknowledged and the wait continues (the pitch
    thread is non-daemon and keeps holding/homing). A return that cannot
    complete HOLDS; the operator uses the physical support / E-stop."""
    if sup is None:
        return "absent"
    sup.request_home_and_release("teleop stop")
    if sup._thread is None and sup.state == IDLE:
        sup.step()
    waited = 0.0
    next_report = 0.0
    while not sup.finished:
        if max_wait_s is not None and waited >= max_wait_s:
            return "timeout"
        try:
            if waited >= next_report:
                next_report += report_every_s
                log("warning", f"[waist-pitch] aguardando o pitch voltar ao neutro q=0.0 (estado={sup.state}); "
                               "ele só solta no neutro.")
            sleep(poll_s)
            waited += poll_s
        except KeyboardInterrupt:
            log("warning", "[waist-pitch] Ctrl-C ignorado: o pitch continua voltando/segurando e só solta no "
                           "neutro q=0.0. Se ele não volta, segure o tronco/use a parada de emergência.")
    if sup._thread is None:
        sup._close_bus()
    return sup.state
