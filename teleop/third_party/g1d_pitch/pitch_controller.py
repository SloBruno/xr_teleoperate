"""Definitive G1-D waist-pitch controller (rt/waistpitchcmd).

Why this exists (root cause, measured on the real robot 2026-10-29 from
~18k logged samples in logs/*.jsonl):

* tau_est on this firmware is NOT a sensed torque: it equals the commanded
  PD law kp*(q_cmd-q) + kd*(dq_cmd-dq) + tau (mean diff ~0 N.m). So torques
  below are commanded torques at steady motion, which is what we need.
* Gear friction grows with load: 16.4 / 17.2 / 18.6 N.m at |g| = 2 / 14 / 24
  N.m (jog1.jsonl) -> F(q) = 16.1 + 0.10*|g(q)|, stiction ~12% higher.
  Reversing at 0.93 rad therefore needs ~ g - F = -53 N.m (jog2.jsonl: stuck
  6 s, then snapped at -0.36 rad/s).
* Holding torque g(q) = 1.162 * g_URDF(q) + 11.73 N.m (URDF RNEA shape is
  right, offset/scale identified; residual <= ~1.3 N.m over q in [0, 0.9]).
* Above ~0.35 rad |g(q)| > friction: an un-commanded joint falls forward on
  its own (observed: crept 0.90 -> 1.21 rad between sessions). Coming back
  from 0.5 rad needs ~36 N.m, but the old position-only loop could only make
  kp*lead <= 22.5 N.m -- so it could not return, and every stiction
  breakaway near zero was a stored-error "snap" that tripped the brake.

Design (standard, not a patch):
  tau_cmd = g(q_meas) + c_f*F*tanh(v_ref/v_eps)        (feedforward)
          + kp*(q_ref - q) + kd*(v_ref - dq)            (firmware PD)
  q_ref/v_ref follow a velocity- and acceleration-limited (trapezoidal)
  profile toward the operator target, and v_ref is sent as dq so kd damps
  deviation from the PLAN, not motion itself.
  Faults never release a leaned joint: they re-anchor to HOLD at the
  measured position (the motor closes its own encoder loop, so publishing a
  fixed q holds even if our feedback goes stale). Release (stop publishing)
  is only allowed where friction alone holds the joint; otherwise the
  controller drives home first.

Return-verified envelope (jog3.jsonl, 2026-10-30): it HELD 1.306 rad with
-42 N.m, but lifting back needed > 85 N.m (motor limit ~80 N.m): commanded
-77..-85 N.m, zero motion. Holding cannot reveal this (friction helps hold and
fights lifting), and above 0.6 rad no controlled return had ever been
measured. So forward motion is only allowed up to `probe_margin` beyond the
highest angle from which the joint has *actually* driven itself back under
control (`q_verified`, persisted by the tool). Extending the envelope is the
measurement: go a little further, press s, it comes back -> verified.
If lifting saturates the motor the reference freezes and it HOLDS with a
clear message instead of winding up error.

Pure logic, no I/O, injectable clock: fully testable against plant_sim.py.
"""
from dataclasses import dataclass, field
from math import isfinite, sqrt, tanh

from g1d_dds_fake import MotorCmd_
from gravity_model import gravity_feedforward_tau

# ---- identified plant (see module docstring) --------------------------------
G_ID_SCALE = 1.162
G_ID_OFFSET_NM = 11.73
COULOMB_NM = 16.1          # unloaded gear friction
LOAD_FRICTION = 0.10       # + this fraction of the load when lowering (with gravity)
LIFT_LOAD_FRICTION = 0.35  # lifting against gravity: jog4.jsonl, 16.3/16.8/20.3/26.6
                           # N.m excess at |g| = 3/7/17/27 N.m
VALIDATED_Q_MAX = 0.95   # identification data covered q in [0, 0.93]
EXTENDED_Q_MAX = 2.3062  # vendor max - 0.05. The real limit is the return-verified
                         # envelope: lifting saturates the motor somewhere in 0.9..1.3

VENDOR_Q_MIN = -0.04363
VENDOR_Q_MAX = 2.3562

VERIFIED_SEED_Q = 0.66   # highest controlled return measured (jog4.jsonl)
# torque1.jsonl (2026-11-04): at q=0.90 lifting did NOT break away with 75 N.m
# but did with 80 N.m (repeatable, 2 sessions, max temp 41 C). So the motor
# delivers at least ~78 N.m; the old 72 N.m saturation / 60 N.m budget were
# software limits, not the motor's.
TAU_MOTOR_NM = 80.0      # demonstrated (torque1.jsonl); true peak still unknown
TAU_SAT_NM = 90.0        # commanded |tau| above this with no motion = saturated
PROBE_MIN_LEVEL_NM = 70.0
EXPLORE_SAT_NM = 120.0   # --explore: saturation threshold (explore1: moved with ~102 N.m at 1.20)
EXPLORE_BUDGET_NM = 115.0 # --explore: accept returns that needed up to this
# A return only counts as proven if it needed <= LIFT_BUDGET_NM, so that the
# next +probe_margin step (breakaway grows ~2-3 N.m per 0.05 rad near 0.9)
# still fits the DEMONSTRATED torque. Proving more torque needs exploration
# (--explore): go a bit beyond, and if it cannot lift back it HOLDS there.
LIFT_BUDGET_NM = 74.0    # demonstrated ~78 N.m - 4 N.m headroom:
                         # keeps >= 20 N.m headroom for the next probe_margin

# Torque-capability probe ('p' key): measures what the motor really delivers
# when lifting back, instead of assuming the URDF 80 N.m. Operator-paced,
# one level per key press, lifting (safe) direction only, self-limiting
# position command (torque falls as soon as the joint moves).
PROBE_LEVELS_NM = (75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 110.0, 120.0)
PROBE_HARD_MAX_NM = 120.0  # Unitree: largest G1 joint motor peak torque
PROBE_HOLD_S = 1.5
PROBE_MIN_Q = 0.3
PROBE_TEMP_ABORT = 65.0

KP_HARD_CEILING = 500.0  # vendor demo value
KD_HARD_CEILING = 12.0
TAU_FEEDFORWARD_CEILING_NM = 110.0  # unchanged normal feed-forward ceiling
TAU_HARD_CEILING_NM = 120.0  # cap on the complete commanded PD+FF snapshot


def g_hold(q):
    """Identified torque needed to hold the joint still at q (N.m)."""
    return G_ID_SCALE * gravity_feedforward_tau(q, scale=1.0, ceiling=None) + G_ID_OFFSET_NM


def friction_nm(q, lifting=False):
    """Identified kinetic gear friction at q (N.m); larger when lifting."""
    mu = LIFT_LOAD_FRICTION if lifting else LOAD_FRICTION
    return COULOMB_NM + mu * abs(g_hold(q))


# Lifting BREAKAWAY (static) friction under load, fitted to the measured
# breakaways: sombra1 0.73 rad ~67 N.m, torque1 0.90 rad 75<b<=80, jog3
# 1.30 rad >85 N.m (no motion). Lift-from-hold = |g| + (1.14|g| + 3.9):
# 67.0 / 77.5 / 90.6 N.m. Kinetic lifting friction is much lower (~0.3|g|),
# a big stiction step under load: it sticks, then slides.
LIFT_BREAK_SLOPE, LIFT_BREAK_OFFSET = 1.14, 3.9
ASSIST_LEAD_RAD = 0.008  # lag window for 'returning under our torque' bookkeeping
PUSH_GRAVITY_FRAC = 0.8  # moving back while commanding < 80% of the gravity hold = a push


def lift_breakaway_friction(g_abs):
    return LIFT_BREAK_SLOPE * g_abs + LIFT_BREAK_OFFSET


def friction_holds(q, margin=0.6):
    """True where friction alone keeps the joint from falling, with margin.
    Only there may we stop publishing. Uses unloaded friction (conservative)."""
    return abs(g_hold(q)) < margin * COULOMB_NM


@dataclass
class Params:
    kp: float = 300.0
    kd: float = 12.0
    v_max: float = 0.15          # rad/s reference speed
    a_max: float = 0.30          # rad/s^2 reference accel
    step: float = 0.02           # rad per accepted keypress
    friction_comp: float = 0.8   # fraction of identified kinetic friction compensated
    v_eps: float = 0.002         # rad/s: full compensation from the first
                                 # reference step (a_max*dt = 0.006 rad/s)
    gravity_comp: float = 1.0
    q_lo: float = VENDOR_Q_MIN + 0.01
    q_hi: float = VALIDATED_Q_MAX
    home_q: float = 0.0
    state_max_age: float = 0.1
    dq_fault: float = 0.35       # rad/s measured -> re-anchor HOLD
    track_fault: float = 0.06    # rad |q_ref-q| ...
    track_fault_time: float = 0.3  # ... sustained this long -> re-anchor HOLD
    lockout_faults: int = 3      # this many faults ...
    lockout_window: float = 10.0  # ... within this window -> motion locked out
    temp_warn: float = 70.0      # deg C: refuse moves away from home
    temp_home: float = 78.0      # deg C: auto home + release
    session_max: float = 1800.0  # s: auto home + release
    verified_q: float = VERIFIED_SEED_Q  # highest proven-returnable angle
    lift_budget: float = LIFT_BUDGET_NM  # --explore raises it
    sat_nm: float = TAU_SAT_NM           # commanded |tau| w/o motion = saturated
    probe_margin: float = 0.05   # may go this far beyond verified_q
    sat_time: float = 0.3        # s of saturation before HOLD + message
    arrival_tolerance: float = 0.01  # measured q must reach target within this error
    arrival_timeout: float = 1.5     # bounded breakaway window before measured HOLD
    arrival_velocity: float = 0.02   # measured velocity must be low to latch arrival
    arrival_dwell: float = 0.1       # stable low-velocity dwell before latching
    # explore2.jsonl: short return steps at 1.24-1.40 rad never broke away;
    # feed-forward (<100% friction by design) + kp*step stayed below the
    # breakaway. While an explicit move is stuck short of its target, ramp a
    # directional breakaway torque; drop it the instant the joint moves.
    breakaway_rate: float = 40.0     # N.m/s ramp while stuck
    breakaway_max: float = 60.0      # N.m extra, total still capped by TAU_HARD_CEILING_NM
    # explore3.jsonl (1.44 -> 1.18 rad): every breakaway was followed by an
    # instant drop of ~30-50 N.m (assist and static-friction FF were switched
    # off at dq < -0.01), so the joint slid a few mrad and re-stuck (stick-
    # slip), while the reference ran 0.06-0.08 rad ahead -> tracking-fault
    # lockouts, and the stored kp*lead snapped it at 0.34 rad/s. Now:
    #  - the reference may lead the measured joint by at most lead_max;
    #  - the extra torque is a bounded velocity regulator: it grows while the
    #    joint is slower than v_slow, holds while moving, and decays only when
    #    the joint goes faster than v_fast.
    lead_max: float = 0.025          # rad, reference lead over measured q
    v_slow: float = 0.02             # rad/s: below this the move counts as stuck
    v_fast_frac: float = 1.3         # x v_max: above this the extra torque decays
    breakaway_decay: float = 400.0   # N.m/s decay when too fast (x up to 3 by overspeed)
    stall_timeout: float = 3.0       # s without 3 mrad of progress -> HOLD (ramp to max takes 1.5 s)

    def validate(self):
        if not (0 <= self.kp <= KP_HARD_CEILING):
            raise ValueError('kp must be in [0, %.0f]' % KP_HARD_CEILING)
        if not (0 <= self.kd <= KD_HARD_CEILING):
            raise ValueError('kd must be in [0, %.0f]' % KD_HARD_CEILING)
        if not (0 < self.v_max <= 0.5):
            raise ValueError('v_max must be in (0, 0.5] rad/s')
        if not (0 < self.a_max <= 2.0):
            raise ValueError('a_max must be in (0, 2.0] rad/s^2')
        if not (0 < self.step <= 0.05):
            raise ValueError('step must be in (0, 0.05] rad')
        if not (0 <= self.friction_comp <= 1.0):
            raise ValueError('friction_comp must be in [0, 1]')
        if not (0 <= self.gravity_comp <= 1.2):
            raise ValueError('gravity_comp must be in [0, 1.2]')
        if not (VENDOR_Q_MIN <= self.q_lo < self.q_hi <= VENDOR_Q_MAX - 0.05):
            raise ValueError('need vendor_min <= q_lo < q_hi <= vendor_max-0.05')
        if not (self.q_lo <= self.home_q <= self.q_hi):
            raise ValueError('home_q outside [q_lo, q_hi]')
        if not friction_holds(self.home_q):
            raise ValueError('home_q must be a pose where friction alone holds the joint')
        if not (70.0 <= self.sat_nm <= EXPLORE_SAT_NM):
            raise ValueError('sat_nm must be in [70, %.0f] N.m' % EXPLORE_SAT_NM)
        if not (40.0 <= self.lift_budget <= self.sat_nm):
            raise ValueError('lift_budget must be in [40, sat_nm] N.m')
        if not (0.0 < self.probe_margin <= 0.10):
            raise ValueError('probe_margin must be in (0, 0.10] rad')
        if self.dq_fault <= self.v_max:
            raise ValueError('dq_fault must exceed v_max')
        if not (0.0 < self.arrival_tolerance <= self.track_fault):
            raise ValueError('arrival_tolerance must be in (0, track_fault]')
        if not (0.0 < self.arrival_timeout <= 2.0):
            raise ValueError('arrival_timeout must be in (0, 2] s')
        if not (0.0 < self.arrival_velocity <= self.dq_fault):
            raise ValueError('arrival_velocity must be in (0, dq_fault]')
        if not (0.0 < self.arrival_dwell <= 1.0):
            raise ValueError('arrival_dwell must be in (0, 1] s')
        if not (0.0 <= self.breakaway_rate <= 100.0):
            raise ValueError('breakaway_rate must be in [0, 100] N.m/s')
        if not (0.0 <= self.breakaway_max <= TAU_HARD_CEILING_NM):
            raise ValueError('breakaway_max must be in [0, TAU_HARD_CEILING_NM]')
        if not (0.005 <= self.lead_max < self.track_fault):
            raise ValueError('lead_max must be in [0.005, track_fault)')
        if not (0.0 < self.v_slow < self.v_max):
            raise ValueError('v_slow must be in (0, v_max)')
        if not (1.0 < self.v_fast_frac and self.v_fast_frac * self.v_max < self.dq_fault):
            raise ValueError('v_fast_frac*v_max must stay below dq_fault')
        if not (0.0 < self.breakaway_decay <= 1000.0):
            raise ValueError('breakaway_decay must be in (0, 1000] N.m/s')
        if not (0.5 <= self.stall_timeout <= 10.0):
            raise ValueError('stall_timeout must be in [0.5, 10] s')


@dataclass
class Event:
    t: float
    kind: str
    msg: str


class PitchController:
    """States: HOLD/MOVE are one tracking mode (reference toward target);
    HOMING drives to home then releases; RELEASED publishes nothing."""

    def __init__(self, params=None):
        self.p = params or Params()
        self.p.validate()
        self.armed = False
        self.released = False
        self.homing_then_release = False
        self.locked_out = False
        self.q_ref = self.v_ref = None
        self._target = None
        self.move_pending = False
        self.arrived_latched = False
        self.last_t = None
        self.armed_at = None
        self.last_good_q = None
        self.track_bad_since = None
        self.fault_times = []
        self.events = []
        self.release_reason = None
        self.q_verified = self.p.verified_q
        self.sat_since = None
        self.saturated_at_q = None
        self.frontier_reported = False
        self.lift_peak = 0.0
        self.probe = None
        self.probe_results = []
        self._dq_meas = 0.0
        self.arrival_wait_since = None
        self.arrival_stable_since = None
        self.breakaway_direction = 0.0
        self.return_origin_q = None
        self.return_lag_since = None
        self.return_lag_observed = False
        self.return_assisted = False
        self.arrival_assist_done = False
        self.assist_nm = 0.0
        self.assist_sign = 0.0
        self.progress_q = None
        self.progress_t = None
        self.saw_return_motion = False
        self._ret_since = None
        self._push_since = None

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        # Direct assignment remains a deliberate compatibility entry point for
        # tests and offline callers; internal lifecycle changes use _set_target.
        old = self._target
        self._target = value
        if hasattr(self, 'move_pending') and value is not None:
            self._begin_move(self.move_pending and old is not None and
                             (value - old) * (old - (self.last_good_q if self.last_good_q is not None else self.q_ref)) >= 0)

    def _set_target(self, value, pending=False):
        old = self._target
        self._target = value
        if pending:
            self._begin_move(self.move_pending and old is not None and
                             (value - old) * (old - (self.last_good_q if self.last_good_q is not None else self.q_ref)) >= 0)

    def _begin_move(self, preserve=False):
        self.assist_nm, self.assist_sign = 0.0, 0.0
        self.saw_return_motion = False
        self._ret_since = None
        self._was_stuck = True
        self.progress_q = self.progress_t = None
        origin = self.last_good_q if self.last_good_q is not None else self.q_ref
        if preserve:
            self.arrival_stable_since = None
            self.breakaway_direction = 0.0
            return
        self.move_pending = True
        self.arrived_latched = False
        self.arrival_wait_since = None
        self.arrival_stable_since = None
        self.breakaway_direction = 0.0
        self.return_origin_q = origin
        self.return_lag_since = None
        self.return_lag_observed = False
        self.return_assisted = False
        self.arrival_assist_done = False
        self.lift_peak = 0.0

    def _cancel_movement(self, q=None):
        self.assist_nm, self.assist_sign = 0.0, 0.0
        self.progress_q = self.progress_t = None
        if q is None:
            q = self.last_good_q if self.last_good_q is not None else self.q_ref
        self.q_ref = q
        self.v_ref = 0.0
        if hasattr(self, 'a_ref'):
            self.a_ref = 0.0
        self._set_target(self._clip(q), pending=False)
        self.move_pending = False
        self.arrived_latched = False
        self.arrival_wait_since = None
        self.arrival_stable_since = None
        self.breakaway_direction = 0.0
        self.return_origin_q = None
        self.return_lag_since = None
        self.return_lag_observed = False
        self.return_assisted = False
        self.arrival_assist_done = False
        self.lift_peak = 0.0
        self.probe = None

    # ---- helpers --------------------------------------------------------
    def _emit(self, t, kind, msg):
        self.events.append(Event(t, kind, msg))

    def drain_events(self):
        ev, self.events = self.events, []
        return ev

    def _valid(self, state, stamp, now):
        if state is None or stamp is None:
            return False, 'sem amostra de estado'
        if not all(isfinite(x) for x in (state.q, state.dq, stamp)):
            return False, 'valor não finito'
        if not (0 <= now - stamp <= self.p.state_max_age):
            return False, 'estado velho (idade=%.3fs)' % (now - stamp)
        if state.mode != 1 or state.motorstate != 0:
            return False, 'mode=%r motorstate=%r' % (state.mode, state.motorstate)
        return True, None

    def _clip(self, q):
        return max(self.p.q_lo, min(self.p.q_hi, q))

    def _stop_point(self):
        v = self.v_ref
        return self.q_ref + (v * abs(v)) / (2.0 * self.p.a_max)

    def _anchor_hold(self, q, now, why):
        self._cancel_movement(q)
        self.track_bad_since = None
        self.fault_times = [t for t in self.fault_times if now - t <= self.p.lockout_window]
        self.fault_times.append(now)
        self._emit(now, 'fault', 'HOLD na posição medida (q=%+.4f): %s' % (q, why))
        if len(self.fault_times) >= self.p.lockout_faults and not self.locked_out:
            self.locked_out = True
            self._emit(now, 'lockout', '%d falhas em %.0fs: movimento bloqueado, segurando. '
                       'Aperte r para liberar.' % (len(self.fault_times), self.p.lockout_window))

    # ---- lifecycle ------------------------------------------------------
    def arm(self, now, state, stamp):
        if stamp is not None:
            now = max(now, stamp)
        ok, why = self._valid(state, stamp, now)
        if not ok:
            raise RuntimeError('recusado ao armar: %s' % why)
        if not (VENDOR_Q_MIN - 0.01 <= state.q <= VENDOR_Q_MAX):
            raise RuntimeError('q medido %.4f fora da faixa do fabricante' % state.q)
        self.q_ref, self.v_ref = float(state.q), 0.0
        self._set_target(self._clip(state.q), pending=False)
        self.move_pending = False
        self.arrived_latched = True
        self.arrival_wait_since = None
        self.last_t = self.armed_at = now
        self.last_good_q = float(state.q)
        self.armed = True
        self._emit(now, 'armed', 'ARMADO segurando q=%+.4f (gravidade compensada)' % state.q)
        if not (self.p.q_lo <= state.q <= self.p.q_hi):
            self._emit(now, 'warn', 'q=%+.4f fora da faixa de operação [%+.3f, %+.3f]; '
                       'só aceita mover de volta para dentro' % (state.q, self.p.q_lo, self.p.q_hi))
        return self.q_ref

    # ---- operator commands -----------------------------------------------
    def jog(self, direction, now, temperature=None):
        if not self.armed or self.released or self.homing_then_release:
            return False, 'indisponível'
        if self.locked_out:
            return False, 'bloqueado (aperte r)'
        if direction not in (1, -1):
            return False, 'direção inválida'
        hot = temperature is not None and max(temperature) >= self.p.temp_warn
        away = (direction > 0) == (self.target >= self.p.home_q)
        if hot and away:
            return False, 'temperatura alta: só aceita ir em direção ao neutro'
        base = self.target
        new = self._clip(base + direction * self.p.step)
        if direction > 0:
            reach = self.q_verified + self.p.probe_margin
            if new > reach:
                new = max(base, reach)
            if new <= base + 1e-9:
                return False, ('ainda não provei que consigo VOLTAR de mais longe (verificado até '
                               '%+.3f). Aperte s para ela voltar um pouco daqui; se voltar, '
                               'libera mais %.2f rad.' % (self.q_verified, self.p.probe_margin))
        if new == base:
            return False, 'limite da faixa (%+.3f..%+.3f)' % (self.p.q_lo, self.p.q_hi)
        if self.probe is not None:
            self._cancel_movement()
        self._set_target(new, pending=True)
        return True, None

    def stop(self, now):
        if self.armed and not self.released:
            self._cancel_movement()
            self.homing_then_release = False
            self._emit(now, 'stop', 'PARAR: segurando em q medido~%+.4f' % self.q_ref)

    def home(self, now):
        if self.armed and not self.released:
            self.probe = None
            self._set_target(self.p.home_q, pending=True)
            self._begin_move()  # explicit home starts a new bounded move
            self._emit(now, 'home', 'Voltando ao neutro q=%+.3f' % self.p.home_q)

    def reset_lockout(self, now):
        if self.locked_out:
            self.locked_out = False
            self.fault_times = []
            self._emit(now, 'reset', 'Bloqueio liberado.')

    def request_release(self, now, q_meas):
        """Stop publishing only where friction alone holds the joint."""
        if self.released:
            return True, None
        if self.move_pending or self.probe is not None:
            return False, 'movimento pendente: pare antes de soltar'
        if q_meas is None or not friction_holds(q_meas) or abs(self.v_ref) > 1e-3:
            return False, ('q=%+.3f: aqui a gravidade venceria o atrito e ela cairia. '
                           'Aperte h para voltar ao neutro antes de soltar.' % (q_meas if q_meas is not None else float('nan')))
        self._release(now, 'operador')
        return True, None

    def start_probe(self, level, now, q_meas):
        """One lifting-torque probe step at `level` N.m (total commanded)."""
        if not self.armed or self.released or self.homing_then_release:
            return False, 'indisponível'
        if self.locked_out:
            return False, 'bloqueado (aperte r)'
        if self.probe is not None:
            return False, 'teste já em andamento'
        if self.move_pending or abs(self.v_ref) > 1e-4 or abs(self.target - self.q_ref) > 1e-4:
            return False, 'pare primeiro (espaço) e espere ela ficar SEGURANDO'
        if q_meas is None or q_meas < PROBE_MIN_Q:
            return False, 'teste só faz sentido inclinado (q >= %.1f rad)' % PROBE_MIN_Q
        if not (PROBE_MIN_LEVEL_NM <= level <= PROBE_HARD_MAX_NM):
            return False, 'nível fora de [%.0f, %.0f] N.m' % (PROBE_MIN_LEVEL_NM, PROBE_HARD_MAX_NM)
        ff = self.feedforward(q_meas)            # hold torque at v_ref = 0 (negative)
        delta = (level + ff) / KP_HARD_CEILING   # kp*delta - ff_hold_magnitude = level
        if delta <= 0:
            return False, 'segurar aqui já exige mais que %.0f N.m' % level
        self.probe = {'level': level, 't0': now, 'q0': q_meas, 'q_hold': self.q_ref,
                      'q_ref': q_meas - delta}
        self._emit(now, 'probe', 'TESTE: tentando levantar com %.0f N.m por %.1f s (q=%+.3f)...'
                   % (level, PROBE_HOLD_S, q_meas))
        return True, None

    def _probe_tick(self, now, q, dq, temperature):
        pr = self.probe
        moved = q < pr['q0'] - 0.01 or dq < -0.05
        hot = temperature is not None and max(temperature) >= PROBE_TEMP_ABORT
        if moved:
            self._cancel_movement(q)
            self.probe = None
            self.probe_results.append((pr['level'], True, pr['q0']))
            self._emit(now, 'probe', 'DESCOLOU com comando de ~%.0f N.m em q=%+.3f -> '
                       'segurando em q=%+.3f.' % (pr['level'], pr['q0'], q))
        elif hot or now - pr['t0'] >= PROBE_HOLD_S:
            self._cancel_movement(pr['q_hold'])
            self.probe = None
            self.probe_results.append((pr['level'], False, pr['q0']))
            self._emit(now, 'probe', ('ABORTADO por temperatura %s' % list(temperature)) if hot else
                       'não descolou com %.0f N.m (q=%+.3f). Aperte p para o próximo nível.'
                       % (pr['level'], pr['q0']))

    def safe_shutdown(self, now):
        """Ctrl-C / terminal lost / timeouts: home, then release. Never free-fall."""
        if self.armed and not self.released:
            self.homing_then_release = True
            self.locked_out = False
            self._set_target(self.p.home_q, pending=True)
            self._begin_move()  # explicit home starts a new bounded move
            self.probe = None
            self._emit(now, 'shutdown', 'Encerrando com segurança: voltando ao neutro e depois soltando.')

    def force_release(self, now, reason):
        """Only for a concurrent publisher (we must not fight another writer)."""
        self._release(now, reason)

    def _release(self, now, reason):
        self._cancel_movement()
        self.released = True
        self.release_reason = reason
        self._emit(now, 'released', 'Parou de publicar (motivo=%s).' % reason)

    # ---- control tick ------------------------------------------------------
    def tick(self, now, state, stamp, temperature=None):
        """Returns MotorCmd_ to publish, or None once released."""
        if not self.armed or self.released:
            return None
        if stamp is not None:
            now = max(now, stamp)
        dt = max(0.0, min(now - self.last_t, 0.05))
        self.last_t = now
        p = self.p

        ok, why = self._valid(state, stamp, now)
        if ok:
            q, dq = float(state.q), float(state.dq)
            self.last_good_q = q
            self._dq_meas = dq
        else:
            # Feedback unusable: freeze the reference. The motor's own encoder
            # loop still holds q_ref, which is far safer than going silent.
            if self.move_pending or self.v_ref != 0.0 or self.probe is not None:
                self._anchor_hold(self.last_good_q, now, 'feedback inválido: %s' % why)
            q, dq = self.last_good_q, 0.0
            return self._cmd(q)

        if self.probe is not None:
            if abs(dq) > p.dq_fault:
                self._anchor_hold(q, now, 'velocidade %.3f no teste de torque' % abs(dq))
                return self._cmd(q)
            self._probe_tick(now, q, dq, temperature)
            if self.probe is not None:
                return self._cmd(q)

        if now - self.armed_at > p.session_max and not self.homing_then_release:
            self.safe_shutdown(now)
        if temperature is not None and max(temperature) >= p.temp_home and not self.homing_then_release:
            self._emit(now, 'thermal', 'Temperatura %s >= %.0f C.' % (list(temperature), p.temp_home))
            self.safe_shutdown(now)

        self._update_breakaway(q, dq, dt, now)
        kp_now, kd_now = self.gains(q)
        tau_total = (self.feedforward(q) + kp_now * (self.q_ref - q)
                     + kd_now * (self.v_ref - dq))
        saturated = abs(tau_total) > p.sat_nm and abs(dq) < 0.01
        if saturated:
            if self.sat_since is None:
                self.sat_since = now
        else:
            self.sat_since = None

        # Proven return: moving back under OUR torque -> the joint lags the
        # reference (q_ref < q). Pushed by a hand it runs AHEAD (q_ref > q),
        # as in jog3.jsonl; that must never extend the envelope.
        lag = q - self.q_ref
        # Upper bound = tracking-fault threshold: jog4.jsonl lifted itself back
        # from 0.66 rad with 0.035-0.040 lag, which a 0.03 bound wrongly rejected.
        # Budget is judged on the PEAK torque of this return (the breakaway),
        # not the lower torque once it is already sliding.
        motion_direction = self.v_ref if abs(self.v_ref) > 1e-9 else self.breakaway_direction
        if self.move_pending:
            # Keep the complete episode peak through zero-speed settling.
            self.lift_peak = max(self.lift_peak, abs(tau_total))
            # external help = joint moving back AHEAD of our reference (not the
            # initial reference offset left by a reversed forward move)
            # With the lead governor our own return tracks q_ref within a few
            # mrad (explore3 replay); a hand pushing it back drives it clearly
            # AHEAD of the reference against kp.
            # tuck1.jsonl: a return starting with the joint short of a forward
            # target already 'leads' q_ref by ~9 mrad, and tucked-arm CT returns
            # slip ahead of q_ref, so the lead test flagged OUR OWN returns as
            # pushes (commanded -50..-74 N.m vs gravity -38). Physics instead:
            # the joint can only rise by itself if our lifting torque exceeds
            # gravity; moving back while we command less than ~80% of the
            # gravity hold torque means somebody else lifted it.
            weak = tau_total > PUSH_GRAVITY_FRAC * self._gravity_now(q)
            # or: driven back faster than we ever drive it, clearly ahead of q_ref
            driven = dq < -self.p.v_fast_frac * self.p.v_max and lag < -ASSIST_LEAD_RAD
            if self.target < self.return_origin_q and dq < -0.01 and lag < -self.p.lead_max:
                # further ahead of q_ref than our own lead governor ever lets
                # it be pulled: not our torque (conservative, no dwell)
                self.return_assisted = True
            if self.target < self.return_origin_q and dq < -0.01 and (weak or driven):
                if self._push_since is None:
                    self._push_since = now
                elif now - self._push_since >= self.p.arrival_dwell:
                    self.return_assisted = True
            else:
                self._push_since = None
        if self.move_pending and motion_direction < 0 and dq < -0.01 and \
                -ASSIST_LEAD_RAD / 2 < lag < self.p.track_fault:
            if self.return_lag_since is None:
                self.return_lag_since = now
            elif now - self.return_lag_since >= self.p.arrival_dwell:
                self.return_lag_observed = True
        else:
            self.return_lag_since = None

        if self.move_pending and dq < -0.01:
            self.saw_return_motion = True
        self._verify_return_progress(q, now)

        if abs(dq) > p.dq_fault:
            self._anchor_hold(q, now, 'velocidade medida %.3f > %.3f rad/s' % (abs(dq), p.dq_fault))
            return self._cmd(q)
        elif self.sat_since is not None and now - self.sat_since > p.sat_time:
            self.saturated_at_q = q
            self.sat_since = None
            self._anchor_hold(q, now, 'LIMITE DE COMANDO (%.0f N.m) sem movimento medido. '
                              'Segurando; torque físico não aferido. Não empurre a junta energizada.'
                              % abs(tau_total))
            return self._cmd(q)
        elif abs(self.q_ref - q) > p.track_fault:
            if self.track_bad_since is None:
                self.track_bad_since = now
            elif now - self.track_bad_since > p.track_fault_time:
                self._anchor_hold(q, now, 'erro de rastreamento %.3f rad' % abs(self.q_ref - q))
                return self._cmd(q)
        else:
            self.track_bad_since = None

        if (self.move_pending and self.progress_t is not None and
                now - self.progress_t > p.stall_timeout):
            self._emit(now, 'arrival_failed',
                       'não descolou em %.1fs em q=%+.4f; segurando' %
                       (p.stall_timeout, q))
            self._anchor_hold(q, now, 'sem progresso em %.1fs' % p.stall_timeout)
            return self._cmd(q)

        if self.sat_since is None and self.move_pending:
            self._advance_reference(dt)   # saturated: freeze, don't wind up error
            self._govern_reference(q)

        # A timed reference reaching its target is not measured arrival.  Keep
        # a bounded directional breakaway request while the joint is still
        # outside the arrival band; if it remains blocked, re-anchor HOLD at
        # the measured position instead of declaring success with v_ref=0.
        planned_done = (self.move_pending and abs(self.target - self.q_ref) < 1e-4 and
                        abs(self.v_ref) < p.a_max * max(dt, 1e-3) * 1.5)
        measured_arrived = abs(self.target - q) <= p.arrival_tolerance
        if self.move_pending and self.return_origin_q is not None:
            # A small overshoot in the commanded direction (gravity carrying it
            # a few mrad past a forward target) is arrival, not a fault: the
            # move completes by holding where the joint actually stopped.
            move_dir = 1.0 if self.target > self.return_origin_q else -1.0
            over = move_dir * (q - self.target)
            if 0.0 < over <= p.track_fault:
                measured_arrived = True
        stable_arrived = measured_arrived and abs(dq) <= p.arrival_velocity
        if planned_done and self.arrival_wait_since is None:
            self.arrival_wait_since = now
        self.breakaway_direction = 0.0
        if self.move_pending:
            if planned_done and stable_arrived:
                self.arrival_assist_done = True
                if self.arrival_stable_since is None:
                    self.arrival_stable_since = now
                elif now - self.arrival_stable_since >= p.arrival_dwell:
                    self._complete_move(now)
            else:
                self.arrival_stable_since = None
                if planned_done and measured_arrived:
                    self.arrival_assist_done = True
                direction = 1.0 if self.target > self.return_origin_q else -1.0
                # One bounded terminal assist, only in the requested direction.
                # Once the arrival band is touched, never rearm from noise.
                if (planned_done and not self.arrival_assist_done and
                        direction * (self.target - q) > p.arrival_tolerance):
                    self.breakaway_direction = direction
            if (self.move_pending and self.arrival_wait_since is not None and
                    now - self.arrival_wait_since >= p.arrival_timeout):
                self._emit(now, 'arrival_failed',
                           'alvo q=%+.4f não alcançado; segurando q medido=%+.4f' %
                           (self.target, q))
                self._anchor_hold(q, now, 'alvo não alcançado em %.2fs' % p.arrival_timeout)
                return self._cmd(q)

        if self.homing_then_release and not self.move_pending and self.arrived_latched and \
                abs(q - p.home_q) <= p.arrival_tolerance and abs(self.q_ref - p.home_q) < 1e-4 and \
                self.v_ref == 0.0 and friction_holds(q) and abs(dq) < p.arrival_velocity:
            self._release(now, 'neutro alcançado')
            return None
        return self._cmd(q)

    def _advance_reference(self, dt):
        p = self.p
        err = self.target - self.q_ref
        v_des = (1 if err > 0 else -1) * min(p.v_max, sqrt(2.0 * p.a_max * abs(err)))
        dv = max(-p.a_max * dt, min(p.a_max * dt, v_des - self.v_ref))
        self.v_ref += dv
        self.q_ref += self.v_ref * dt
        if abs(self.target - self.q_ref) < 1e-4 and abs(self.v_ref) < p.a_max * max(dt, 1e-3) * 1.5:
            self.q_ref, self.v_ref = self.target, 0.0

    def _verify_return_progress(self, q, now):
        """explore4.jsonl: it lifted itself back 1.433 -> 1.298 rad with
        <= 114 N.m, but every move ended in a stall-HOLD before measured
        arrival, so the proof (only checked at completion) was never granted.
        The proof is the motion itself: >= 0.01 rad back from the origin under
        our own torque, never pushed ahead of q_ref, within the budget."""
        origin = self.return_origin_q
        if (not self.move_pending or origin is None or self.return_assisted
                or self.target >= origin or origin <= self.q_verified):
            return
        if origin - q < 0.01:
            self._ret_since = None
            return
        if getattr(self, '_ret_since', None) is None:
            self._ret_since = now          # must STAY back (not a 1-sample glitch)
            return
        if now - self._ret_since >= 2 * self.p.arrival_dwell and self.saw_return_motion:
            if self.lift_peak <= self.p.lift_budget:
                self.q_verified = origin
                self._emit(now, 'verified', 'volta comprovada a partir de q=%+.3f' % origin)
            elif not self.frontier_reported:
                self.frontier_reported = True
                self._emit(now, 'frontier', 'FRONTEIRA: retorno exigiu %.0f N.m; não ampliando o envelope.'
                           % self.lift_peak)

    def _gravity_now(self, q):
        """Gravity hold torque at q (negative = holds against falling forward)."""
        return self.p.gravity_comp * g_hold(q)

    def _govern_reference(self, q):
        """The reference may lead the MEASURED joint by at most lead_max in
        the move direction: no stored kp torque to snap with, no tracking
        fault from a reference that ran away from a stuck joint."""
        if self.return_origin_q is None:
            return
        d = 1.0 if self.target > self.return_origin_q else -1.0
        lead = d * (self.q_ref - q)
        if lead > self.p.lead_max:
            self.q_ref = q + d * self.p.lead_max

    def _complete_move(self, now):
        self.assist_nm, self.assist_sign = 0.0, 0.0
        self.progress_q = self.progress_t = None
        origin = self.return_origin_q
        # Proven: it came back >= 1 cm-rad from the origin under OUR torque
        # (never driven ahead of q_ref by a push). The old sustained-lag
        # requirement could not be met once the lead governor made our own
        # returns track the reference within a few mrad, nor by short slips.
        if (not self.return_assisted and origin is not None
                and origin > self.q_verified and origin - self.last_good_q >= 0.01):
            if self.lift_peak <= self.p.lift_budget:
                self.q_verified = origin
                self._emit(now, 'verified', 'volta comprovada a partir de q=%+.3f' % origin)
            elif not self.frontier_reported:
                self.frontier_reported = True
                self._emit(now, 'frontier', 'FRONTEIRA: retorno exigiu %.0f N.m; não ampliando o envelope.'
                           % self.lift_peak)
        q_end = self.last_good_q if self.last_good_q is not None else self.target
        if abs(q_end - self.target) > self.p.arrival_tolerance:
            self._set_target(self._clip(q_end), pending=False)   # overshoot: hold where it stopped
        self.q_ref, self.v_ref = self.target, 0.0
        self.move_pending = False
        self.arrived_latched = True
        self.arrival_wait_since = None
        self.arrival_stable_since = None
        self.breakaway_direction = 0.0
        self.return_origin_q = None
        self.return_lag_since = None
        self.return_lag_observed = False
        self.return_assisted = False
        self.arrival_assist_done = False
        self.lift_peak = 0.0

    def gains(self, q):
        """PD gains sent to the motor driver (overridden by computed torque)."""
        if self.probe is not None:
            return KP_HARD_CEILING, self.p.kd
        return self.p.kp, self.p.kd

    def _update_breakaway(self, q, dq, dt, now=None):
        """Bounded velocity regulator for the extra (unmodelled) friction.

        Grows while the joint is stuck/slower than v_slow, HOLDS while it
        moves (dropping it the instant it slides is what made explore3.jsonl
        stick-slip), decays only when it moves faster than v_fast. Never
        exceeds breakaway_max nor lets the total command pass the hard cap."""
        p = self.p
        if not self.move_pending or self.probe is not None or self.return_origin_q is None:
            self.assist_nm, self.assist_sign = 0.0, 0.0
            return
        direction = 1.0 if self.target > self.return_origin_q else -1.0
        remaining = direction * (self.target - q)
        if remaining <= p.arrival_tolerance:
            self.assist_nm, self.assist_sign = 0.0, 0.0   # arrived (or past it)
            return
        self.assist_sign = direction
        speed = direction * dq
        if now is not None:
            if self.progress_q is None or direction * (q - self.progress_q) > 0.003:
                self.progress_q, self.progress_t = q, now
        stuck_now = abs(dq) < 0.01
        was_stuck = getattr(self, '_was_stuck', True)
        self._was_stuck = stuck_now
        v_target = max(abs(self.v_ref), p.v_slow)
        if speed > v_target:
            # faster than the reference: decay, harder the faster it goes
            excess = min(3.0, (speed - v_target) / v_target)
            self.assist_nm = max(0.0, self.assist_nm - p.breakaway_decay * excess * dt)
        elif speed < p.v_slow:
            headroom = TAU_HARD_CEILING_NM - abs(self._base_total(q, dq))
            step = p.breakaway_rate * dt
            self.assist_nm = max(0.0, min(p.breakaway_max, self.assist_nm + min(step, headroom)))

    def _base_total(self, q, dq):
        """Commanded total (FF incl. current assist + PD) at this state."""
        kp, kd = self.gains(q)
        return self.feedforward(q) + kp * (self.q_ref - q) + kd * (self.v_ref - dq)

    def _assist_torque(self):
        # +q = forward (gravity side); lifting back (q decreasing) needs -tau.
        return getattr(self, 'assist_sign', 0.0) * getattr(self, 'assist_nm', 0.0)

    def _friction_signal(self):
        # Assist changes friction only, never the profiled velocity/CT acceleration.
        return self.v_ref if abs(self.v_ref) > 1e-9 else self.breakaway_direction * 10.0 * self.p.v_eps

    def feedforward(self, q):
        p = self.p
        tau = p.gravity_comp * g_hold(q)
        # Friction comp follows the planned direction of motion, at full
        # strength from the first reference step: otherwise kp has to store
        # the missing torque as position error and releases it as a snap.
        # < 100% of kinetic friction, so it can never drive the joint alone.
        motion_direction = self._friction_signal()
        F = friction_nm(q, lifting=motion_direction < 0)
        if motion_direction < 0 and abs(self._dq_meas) < 0.01:
            # still stuck while starting to lift: compensate the (much larger)
            # breakaway friction. When it starts sliding this static part is
            # handed over to the regulated assist (no step drop -> no re-stick).
            F = max(F, lift_breakaway_friction(abs(g_hold(q))))
        tau += p.friction_comp * F * tanh(motion_direction / p.v_eps)
        tau += self._assist_torque()
        return max(-TAU_FEEDFORWARD_CEILING_NM, min(TAU_FEEDFORWARD_CEILING_NM, tau))

    def _cmd(self, q_for_ff):
        kp, kd = self.gains(q_for_ff)
        q_cmd = self.probe['q_ref'] if self.probe is not None else self.q_ref
        dq_cmd = self.v_ref
        tau_ff = self.feedforward(q_for_ff)
        pd = kp * (q_cmd - q_for_ff) + kd * (dq_cmd - self._dq_meas)
        total = tau_ff + pd
        if abs(total) > TAU_HARD_CEILING_NM and abs(pd) > 1e-12:
            desired_pd = (TAU_HARD_CEILING_NM if total > 0 else -TAU_HARD_CEILING_NM) - tau_ff
            scale = desired_pd / pd
            q_cmd = q_for_ff + scale * (q_cmd - q_for_ff)
            dq_cmd = self._dq_meas + scale * (dq_cmd - self._dq_meas)
        return MotorCmd_(mode=1, q=q_cmd, dq=dq_cmd,
                         tau=tau_ff, kp=kp, kd=kd)
