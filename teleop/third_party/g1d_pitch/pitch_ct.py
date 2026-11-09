"""Computed-torque waist-pitch controller (whole-body URDF, Pinocchio).

Same supervisor / envelope / safety as PitchController; the torque law and
the forward-motion admission change:

  tau_ff = S*RNEA(q_body, [v_ref, dq_arms], [a_ref, ddq_arms])[pitch] + OFFSET
           + c_f * F(g_body, sentido) * tanh(v_ref / v_eps)
  kp = M_pp(q_body) * wn^2,   kd = 2*zeta*wn*M_pp(q_body)   (clipped to vendor max)

q_body uses the MEASURED arm joints (rt/lowstate slots 15..28), so gravity and
inertia follow the real arm pose instead of the arms-at-zero table.

Forward motion is admitted only if BOTH hold:
  * proven envelope (as before): target <= q_verified + probe_margin;
  * model envelope: predicted breakaway torque to LIFT BACK from the target
    with the CURRENT arms (gravity + identified lifting friction) <= budget.
If the arms move to a heavier pose while leaned so that the return is no
longer predicted to fit, the controller warns ('BRAÇOS PESADOS').
Arm state stale/invalid -> forward motion refused; torque keeps the last
valid arm pose (holding is still correct for a frozen arm pose).
"""
from math import sqrt, tanh

from g1d_dds_fake import MotorCmd_
from pitch_controller import (PitchController, Params, COULOMB_NM, LOAD_FRICTION,
                              LIFT_LOAD_FRICTION, TAU_FEEDFORWARD_CEILING_NM,
                              KP_HARD_CEILING, KD_HARD_CEILING, LIFT_BUDGET_NM)

# Same closed-loop dynamics as kp=300/kd=12 at arms-zero inertia.
M_NOMINAL = 1.754
WN = sqrt(300.0 / M_NOMINAL)          # ~13.1 rad/s
ZETA = 12.0 / (2 * WN * M_NOMINAL)    # ~0.26
STICTION_RATIO = 1.12
# Lifting BREAKAWAY (static) friction, fitted to the measured breakaways:
# sombra1 0.73 rad ~67 N.m, torque1 0.90 rad 75<b<=80, jog3 1.30 rad >85 (no
# motion). Total lift-from-hold = |g| + (1.14|g| + 3.9) = 2.14|g| + 3.9:
# 67.0 / 77.5 / 90.6 N.m (jog4 0.66: predicts 61.9 vs 54 -> conservative).
# Kinetic lifting friction is much lower (identify_ct: ~0.3|g|), i.e. a big
# stiction step under load -- why it sticks, then slides.
from pitch_controller import LIFT_BREAK_SLOPE as JOG3_LIFT_SLOPE, LIFT_BREAK_OFFSET as JOG3_LIFT_OFFSET  # noqa: E402
HEAVY_WARN_NM, HEAVY_CLEAR_NM = 90.0, 82.0


# (F0, mu_lower, mu_lift, basis). basis 'real': mu multiplies the corrected
# gravity (first identification); 'raw': mu multiplies the raw URDF gravity
# (identify_ct.py) -- then per-direction torques equal a_up/a_dn exactly.
DEFAULT_FRICTION = (COULOMB_NM, LOAD_FRICTION, LIFT_LOAD_FRICTION, 'real')


def lift_breakaway_nm(g, friction=DEFAULT_FRICTION, g_basis=None):
    """Predicted torque to start lifting back from a hold with gravity g
    (g_basis = gravity in the friction's basis; defaults to g).
    Kinetic lifting friction x stiction ratio, never below the jog3.jsonl
    calibration (the only measurement at high load: >85 N.m at 1.3 rad)."""
    F0, _, mu_lift = friction[:3]
    gb = g if g_basis is None else g_basis
    Fs = STICTION_RATIO * (F0 + mu_lift * abs(gb))
    Fs = max(Fs, JOG3_LIFT_SLOPE * abs(g) + JOG3_LIFT_OFFSET)
    return abs(g - Fs)


class ComputedTorqueController(PitchController):
    def __init__(self, body, params=None, schedule_gains=True, inertial=True,
                 model_budget=None, friction=DEFAULT_FRICTION):
        super().__init__(params or Params())
        self.body = body
        self.friction = tuple(friction) + (('real',) if len(friction) == 3 else ())
        self.schedule_gains = schedule_gains
        self.inertial = inertial
        self.model_budget = model_budget if model_budget is not None else self.p.lift_budget
        self.arms_q, self.arms_dq, self.arms_ddq = {}, {}, {}
        self.body_fresh = True   # sim/tests that never call set_body_state: arms at zero
        self.a_ref = 0.0
        self.last_g = None
        self._heavy_warned = False

    def set_body_state(self, arms_q, arms_dq=None, arms_ddq=None, fresh=True):
        """Measured arm joints (dicts by URDF joint name). fresh=False keeps
        the last valid pose for torque but blocks forward motion."""
        self.body_fresh = bool(fresh) and arms_q is not None
        if self.body_fresh:
            self.arms_q = arms_q
            self.arms_dq = arms_dq or {}
            self.arms_ddq = arms_ddq or {}
        else:
            self.arms_dq, self.arms_ddq = {}, {}
            if self.armed and not self.released and (self.move_pending or self.probe is not None or self.v_ref):
                # Do not resume a base-controller breakaway with stale arm
                # state.  A later explicit jog may start a fresh lifecycle.
                self._cancel_movement()

    # ---- admission ------------------------------------------------------------
    def _g_basis(self, g):
        if self.friction[3] == 'raw':
            return (g - self.body.offset) / self.body.scale
        return g

    def predicted_lift(self, q):
        g = self.body.gravity(q, self.arms_q)
        return lift_breakaway_nm(g, self.friction, self._g_basis(g))

    def jog(self, direction, now, temperature=None):
        if direction > 0 and self.armed and not self.released:
            if not self.body_fresh:
                return False, 'estado dos braços (rt/lowstate) indisponível: só aceita voltar'
            nxt = min(self.p.q_hi, self.target + self.p.step)
            need = self.predicted_lift(nxt)
            if need > self.model_budget:
                return False, ('com os braços nesta pose, voltar de q=%+.3f exigiria ~%.0f N.m '
                               '(orçamento %.0f). Recolha os braços para ir além.'
                               % (nxt, need, self.model_budget))
        return super().jog(direction, now, temperature)

    # ---- torque law -------------------------------------------------------------
    def _advance_reference(self, dt):
        v0 = self.v_ref
        super()._advance_reference(dt)
        self.a_ref = (self.v_ref - v0) / dt if dt > 0 else 0.0

    def tick(self, now, state, stamp, temperature=None):
        cmd = super().tick(now, state, stamp, temperature)
        if cmd is not None and state is not None:
            need = self.predicted_lift(state.q)
            if need > HEAVY_WARN_NM and not self._heavy_warned:
                self._heavy_warned = True
                self._emit(now, 'heavy', 'BRAÇOS PESADOS: com esta pose, voltar de q=%+.3f exigiria '
                           '~%.0f N.m (motor ~80). Recolha os braços antes de voltar.' % (state.q, need))
            elif need < HEAVY_CLEAR_NM:
                self._heavy_warned = False
        return cmd

    def feedforward(self, q):
        p = self.p
        g = self.body.gravity(q, self.arms_q)
        self.last_g = g
        if self.inertial:
            tau = self.body.pitch_torque(q, self.v_ref, self.a_ref,
                                         self.arms_q, self.arms_dq, self.arms_ddq)
        else:
            tau = g
        tau = p.gravity_comp * tau
        F0, mu_lower, mu_lift = self.friction[:3]
        motion = self._friction_signal()
        mu = mu_lift if motion < 0 else mu_lower
        F = F0 + mu * abs(self._g_basis(g))
        if motion < 0 and abs(self._dq_meas) < 0.01:
            F = max(F, JOG3_LIFT_SLOPE * abs(g) + JOG3_LIFT_OFFSET)
        tau += p.friction_comp * F * tanh(motion / p.v_eps)
        tau += self._assist_torque()
        return max(-TAU_FEEDFORWARD_CEILING_NM, min(TAU_FEEDFORWARD_CEILING_NM, tau))

    def _gravity_now(self, q):
        g = self.last_g if self.last_g is not None else self.body.gravity(q, self.arms_q)
        return self.p.gravity_comp * g

    def gains(self, q):
        if self.probe is not None:
            return super().gains(q)
        if not self.schedule_gains:
            return self.p.kp, self.p.kd
        M = self.body.M_pp(q, self.arms_q)
        return (min(KP_HARD_CEILING, M * WN * WN),
                min(KD_HARD_CEILING, 2 * ZETA * WN * M))

    def _cmd(self, q_for_ff):
        return super()._cmd(q_for_ff)
