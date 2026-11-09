"""Whole-body physics plant for the G1-D waist pitch (simulation only).

Pitch dynamics from Pinocchio RNEA of the full body (arms follow their own
prescribed trajectories, as their stiff joint controllers would):

    M_pp * ddq_p = tau_motor - b_p(q, v, a_arms) - friction
    b_p = S*RNEA(q, v, [0 at pitch, a_arms])[pitch] + OFFSET
    M_pp = S*M_URDF[pitch, pitch]

Firmware PD at 1 kHz on the latest command, motor torque clipped.

Friction hypotheses (all fitted to real logs):
  'linear'  kinetic F = 16.1 + mu*|g| (mu 0.10 lowering / 0.35 lifting),
            stiction x1.12. Fits jog1/jog4 (<= 0.66 rad) but predicts only
            ~76 N.m to lift from 1.3 rad.
  'jog3'    same, plus lifting stiction >= 1.107*|g| - 2.0, which also
            matches jog3.jsonl (>85 N.m commanded, no motion at 1.3 rad)
            and the 54 N.m breakaway at 0.66 rad. Default: matches ALL data.
"""
from math import copysign


class BodyPlant:
    def __init__(self, body, q0=0.0, arm_traj=None, friction='jog3', coulomb=16.1,
                 mu_lower=0.10, mu_lift=0.35, stiction_ratio=1.12, visc=3.0,
                 tau_limit=80.0, dt=0.001, g_bias=0.0):
        self.body = body
        self.q, self.dq = q0, 0.0
        self.arm_traj = arm_traj or (lambda t: ({}, {}, {}))
        self.friction, self.F0 = friction, coulomb
        self.mu_lower, self.mu_lift, self.sr = mu_lower, mu_lift, stiction_ratio
        self.visc, self.tau_limit, self.dt = visc, tau_limit, dt
        self.g_bias = g_bias
        self.cmd = None
        self.t = 0.0
        self.max_abs_dq = 0.0
        self.tau_m = 0.0
        self.log = []

    def arms(self, t=None):
        return self.arm_traj(self.t if t is None else t)

    def _friction(self, g, lifting):
        mu = self.mu_lift if lifting else self.mu_lower
        Fc = self.F0 + mu * abs(g)
        Fs = self.sr * Fc
        if lifting and self.friction == 'jog3':
            Fs = max(Fs, 1.107 * abs(g) - 2.0)
            Fc = max(Fc, Fs / self.sr)
        return Fc, Fs

    def step(self, duration):
        n = max(1, int(round(duration / self.dt)))
        for _ in range(n):
            aq, adq, addq = self.arms()
            b = self.body.pitch_torque(self.q, self.dq, 0.0, aq, adq, addq) + self.g_bias
            M = self.body.M_pp(self.q, aq) if (int(self.t / self.dt) % 20 == 0 or not hasattr(self, '_M')) else self._M
            self._M = M
            g = self.body.gravity(self.q, aq) + self.g_bias
            if self.cmd is None:
                tau_m = 0.0
            else:
                c = self.cmd
                tau_m = c.kp * (c.q - self.q) + c.kd * (c.dq - self.dq) + c.tau
                tau_m = max(-self.tau_limit, min(self.tau_limit, tau_m))
            self.tau_m = tau_m
            drive = tau_m - b - self.visc * self.dq
            lifting = self.dq < 0 or (abs(self.dq) < 1e-4 and drive < 0)
            Fc, Fs = self._friction(g, lifting)
            if abs(self.dq) < 1e-4:
                if abs(drive) <= Fs:
                    self.dq = 0.0
                    self.t += self.dt
                    continue
                fric = -copysign(Fc, drive)
            else:
                fric = -copysign(Fc, self.dq)
            ddq = (drive + fric) / M
            new_dq = self.dq + ddq * self.dt
            if self.dq != 0.0 and (new_dq > 0) != (self.dq > 0):
                new_dq = 0.0
            self.dq = new_dq
            self.q += self.dq * self.dt
            self.t += self.dt
            self.max_abs_dq = max(self.max_abs_dq, abs(self.dq))

    def state(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from g1d_dds_fake import MotorState_
        return MotorState_(mode=1, q=self.q, dq=self.dq, motorstate=0)
