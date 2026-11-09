"""Whole-body state for the computed-torque waist-pitch controller.

Reads the 14 arm joints from rt/lowstate (unitree_hg LowState_), slots
15..28 -- the same G1_29_JointArmIndex mapping the G1-D teleop
(G1D_ArmController) already uses: per side shoulder pitch/roll/yaw, elbow,
wrist roll/pitch/yaw. Torso yaw is NOT read by default: its lowstate slot on
the G1-D has not been verified (pass yaw_slot only after a passive check).

Pure parsing + filtering lives in ArmStateFilter (testable with fakes);
LowStateReader is the thin DDS wrapper.
"""
import math
import time

ARM_SLOTS = {
    'left_shoulder_pitch_joint': 15, 'left_shoulder_roll_joint': 16,
    'left_shoulder_yaw_joint': 17, 'left_elbow_joint': 18,
    'left_wrist_roll_joint': 19, 'left_wrist_pitch_joint': 20,
    'left_wrist_yaw_joint': 21,
    'right_shoulder_pitch_joint': 22, 'right_shoulder_roll_joint': 23,
    'right_shoulder_yaw_joint': 24, 'right_elbow_joint': 25,
    'right_wrist_roll_joint': 26, 'right_wrist_pitch_joint': 27,
    'right_wrist_yaw_joint': 28,
}
# URDF limits (g1_d.urdf) with a small tolerance, used to reject garbage.
LIMIT_TOL = 0.05


class ArmStateFilter:
    """Turns raw lowstate samples into (q, dq, ddq) dicts with freshness.

    ddq is a first-order-filtered finite difference of the measured dq
    (only used for the small inertial coupling term; it is clipped).
    """

    def __init__(self, limits=None, max_age=0.10, ddq_tau=0.05, ddq_clip=20.0, yaw_slot=None):
        self.limits = limits or {}
        self.max_age, self.ddq_tau, self.ddq_clip = max_age, ddq_tau, ddq_clip
        self.yaw_slot = yaw_slot
        self.q = self.dq = self.ddq = None
        self.stamp = None
        self._prev_dq, self._prev_t = None, None
        self.reject_reason = None

    def ingest(self, motor_states, stamp):
        q, dq = {}, {}
        slots = dict(ARM_SLOTS)
        if self.yaw_slot is not None:
            slots['torso_Joint'] = self.yaw_slot
        for name, i in slots.items():
            try:
                ms = motor_states[i]
                qi, dqi = float(ms.q), float(ms.dq)
            except (IndexError, AttributeError, TypeError, ValueError):
                self.reject_reason = 'slot %d ausente' % i
                return False
            if not (math.isfinite(qi) and math.isfinite(dqi)):
                self.reject_reason = '%s não finito' % name
                return False
            lo, hi = self.limits.get(name, (-math.inf, math.inf))
            if not (lo - LIMIT_TOL <= qi <= hi + LIMIT_TOL):
                self.reject_reason = '%s=%.3f fora do limite do URDF' % (name, qi)
                return False
            q[name], dq[name] = qi, dqi
        ddq = {}
        if self._prev_dq is not None and stamp > self._prev_t:
            dt = stamp - self._prev_t
            a = dt / (self.ddq_tau + dt)
            for n in dq:
                raw = (dq[n] - self._prev_dq[n]) / dt
                prev = (self.ddq or {}).get(n, 0.0)
                ddq[n] = max(-self.ddq_clip, min(self.ddq_clip, prev + a * (raw - prev)))
        else:
            ddq = {n: 0.0 for n in dq}
        self.q, self.dq, self.ddq = q, dq, ddq
        self._prev_dq, self._prev_t = dq, stamp
        self.stamp = stamp
        self.reject_reason = None
        return True

    def fresh(self, now):
        return self.stamp is not None and 0.0 <= now - self.stamp <= self.max_age

    def get(self):
        return self.q, self.dq, self.ddq


class LowStateReader:
    """Subscribes rt/lowstate; never publishes anything."""

    def __init__(self, filt):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        self.filt = filt
        self._sub = ChannelSubscriber('rt/lowstate', LowState_)
        self._sub.Init(self._on, 10)

    def _on(self, msg):
        self.filt.ingest(msg.motor_state, time.monotonic())

    def close(self):
        self._sub.Close()
