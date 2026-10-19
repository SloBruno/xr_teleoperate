"""G1D teach/replay DDS adapter: passive state readers + single deferred writer.

Pre-start this object owns ONLY subscribers (``rt/lowstate`` and the enabled
``rt/dex3/<side>/state`` topics), each with its own callback and monotonic
timestamp so one silent Dex3 cannot freeze the other side. ``activate`` is the
one-way construction point for the ``rt/lowcmd`` publisher and the enabled
Dex3 command publishers; it requires a fresh LowState and seeds the complete
LowCmd (all 29 G1D slots) from that measured sample, using exactly the same
non-arm slot configuration as ``G1D_ArmController`` (waist passive kp=kd=0).
Activation writes nothing: the session's first write is already seeded from
measured state. Only the session's ticking thread calls ``write_*``.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from teleop.utils.teach_replay import ARM_DOF, HAND_DOF, HAND_SIDES, ArmState, HandState

LOWSTATE_TOPIC = "rt/lowstate"
LOWCMD_TOPIC = "rt/lowcmd"            # debug mode (same as G1D_ArmController)
DEX3_STATE_TOPIC = "rt/dex3/{side}/state"
DEX3_CMD_TOPIC = "rt/dex3/{side}/cmd"

G1D_MOTOR_COUNT = 29
ARM_SLOTS = tuple(range(15, 29))      # G1_29_JointArmIndex order
WAIST_SLOTS = (12, 13, 14)
WRIST_SLOTS = (19, 20, 21, 26, 27, 28)
WEAK_SLOTS = (4, 10, 15, 16, 17, 18, 22, 23, 24, 25)   # G1_29 _Is_weak_motor
KP_HIGH, KD_HIGH = 300.0, 3.0
KP_LOW, KD_LOW = 80.0, 3.0
KP_WRIST, KD_WRIST = 40.0, 1.5
ACTIVATE_MAX_AGE_S = 0.1


def dex3_mode_byte(joint_id, status=0x01, timeout=0):
    """Same bit layout as Dex3_1_Controller._RIS_Mode._mode_to_uint8."""
    return (int(joint_id) & 0x0F) | ((int(status) & 0x07) << 4) | ((int(timeout) & 0x01) << 7)


def _default_dds():
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_, unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    return dict(publisher_cls=ChannelPublisher, subscriber_cls=ChannelSubscriber,
                lowcmd_factory=unitree_hg_msg_dds__LowCmd_, handcmd_factory=unitree_hg_msg_dds__HandCmd_,
                crc_factory=CRC, lowcmd_type=LowCmd_, lowstate_type=LowState_,
                handcmd_type=HandCmd_, handstate_type=HandState_)


def _protection_snapshot(msg, timestamp):
    try:
        from teleop.utils.dex3_protection import extract_protection_state
        return extract_protection_state(msg, range(HAND_DOF), timestamp)
    except Exception:
        return None


class G1DTeachReplayIO:
    def __init__(self, hand_sides=HAND_SIDES, *, clock=time.monotonic, pre_activate=None, **dds):
        if not dds:
            dds = _default_dds()
        self._dds = dds
        self.clock = clock
        # Called once inside activate(), before any publisher exists (e.g. the
        # debug-mode switch that rt/lowcmd needs). Must raise on failure.
        self._pre_activate = pre_activate
        self.hand_sides = tuple(s for s in HAND_SIDES if s in tuple(hand_sides))
        self._lock = threading.Lock()
        self._arm = None
        self._hands = {}
        self._lowcmd_pub = None
        self._lowcmd_msg = None
        self._crc = None
        self._hand_pub = {}
        self._hand_msg = {}
        self._active = False
        self._released = False
        sub_cls = dds["subscriber_cls"]
        self._low_sub = sub_cls(LOWSTATE_TOPIC, dds["lowstate_type"])
        self._low_sub.Init(self._on_lowstate, 10)
        self._hand_subs = {}
        for side in self.hand_sides:
            sub = sub_cls(DEX3_STATE_TOPIC.format(side=side), dds["handstate_type"])
            sub.Init(lambda msg, side=side: self._on_handstate(side, msg), 10)
            self._hand_subs[side] = sub

    # ------------------------------------------------------------ readers
    def _on_lowstate(self, msg):
        try:
            ts = self.clock()
            q = np.array([float(msg.motor_state[i].q) for i in range(G1D_MOTOR_COUNT)])
            dq = np.array([float(msg.motor_state[i].dq) for i in range(G1D_MOTOR_COUNT)])
            mode_machine = msg.mode_machine
        except Exception:
            return
        with self._lock:
            self._arm = (q, dq, ts, mode_machine)

    def _on_handstate(self, side, msg):
        try:
            ts = self.clock()
            q = np.array([float(msg.motor_state[j].q) for j in range(HAND_DOF)])
            dq = np.array([float(msg.motor_state[j].dq) for j in range(HAND_DOF)])
        except Exception:
            return
        protection = _protection_snapshot(msg, ts)
        with self._lock:
            self._hands[side] = HandState(q=q, dq=dq, timestamp=ts, protection=protection)

    def arm_state(self):
        with self._lock:
            sample = self._arm
        if sample is None:
            return None
        q, dq, ts, _ = sample
        return ArmState(q=q[list(ARM_SLOTS)].copy(), dq=dq[list(ARM_SLOTS)].copy(), timestamp=ts)

    def hand_state(self, side):
        with self._lock:
            return self._hands.get(side)

    # ------------------------------------------------------------ writers
    def activate(self, hand_sides):
        if self._active or self._released:
            raise RuntimeError("teach/replay outputs can be activated only once")
        hand_sides = tuple(s for s in HAND_SIDES if s in tuple(hand_sides))
        if any(s not in self.hand_sides for s in hand_sides):
            raise RuntimeError("cannot activate a hand that has no state subscriber")
        self._check_fresh_lowstate()
        if self._pre_activate is not None:
            self._pre_activate()
        sample = self._check_fresh_lowstate()
        q_all, _, _, mode_machine = sample
        dds = self._dds
        msg = dds["lowcmd_factory"]()
        msg.mode_pr = 0
        msg.mode_machine = mode_machine
        for slot in range(G1D_MOTOR_COUNT):
            cmd = msg.motor_cmd[slot]
            cmd.mode = 1
            cmd.q = float(q_all[slot])
            cmd.dq = 0.0
            cmd.tau = 0.0
            if slot in WAIST_SLOTS:
                cmd.kp, cmd.kd = 0.0, 0.0
            elif slot in ARM_SLOTS:
                cmd.kp, cmd.kd = (KP_WRIST, KD_WRIST) if slot in WRIST_SLOTS else (KP_LOW, KD_LOW)
            elif slot in WEAK_SLOTS:
                cmd.kp, cmd.kd = KP_LOW, KD_LOW
            else:
                cmd.kp, cmd.kd = KP_HIGH, KD_HIGH
        hand_msgs = {}
        for side in hand_sides:
            hmsg = dds["handcmd_factory"]()
            for j in range(HAND_DOF):
                hc = hmsg.motor_cmd[j]
                hc.mode = dex3_mode_byte(j)
                hc.q, hc.dq, hc.tau, hc.kp, hc.kd = 0.0, 0.0, 0.0, 0.0, 0.0
            hand_msgs[side] = hmsg
        pub = dds["publisher_cls"](LOWCMD_TOPIC, dds["lowcmd_type"])
        pub.Init()
        hand_pubs = {}
        for side in hand_sides:
            hp = dds["publisher_cls"](DEX3_CMD_TOPIC.format(side=side), dds["handcmd_type"])
            hp.Init()
            hand_pubs[side] = hp
        self._crc = dds["crc_factory"]()
        self._lowcmd_msg, self._lowcmd_pub = msg, pub
        self._hand_msg, self._hand_pub = hand_msgs, hand_pubs
        self._active = True

    def _check_fresh_lowstate(self):
        with self._lock:
            sample = self._arm
        if sample is None or not (0.0 <= self.clock() - sample[2] <= ACTIVATE_MAX_AGE_S):
            raise RuntimeError("LowState missing/stale; refusing output activation")
        return sample

    @staticmethod
    def _check(name, array, size):
        a = np.asarray(array, dtype=float).reshape(-1)
        if a.shape != (size,) or not np.all(np.isfinite(a)):
            raise ValueError(f"{name}: expected {size} finite values")
        return a

    def write_arm(self, cmd):
        q = self._check("arm q", cmd.q, ARM_DOF)
        dq = self._check("arm dq", cmd.dq, ARM_DOF)
        kp = self._check("arm kp", cmd.kp, ARM_DOF)
        kd = self._check("arm kd", cmd.kd, ARM_DOF)
        tau = self._check("arm tau", cmd.tau, ARM_DOF)
        if not self._active:
            return
        msg = self._lowcmd_msg
        for i, slot in enumerate(ARM_SLOTS):
            m = msg.motor_cmd[slot]
            m.q, m.dq, m.kp, m.kd, m.tau = float(q[i]), float(dq[i]), float(kp[i]), float(kd[i]), float(tau[i])
        msg.crc = self._crc.Crc(msg)
        self._lowcmd_pub.Write(msg)

    def write_hand(self, side, cmd):
        q = self._check(f"{side} hand q", cmd.q, HAND_DOF)
        kp = self._check(f"{side} hand kp", cmd.kp, HAND_DOF)
        kd = self._check(f"{side} hand kd", cmd.kd, HAND_DOF)
        if not self._active:
            return
        if side not in self._hand_pub:
            raise RuntimeError(f"{side} Dex3 output is not active")
        msg = self._hand_msg[side]
        for j in range(HAND_DOF):
            m = msg.motor_cmd[j]
            m.q, m.dq, m.tau, m.kp, m.kd = float(q[j]), 0.0, 0.0, float(kp[j]), float(kd[j])
        self._hand_pub[side].Write(msg)

    def release(self):
        """Stop writing. The firmware keeps the last frame received."""
        self._active = False
        self._released = True
