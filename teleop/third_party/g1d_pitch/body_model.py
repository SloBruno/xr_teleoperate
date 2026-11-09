"""Whole-body rigid model of the G1-D for waist-pitch computed-torque work.

Pinocchio model built from g1_d.urdf with the base fixed (stand / AGV).
The column (LZ_*), wheels and fingers are locked: column height is a
vertical translation and does not change the pitch gravity moment; fingers
are ~0.3 kg per hand and are treated as part of the hand.

Identified correction (from real logs, see pitch_controller.py):
    tau_real = S * tau_URDF + OFFSET,   S = 1.162, OFFSET = 11.73 N.m
S scales every inertial parameter equally (equivalent to ~16% heavier
links); OFFSET is a constant joint torque (encoder zero / cable / spring --
still unexplained, see the identification plan).
"""
import os

import numpy as np
import pinocchio as pin

URDF = os.environ.get('G1D_URDF', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'g1_d.urdf'))
PITCH = 'Yaw_Joint'          # URDF's misleading name for the pitch axis (RY)
YAW = 'torso_Joint'          # torso yaw (RZ), child of pitch
S_ID = 1.162
OFFSET_ID = 11.73

ARM_JOINTS = [f'{s}_{j}' for s in ('left', 'right') for j in (
    'shoulder_pitch_joint', 'shoulder_roll_joint', 'shoulder_yaw_joint',
    'elbow_joint', 'wrist_roll_joint', 'wrist_pitch_joint', 'wrist_yaw_joint')]

# Arm poses (same values both sides; roll mirrored where it matters).
POSES = {
    'zero (bracos para baixo)': {},
    'manipulacao a frente': {'shoulder_pitch_joint': -0.8, 'elbow_joint': 1.2},
    'bracos estendidos a frente': {'shoulder_pitch_joint': -1.57, 'elbow_joint': 0.0},
    'recolhidos (cotovelo dobrado)': {'shoulder_pitch_joint': 0.3, 'elbow_joint': 1.6},
}


def pose_vector(pose):
    return {f'{s}_{k}': v for s in ('left', 'right') for k, v in pose.items()}


class BodyModel:
    def __init__(self, payload_per_hand=0.0, scale=1.0, offset=0.0, urdf=URDF):
        full = pin.buildModelFromUrdf(urdf)
        keep = {PITCH, YAW, *ARM_JOINTS}
        lock = [i for i in range(1, full.njoints) if full.names[i] not in keep]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full))
        if payload_per_hand > 0:
            for side in ('left', 'right'):
                fid = self.model.getFrameId(f'{side}_hand_palm_link')
                fr = self.model.frames[fid]
                jid = fr.parentJoint
                self.model.inertias[jid] = self.model.inertias[jid] + pin.Inertia(
                    payload_per_hand, fr.placement.translation, np.eye(3) * 1e-4)
        self.data = self.model.createData()
        self.scale, self.offset = scale, offset
        j = self.model.getJointId(PITCH)
        self.ip, self.vp = self.model.joints[j].idx_q, self.model.joints[j].idx_v
        self.arm_idx = {n: (self.model.joints[self.model.getJointId(n)].idx_q,
                            self.model.joints[self.model.getJointId(n)].idx_v)
                        for n in ARM_JOINTS + [YAW]}

    def config(self, q_pitch, arms=None):
        q = pin.neutral(self.model)
        q[self.ip] = q_pitch
        for n, v in (arms or {}).items():
            q[self.arm_idx[n][0]] = v
        return q

    def vec(self, pitch_val, arms=None):
        v = np.zeros(self.model.nv)
        v[self.vp] = pitch_val
        for n, x in (arms or {}).items():
            v[self.arm_idx[n][1]] = x
        return v

    def pitch_torque(self, q_p, dq_p, ddq_p, arms_q=None, arms_dq=None, arms_ddq=None):
        """Torque the pitch joint must produce for the given whole-body motion
        (inertia + coupling + Coriolis + gravity), real-robot scaled."""
        tau = pin.rnea(self.model, self.data, self.config(q_p, arms_q),
                       self.vec(dq_p, arms_dq), self.vec(ddq_p, arms_ddq))[self.vp]
        return self.scale * tau + self.offset

    def gravity(self, q_p, arms_q=None):
        tau = pin.computeGeneralizedGravity(self.model, self.data, self.config(q_p, arms_q))[self.vp]
        return self.scale * tau + self.offset

    def M_pp(self, q_p, arms_q=None):
        M = pin.crba(self.model, self.data, self.config(q_p, arms_q))
        return self.scale * M[self.vp, self.vp]
