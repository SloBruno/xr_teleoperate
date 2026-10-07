"""Forward kinematics of the G1_29 wrist frames used by the teleop IK.

Builds the SAME reduced model as ``teleop.robot_control.robot_arm_ik.G1_29_ArmIK``
(``assets/g1/g1_body29_hand14.urdf``, legs/waist/Dex3 finger joints locked at 0,
frames ``L_ee``/``R_ee`` = +0.05 m along x of ``{left,right}_wrist_yaw_joint``),
so FK(q) lives in the same frame as the IK targets (``tele_data.*_wrist_pose``:
robot waist frame, metres). Read-only: no casadi, no solver, no DDS.

q layout: 14 arm joints, left 7 then right 7 (``G1_29_JointArmIndex`` order),
identical to the reduced model's configuration vector.
"""
from __future__ import annotations

import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH = os.path.join(REPO, "assets", "g1", "g1_body29_hand14.urdf")
MODEL_DIR = os.path.join(REPO, "assets", "g1")

# Must stay identical to G1_29_ArmIK.mixed_jointsToLockIDs (checked by tests).
LOCKED_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
EE_OFFSET = np.array([0.05, 0.0, 0.0])
EE_FRAMES = (("L_ee", "left_wrist_yaw_joint"), ("R_ee", "right_wrist_yaw_joint"))
SKELETON_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                   "wrist_roll", "wrist_pitch", "wrist_yaw")
SKELETON_BASE_FRAMES = ("pelvis", "torso_link")
FRAME_DESC = ("G1_29 IK frame: robot waist (pelvis-fixed reduced model, legs/waist locked at 0), "
              "metres, x forward, y left, z up; point = L_ee/R_ee (+0.05 m x of wrist_yaw_joint)")


class G1_29_WristFK:
    def __init__(self, urdf_path: str = URDF_PATH, model_dir: str = MODEL_DIR):
        import pinocchio as pin
        self._pin = pin
        robot = pin.RobotWrapper.BuildFromURDF(urdf_path, model_dir)
        reduced = robot.buildReducedRobot(list_of_joints_to_lock=LOCKED_JOINTS,
                                          reference_configuration=np.zeros(robot.model.nq))
        for name, joint in EE_FRAMES:
            reduced.model.addFrame(pin.Frame(name, reduced.model.getJointId(joint),
                                             pin.SE3(np.eye(3), EE_OFFSET.copy()), pin.FrameType.OP_FRAME))
        self.model = reduced.model
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.l_id = self.model.getFrameId("L_ee")
        self.r_id = self.model.getFrameId("R_ee")
        # Skeleton: joint origins shoulder -> wrist (7 per arm, model order) + L_ee/R_ee.
        self.arm_joint_ids = tuple(
            tuple(self.model.getJointId(f"{side}_{j}_joint") for j in SKELETON_JOINTS)
            for side in ("left", "right"))
        self.base_frame_ids = tuple(self.model.getFrameId(n) for n in SKELETON_BASE_FRAMES)

    def skeleton(self, q):
        """Arm skeleton for display, in the same frame as :meth:`wrist_xyz`.

        Returns ``{"l": [[x,y,z]*8], "r": [...], "b": [pelvis, torso]}`` (lists of
        floats): per arm the origins of shoulder_pitch, shoulder_roll,
        shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw joints and the
        IK end-effector point (L_ee/R_ee). ``None`` for a bad q.
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.nq or not np.all(np.isfinite(q)):
            return None
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        out = {}
        for key, jids, ee in (("l", self.arm_joint_ids[0], self.l_id), ("r", self.arm_joint_ids[1], self.r_id)):
            pts = [self.data.oMi[j].translation for j in jids] + [self.data.oMf[ee].translation]
            out[key] = [[float(v) for v in p] for p in pts]
        out["b"] = [[float(v) for v in self.data.oMf[f].translation] for f in self.base_frame_ids]
        return out

    def wrist_xyz(self, q):
        """Return (left_xyz, right_xyz) as numpy (3,), or (None, None) for a bad q."""
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.nq or not np.all(np.isfinite(q)):
            return None, None
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return (self.data.oMf[self.l_id].translation.copy(),
                self.data.oMf[self.r_id].translation.copy())
