"""FK used by the pose compare web must match the teleop IK model and frame.

Needs pinocchio (+ casadi for the IK check); skipped otherwise (validate on the
robot env tv_inspire instead).
"""
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np

try:
    import pinocchio  # noqa: F401
    HAVE_PIN = True
except Exception:  # pragma: no cover
    HAVE_PIN = False

try:
    from pinocchio import casadi as _cpin  # noqa: F401
    import casadi  # noqa: F401
    HAVE_CASADI = True
except Exception:  # pragma: no cover
    HAVE_CASADI = False

REPO = Path(__file__).parents[1]


def _load_ik_module():
    # robot_arm_ik imports meshcat/logging_mp only for visualisation/logging.
    for name in ("meshcat", "meshcat.geometry", "matplotlib", "matplotlib.pyplot"):
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)
    if not hasattr(sys.modules["meshcat"], "geometry"):
        sys.modules["meshcat"].geometry = sys.modules["meshcat.geometry"]
    if "logging_mp" not in sys.modules:
        import logging
        lm = types.ModuleType("logging_mp")
        lm.getLogger = logging.getLogger
        sys.modules["logging_mp"] = lm
    try:
        import pinocchio.visualize  # noqa: F401
    except Exception:
        pass
    from teleop.robot_control import robot_arm_ik
    return robot_arm_ik


@unittest.skipUnless(HAVE_PIN, "pinocchio indisponível neste ambiente; validar no robô (tv_inspire)")
class WristFKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from teleop.utils.arm_fk import G1_29_WristFK
        cls.fk = G1_29_WristFK()

    def test_nq_and_finite(self):
        self.assertEqual(self.fk.nq, 14)
        l, r = self.fk.wrist_xyz(np.zeros(14))
        self.assertTrue(np.all(np.isfinite(l)) and np.all(np.isfinite(r)))
        # zero pose: wrists in front of the waist, left at +y, right at -y, symmetric
        self.assertGreater(l[1], 0.05)
        self.assertLess(r[1], -0.05)
        np.testing.assert_allclose(l * [1, -1, 1], r, atol=1e-3)  # URDF is symmetric to ~10 um

    def test_bad_q(self):
        self.assertEqual(self.fk.wrist_xyz(np.zeros(13)), (None, None))
        q = np.zeros(14)
        q[3] = np.nan
        self.assertEqual(self.fk.wrist_xyz(q), (None, None))

    def test_skeleton_points(self):
        from teleop.utils import arm_fk
        rng = np.random.default_rng(1)
        lo, hi = self.fk.model.lowerPositionLimit, self.fk.model.upperPositionLimit
        for q in (np.zeros(14), np.clip(rng.normal(0, 0.4, 14), lo, hi)):
            sk = self.fk.skeleton(q)
            self.assertEqual(set(sk), {"l", "r", "b"})
            self.assertEqual(len(sk["b"]), len(arm_fk.SKELETON_BASE_FRAMES))
            wl, wr = self.fk.wrist_xyz(q)
            for key, side, w in (("l", "left", wl), ("r", "right", wr)):
                pts = np.array(sk[key])
                self.assertEqual(pts.shape, (len(arm_fk.SKELETON_JOINTS) + 1, 3))
                self.assertTrue(np.all(np.isfinite(pts)))
                # first point = shoulder (pitch joint origin, does not move with arm q)
                jid = self.fk.model.getJointId(f"{side}_shoulder_pitch_joint")
                import pinocchio as pin
                d = self.fk.model.createData()
                pin.forwardKinematics(self.fk.model, d, q)
                np.testing.assert_allclose(pts[0], d.oMi[jid].translation, atol=1e-9)
                self.assertGreater(pts[0][2], 0.2)              # shoulder above the waist
                self.assertGreater(pts[0][1] * (1 if key == "l" else -1), 0.05)
                # last point == wrist point of arm_fk (same frame as IK target)
                np.testing.assert_allclose(pts[-1], w, atol=1e-6)
                # wrist_yaw joint origin is 5 cm from the ee point
                self.assertAlmostEqual(float(np.linalg.norm(pts[-1] - pts[-2])), 0.05, places=6)
        self.assertIsNone(self.fk.skeleton(np.zeros(13)))
        q = np.zeros(14); q[0] = np.inf
        self.assertIsNone(self.fk.skeleton(q))

    def test_pose_web_hub_with_real_fk_sends_skeleton(self):
        sys.path.insert(0, str(REPO / "tools"))
        import pose_compare_web as web
        from teleop.utils import pose_stream as ps
        t = [10.0]
        hub = web.PoseHub(fk=self.fk.wrist_xyz, skeleton=self.fk.skeleton, skeleton_hz=20.0, clock=lambda: t[0])
        q = np.zeros(14)
        for i in range(50):  # 1 s at 50 Hz -> ~20 skeletons
            t[0] += 0.02
            hub.ingest(ps.pack_sample(i, t[0], 1.7e9, True, True, (0.3, 0.2, 0.1), (0.3, -0.2, 0.1), q, q))
        smp = hub.samples_since(0)
        sk = [s["sk"] for s in smp if "sk" in s]
        self.assertTrue(19 <= len(sk) <= 26, len(sk))
        self.assertEqual(len(sk[-1]["l"]), 8)
        np.testing.assert_allclose(sk[-1]["r"][-1], smp[-1]["mr"], atol=1e-4)
        self.assertTrue(hub.status()["skeleton"])

    @unittest.skipUnless(HAVE_CASADI, "pinocchio.casadi indisponível")
    def test_same_model_and_fk_matches_ik_target(self):
        import pinocchio as pin
        mod = _load_ik_module()
        from teleop.utils import arm_fk
        import tempfile
        cwd = os.getcwd()
        tmp = Path(tempfile.mkdtemp())
        (tmp / "assets").symlink_to(REPO / "assets")
        work = tmp / "a" / "b"
        work.mkdir(parents=True)
        os.chdir(work)  # Unit_Test URDF path is ../../assets; no stale cwd cache
        try:
            ik = mod.G1_29_ArmIK(Unit_Test=True, Visualization=False)
        finally:
            os.chdir(cwd)
        self.assertEqual(list(ik.mixed_jointsToLockIDs), arm_fk.LOCKED_JOINTS)
        m_ik = ik.reduced_robot.model
        self.assertEqual(list(m_ik.names), list(self.fk.model.names))
        for name in ("L_ee", "R_ee"):
            a = m_ik.frames[m_ik.getFrameId(name)]
            b = self.fk.model.frames[self.fk.model.getFrameId(name)]
            self.assertEqual(a.parentJoint if hasattr(a, "parentJoint") else a.parent,
                             b.parentJoint if hasattr(b, "parentJoint") else b.parent)
            np.testing.assert_allclose(a.placement.homogeneous, b.placement.homogeneous)

        # Reachable targets: FK of a known q -> IK -> FK(sol_q) ~= target.
        rng = np.random.default_rng(0)
        d_ik = m_ik.createData()
        for _ in range(3):
            q_true = np.clip(rng.normal(0, 0.3, 14), m_ik.lowerPositionLimit, m_ik.upperPositionLimit)
            pin.framesForwardKinematics(m_ik, d_ik, q_true)
            tl = d_ik.oMf[m_ik.getFrameId("L_ee")].homogeneous.copy()
            tr = d_ik.oMf[m_ik.getFrameId("R_ee")].homogeneous.copy()
            sol_q = q_true
            ik.smooth_filter = type(ik.smooth_filter)(np.array([1.0]), 14)  # no averaging for the check
            for _ in range(3):
                sol_q, _tau = ik.solve_ik(tl, tr, sol_q, np.zeros(14))
            l, r = self.fk.wrist_xyz(sol_q)
            self.assertLess(np.linalg.norm(l - tl[:3, 3]), 0.01)
            self.assertLess(np.linalg.norm(r - tr[:3, 3]), 0.01)


if __name__ == "__main__":
    unittest.main()
