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
