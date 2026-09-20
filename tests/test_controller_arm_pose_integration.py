import ast
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "teleop" / "teleop_hand_and_arm.py"


class ControllerArmPoseIntegrationTest(unittest.TestCase):
    def test_dex3_launcher_selects_quest_controller_pose_for_arm_ik(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        wrapper_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TeleVuerWrapper"
        ]
        self.assertEqual(len(wrapper_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in wrapper_calls[0].keywords}
        arm_pose_source = keywords.get("arm_pose_source")
        self.assertIsInstance(arm_pose_source, ast.Constant)
        self.assertEqual(arm_pose_source.value, "controller")
    def test_prearm_waits_for_a_fresh_controller_sample_after_r(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("while not STOP:", source)
        self.assertIn("controller_sample_is_fresh(ready_tele_data.controller_sample_timestamp)", source)

    def test_stale_controller_holds_measured_arm_position_without_solving_ik(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("if controller_is_fresh:", source)
        self.assertIn("sol_q = current_lr_arm_q.copy()", source)
        self.assertIn("sol_tauff = np.zeros_like(current_lr_arm_q)", source)
    def test_prearm_requires_controller_pose_received_after_terminal_r(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ARM_REQUEST_TIMESTAMP = 0.0", source)
        self.assertIn("ARM_REQUEST_TIMESTAMP = time.monotonic()", source)
        self.assertIn(
            "ready_tele_data.controller_sample_timestamp >= ARM_REQUEST_TIMESTAMP",
            source,
        )

    def test_controller_freshness_is_rechecked_after_robot_state_read_and_before_ik(self):
        source = SCRIPT.read_text(encoding="utf-8")
        state_read = source.index("current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()")
        ik_branch = source.index("if controller_is_fresh:", state_read)
        recheck = source.index(
            "controller_is_fresh = controller_sample_is_fresh(tele_data.controller_sample_timestamp)",
            state_read,
        )
        self.assertLess(recheck, ik_branch)


if __name__ == "__main__":
    unittest.main()
