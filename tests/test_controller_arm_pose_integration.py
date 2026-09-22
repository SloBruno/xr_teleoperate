import ast
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "teleop" / "teleop_hand_and_arm.py"


class ControllerArmPoseIntegrationTest(unittest.TestCase):
    def test_hand_mode_selects_hand_poses_for_arm_ik(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        wrapper_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TeleVuerWrapper"
        ]
        self.assertEqual(len(wrapper_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in wrapper_calls[0].keywords}
        self.assertEqual(keywords["arm_pose_source"].value, "hand")

    def test_prearm_requires_a_strictly_post_r_hand_sample(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ARM_REQUEST_TIMESTAMP = time.monotonic()", source)
        self.assertIn(
            "ready_tele_data.hand_sample_timestamp > ARM_REQUEST_TIMESTAMP",
            source,
        )
        self.assertIn("hand_sample_is_fresh(ready_tele_data.hand_sample_timestamp)", source)

    def test_stale_hand_pose_holds_measured_arm_position_without_solving_ik(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("if hand_pose_is_fresh:", source)
        self.assertIn("sol_q = current_lr_arm_q.copy()", source)
        self.assertIn("sol_tauff = np.zeros_like(current_lr_arm_q)", source)

    def test_hand_freshness_is_rechecked_before_and_after_ik(self):
        source = SCRIPT.read_text(encoding="utf-8")
        state_read = source.index("current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()")
        ik_call = source.index("arm_ik.solve_ik")
        arm_write = source.index("arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)")
        pre_ik_check = source.index(
            "hand_pose_is_fresh = hand_sample_is_fresh(tele_data.hand_sample_timestamp)",
            state_read,
        )
        post_ik_check = source.index(
            "and hand_sample_is_fresh(tele_data.hand_sample_timestamp)",
            ik_call,
        )
        self.assertLess(state_read, pre_ik_check)
        self.assertLess(pre_ik_check, ik_call)
        self.assertLess(ik_call, post_ik_check)
        self.assertLess(post_ik_check, arm_write)


if __name__ == "__main__":
    unittest.main()
