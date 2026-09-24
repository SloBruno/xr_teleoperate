import ast
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "teleop" / "teleop_hand_and_arm.py"
IK_SCRIPT = Path(__file__).parents[1] / "teleop" / "robot_control" / "robot_arm_ik.py"


class ControllerArmPoseIntegrationTest(unittest.TestCase):
    def test_launcher_selects_controller_poses_for_arm_ik(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        wrapper_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TeleVuerWrapper"
        ]
        self.assertEqual(len(wrapper_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in wrapper_calls[0].keywords}
        self.assertEqual(keywords["arm_pose_source"].value, "controller")

    def test_prearm_requires_a_strictly_post_r_controller_sample(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ARM_REQUEST_TIMESTAMP = time.monotonic()", source)
        self.assertIn(
            "ready_tele_data.controller_sample_timestamp > ARM_REQUEST_TIMESTAMP",
            source,
        )
        self.assertIn(
            "controller_sample_is_fresh(ready_tele_data.controller_sample_timestamp)",
            source,
        )

    def test_stale_controller_pose_holds_measured_arm_position_without_solving_ik(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("run_arm_tracking_cycle(", source)
        self.assertIn("sample_timestamp=tele_data.controller_sample_timestamp", source)

    def test_controller_freshness_is_rechecked_before_and_after_ik(self):
        source = SCRIPT.read_text(encoding="utf-8")
        helper = (Path(__file__).parents[1] / "teleop/utils/arm_tracking_orchestration.py").read_text()
        state_read = source.index("current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()")
        cycle = source.index("run_arm_tracking_cycle(", state_read)
        self.assertLess(state_read, cycle)
        self.assertEqual(helper.count("controller_sample_is_fresh(sample_timestamp, now)"), 2)
        self.assertIn("publish_if_authorized(", helper)

    def test_g1_29_uses_measured_fk_for_per_start_controller_wrist_calibration(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ControllerWristCalibrator", source)
        self.assertIn("arm_ik.forward_kinematics(measured_lr_arm_q)", source)
        self.assertIn("arm_calibration.reset_for_start_request(ARM_REQUEST_TIMESTAMP)", source)
        self.assertIn("arm_calibration.calibrate(", source)

    def test_arm_ik_is_never_called_until_calibration_has_succeeded(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("calibration = arm_calibration if args.arm == \"G1_29\" else None", source)
        self.assertIn("calibrator=calibration", source)

    def test_invalid_controller_target_holds_measured_q_without_new_ik_or_arm_write(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("target_accepted=target is not None", (Path(__file__).parents[1] / "teleop/utils/arm_tracking_orchestration.py").read_text())
        self.assertIn("calibrator.targets(controller_poses, sample_timestamp", (Path(__file__).parents[1] / "teleop/utils/arm_tracking_orchestration.py").read_text())

    def test_alternate_arm_profiles_do_not_use_controller_wrist_calibration(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('if args.arm == "G1_29":', source)
        self.assertIn('if args.arm == "G1_29" and not arm_calibration.calibrated:', source)
        self.assertIn('if args.arm != "G1_29" and controller_pose_is_fresh:', source)

    def test_prearm_sample_is_carried_into_first_tracking_command(self):
        source = SCRIPT.read_text(encoding="utf-8")
        calibrate = source.index("calibrated = arm_calibration.calibrate(")
        consume = source.index("first_controller_targets = arm_calibration.consume_first_target()", calibrate)
        first_break = source.index("break", consume)
        tracking_loop = source.index("# main loop. robot start to follow VR user's motion")
        first_use = source.index("first_target = first_controller_targets", tracking_loop)
        self.assertLess(calibrate, consume)
        self.assertLess(consume, first_break)
        self.assertLess(first_break, first_use)
        self.assertIn("if first_controller_targets is not None:", source)

    def test_fk_order_is_static_checked_when_pinocchio_is_unavailable(self):
        source = IK_SCRIPT.read_text(encoding="utf-8")
        fk = source[source.index("def forward_kinematics"):source.index("def solve_ik")]
        self.assertLess(fk.index("self.L_hand_id"), fk.index("self.R_hand_id"))
        self.assertIn("framesForwardKinematics", fk)
        urdf = (Path(__file__).parents[1] / "assets" / "g1" / "g1_body29_hand14.urdf").read_text(encoding="utf-8")
        self.assertLess(urdf.index('name="left_wrist_yaw_joint"'), urdf.index('name="right_wrist_yaw_joint"'))

    def test_final_arm_publication_is_serialized_and_stop_holds_measured_q(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("run_arm_tracking_cycle(", source)
        self.assertIn("lifecycle_lock=LIFECYCLE_LOCK", source)
        self.assertIn("is_stopped=lambda: STOP", source)
        self.assertNotIn("arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)", source)

    def test_rejected_target_path_does_not_publish_ik_solution(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("cycle.target_accepted", source)
        self.assertIn("run_arm_tracking_cycle", source)


if __name__ == "__main__":
    unittest.main()
