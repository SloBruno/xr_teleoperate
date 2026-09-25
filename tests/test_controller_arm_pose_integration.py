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
        self.assertIn("publish_arm_command(", helper)

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

    def test_telemetry_correlates_selected_command_with_async_exact_publication(self):
        source = SCRIPT.read_text(encoding="utf-8")
        telemetry = source.index("emit_pose_record_best_effort(", source.index("run_arm_tracking_cycle("))
        self.assertIn("arm_command_request_id=arm_request_id", source[telemetry:])
        self.assertIn("requested_arm_q=cycle.requested_q", source[telemetry:])
        self.assertIn("selected_arm_q=cycle.selected_q", source[telemetry:])
        self.assertIn("commanded_arm_q=None", source[telemetry:])
        self.assertIn("arm_publication_telemetry.emit_cycle(record, arm_ctrl)", source[telemetry:])

    def test_lifecycle_events_cover_requested_accepted_tracking_and_shutdown(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for event in (
            "preparation_ready",
            "start_requested",
            "start_accepted",
            "tracking_started",
            "stop_requested",
            "shutdown_finalization",
        ):
            self.assertIn(event, source)

    def test_shutdown_records_interruption_or_exception_cause_before_finalization(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('shutdown_cause = "shutdown_interrupted"', source)
        self.assertIn('shutdown_cause = "shutdown_exception"', source)
        self.assertLess(source.index('shutdown_cause = "shutdown_interrupted"'), source.index('"shutdown_finalization"'))
        self.assertLess(source.index('shutdown_cause = "shutdown_exception"'), source.index('"shutdown_finalization"'))
        exception_handler = source[source.index("except Exception:"):source.index("finally:")]
        self.assertNotIn("raise", exception_handler)

    def test_keyboard_interrupt_continues_to_finally_cleanup(self):
        source = SCRIPT.read_text(encoding="utf-8")
        interrupt_handler = source[source.index("except KeyboardInterrupt:"):source.index("except Exception:")]
        self.assertNotIn("raise", interrupt_handler)

    def test_lifecycle_shutdown_emits_are_guarded_individually(self):
        source = SCRIPT.read_text(encoding="utf-8")
        finally_block = source[source.index("finally:"):]
        self.assertIn("emit_lifecycle_event_best_effort", source)
        self.assertIn("hand_ctrl.deactivate()", finally_block)
        self.assertIn("arm_ctrl.deactivate()", finally_block)

    def test_control_loop_pose_publication_uses_best_effort_builder_boundary(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("emit_pose_record_best_effort("), 2)
        self.assertNotIn("pose_telemetry_sink.emit(build_pose_record(", source)
        self.assertNotIn("arm_publication_telemetry.emit_cycle(build_pose_record(", source)

    def test_pose_telemetry_close_cannot_skip_actuator_cleanup(self):
        source = SCRIPT.read_text(encoding="utf-8")
        helper_start = source.index("def _close_telemetry_best_effort")
        helper_end = source.index("\ndef on_press", helper_start)
        self.assertIn("except BaseException", source[helper_start:helper_end])
        self.assertLess(source.index("arm_ctrl.ctrl_dual_arm_go_home"), source.index("_close_telemetry_best_effort(pose_telemetry_sink"))

    def test_cleanup_logging_cannot_interrupt_later_cleanup(self):
        source = SCRIPT.read_text(encoding="utf-8")
        finally_block = source[source.index("finally:"):]
        self.assertIn("def _log_best_effort", source)
        self.assertNotIn("logger_mp.error", finally_block)
        self.assertNotIn("logger_mp.warning", finally_block)
        self.assertNotIn("\n        logger_mp.info", finally_block)

if __name__ == "__main__":
    unittest.main()
