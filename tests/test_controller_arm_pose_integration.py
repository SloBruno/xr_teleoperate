import ast
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "teleop" / "teleop_hand_and_arm.py"


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
        self.assertIn("if controller_pose_is_fresh:", source)
        self.assertIn("sol_q = current_lr_arm_q.copy()", source)
        self.assertIn("sol_tauff = np.zeros_like(current_lr_arm_q)", source)

    def test_controller_freshness_is_rechecked_before_and_after_ik(self):
        source = SCRIPT.read_text(encoding="utf-8")
        state_read = source.index("current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()")
        ik_call = source.index("arm_ik.solve_ik")
        arm_write = source.index("publish_arm_command_for_telemetry(")
        pre_ik_check = source.index(
            "controller_pose_is_fresh = controller_sample_is_fresh(tele_data.controller_sample_timestamp)",
            state_read,
        )
        post_ik_check = source.index(
            "and controller_sample_is_fresh(tele_data.controller_sample_timestamp)",
            ik_call,
        )
        self.assertLess(state_read, pre_ik_check)
        self.assertLess(pre_ik_check, ik_call)
        self.assertLess(ik_call, post_ik_check)
        self.assertLess(post_ik_check, arm_write)

    def test_telemetry_uses_the_exact_command_sent_after_stale_ik_hold(self):
        source = SCRIPT.read_text(encoding="utf-8")
        hold = source.index("# The sample expired during IK")
        telemetry = source.index("commanded_arm_q=commanded_arm_q", hold)
        self.assertIn("hold_q = arm_ctrl.get_current_dual_arm_q().copy()", source[hold:telemetry])
        self.assertIn("commanded_arm_q_reason=commanded_arm_q_reason", source[telemetry:])

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

if __name__ == "__main__":
    unittest.main()
