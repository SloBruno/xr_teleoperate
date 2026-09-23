import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
ARM = ROOT / "teleop" / "robot_control" / "robot_arm.py"
HAND = ROOT / "teleop" / "robot_control" / "robot_hand_unitree.py"
TELEOP = ROOT / "teleop" / "teleop_hand_and_arm.py"


def class_method(source_path, class_name, method_name):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def starts_named_output(function, output_name):
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "start"
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
        and node.func.value.attr == output_name
        for node in ast.walk(function)
    )


class PrearmActuatorGateTest(unittest.TestCase):
    def test_g1_arm_constructor_never_starts_a_publisher_thread(self):
        self.assertFalse(starts_named_output(class_method(ARM, "G1_29_ArmController", "__init__"), "publish_thread"))
        self.assertTrue(starts_named_output(class_method(ARM, "G1_29_ArmController", "activate"), "publish_thread"))

    def test_dex3_constructor_never_starts_a_command_process(self):
        self.assertFalse(starts_named_output(class_method(HAND, "Dex3_1_Controller", "__init__"), "hand_control_process"))
        self.assertTrue(starts_named_output(class_method(HAND, "Dex3_1_Controller", "activate"), "hand_control_process"))

    def test_launcher_prepares_arms_before_the_prearm_loop(self):
        source = TELEOP.read_text(encoding="utf-8")
        prearm = source.index("        READY = True")
        activate = source.index("arm_ctrl.activate()")
        prepare = source.index("arm_ctrl.ctrl_dual_arm_go_home()")
        self.assertLess(activate, prearm)
        self.assertLess(prepare, prearm)

    def test_launcher_releases_r_only_after_confirmed_preparation_pose(self):
        source = TELEOP.read_text(encoding="utf-8")
        prepare = source[source.index("Match the original launcher behavior"):source.index("# Initialize before the pre-arm loop")]
        self.assertIn("preparation_confirmed = arm_ctrl.ctrl_dual_arm_go_home()", prepare)
        self.assertIn('if args.arm == "G1_29":', prepare)
        self.assertLess(prepare.index('if args.arm == "G1_29":'), prepare.index("arm_ctrl.activate()"))
        self.assertIn("if not preparation_confirmed:", prepare)
        self.assertLess(prepare.index("if not preparation_confirmed:"), prepare.index("PREPARATION_COMPLETE = True"))
        arm_source = ARM.read_text(encoding="utf-8")
        home_start = arm_source.index("    def ctrl_dual_arm_go_home(self, release_motion_authority=False):")
        home = arm_source[home_start:arm_source.index("    def speed_gradual_max", home_start)]
        self.assertIn("if self.motion_mode and release_motion_authority:", home)
        self.assertIn("return True", home)
        self.assertIn("return False", home)
        self.assertIn("np.all(np.abs(current_q) <= tolerance)", home)

        shutdown = source[source.index("    finally:"):]
        self.assertIn("arm_ctrl.ctrl_dual_arm_go_home(release_motion_authority=True)", shutdown)

    def test_startup_inputs_are_serialized_with_launcher_preparation(self):
        source = TELEOP.read_text(encoding="utf-8")
        on_press = source[source.index("def on_press"):source.index("def get_state")]
        self.assertIn("if not PREPARATION_COMPLETE:", on_press)
        self.assertLess(
            on_press.index("if not PREPARATION_COMPLETE:"),
            on_press.index("ARM_REQUEST_TIMESTAMP = time.monotonic()"),
        )
        prepare = source[source.index("Match the original launcher behavior"):source.index("# Initialize before the pre-arm loop")]
        self.assertIn("with LIFECYCLE_LOCK:", prepare)
        self.assertLess(prepare.index("if STOP:"), prepare.index("arm_ctrl.activate()"))
        self.assertLess(prepare.index("arm_ctrl.ctrl_dual_arm_go_home()"), prepare.index("PREPARATION_COMPLETE = True"))

    def test_dex3_activates_only_after_the_post_r_controller_gate(self):
        source = TELEOP.read_text(encoding="utf-8")
        gate = source.index("ready_tele_data.controller_sample_timestamp > ARM_REQUEST_TIMESTAMP")
        hand_activate = source.index("hand_ctrl.activate()")
        self.assertLess(gate, hand_activate)
        dex3_gate = source[gate:hand_activate]
        self.assertIn("controller_sample_is_fresh", dex3_gate)
        self.assertNotIn("hand_sample_timestamp", dex3_gate)

    def test_r_prepares_arms_before_tracking(self):
        source = TELEOP.read_text(encoding="utf-8")
        activate = source.index("arm_ctrl.activate()")
        prepare = source.index("arm_ctrl.ctrl_dual_arm_go_home()")
        tracking = source.index("start Tracking")
        self.assertLess(activate, prepare)
        self.assertLess(prepare, tracking)

    def test_q_returns_arms_to_prepared_pose_before_deactivation(self):
        source = TELEOP.read_text(encoding="utf-8")
        finally_block = source[source.index("    finally:"):]
        g1_29_shutdown = finally_block[finally_block.index('if args.arm == "G1_29":'):]
        # In the lifecycle-gated path, the arms return to the prepared pose before
        # the arm DDS output is released.
        self.assertLess(
            g1_29_shutdown.index("arm_ctrl.ctrl_dual_arm_go_home(release_motion_authority=True)"),
            g1_29_shutdown.index("arm_ctrl.deactivate()"),
        )

    def test_prearm_exit_never_requests_arm_home_motion(self):
        source = TELEOP.read_text(encoding="utf-8")
        prearm = source[:source.index("arm_ctrl.activate()")]
        self.assertNotIn("arm_ctrl.ctrl_dual_arm_go_home()", prearm)
    def test_arm_activate_holds_measured_pose_before_the_first_publish(self):
        source = ARM.read_text(encoding="utf-8")
        activate = source[source.index("    def activate(self):"):source.index("    def _subscribe_motor_state", source.index("    def activate(self):"))]
        hold = activate.index("lowstate.motor_state[id].q for id in G1_29_JointArmIndex")
        start = activate.index("self.publish_thread.start()")
        self.assertLess(hold, start)
    def test_constructors_create_no_dds_command_publishers(self):
        arm_init = ast.get_source_segment(
            ARM.read_text(encoding="utf-8"),
            class_method(ARM, "G1_29_ArmController", "__init__"),
        )
        hand_init = ast.get_source_segment(
            HAND.read_text(encoding="utf-8"),
            class_method(HAND, "Dex3_1_Controller", "__init__"),
        )
        self.assertNotIn("ChannelPublisher", arm_init)
        self.assertNotIn("ChannelPublisher", hand_init)
    def test_q_prearm_exit_serializes_with_dex3_activation(self):
        source = TELEOP.read_text(encoding="utf-8")
        gate = source.index("ready_tele_data.controller_sample_timestamp > ARM_REQUEST_TIMESTAMP")
        hand_activate = source.index("hand_ctrl.activate()")
        dex3_gate = source[gate:hand_activate]
        self.assertIn("with LIFECYCLE_LOCK:", dex3_gate)
        self.assertIn("if not STOP and not hand_outputs_activated:", dex3_gate)
        on_press = source[source.index("def on_press"):source.index("def get_state")]
        self.assertEqual(on_press.count("with LIFECYCLE_LOCK:"), 2)
    def test_only_g1_29_arm_controller_implements_lifecycle_gate_methods(self):
        tree = ast.parse(ARM.read_text(encoding="utf-8"))
        lifecycle_methods = ("activate", "deactivate")
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
            for method in lifecycle_methods:
                self.assertEqual(
                    method in methods,
                    node.name == "G1_29_ArmController",
                    f"{node.name} must {'define' if node.name == 'G1_29_ArmController' else 'not define'} {method}",
                )

    def test_alternate_arm_profiles_skip_the_g1_29_lifecycle_gate(self):
        source = TELEOP.read_text(encoding="utf-8")
        prep = source[source.index("Match the original launcher behavior"):source.index("# Initialize before the pre-arm loop")]
        g1_29_guard = prep.index('if args.arm == "G1_29":')
        activate = prep.index("arm_ctrl.activate()")
        # activate()/deactivate() belong to the G1_29 lifecycle contract only;
        # they must never run for the legacy arm profiles.
        self.assertLess(g1_29_guard, activate)
        self.assertNotIn("arm_ctrl.activate()", prep[:g1_29_guard])
        self.assertNotIn("arm_ctrl.deactivate()", prep[:g1_29_guard])

    def test_legacy_arm_profiles_retain_shutdown_home_motion(self):
        source = TELEOP.read_text(encoding="utf-8")
        finally_block = source[source.index("    finally:"):]
        legacy_home = finally_block.index("arm_ctrl.ctrl_dual_arm_go_home()")
        # The legacy shutdown path restores the arms home without touching the
        # G1_29-only activate()/deactivate() methods.
        self.assertIn("arm_ctrl.ctrl_dual_arm_go_home(release_motion_authority=True)", finally_block)
        self.assertGreater(legacy_home, finally_block.index('if args.arm == "G1_29":'))
        self.assertLess(finally_block.index("arm_ctrl.deactivate()"), legacy_home)

    def test_launcher_keeps_every_arm_profile_selectable(self):
        source = TELEOP.read_text(encoding="utf-8")
        profiles = [
            ("G1_29", "G1_29_ArmController"),
            ("G1_23", "G1_23_ArmController"),
            ("H1_2", "H1_2_ArmController"),
            ("H1", "H1_ArmController"),
            ("H2", "H2_ArmController"),
        ]
        for index, (flag, controller) in enumerate(profiles):
            guard = f'if args.arm == "{flag}":' if index == 0 else f'elif args.arm == "{flag}":'
            self.assertIn(guard, source)
            self.assertIn(f"arm_ctrl = {controller}(", source)
            self.assertLess(source.index(guard), source.index(f"arm_ctrl = {controller}("))

    def test_activation_rejects_stale_lowstate_and_q_deactivates_outputs(self):
        arm = ARM.read_text(encoding="utf-8")
        teleop = TELEOP.read_text(encoding="utf-8")
        hand = HAND.read_text(encoding="utf-8")
        self.assertIn("LowState is stale; refusing activation", arm)
        self.assertIn("self.output_enabled.clear()", arm)
        self.assertIn("def deactivate(self):", hand)
        self.assertLess(
            teleop.index("arm_ctrl.deactivate()"),
            teleop.index("stop_listening()"),
        )


if __name__ == "__main__":
    unittest.main()
