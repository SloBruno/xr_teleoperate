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

    def test_teleop_activates_outputs_only_after_the_post_r_prearm_gate(self):
        source = TELEOP.read_text(encoding="utf-8")
        gate = source.index("ready_tele_data.controller_sample_timestamp > ARM_REQUEST_TIMESTAMP")
        arm_activate = source.index("arm_ctrl.activate()")
        hand_activate = source.index("hand_ctrl.activate()")
        self.assertLess(gate, arm_activate)
        self.assertLess(gate, hand_activate)

    def test_prearm_exit_never_requests_arm_home_motion(self):
        source = TELEOP.read_text(encoding="utf-8")
        self.assertNotIn("arm_ctrl.ctrl_dual_arm_go_home()", source)
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
    def test_q_prearm_exit_serializes_with_output_activation(self):
        source = TELEOP.read_text(encoding="utf-8")
        prearm = source.index("while not STOP:")
        arm_activate = source.index("arm_ctrl.activate()")
        lock = source.index("with LIFECYCLE_LOCK:", prearm, arm_activate)
        cancellation = source.index("prearm_cancelled = STOP", lock, arm_activate)
        self.assertLess(prearm, lock)
        self.assertLess(lock, cancellation)
        self.assertLess(cancellation, arm_activate)
        on_press = source[source.index("def on_press"):source.index("def get_state")]
        self.assertEqual(on_press.count("with LIFECYCLE_LOCK:"), 2)
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
