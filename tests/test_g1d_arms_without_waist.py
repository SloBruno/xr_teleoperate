"""G1D profile safety: Quest controls arms/Dex3, never the waist."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "teleop" / "robot_control" / "robot_arm.py"
TELEOP = ROOT / "teleop" / "teleop_hand_and_arm.py"


def _class_node(path, name):
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def test_g1d_controller_is_a_g1_29_arm_only_profile():
    node = _class_node(ARM, "G1D_ArmController")
    assert any(
        isinstance(base, ast.Name) and base.id == "G1_29_ArmController"
        for base in node.bases
    )
    values = {
        item.targets[0].id: item.value.value
        for item in node.body
        if isinstance(item, ast.Assign)
        and isinstance(item.targets[0], ast.Name)
        and isinstance(item.value, ast.Constant)
    }
    assert values["command_waist"] is False
    assert values["prepare_to_zero"] is False


def test_g1d_lowcmd_initializes_exactly_29_slots_with_passive_waist():
    source = ARM.read_text()
    g1d = source[source.index("class G1D_ArmController"):source.index("class G1_23_ArmController")]
    assert "motor_indices = tuple(range(29))" in g1d
    assert "motion_mode is unsupported" in g1d

    activate = source[source.index("    def activate(self):"):source.index("    def _configure_joint_command", source.index("    def activate(self):"))]
    assert "for id in self.motor_indices:" in activate
    assert "for id in G1_29_JointIndex:" not in activate

    passive = source[source.index("def _configure_joint_command"):source.index("def deactivate", source.index("def _configure_joint_command"))]
    assert "self.command_waist" in passive
    assert "cmd.kp = 0.0" in passive
    assert "cmd.kd = 0.0" in passive
    assert "cmd.tau = 0.0" in passive
    assert "G1_29_JointIndex.kWaistYaw" in passive
    assert "G1_29_JointIndex.kWaistRoll" in passive
    assert "G1_29_JointIndex.kWaistPitch" in passive


def test_g1d_requires_explicit_ik_geometry_assumption_and_rejects_motion_mode():
    source = TELEOP.read_text()
    assert 'parser.add_argument("--g1d-ik-geometry-assumption"' in source
    assert 'if args.arm == "G1D" and args.motion:' in source
    assert 'if args.arm == "G1D" and not args.g1d_ik_geometry_assumption:' in source


def test_g1d_profile_reuses_quest_arm_calibration_but_not_home_or_open_hands():
    source = TELEOP.read_text()
    assert "G1_ARM_7DOF_PROFILES = (\"G1_29\", \"G1D\")" in source
    assert "elif args.arm == \"G1D\":" in source
    g1d_shutdown = source[source.index('elif args.arm == "G1D":', source.index("    finally:")):]
    assert "attempt_return=False" in g1d_shutdown
    assert "open_hands=False" in g1d_shutdown


def test_g1d_operator_command_is_arm_only_and_acknowledges_ik_assumption():
    guide = (ROOT / "teleop" / "G1D_QUEST_DEX3.md").read_text()
    assert "--arm G1D" in guide
    assert "--ee dex3" in guide
    assert "--input-mode hand" in guide
    assert "--g1d-ik-geometry-assumption" in guide
    assert "--motion" not in guide


def test_g1d_lifecycle_accepts_terminal_r_q_and_quest_a_start_without_profile_gate():
    source = TELEOP.read_text()
    assert 'if args.arm != "G1D":\n                right_a_was_pressed' not in source
    assert source.count("right_a_was_pressed, right_b_was_pressed = poll_controller_lifecycle(") == 2
    assert "if key == 'r':\n            _request_start_locked()" in source
