"""G1D preparation must reuse G1's exact arm-only home contract."""
import ast
import threading
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "teleop" / "robot_control" / "robot_arm.py"
TELEOP = ROOT / "teleop" / "teleop_hand_and_arm.py"


def _method_source(class_name, method_name):
    tree = ast.parse(ARM.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    return ast.get_source_segment(ARM.read_text(encoding="utf-8"), method)


def test_g1d_uses_the_normal_g1_home_method_with_the_same_target_timeout_and_arrival_check():
    normal = _method_source("G1_29_ArmController", "ctrl_dual_arm_go_home")
    g1d = ARM.read_text(encoding="utf-8").split("class G1D_ArmController", 1)[1].split("class G1_23_ArmController", 1)[0]

    assert "self.q_target = np.zeros(14)" in normal
    assert "max_attempts = 100" in normal
    assert "time.sleep(0.05)" in normal
    assert "np.all(np.abs(current_q) <= tolerance)" in normal
    launcher = TELEOP.read_text(encoding="utf-8")
    assert "ctrl_dual_arm_go_home(cancelled=lambda: STOP, require_fresh_lowstate=True)" in launcher
    assert "def ctrl_dual_arm_go_home" not in g1d


def test_normal_home_path_keeps_the_normal_target_and_measured_arrival_behavior():
    source = ARM.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1_29_ArmController")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "ctrl_dual_arm_go_home")
    namespace = {"np": np, "time": type("Time", (), {"sleep": staticmethod(lambda _s: None)})(), "logger_mp": type("Log", (), {"info": staticmethod(lambda *_a: None), "error": staticmethod(lambda *_a: None)})()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ARM), "exec"), namespace)

    class FakeArm:
        motion_mode = False
        ctrl_lock = threading.Lock()
        q_target = np.full(14, 9.0)

        def __init__(self):
            self.samples = [np.ones(14), np.zeros(14)]

        def get_current_dual_arm_q(self):
            return self.samples.pop(0)

    arm = FakeArm()
    assert namespace["ctrl_dual_arm_go_home"](arm)
    np.testing.assert_array_equal(arm.q_target, np.zeros(14))


def test_q_cancels_the_shared_home_path_before_a_measured_arrival():
    source = ARM.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1_29_ArmController")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "ctrl_dual_arm_go_home")
    namespace = {"np": np, "time": type("Time", (), {"sleep": staticmethod(lambda _s: None)})(), "logger_mp": type("Log", (), {"info": staticmethod(lambda *_a: None), "error": staticmethod(lambda *_a: None)})()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ARM), "exec"), namespace)

    class FakeArm:
        motion_mode = False
        ctrl_lock = threading.Lock()
        q_target = np.full(14, 9.0)
        last_preparation_status = ""

        def get_current_dual_arm_q(self):
            raise AssertionError("cancelled prep must not wait for arrival")

    arm = FakeArm()
    assert not namespace["ctrl_dual_arm_go_home"](arm, cancelled=lambda: True)
    assert arm.last_preparation_status == "preparation_cancelled"
    np.testing.assert_array_equal(arm.q_target, np.zeros(14))


def test_r_is_blocked_until_preparation_is_explicitly_complete():
    source = TELEOP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_request_start_locked")
    namespace = {
        "time": __import__("time"),
        "logger_mp": type("Log", (), {"warning": staticmethod(lambda *_a: None)})(),
        "PREPARATION_COMPLETE": False,
        "START": False,
        "STOP": False,
        "ARM_REQUEST_TIMESTAMP": 0.0,
        "arm_calibration": None,
        "LIFECYCLE_EVENTS": [],
        "HUMAN_SWEEP_ACTIVE": False,
        "HUMAN_SWEEP_REQUESTED": False,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TELEOP), "exec"), namespace)

    assert not namespace["_request_start_locked"]()
    assert namespace["START"] is False


def test_g1d_never_commands_waist_and_q_cancels_preparation_before_r_is_accepted():
    arm_source = ARM.read_text(encoding="utf-8")
    teleop_source = TELEOP.read_text(encoding="utf-8")
    g1d = arm_source.split("class G1D_ArmController", 1)[1].split("class G1_23_ArmController", 1)[0]
    preparation = teleop_source[teleop_source.index("# Match the original launcher behavior"):teleop_source.index("# Initialize before the pre-arm loop")]

    assert "command_waist = False" in g1d
    assert "cmd.q = measured_q" in arm_source
    assert "cmd.kp = 0.0" in arm_source and "cmd.kd = 0.0" in arm_source and "cmd.tau = 0.0" in arm_source
    assert "preparation_cancelled" in preparation
    assert "if STOP:" in preparation
    start = teleop_source[teleop_source.index("def _request_start_locked"):teleop_source.index("def _request_stop_locked")]
    assert start.index("if not PREPARATION_COMPLETE:") < start.index("START = True")


def test_g1d_keeps_motion_disabled_and_dex3_inert_during_preparation():
    source = TELEOP.read_text(encoding="utf-8")
    preparation = source[source.index("# Match the original launcher behavior"):source.index("# Initialize before the pre-arm loop")]

    assert 'if args.arm == "G1D" and args.motion:' in source
    assert "hand_ctrl.activate()" not in preparation
    assert "open_hands=False" in source[source.index('elif args.arm == "G1D":', source.index("    finally:")):]
