import ast
import importlib
from pathlib import Path
import sys
import types

import numpy as np


TV_WRAPPER = (
    Path(__file__).parents[1]
    / "teleop/televuer/src/televuer/tv_wrapper.py"
)


def _load_tv_wrapper_module(monkeypatch):
    vuer = types.ModuleType("vuer")
    vuer.Vuer = object
    schemas = types.ModuleType("vuer.schemas")
    for name in ("ImageBackground", "Hands", "MotionControllers", "WebRTCVideoPlane", "WebRTCStereoVideoPlane"):
        setattr(schemas, name, object)
    monkeypatch.setitem(sys.modules, "vuer", vuer)
    monkeypatch.setitem(sys.modules, "vuer.schemas", schemas)
    sys.path.insert(0, str(TV_WRAPPER.parents[1]))
    sys.modules.pop("televuer", None)
    sys.modules.pop("televuer.tv_wrapper", None)
    return importlib.import_module("televuer.tv_wrapper")


class _FakeControllerTeleVuer:
    motion_data_ready = True
    left_hand_positions = np.zeros((25, 3))
    right_hand_positions = np.zeros((25, 3))
    left_hand_orientations = np.tile(np.eye(3)[None, :, :], (25, 1, 1))
    right_hand_orientations = np.tile(np.eye(3)[None, :, :], (25, 1, 1))
    left_hand_pinch = right_hand_pinch = False
    left_hand_pinchValue = right_hand_pinchValue = 0.0
    left_hand_squeeze = right_hand_squeeze = False
    left_hand_squeezeValue = right_hand_squeezeValue = 0.0
    left_ctrl_trigger = right_ctrl_trigger = False
    left_ctrl_triggerValue = right_ctrl_triggerValue = 0.0
    left_ctrl_squeeze = right_ctrl_squeeze = False
    left_ctrl_squeezeValue = right_ctrl_squeezeValue = 0.0
    left_ctrl_thumbstick = right_ctrl_thumbstick = False
    left_ctrl_thumbstickValue = right_ctrl_thumbstickValue = np.zeros(2)
    left_ctrl_aButton = right_ctrl_aButton = False
    left_ctrl_bButton = right_ctrl_bButton = False

    def __init__(self, head_pose, controller_pair):
        self.head_pose = head_pose
        self.hand_pose_sample = (np.eye(4), np.eye(4), 1.0)
        self.controller_pose_sample = (*controller_pair, 2.0)


def _load_world_transform():
    tree = ast.parse(TV_WRAPPER.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "transform_controller_world_arm_to_calibration_origin"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TV_WRAPPER), "exec"), namespace)
    return namespace[function.name]


def test_stationary_world_controller_is_independent_of_head_motion():
    transform = _load_world_transform()
    controller_world = np.eye(4)
    controller_world[:3, 3] = [0.4, 0.2, 1.1]
    head_a = np.eye(4)
    head_b = np.eye(4)
    head_b[:3, 3] = [0.1, -0.2, 0.3]

    first = transform(controller_world, head_a)
    second = transform(controller_world, head_b)

    np.testing.assert_array_equal(second, first)

    head_b[:3, :3] = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_array_equal(transform(controller_world, head_b), first)

    source = TV_WRAPPER.read_text()
    controller_selection = source.split(
        "# Hand skeleton control keeps the reference implementation's live", 1
    )[1].split("# -----------------------------------hand position", 1)[0]
    assert "transform_controller_world_arm_to_calibration_origin" in controller_selection

    controller_path = source.rsplit("# controller tracking", 1)[1]
    assert "left_controller_pose" in controller_path
    assert "right_controller_pose" in controller_path
    assert "transform_controller_world_arm_to_calibration_origin" in controller_path
    assert "transform_IPunitree_Brobot_world_arm_to_head_then_waist" not in controller_path


def test_world_frame_transform_keeps_left_and_right_controller_offsets_independent():
    transform = _load_world_transform()
    left = np.eye(4)
    right = np.eye(4)
    left[:3, 3] = [0.4, 0.2, 1.1]
    right[:3, 3] = [0.4, -0.2, 1.1]

    left_result = transform(left)
    right_result = transform(right)

    np.testing.assert_allclose(left_result[:3, 3], [0.55, 0.2, 1.55])
    np.testing.assert_allclose(right_result[:3, 3], [0.55, -0.2, 1.55])


def test_both_controller_modes_return_same_target_when_headset_moves(monkeypatch):
    module = _load_tv_wrapper_module(monkeypatch)
    left = np.eye(4)
    right = np.eye(4)
    left[:3, 3] = [0.4, 0.2, 1.1]
    right[:3, 3] = [0.4, -0.2, 1.1]
    head_a = np.eye(4)
    head_b = np.eye(4)
    head_b[:3, 3] = [0.2, -0.1, 0.3]
    head_b[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

    outputs = []
    for use_hand_tracking in (False, True):
        wrapper = module.TeleVuerWrapper.__new__(module.TeleVuerWrapper)
        wrapper.use_hand_tracking = use_hand_tracking
        wrapper.return_hand_rot_data = False
        wrapper.arm_reference_mode = "head_yaw"
        wrapper.arm_pose_source = "controller"
        wrapper.tvuer = _FakeControllerTeleVuer(head_a, (left, right))
        first = wrapper.get_tele_data()
        wrapper.tvuer.head_pose = head_b
        second = wrapper.get_tele_data()
        np.testing.assert_allclose(second.left_wrist_pose, first.left_wrist_pose)
        np.testing.assert_allclose(second.right_wrist_pose, first.right_wrist_pose)
        outputs.append((first.left_wrist_pose, first.right_wrist_pose))

    np.testing.assert_allclose(outputs[0][0], outputs[1][0])
    np.testing.assert_allclose(outputs[0][1], outputs[1][1])
