import ast
from pathlib import Path

import numpy as np


TV_WRAPPER = (
    Path(__file__).parents[1]
    / "teleop/televuer/src/televuer/tv_wrapper.py"
)


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

    source = TV_WRAPPER.read_text()
    controller_selection = source.split(
        "# Hand skeleton control keeps the reference implementation's live", 1
    )[1].split("# -----------------------------------hand position", 1)[0]
    assert "transform_controller_world_arm_to_calibration_origin" in controller_selection