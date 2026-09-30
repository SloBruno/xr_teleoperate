import copy
import importlib.util
from pathlib import Path

MODULE = Path(__file__).parents[1] / "teleop" / "utils" / "teleimager_head_only_server.py"


def load():
    spec = importlib.util.spec_from_file_location("teleimager_head_only_server", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = {
    "head_camera": {"enable_zmq": True, "enable_webrtc": False, "zmq_port": 55555, "serial_number": "243122072230"},
    "left_wrist_camera": {"enable_zmq": True, "enable_webrtc": True, "zmq_port": 55556},
    "right_wrist_camera": {"enable_zmq": False, "enable_webrtc": False, "zmq_port": 55557},
}


def test_head_only_config_disables_every_wrist_camera_without_mutating_input():
    module = load()
    original = copy.deepcopy(CONFIG)
    result = module.head_only_config(CONFIG)
    assert CONFIG == original
    assert result["head_camera"] == CONFIG["head_camera"]
    for name in ("left_wrist_camera", "right_wrist_camera"):
        assert result[name]["enable_zmq"] is False
        assert result[name]["enable_webrtc"] is False
    assert result["left_wrist_camera"]["zmq_port"] == 55556


def test_head_only_config_requires_enabled_head_camera():
    module = load()
    bad = copy.deepcopy(CONFIG)
    bad["head_camera"]["enable_zmq"] = False
    try:
        module.head_only_config(bad)
    except ValueError as exc:
        assert "head_camera" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_wrapper_overrides_server_config_path_not_the_yaml():
    source = MODULE.read_text(encoding="utf-8")
    assert "CONFIG_PATH" in source
    assert "TELEIMAGER_HEAD_ONLY_CONFIG" in source
    assert "SOMENTE CABEÇA" in source
