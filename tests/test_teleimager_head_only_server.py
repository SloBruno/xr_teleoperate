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


# ---- modo any / câmera única ----
HEAD, WRIST = "243122072230", "233622070789"
FULL = copy.deepcopy(CONFIG)
FULL["left_wrist_camera"].update({"serial_number": WRIST, "image_shape": [480, 640], "fps": 30, "binocular": True})


def test_select_single_camera_none_fails_clearly():
    module = load()
    try:
        module.select_single_camera(set())
    except RuntimeError as exc:
        assert "nenhuma" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError")


def test_select_single_camera_head_wrist_and_both():
    module = load()
    assert module.select_single_camera({HEAD}) == ("head", HEAD, False)
    assert module.select_single_camera({WRIST}) == ("left_wrist", WRIST, False)
    assert module.select_single_camera({HEAD, WRIST}) == ("head", HEAD, True)
    assert module.select_single_camera({"other", WRIST}) == ("left_wrist", WRIST, False)


def test_single_camera_config_wrist_becomes_head_and_wrist_disabled():
    module = load()
    original = copy.deepcopy(FULL)
    result = module.single_camera_config(FULL, "left_wrist", WRIST)
    assert FULL == original
    head = result["head_camera"]
    assert head["serial_number"] == WRIST
    assert head["image_shape"] == [720, 1280] and head["fps"] == 15 and head["binocular"] is False
    assert head["enable_zmq"] is True and head["zmq_port"] == 55555 and head["type"] == "realsense"
    for name in ("left_wrist_camera", "right_wrist_camera"):
        assert result[name]["enable_zmq"] is False and result[name]["enable_webrtc"] is False


def test_single_camera_config_head_keeps_head():
    module = load()
    result = module.single_camera_config(FULL, "head", HEAD)
    assert result["head_camera"]["serial_number"] == HEAD
    assert result["left_wrist_camera"]["enable_zmq"] is False


def test_banner_and_warning_text():
    module = load()
    assert module.single_banner("head", HEAD) == f"TELEIMAGER: modo CÂMERA ÚNICA (cabeça serial {HEAD} publicada como imagem principal)"
    assert module.single_banner("left_wrist", WRIST) == f"TELEIMAGER: modo CÂMERA ÚNICA (pulso esquerdo serial {WRIST} publicada como imagem principal)"
    assert "mão" in module.WRIST_WARNING and "cabeça" in module.WRIST_WARNING
