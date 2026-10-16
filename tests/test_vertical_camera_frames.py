import importlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace


def _install_import_stubs(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    module_names = (
        "logging_mp",
        "unitree_sdk2py",
        "unitree_sdk2py.core",
        "unitree_sdk2py.core.channel",
        "unitree_sdk2py.idl",
        "unitree_sdk2py.idl.std_msgs",
        "unitree_sdk2py.idl.std_msgs.msg",
        "unitree_sdk2py.idl.std_msgs.msg.dds_",
        "televuer",
        "teleimager",
        "teleimager.image_client",
        "sshkeyboard",
        "teleop.robot_control.robot_arm",
        "teleop.robot_control.robot_arm_ik",
        "teleop.utils.episode_writer",
        "teleop.utils.ipc",
        "teleop.utils.motion_switcher",
    )
    for name in module_names:
        module = types.ModuleType(name)
        module.__path__ = []
        module.__getattr__ = lambda name: object
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["logging_mp"].basicConfig = lambda **kwargs: None
    sys.modules["logging_mp"].getLogger = lambda name: types.SimpleNamespace()
    sys.modules["logging_mp"].INFO = 20
    channel = sys.modules["unitree_sdk2py.core.channel"]
    channel.ChannelFactoryInitialize = object
    channel.ChannelPublisher = object
    sys.modules["unitree_sdk2py.idl.std_msgs.msg.dds_"].String_ = object
    sys.modules["televuer"].TeleVuerWrapper = object
    sys.modules["teleimager.image_client"].ImageClient = object
    sys.modules["sshkeyboard"].listen_keyboard = object
    sys.modules["sshkeyboard"].stop_listening = object


def test_vertical_stack_skips_missing_camera_frames(monkeypatch):
    _install_import_stubs(monkeypatch)
    module = importlib.import_module("teleop.teleop_hand_and_arm")

    assert module.stack_camera_images_vertical(None, SimpleNamespace(bgr=None)) is None
    assert module.stack_camera_images_vertical(SimpleNamespace(bgr=None), SimpleNamespace(bgr=None)) is None


REAL_HEAD_CFG = {"enable_zmq": True, "enable_webrtc": True, "image_shape": [480, 1280], "binocular": True}


def test_real_frames_stack_to_planned_display_shape(monkeypatch):
    import numpy as np
    _install_import_stubs(monkeypatch)
    module = importlib.import_module("teleop.teleop_hand_and_arm")
    head = SimpleNamespace(bgr=np.zeros((480, 1280, 3), np.uint8))
    wrist = SimpleNamespace(bgr=np.zeros((480, 640, 3), np.uint8))
    out = module.stack_camera_images_vertical(head, wrist, 0.5, 0.89, 0.11, 4)
    assert out.shape == (645, 640, 3)


def test_robot_webrtc_flag_must_not_disable_zmq_headset_view(monkeypatch):
    # Real G1D server advertises enable_webrtc=true on a robot-LAN-only URL the
    # Wi-Fi Quest cannot reach; the stacked ZMQ view must still be rendered.
    _install_import_stubs(monkeypatch)
    module = importlib.import_module("teleop.teleop_hand_and_arm")
    use_webrtc, need_local = module.resolve_xr_image_transport("immersive", REAL_HEAD_CFG, True)
    assert use_webrtc is False
    assert need_local is True
    assert module.resolve_xr_image_transport("pass-through", REAL_HEAD_CFG, True) == (False, False)
    assert module.resolve_xr_image_transport("immersive", REAL_HEAD_CFG, False) == (True, False)
