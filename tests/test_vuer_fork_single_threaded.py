"""Regression: TeleVuer forks its Vuer server Process in __init__.  Forking after
DDS / keyboard / ZMQ threads exist hands the child a locked mutex (G1D head video
incident: child stuck in futex_wait, port 8012 never listened).

Runs the real teleop main path in a clean subprocess with stubbed DDS, keyboard,
image client and TeleVuer, and asserts that at the moment TeleVuerWrapper is
constructed the process has exactly one Python thread, no native thread beyond
the import-time numpy/OpenCV pools, and no live ZMQ context -- for head, legacy vertical and no-video modes."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

DRIVER = textwrap.dedent(r'''
    import gc, json, os, runpy, sys, threading, types
    import zmq
    import numpy, cv2  # OpenBLAS/OpenCV pools exist at import (fork-safe via pthread_atfork)
    BASELINE_NATIVE = len(os.listdir("/proc/self/task"))

    ROOT, OUT = sys.argv[1], sys.argv[2]
    sys.argv = ["teleop_hand_and_arm.py"] + json.loads(sys.argv[3])
    sys.path.insert(0, ROOT)
    events, record = [], {}

    def live_zmq_contexts():
        return sum(1 for o in gc.get_objects() if isinstance(o, zmq.Context) and not o.closed)

    def native_tasks():
        return len(os.listdir("/proc/self/task"))

    HEAD = {"enable_zmq": True, "enable_webrtc": True, "image_shape": [480, 1280],
            "binocular": True, "webrtc_port": 60001, "zmq_port": 55555}
    WRIST = {"enable_zmq": True, "enable_webrtc": False, "image_shape": [480, 640],
             "binocular": False, "webrtc_port": 0, "zmq_port": 55556}
    CONFIG = {"head_camera": HEAD, "left_wrist_camera": WRIST, "right_wrist_camera": dict(WRIST, zmq_port=55557)}

    class FakeRequester:
        """Real zmq Context + REQ socket, like teleimager's ZMQ_Requester."""
        def __init__(self, host, port):
            events.append("config_request")
            self._ctx = zmq.Context()
            self._sock = self._ctx.socket(zmq.REQ)
            self._sock.setsockopt(zmq.LINGER, 0)
            self._sock.connect("tcp://127.0.0.1:1")
        def request(self):
            return json.loads(json.dumps(CONFIG))
        def close(self):
            self._sock.close()
            self._ctx.term()

    class FakeImageClient:
        """Starts real subscriber threads + zmq context, like teleimager's ImageClient."""
        def __init__(self, host, request_bgr=False, camera_names=None, **kw):
            events.append("image_client")
            record["camera_names"] = None if camera_names is None else list(camera_names)
            self._ctx = zmq.Context()
            self._stop = threading.Event()
            self._t = threading.Thread(target=self._stop.wait, daemon=True)
            self._t.start()
        def get_cam_config(self):
            return json.loads(json.dumps(CONFIG))
        def close(self):
            self._stop.set()
            self._ctx.term()

    class FakeTeleVuerWrapper:
        def __init__(self, **kwargs):
            events.append("televuer")
            record["at_fork"] = {
                "py_threads": threading.active_count(),
                "thread_names": [t.name for t in threading.enumerate()],
                "native_tasks": native_tasks(),
                "baseline_native_tasks": BASELINE_NATIVE,
                "live_zmq_contexts": live_zmq_contexts(),
            }
            record["tv_kwargs"] = {k: (v if isinstance(v, (bool, int, float, str, type(None))) else list(v))
                                   for k, v in kwargs.items()}
        def close(self):
            pass

    dds_stop = threading.Event()
    def fake_dds_init(*a, **k):
        events.append("dds")
        threading.Thread(target=dds_stop.wait, daemon=True, name="fake-dds").start()

    kb_stop = threading.Event()
    def fake_listen_keyboard(**kwargs):
        kb_stop.wait()
    def fake_stop_listening():
        kb_stop.set()
    _real_thread_start = threading.Thread.start

    class StopHere(KeyboardInterrupt):
        pass

    class FakeMotionSwitcher:
        def __init__(self):
            events.append("motion_switcher")
            raise StopHere()

    logger = types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None,
                                   warning=lambda *a, **k: None, error=lambda *a, **k: None)
    def ns_module(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        for k, v in attrs.items():
            setattr(m, k, v)
        return m
    modules = {
        "logging_mp": ns_module("logging_mp", basicConfig=lambda **k: None, getLogger=lambda n: logger, INFO=20),
        "unitree_sdk2py": ns_module("unitree_sdk2py"),
        "unitree_sdk2py.core": ns_module("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": ns_module("unitree_sdk2py.core.channel",
                                                 ChannelFactoryInitialize=fake_dds_init, ChannelPublisher=object),
        "unitree_sdk2py.idl": ns_module("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.std_msgs": ns_module("unitree_sdk2py.idl.std_msgs"),
        "unitree_sdk2py.idl.std_msgs.msg": ns_module("unitree_sdk2py.idl.std_msgs.msg"),
        "unitree_sdk2py.idl.std_msgs.msg.dds_": ns_module("unitree_sdk2py.idl.std_msgs.msg.dds_", String_=object),
        "televuer": ns_module("televuer", TeleVuerWrapper=FakeTeleVuerWrapper),
        "teleimager": ns_module("teleimager"),
        "teleimager.image_client": ns_module("teleimager.image_client",
                                             ImageClient=FakeImageClient, ZMQ_Requester=FakeRequester),
        "sshkeyboard": ns_module("sshkeyboard", listen_keyboard=fake_listen_keyboard,
                                 stop_listening=fake_stop_listening),
        "teleop.robot_control.robot_arm": ns_module("teleop.robot_control.robot_arm",
            G1_29_ArmController=object, G1_23_ArmController=object, H1_2_ArmController=object,
            H1_ArmController=object, H2_ArmController=object, G1D_ArmController=object),
        "teleop.robot_control.robot_arm_ik": ns_module("teleop.robot_control.robot_arm_ik",
            G1_29_ArmIK=object, G1_23_ArmIK=object, H1_2_ArmIK=object, H1_ArmIK=object, H2_ArmIK=object),
        "teleop.utils.episode_writer": ns_module("teleop.utils.episode_writer", EpisodeWriter=object),
        "teleop.utils.ipc": ns_module("teleop.utils.ipc", IPC_Server=object),
        "teleop.utils.motion_switcher": ns_module("teleop.utils.motion_switcher",
            MotionSwitcher=FakeMotionSwitcher, LocoClientWrapper=object, is_walk_fsm=lambda f: False),
    }
    sys.modules.update(modules)
    try:
        runpy.run_path(os.path.join(ROOT, "teleop", "teleop_hand_and_arm.py"), run_name="__main__")
    except SystemExit as e:
        record["exit"] = str(e.code)
    record["events"] = events
    with open(OUT, "w") as f:
        json.dump(record, f)
''')

G1D = ["--arm", "G1D", "--g1d-ik-geometry-assumption", "--ee", "dex3", "--input-mode", "hand"]
MODES = {
    "head": G1D + ["--camera-layout", "head", "--force-zmq-video", "--video-fps", "10", "--jpeg-quality", "60"],
    "vertical": G1D + ["--camera-layout", "vertical"],
    "no_video": G1D + ["--display-mode", "pass-through", "--no-video"],
}


def _run_main(tmp_path, argv):
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    out = tmp_path / "out.json"
    env = {**os.environ, "XR_TELEOP_STATUS_LOG": str(tmp_path / "s.jsonl"),
           "XR_TELEOP_POSE_LOG_DIR": str(tmp_path)}
    proc = subprocess.run([sys.executable, str(driver), str(ROOT), str(out), json.dumps(argv)],
                          capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60)
    assert out.exists(), proc.stdout + proc.stderr
    return json.loads(out.read_text())


@pytest.mark.parametrize("mode", sorted(MODES))
def test_televuer_forks_while_process_is_single_threaded(tmp_path, mode):
    rec = _run_main(tmp_path, MODES[mode])
    assert "at_fork" in rec, rec
    at_fork = rec["at_fork"]
    assert at_fork["py_threads"] == 1, at_fork
    # No native threads beyond the import-time library pools (catches DDS/ZMQ I/O threads).
    assert at_fork["native_tasks"] <= at_fork["baseline_native_tasks"], at_fork
    assert at_fork["live_zmq_contexts"] == 0, at_fork
    events = rec["events"]
    tv = events.index("televuer")
    assert tv < events.index("dds"), events
    if mode == "no_video":
        assert "image_client" not in events and "config_request" not in events, events
    else:
        assert events.index("config_request") < tv < events.index("image_client"), events
        assert events.index("dds") < events.index("image_client"), events


def test_head_mode_keeps_head_only_binocular_zmq_10hz_jpeg60(tmp_path):
    rec = _run_main(tmp_path, MODES["head"])
    kw = rec["tv_kwargs"]
    assert rec["camera_names"] == ["head_camera"]
    assert kw["binocular"] is True and kw["img_shape"] == [480, 1280]
    assert kw["zmq"] is True and kw["webrtc"] is False
    assert kw["display_fps"] == 10.0 and kw["jpeg_quality"] == 60


def test_vertical_mode_keeps_legacy_stacked_monocular_layout(tmp_path):
    rec = _run_main(tmp_path, MODES["vertical"])
    kw = rec["tv_kwargs"]
    assert rec["camera_names"] is None
    assert kw["binocular"] is False and kw["img_shape"] == [645, 640]
    assert kw["zmq"] is True and kw["webrtc"] is False
    assert kw["display_fps"] == 30.0 and kw["jpeg_quality"] == 80


def test_no_video_keeps_passthrough_wrapper(tmp_path):
    kw = _run_main(tmp_path, MODES["no_video"])["tv_kwargs"]
    assert kw["display_mode"] == "pass-through"
    assert kw["zmq"] is False and kw["webrtc"] is False
