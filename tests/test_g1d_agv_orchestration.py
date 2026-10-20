"""Real teleop main-path orchestration for the opt-in G1-D AGV (in-process
runpy harness, fake RPC client and height feedback; no DDS, no robot)."""
import json
import runpy
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teleop.utils import g1d_agv  # noqa: E402

BASE_ARGV = ["teleop_hand_and_arm.py", "--arm", "G1D", "--g1d-ik-geometry-assumption", "--input-mode", "hand",
             "--display-mode", "pass-through", "--no-video", "--ipc"]


def _run(monkeypatch, tmp_path, script, *, agv=True, extra_env=None):
    """script(i) -> dict for tracking-loop call i (left/right/age/q/raise)."""
    log = []                    # ordered events across fakes
    agv_calls = []
    callbacks = []
    state = {"tracking_i": -1, "phase": "prearm"}

    class FakeAgvClient:
        def Move(self, vx, vy, vyaw):
            agv_calls.append((state["tracking_i"], state["phase"], "move", vx, vy, vyaw))
            log.append("agv_move")
            return 0

        def HeightAdjust(self, vz):
            agv_calls.append((state["tracking_i"], state["phase"], "height", vz))
            return 0

    class FakeHeight:
        def start(self):
            log.append("height_sub")

        def is_fresh(self, now=None):
            return True

        def snapshot(self, now=None):
            return {"height_m": 0.18, "age_s": 0.0, "fresh": True, "samples": 1}

        def close(self):
            log.append("height_close")

    def client_factory():
        log.append("agv_client")
        return FakeAgvClient()

    monkeypatch.setattr(g1d_agv, "make_agv_client", client_factory)
    monkeypatch.setattr(g1d_agv, "HeightMonitor", FakeHeight)

    class FakeArm:
        arm_joint_split = (7, 7)

        def __init__(self, **kw):
            self.preparation_progress = None

        def activate(self):
            log.append("arm_activate")

        def ctrl_dual_arm_go_home(self, **kw):
            return True

        def get_current_dual_arm_q(self):
            return np.zeros(14)

        def get_current_dual_arm_dq(self):
            return np.zeros(14)

        def speed_gradual_max(self):
            pass

        def deactivate(self):
            log.append("arm_deactivate")

    class FakeIK:
        def forward_kinematics(self, q):
            return np.eye(4), np.eye(4)

        def reset_warm_start(self):
            pass

    class FakeCalibrator:
        human_calibration = None

        def __init__(self, *a, **k):
            self.calibrated = False

        def calibrate(self, *a, **k):
            self.calibrated = True
            return True

        def reset_for_start_request(self, ts):
            pass

        def consume_first_target(self):
            return (np.eye(4), np.eye(4))

        def check_l_pose(self, *a):
            return types.SimpleNamespace(accepted=True)

    def tele(ts, left=(0.0, 0.0), right=(0.0, 0.0)):
        return types.SimpleNamespace(
            head_pose=np.eye(4), left_wrist_pose=np.eye(4), right_wrist_pose=np.eye(4),
            controller_sample_timestamp=ts, right_ctrl_aButton=False, right_ctrl_bButton=False,
            left_ctrl_aButton=False, left_ctrl_thumbstickValue=np.array(left, float),
            right_ctrl_thumbstickValue=np.array(right, float), left_ctrl_triggerValue=0.0,
            right_ctrl_triggerValue=0.0, motion_data_ready=True)

    class FakeTV:
        def __init__(self, **kw):
            pass

        def get_tele_data(self):
            if state["phase"] == "prearm":
                if callbacks:
                    callbacks.pop()("r")
                    state["phase"] = "arming"
                    time.sleep(0.002)
                return tele(time.monotonic(), (0.0, -1.0), (0.0, -1.0))
            state["phase"] = "tracking"
            state["tracking_i"] += 1
            step = script(state["tracking_i"])
            if step.get("raise") is not None:
                raise step["raise"]
            if step.get("q"):
                ipc_on_press[0]("q")
            return tele(time.monotonic() - step.get("age", 0.0), step.get("left", (0.0, 0.0)),
                        step.get("right", (0.0, 0.0)))

        def close(self):
            log.append("tv_close")

    ipc_on_press = []

    class FakeIPC:
        def __init__(self, on_press, get_state):
            ipc_on_press.append(on_press)
            self.on_press = on_press

        def start(self):
            callbacks.append(self.on_press)

        def stop(self):
            log.append("ipc_stop")

    def fake_cycle(**kw):
        from teleop.utils.arm_tracking_orchestration import ArmTrackingCycleResult
        return ArmTrackingCycleResult(target_accepted=False, published=False, hold=True, requested_q=None,
                                      requested_tauff=None, selected_q=None, selected_tauff=None,
                                      decision_reason="test")

    def fake_shutdown(arm, **kw):
        log.append("arm_shutdown")
        arm.deactivate()
        return types.SimpleNamespace(deactivated=True, hands_opened=True)

    logger = types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None,
                                   warning=lambda *a, **k: None,
                                   error=lambda *a, **k: log.append(("error", str(a[0])[-400:])))
    modules = {
        "logging_mp": types.SimpleNamespace(basicConfig=lambda **k: None, getLogger=lambda n: logger, INFO=20),
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": types.SimpleNamespace(
            ChannelFactoryInitialize=lambda *a, **k: log.append("dds"), ChannelPublisher=object),
        "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.std_msgs": types.ModuleType("unitree_sdk2py.idl.std_msgs"),
        "unitree_sdk2py.idl.std_msgs.msg": types.ModuleType("unitree_sdk2py.idl.std_msgs.msg"),
        "unitree_sdk2py.idl.std_msgs.msg.dds_": types.SimpleNamespace(String_=object),
        "televuer": types.SimpleNamespace(TeleVuerWrapper=FakeTV),
        "teleimager": types.ModuleType("teleimager"),
        "teleimager.image_client": types.SimpleNamespace(ImageClient=object, ZMQ_Requester=object),
        "sshkeyboard": types.SimpleNamespace(listen_keyboard=lambda **k: None, stop_listening=lambda: None),
        "teleop.robot_control.robot_arm": types.SimpleNamespace(
            G1_29_ArmController=FakeArm, G1_23_ArmController=FakeArm, H1_2_ArmController=FakeArm,
            H1_ArmController=FakeArm, H2_ArmController=FakeArm, G1D_ArmController=FakeArm),
        "teleop.robot_control.robot_arm_ik": types.SimpleNamespace(
            G1_29_ArmIK=FakeIK, G1_23_ArmIK=FakeIK, H1_2_ArmIK=FakeIK, H1_ArmIK=FakeIK, H2_ArmIK=FakeIK),
        "teleop.utils.episode_writer": types.SimpleNamespace(EpisodeWriter=object),
        "teleop.utils.ipc": types.SimpleNamespace(IPC_Server=FakeIPC),
        "teleop.utils.motion_switcher": types.SimpleNamespace(
            MotionSwitcher=lambda: types.SimpleNamespace(Enter_Debug_Mode=lambda: (0, None)),
            LocoClientWrapper=object, is_walk_fsm=lambda f: False),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    import teleop.utils.human_arm_calibration as hac
    import teleop.utils.controller_wrist_calibration as cwc
    import teleop.utils.arm_tracking_orchestration as ato
    import teleop.utils.arm_graceful_shutdown as ags
    import teleop.utils.xr_video_policy as xvp
    monkeypatch.setattr(hac, "HumanCalibratedWristCalibrator", FakeCalibrator)
    monkeypatch.setattr(cwc, "ControllerWristCalibrator", lambda *a, **k: None)
    monkeypatch.setattr(ato, "run_arm_tracking_cycle", fake_cycle)
    monkeypatch.setattr(ags, "run_graceful_arm_shutdown", fake_shutdown)
    # In-process harness: pytest may hold daemon threads; the fork guard runs in
    # a clean subprocess in tests/test_vuer_fork_single_threaded.py.
    monkeypatch.setattr(xvp, "require_single_threaded_for_fork", lambda: None)
    monkeypatch.setenv("XR_TELEOP_STATUS_LOG", str(tmp_path / "status.jsonl"))
    monkeypatch.setenv("XR_TELEOP_POSE_LOG_DIR", str(tmp_path))
    if agv:
        monkeypatch.setenv("G1D_AGV", "1")
    else:
        monkeypatch.delenv("G1D_AGV", raising=False)
    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)
    argv = BASE_ARGV + (["--g1d-agv"] if agv else [])
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(ROOT / "teleop" / "teleop_hand_and_arm.py"), run_name="__main__")
    return log, agv_calls, state


def _forward_then(n_forward, tail):
    def script(i):
        if i < n_forward:
            return {"left": (0.0, -1.0), "right": (0.0, -1.0)}
        return tail(i)
    return script


def _moves(calls):
    return [c for c in calls if c[2] == "move"]


def _no_agv_threads():
    return not any(t.name == "g1d-agv-sender" and t.is_alive() for t in threading.enumerate())


def test_without_flag_nothing_agv_is_constructed(monkeypatch, tmp_path):
    constructed = []
    monkeypatch.setattr(g1d_agv, "G1dAgvController", lambda *a, **k: constructed.append(1))
    log, calls, _ = _run(monkeypatch, tmp_path, _forward_then(10, lambda i: {"q": True}), agv=False)
    assert constructed == [] and calls == []
    assert "agv_client" not in log and "height_sub" not in log
    assert _no_agv_threads()


def test_q_after_forward_drive_zeroes_and_shuts_down_before_arm_cleanup(monkeypatch, tmp_path):
    log, calls, _ = _run(monkeypatch, tmp_path, _forward_then(12, lambda i: {"q": True}))
    # client/sender exist only once tracking starts; before the first tracking
    # sample the sender may only send zeros
    assert all(c[1] in ("arming", "tracking") for c in calls), calls[:3]
    assert all(c[3:] in ((0.0, 0.0, 0.0), (0.0,)) for c in calls if c[1] != "tracking")
    assert log.index("dds") < log.index("height_sub") < log.index("agv_client")
    assert any(c[3:] == (0.6, 0.0, 0.0) for c in _moves(calls))
    heights = [c for c in calls if c[2] == "height"]
    assert any(c[3] == 1.0 for c in heights)
    assert _moves(calls)[-1][3:] == (0.0, 0.0, 0.0) and heights[-1][3] == 0.0
    # zero burst + join happen before the arm writer is shut down
    last_agv = max(i for i, e in enumerate(log) if e == "agv_move")
    assert last_agv < log.index("arm_shutdown")
    assert "height_close" in log
    assert _no_agv_threads()


def test_stale_controller_during_tracking_zeroes_base_and_column(monkeypatch, tmp_path):
    def tail(i):
        if i < 24:
            return {"left": (0.0, -1.0), "right": (0.0, -1.0), "age": 1.0}
        return {"q": True}
    log, calls, _ = _run(monkeypatch, tmp_path, _forward_then(8, tail))
    stale = [c for c in calls if 14 <= c[0] < 24]
    assert stale, calls
    assert all(c[3:] in ((0.0, 0.0, 0.0), (0.0,)) for c in stale)


def test_exception_in_loop_zeroes_immediately(monkeypatch, tmp_path):
    log, calls, _ = _run(monkeypatch, tmp_path, _forward_then(10, lambda i: {"raise": RuntimeError("xr died")}))
    assert any(c[3:] == (0.6, 0.0, 0.0) for c in _moves(calls))
    assert _moves(calls)[-1][3:] == (0.0, 0.0, 0.0)
    assert [c for c in calls if c[2] == "height"][-1][3] == 0.0
    assert _no_agv_threads()


def test_status_log_records_raw_sticks_and_agv_command(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, _forward_then(50, lambda i: {"q": True}))
    time.sleep(0.2)
    records = [json.loads(l) for l in (tmp_path / "status.jsonl").read_text().splitlines() if l.strip()]
    agv = [r["locomotion"]["agv"] for r in records if isinstance(r.get("locomotion"), dict) and "agv" in r["locomotion"]]
    assert agv, records[:2]
    active = [a for a in agv if a.get("submit_reason") == "active"]
    assert active and active[-1]["raw_left_xy"] == [0.0, -1.0] and active[-1]["target"] == [0.6, 0.0, 1.0]
    assert "height" in active[-1]


def test_env_cap_is_applied(monkeypatch, tmp_path):
    _, calls, _ = _run(monkeypatch, tmp_path, _forward_then(10, lambda i: {"q": True}),
                       extra_env={"G1D_AGV_MAX_VX": "1.2"})
    assert any(c[3:] == (1.2, 0.0, 0.0) for c in _moves(calls))


@pytest.mark.parametrize("bad", ["1.6", "nan", "-1", "abc"])
def test_invalid_env_cap_refuses_before_anything_runs(monkeypatch, tmp_path, capsys, bad):
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, lambda i: {"q": True}, extra_env={"G1D_AGV_MAX_VX": bad})
    assert "G1D_AGV_MAX_VX" in capsys.readouterr().err


def test_flag_without_launcher_env_is_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("G1D_AGV", raising=False)
    monkeypatch.setattr(sys, "argv", BASE_ARGV + ["--g1d-agv"])
    with pytest.raises(SystemExit):
        _run_parse_only(monkeypatch)
    assert "launcher-controlled" in capsys.readouterr().err


def test_agv_zeroes_before_arm_cleanup_even_if_loop_raises_keyboardinterrupt(monkeypatch, tmp_path):
    log, calls, _ = _run(monkeypatch, tmp_path, _forward_then(10, lambda i: {"raise": KeyboardInterrupt()}))
    assert _moves(calls)[-1][3:] == (0.0, 0.0, 0.0)
    last_agv = max(i for i, e in enumerate(log) if e == "agv_move")
    assert last_agv < log.index("arm_shutdown") < log.index("ipc_stop") < log.index("tv_close")


def test_flag_rejected_for_non_g1d_arm(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["teleop_hand_and_arm.py", "--arm", "G1_29", "--g1d-agv"])
    with pytest.raises(SystemExit):
        _run_parse_only(monkeypatch)
    assert "--g1d-agv is only valid with --arm G1D" in capsys.readouterr().err


def _run_parse_only(monkeypatch):
    logger = types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None,
                                   warning=lambda *a, **k: None, error=lambda *a, **k: None)
    for name, mod in {
        "logging_mp": types.SimpleNamespace(basicConfig=lambda **k: None, getLogger=lambda n: logger, INFO=20),
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": types.SimpleNamespace(ChannelFactoryInitialize=None, ChannelPublisher=object),
        "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.std_msgs": types.ModuleType("unitree_sdk2py.idl.std_msgs"),
        "unitree_sdk2py.idl.std_msgs.msg": types.ModuleType("unitree_sdk2py.idl.std_msgs.msg"),
        "unitree_sdk2py.idl.std_msgs.msg.dds_": types.SimpleNamespace(String_=object),
        "televuer": types.SimpleNamespace(TeleVuerWrapper=object),
        "teleimager": types.ModuleType("teleimager"),
        "teleimager.image_client": types.SimpleNamespace(ImageClient=object, ZMQ_Requester=object),
        "sshkeyboard": types.SimpleNamespace(listen_keyboard=None, stop_listening=None),
        "teleop.robot_control.robot_arm": types.SimpleNamespace(
            G1_29_ArmController=object, G1_23_ArmController=object, H1_2_ArmController=object,
            H1_ArmController=object, H2_ArmController=object),
        "teleop.robot_control.robot_arm_ik": types.SimpleNamespace(
            G1_29_ArmIK=object, G1_23_ArmIK=object, H1_2_ArmIK=object, H1_ArmIK=object, H2_ArmIK=object),
        "teleop.utils.episode_writer": types.SimpleNamespace(EpisodeWriter=object),
        "teleop.utils.ipc": types.SimpleNamespace(IPC_Server=object),
        "teleop.utils.motion_switcher": types.SimpleNamespace(MotionSwitcher=object, LocoClientWrapper=object,
                                                              is_walk_fsm=None),
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    runpy.run_path(str(ROOT / "teleop" / "teleop_hand_and_arm.py"), run_name="__main__")
