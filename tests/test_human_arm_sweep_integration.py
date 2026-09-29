"""Launcher-level integration of the Phase-2 human-arm sweep (no hardware)."""

import importlib
import sys
import time
import types

import numpy as np
import pytest

from teleop.utils import human_arm_calibration as hac
from teleop.utils.controller_wrist_calibration import ControllerWristCalibrator
from tests.test_human_arm_calibration import Operator, W0, calibrated, pose


STUB_NAMES = (
    "logging_mp", "unitree_sdk2py", "unitree_sdk2py.core", "unitree_sdk2py.core.channel",
    "televuer", "teleimager", "teleimager.image_client", "sshkeyboard",
    "teleop.robot_control.robot_arm", "teleop.robot_control.robot_arm_ik",
    "teleop.utils.episode_writer", "teleop.utils.ipc", "teleop.utils.motion_switcher",
    "unitree_sdk2py.idl", "unitree_sdk2py.idl.std_msgs", "unitree_sdk2py.idl.std_msgs.msg",
    "unitree_sdk2py.idl.std_msgs.msg.dds_",
)


@pytest.fixture
def teleop(monkeypatch):
    messages = []
    for name in STUB_NAMES:
        module = types.ModuleType(name)
        module.__getattr__ = lambda attr: object
        monkeypatch.setitem(sys.modules, name, module)
    logger = types.SimpleNamespace(
        warning=lambda m: messages.append(("warning", m)), info=lambda m: messages.append(("info", m)),
        debug=lambda m: None, error=lambda m: messages.append(("error", m)))
    sys.modules["logging_mp"].basicConfig = lambda **kwargs: None
    sys.modules["logging_mp"].getLogger = lambda name: logger
    sys.modules["logging_mp"].INFO = 20
    sys.modules["sshkeyboard"].listen_keyboard = object
    sys.modules["sshkeyboard"].stop_listening = object
    module = importlib.reload(importlib.import_module("teleop.teleop_hand_and_arm"))
    module.START = False
    module.STOP = False
    module.READY = True
    module.PREPARATION_COMPLETE = True
    module.HUMAN_SWEEP_REQUESTED = False
    module.HUMAN_SWEEP_ACTIVE = False
    module.LIFECYCLE_EVENTS[:] = []
    module.arm_calibration = hac.HumanCalibratedWristCalibrator(ControllerWristCalibrator())
    module.messages = messages
    yield module
    sys.modules.pop("teleop.teleop_hand_and_arm", None)


class Sink:
    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(record)
        return True


class ForbiddenArm:
    """Any command-path call is a test failure."""

    def __getattr__(self, name):
        raise AssertionError(f"arm controller used: {name}")


def tele(left, right, timestamp, **buttons):
    return types.SimpleNamespace(
        left_wrist_pose=pose(left), right_wrist_pose=pose(right), controller_sample_timestamp=timestamp,
        left_ctrl_aButton=buttons.get("x", False), right_ctrl_aButton=False, right_ctrl_bButton=False)


def run_sweep(module, operator, sink, *, start=None, stop_at=None):
    rng = np.random.default_rng(3)
    left, right = operator.sweep("left", rng, count=100), operator.sweep("right", rng, count=100)
    sweep = hac.HumanArmSweep(duration_s=0.3)
    result = None
    base = time.monotonic()
    for index in range(100):
        now = base + index * 0.004
        if stop_at is not None and index == stop_at:
            with module.LIFECYCLE_LOCK:
                module._request_stop_locked()
        result = module.service_human_arm_sweep(
            sweep, module.arm_calibration, tele(left[index], right[index], now - 0.001), now, sink) or result
    return result


def test_c_key_requests_sweep_only_in_prepared_state(teleop):
    teleop.on_press("c")
    assert teleop.HUMAN_SWEEP_REQUESTED
    teleop.HUMAN_SWEEP_REQUESTED = False
    teleop.START = True
    teleop.on_press("c")
    assert not teleop.HUMAN_SWEEP_REQUESTED
    assert "human_sweep_refused" in teleop.LIFECYCLE_EVENTS
    teleop.START = False
    teleop.PREPARATION_COMPLETE = False
    teleop.on_press("c")
    assert not teleop.HUMAN_SWEEP_REQUESTED


def test_left_x_rising_edge_requires_fresh_sample(teleop):
    stale = tele([0, 0, 0], [0, 0, 0], 0.0, x=True)
    assert teleop.poll_human_sweep_button(stale, False) is False
    assert not teleop.HUMAN_SWEEP_REQUESTED
    fresh = tele([0, 0, 0], [0, 0, 0], time.monotonic(), x=True)
    assert teleop.poll_human_sweep_button(fresh, True) is True  # held, no edge
    assert not teleop.HUMAN_SWEEP_REQUESTED
    assert teleop.poll_human_sweep_button(fresh, False) is True
    assert teleop.HUMAN_SWEEP_REQUESTED


def test_sweep_installs_calibration_emits_telemetry_and_never_touches_arm(teleop):
    sink = Sink()
    teleop.on_press("c")
    result = run_sweep(teleop, Operator(), sink)
    assert result is not None and result.accepted, result and result.reason
    assert teleop.arm_calibration.human_calibration is result
    assert not teleop.HUMAN_SWEEP_ACTIVE
    record = next(r for r in sink.records if r["event"] == "human_arm_calibration")
    side = record["human_arm_calibration"]["sides"]["left"]
    assert side["accepted"] and side["human_arm_length_m"] > 0.4 and side["translation_scale"] > 0.3
    assert side["shoulder_estimate_m"] is not None and side["fit_rms_m"] < 0.015
    assert any("ACCEPTED" in m for level, m in teleop.messages)


def test_start_is_refused_while_sweep_runs(teleop):
    teleop.on_press("c")
    teleop.on_press("r")
    assert teleop.START is False
    assert "start_refused_human_sweep_active" in teleop.LIFECYCLE_EVENTS


def test_q_wins_during_sweep_and_nothing_is_installed(teleop):
    sink = Sink()
    teleop.on_press("c")
    result = run_sweep(teleop, Operator(), sink, stop_at=10)
    assert teleop.STOP is True
    assert result is not None and not result.accepted
    assert result.reason == "aborted_tracking_active"
    assert teleop.arm_calibration.human_calibration is None
    assert not teleop.HUMAN_SWEEP_ACTIVE


def test_sweep_never_starts_during_tracking(teleop):
    teleop.HUMAN_SWEEP_REQUESTED = True  # e.g. raced with r
    teleop.START = True
    sweep = hac.HumanArmSweep(duration_s=0.01)
    now = time.monotonic()
    assert teleop.service_human_arm_sweep(sweep, teleop.arm_calibration, tele([0, 0, 0], [0, 0, 0], now), now, Sink()) is None
    assert not sweep.active and not teleop.HUMAN_SWEEP_ACTIVE and not teleop.HUMAN_SWEEP_REQUESTED


def test_failed_sweep_keeps_fixed_scale_and_logs(teleop):
    sink = Sink()
    teleop.on_press("c")
    result = run_sweep(teleop, Operator(length=0.30), sink)
    assert result is not None and not result.accepted
    assert teleop.arm_calibration.human_calibration is None
    assert any("REJECTED" in m for level, m in teleop.messages)
    assert sink.records[-1]["human_arm_calibration"]["accepted"] is False


class FakeArmIK:
    def forward_kinematics(self, q):
        return (pose(W0["left"]), pose(W0["right"]))


class FakeArmState:
    def __init__(self):
        self.calls = []

    def get_current_dual_arm_q(self):
        self.calls.append("q")
        return np.zeros(14)

    def __getattr__(self, name):
        raise AssertionError(f"arm command path used: {name}")


def test_l_pose_gate_refuses_r_then_accepts_when_operator_in_l_pose(teleop):
    operator = Operator()
    calibration = calibrated(operator)
    teleop.arm_calibration.set_human_calibration(calibration)
    sink = Sink()
    arm = FakeArmState()
    teleop.on_press("r")
    assert teleop.START
    far = tele(operator.straight("left", (1, 0, 0)), operator.straight("right", (1, 0, 0)), time.monotonic())
    assert not teleop.enforce_l_pose_start_gate(teleop.arm_calibration, far, arm, FakeArmIK(), sink)
    assert teleop.START is False and teleop.STOP is False
    assert "start_refused_l_pose" in teleop.LIFECYCLE_EVENTS
    assert any("START REFUSED" in m for level, m in teleop.messages)
    gate = sink.records[-1]["l_pose_start_gate"]
    assert gate["accepted"] is False and gate["errors_m"]["left"] > 0.05

    teleop.on_press("r")
    near = tele(operator.l_pose_controller(calibration, "left"), operator.l_pose_controller(calibration, "right"), time.monotonic())
    assert teleop.enforce_l_pose_start_gate(teleop.arm_calibration, near, arm, FakeArmIK(), sink)
    assert teleop.START is True
    assert sink.records[-1]["l_pose_start_gate"]["accepted"] is True


def test_without_human_calibration_r_is_accepted_unchanged(teleop):
    arm = ForbiddenArm()
    teleop.on_press("r")
    assert teleop.enforce_l_pose_start_gate(teleop.arm_calibration, None, arm, None, Sink())
    assert teleop.START is True


def test_q_after_refused_r_still_stops(teleop):
    operator = Operator()
    teleop.arm_calibration.set_human_calibration(calibrated(operator))
    teleop.on_press("r")
    far = tele(operator.straight("left", (1, 0, 0)), operator.straight("right", (1, 0, 0)), time.monotonic())
    teleop.enforce_l_pose_start_gate(teleop.arm_calibration, far, FakeArmState(), FakeArmIK(), Sink())
    teleop.on_press("q")
    assert teleop.STOP is True and teleop.START is False


def test_prearm_integration_source_contract():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py").read_text()
    prearm = source[source.index("        READY = True"):source.index("# main loop. robot start to follow VR user's motion")]
    tracking = source[source.index("# main loop. robot start to follow VR user's motion"):]
    assert "service_human_arm_sweep(" in prearm
    assert "enforce_l_pose_start_gate(" in prearm
    assert prearm.index("enforce_l_pose_start_gate(") < prearm.index("arm_calibration.calibrate(")
    # No sweep servicing or button polling in the tracking loop.
    assert "service_human_arm_sweep" not in tracking
    assert "poll_human_sweep_button" not in tracking
    helper = source[source.index("def service_human_arm_sweep"):source.index("def enforce_l_pose_start_gate")]
    for forbidden in ("arm_ctrl", "arm_ik", "ctrl_dual_arm", "publish"):
        assert forbidden not in helper


def test_calibration_and_gate_telemetry_serialize_through_real_sink(tmp_path):
    import json
    from teleop.utils.full_pose_telemetry import PoseTelemetryJsonlSink

    operator = Operator()
    calibration = calibrated(operator)
    failed = calibrated(Operator(length=0.30))
    decision = hac.evaluate_l_pose_gate(calibration, (np.eye(4), np.eye(4)), (pose(W0["left"]), pose(W0["right"])))
    sink = PoseTelemetryJsonlSink(tmp_path)
    for payload in (calibration.telemetry(), failed.telemetry(), decision.telemetry()):
        assert sink.emit({"event": "x", "payload": payload})
    sink.close()
    lines = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert len(lines) == 3
    assert lines[0]["payload"]["sides"]["right"]["accepted"] is True
    assert lines[1]["payload"]["accepted"] is False
    assert lines[2]["payload"]["accepted"] is False
