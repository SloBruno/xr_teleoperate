import importlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teleop.utils import dex3_telemetry as dt
from teleop.utils.full_pose_telemetry import build_pose_record, PoseTelemetryJsonlSink


def ns(**kw):
    return types.SimpleNamespace(**kw)


def fake_motor(i, **over):
    base = dict(mode=0x11, q=0.1 * i, dq=0.01 * i, ddq=0.0, tau_est=0.5 + i,
                temperature=[30 + i, 31 + i], vol=24.0, sensor=[1, 2],
                motorstate=0, reserve=[0, 0, 0, 0])
    base.update(over)
    return ns(**base)


def fake_state(**over):
    press = [ns(pressure=[0.0] * 11 + [2.5], temperature=[25.0] * 12, lost=0, reserve=0)
             for _ in range(9)]
    base = dict(motor_state=[fake_motor(i) for i in range(7)], press_sensor_state=press,
                imu_state=None, power_v=24.1, power_a=0.7, system_v=12.0, device_v=5.0,
                error=[0, 0], reserve=[0, 0])
    base.update(over)
    return ns(**base)


def test_snapshot_extracts_all_joint_fields_and_hand_fields():
    snap = dt.extract_hand_snapshot(fake_state(), range(7))
    assert len(snap["joints"]) == 7
    j2 = snap["joints"][2]
    assert j2["tau_est"] == pytest.approx(2.5)
    assert j2["temperature"] == [32, 33]
    assert j2["dq"] == pytest.approx(0.02)
    assert j2["mode"] == 0x11
    assert snap["hand"]["power_v"] == pytest.approx(24.1)
    assert snap["hand"]["power_a"] == pytest.approx(0.7)
    assert snap["hand"]["pressure_max"][0] == pytest.approx(2.5)
    assert snap["hand"]["pressure_lost"] == [0] * 9


def test_snapshot_missing_and_nonfinite_become_null_without_raising():
    state = ns(motor_state=[fake_motor(0, tau_est=float("nan"), dq=float("inf"))])
    snap = dt.extract_hand_snapshot(state, range(7))
    assert snap["joints"][0]["tau_est"] is None
    assert snap["joints"][0]["dq"] is None
    assert snap["joints"][3]["q"] is None  # index out of range
    assert snap["hand"]["power_v"] is None
    assert snap["hand"]["pressure_max"] == []
    json.dumps(snap, allow_nan=False)


def test_snapshot_of_garbage_does_not_raise():
    snap = dt.extract_hand_snapshot(None, range(7))
    assert all(v is None for v in snap["joints"][0].values())


def test_published_command_snapshot():
    cmds = [ns(q=1.0 * i, dq=0.0, tau=0.0, kp=1.5, kd=0.2, mode=17) for i in range(7)]
    out = dt.extract_published_command(cmds, range(7))
    assert out["kp"] == [1.5] * 7 and out["kd"] == [0.2] * 7
    assert out["q"][3] == 3.0


def test_rate_estimator():
    rate = dt.RateEstimator(window=10)
    assert rate.rate_hz() is None
    for k in range(11):
        rate.add(k * 0.02)
    assert rate.rate_hz() == pytest.approx(50.0)
    assert rate.count == 11


def test_slow_gate_decimates_unchanged_slow_fields_but_keeps_fast():
    gate = dt.Dex3SlowFieldGate(slow_every=3)
    snap = dt.extract_hand_snapshot(fake_state(), range(7))
    outs = [gate.apply("left", snap) for _ in range(4)]
    assert outs[0]["slow_included"] is True
    assert "temperature" in outs[0]["joints"][0]
    assert outs[1]["slow_included"] is False
    assert "temperature" not in outs[1]["joints"][0]
    assert "power_v" not in outs[1]["hand"]
    assert outs[1]["joints"][0]["tau_est"] is not None  # fast field kept
    assert "q" in outs[1]["joints"][0] and "pressure_max" in outs[1]["hand"]
    assert outs[3]["slow_included"] is True  # periodic refresh


def test_slow_gate_emits_immediately_on_change_and_is_per_side():
    gate = dt.Dex3SlowFieldGate(slow_every=100)
    a = dt.extract_hand_snapshot(fake_state(), range(7))
    gate.apply("left", a)
    assert gate.apply("right", a)["slow_included"] is True
    assert gate.apply("left", a)["slow_included"] is False
    hot = fake_state(motor_state=[fake_motor(i, temperature=[60, 61]) for i in range(7)])
    changed = gate.apply("left", dt.extract_hand_snapshot(hot, range(7)))
    assert changed["slow_included"] is True
    err = fake_state(error=[4, 0])
    assert gate.apply("left", dt.extract_hand_snapshot(err, range(7)))["slow_included"] is True


def _record(**kw):
    return build_pose_record(
        timestamp=1.0, timestamp_monotonic=10.0, lifecycle="tracking",
        controller_sample_timestamp=9.99, left_wrist_pose=None, right_wrist_pose=None,
        measured_arm_q=None, commanded_arm_q=None, dex3_configured=True,
        dex3_measured_q=np.zeros(14), dex3_commanded_q=np.zeros(14),
        dex3_sample_metadata={s: {"state_valid": True, "state_timestamp": 9.9,
                                  "action_valid": True, "action_timestamp": 9.95}
                              for s in ("left", "right")},
        now=10.0, **kw)


def test_record_without_extended_is_unchanged_backward_compatible():
    record = _record()
    assert "extended" not in record["dex3"]["left"]
    assert record["schema_version"] == 1


def test_record_attaches_extended_with_age_rate_and_version():
    snap = dt.extract_hand_snapshot(fake_state(), range(7))
    ext = {"left": {"state": snap, "state_timestamp": 9.9, "state_count": 40, "rate_hz": 99.5,
                    "published_command": dt.extract_published_command(
                        [ns(q=0, dq=0, tau=0, kp=1.5, kd=0.2, mode=17)] * 7, range(7)),
                    "command_timestamp": 9.95, "command_count": 12},
           "right": None}
    record = _record(dex3_extended=ext)
    left = record["dex3"]["left"]["extended"]
    assert left["state_age_ms"] == 100
    assert left["command_age_ms"] == 50
    assert left["rate_hz"] == 99.5
    assert left["state"]["joints"][1]["tau_est"] == pytest.approx(1.5)
    assert left["published_command"]["kp"] == [1.5] * 7
    assert record["dex3"]["right"]["extended"] is None
    assert record["dex3"]["extended_schema_version"] == 1
    json.dumps(record, allow_nan=False)


def test_record_extended_nonfinite_is_null_and_garbage_does_not_raise():
    ext = {"left": {"state": {"joints": [{"tau_est": float("nan")}]}, "state_timestamp": float("nan"),
                    "rate_hz": float("inf")}, "right": "garbage"}
    record = _record(dex3_extended=ext)
    left = record["dex3"]["left"]["extended"]
    assert left["state"]["joints"][0]["tau_est"] is None
    assert left["state_age_ms"] is None and left["rate_hz"] is None
    assert record["dex3"]["right"]["extended"] is None
    json.dumps(record, allow_nan=False)


def test_extended_record_roundtrips_through_async_sink(tmp_path):
    sink = PoseTelemetryJsonlSink(tmp_path)
    snap = dt.extract_hand_snapshot(fake_state(), range(7))
    assert sink.emit(_record(dex3_extended={"left": {"state": snap, "state_timestamp": 9.9}, "right": None}))
    sink.close()
    line = json.loads(sink.path.read_text().splitlines()[0])
    assert line["dex3"]["left"]["extended"]["state"]["joints"][0]["tau_est"] == pytest.approx(0.5)


# ---- controller integration (fake SDK) ----

@pytest.fixture
def module(monkeypatch):
    sdk_channel = types.ModuleType("unitree_sdk2py.core.channel")
    sdk_channel.ChannelPublisher = object
    sdk_channel.ChannelSubscriber = object
    sdk_channel.ChannelFactoryInitialize = object
    sdk_hand = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    sdk_hand.HandCmd_ = object
    sdk_hand.HandState_ = object
    sdk_default = types.ModuleType("unitree_sdk2py.idl.default")
    sdk_default.unitree_hg_msg_dds__HandCmd_ = object
    sdk_default.unitree_go_msg_dds__MotorCmd_ = object
    go = types.ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_")
    go.MotorCmds_ = object
    go.MotorStates_ = object
    for name, mod in {
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": sdk_channel,
        "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.unitree_hg": types.ModuleType("unitree_sdk2py.idl.unitree_hg"),
        "unitree_sdk2py.idl.unitree_hg.msg": types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg"),
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": sdk_hand,
        "unitree_sdk2py.idl.default": sdk_default,
        "unitree_sdk2py.idl.unitree_go": types.ModuleType("unitree_sdk2py.idl.unitree_go"),
        "unitree_sdk2py.idl.unitree_go.msg": types.ModuleType("unitree_sdk2py.idl.unitree_go.msg"),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": go,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    retargeting = types.ModuleType("teleop.robot_control.hand_retargeting")
    retargeting.HandRetargeting = object
    retargeting.HandType = object
    monkeypatch.setitem(sys.modules, "teleop.robot_control.hand_retargeting", retargeting)
    lm = types.ModuleType("logging_mp")
    lm.getLogger = lambda name: types.SimpleNamespace(
        warning=lambda *a, **k: None, info=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "logging_mp", lm)
    sys.modules.pop("teleop.robot_control.robot_hand_unitree", None)
    return importlib.import_module("teleop.robot_control.robot_hand_unitree")


def bare(module):
    c = module.Dex3_1_Controller.__new__(module.Dex3_1_Controller)
    import threading
    c._telemetry_lock = threading.Lock()
    return c


def test_record_hand_state_stores_extended_sample_and_rate(module):
    c = bare(module)
    for k in range(5):
        c._record_extended_state("left", fake_state(), module.Dex3_1_Left_JointIndex, 100.0 + k * 0.01)
    samples = c.get_extended_samples()
    left = samples["left"]
    assert left["state"]["joints"][2]["tau_est"] == pytest.approx(2.5)
    assert left["state_count"] == 5
    assert left["rate_hz"] == pytest.approx(100.0)
    assert left["state_timestamp"] == pytest.approx(100.04)
    assert samples["right"] is None


def test_record_hand_state_failure_is_counted_not_raised(module, monkeypatch):
    c = bare(module)
    monkeypatch.setattr(module, "extract_hand_snapshot", lambda *a, **k: 1 / 0)
    c._record_extended_state("left", fake_state(), module.Dex3_1_Left_JointIndex, 1.0)
    assert c.extended_telemetry_failure_count == 1
    assert c.get_extended_samples()["left"] is None


def test_get_extended_samples_on_bare_controller_is_safe(module):
    c = bare(module)
    assert c.get_extended_samples() == {"left": None, "right": None}


def test_ctrl_dual_hand_records_published_command_without_changing_it(module):
    c = bare(module)
    written = []

    class Pub:
        def Write(self, m):
            written.append([(x.q, x.kp, x.kd) for x in m.motor_cmd])

    def msg():
        return ns(motor_cmd=[ns(q=0.0, dq=0.0, tau=0.0, kp=1.5, kd=0.2, mode=17) for _ in range(7)])

    c.left_msg, c.right_msg = msg(), msg()
    c.LeftHandCmb_publisher = c.RightHandCmb_publisher = Pub()
    import time
    now = time.monotonic()
    c.ctrl_dual_hand(np.arange(7) * 0.1, np.arange(7) * -0.1, now, now)
    assert written[0][2] == (pytest.approx(0.2), 1.5, 0.2)
    cmd = c.get_extended_samples()
    assert cmd["left"]["state"] is None  # no state yet; command tracked separately
    assert cmd["left"]["command_count"] == 1
    assert c._published_command["left"]["published_command"]["q"][2] == pytest.approx(0.2)
    assert c._published_command["right"]["published_command"]["kp"] == [1.5] * 7


def test_published_command_failure_never_blocks_publication(module, monkeypatch):
    c = bare(module)
    written = []
    c.left_msg = ns(motor_cmd=[ns(q=0.0) for _ in range(7)])
    c.right_msg = ns(motor_cmd=[ns(q=0.0) for _ in range(7)])
    c.LeftHandCmb_publisher = c.RightHandCmb_publisher = ns(Write=lambda m: written.append(m))
    monkeypatch.setattr(module, "extract_published_command", lambda *a, **k: 1 / 0)
    import time
    now = time.monotonic()
    c.ctrl_dual_hand(np.zeros(7), np.zeros(7), now, now)
    assert len(written) == 2
    assert c.extended_telemetry_failure_count == 2  # one per side


def test_control_constants_and_publication_path_unchanged(module):
    assert module.Dex3_Left_Closed_Pose[1] == 1.05 and module.Dex3_Left_Closed_Pose[2] == 1.75
    assert module.Dex3_Right_Closed_Pose[1] == -1.05 and module.Dex3_Right_Closed_Pose[2] == -1.75
    src = Path(module.__file__).read_text()
    assert "kp = 1.5" in src and "kd = 0.2" in src


def test_extended_payload_builder_applies_gate_and_handles_none():
    gate = dt.Dex3SlowFieldGate(slow_every=5)
    snap = dt.extract_hand_snapshot(fake_state(), range(7))
    samples = {"left": {"state": snap, "state_timestamp": 1.0}, "right": None}
    out = dt.build_extended_payload(samples, gate)
    assert out["right"] is None
    assert out["left"]["state"]["slow_included"] is True
    out2 = dt.build_extended_payload(samples, gate)
    assert out2["left"]["state"]["slow_included"] is False
    assert dt.build_extended_payload(None, gate) is None
    assert dt.build_extended_payload({"left": {"state": 1 / 1}}, gate) is not None


def test_publication_survives_controller_without_telemetry_state(module):
    c = module.Dex3_1_Controller.__new__(module.Dex3_1_Controller)  # no lock at all
    written = []
    c.left_msg = ns(motor_cmd=[ns(q=0.0) for _ in range(7)])
    c.right_msg = ns(motor_cmd=[ns(q=0.0) for _ in range(7)])
    c.LeftHandCmb_publisher = c.RightHandCmb_publisher = ns(Write=lambda m: written.append(m))
    import time
    now = time.monotonic()
    c.ctrl_dual_hand(np.zeros(7), np.zeros(7), now, now)
    assert len(written) == 2
