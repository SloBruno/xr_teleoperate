"""Balance/IMU side-channel telemetry: fakes only, no DDS, no robot."""
import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.utils import balance_telemetry as bt
from teleop.utils.full_pose_telemetry import build_pose_record


def imu(r=0.0, p=0.0, y=0.0):
    return NS(rpy=[r, p, y], quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=[0.1, 0.2, 0.3],
              accelerometer=[0.0, 0.0, 9.8], temperature=40)


def motor(i):
    return NS(q=0.01 * i, dq=0.1, tau_est=float(i), temperature=[30, 31], mode=1)


def lowstate(pitch=0.0):
    return NS(tick=7, mode_machine=5, imu_state=imu(p=pitch), motor_state=[motor(i) for i in range(35)],
              wireless_remote=[0xAB, 1] + [0] * 38)


def sport(vx=0.0, vy=0.0, force=(0, 0, 0, 0)):
    return NS(mode=1, gait_type=0, progress=0.0, error_code=0, body_height=0.76,
              position=[1.0, 2.0, 0.76], velocity=[vx, vy, 0.0], yaw_speed=0.0,
              foot_force=list(force), foot_position_body=[0.0] * 12, foot_speed_body=[0.0] * 12,
              imu_state=imu())


def odom(x=0.0):
    v3 = lambda a, b, c: NS(x=a, y=b, z=c)
    return NS(header=NS(stamp=NS(sec=1, nanosec=5), frame_id="odom"), child_frame_id="pelvis",
              pose=NS(pose=NS(position=v3(x, 0.0, 0.7), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
              twist=NS(twist=NS(linear=v3(0.2, 0.0, 0.0), angular=v3(0.0, 0.0, 0.1))))


def test_extract_lowstate_legs_waist_arms_and_pelvis_imu():
    out = bt.extract_lowstate(lowstate(pitch=0.05))
    assert out["imu_pelvis"]["rpy"] == pytest.approx([0.0, 0.05, 0.0])
    assert len(out["legs"]["q"]) == 12 and len(out["waist"]["q"]) == 3
    assert out["waist"]["tau_est"] == [12.0, 13.0, 14.0]
    assert out["arms"]["tau_est"][0] == 15.0 and len(out["arms"]["tau_est"]) == 14
    assert out["legs"]["temperature"][0] == [30, 31]
    assert out["tick"] == 7 and out["mode_machine"] == 5
    assert out["wireless_remote_hex"].startswith("ab01") and len(out["wireless_remote_hex"]) == 80


def test_extract_sport_odom_config_use_real_idl_fields():
    s = bt.extract_sport(sport(vx=0.3, force=(10, 0, 0, 0)))
    assert s["velocity"] == [0.3, 0.0, 0.0] and s["foot_force"] == [10, 0, 0, 0]
    assert s["body_height"] == pytest.approx(0.76) and "imu" in s
    o = bt.extract_odom(odom(x=0.5))
    assert o["position"] == [0.5, 0.0, 0.7] and o["orientation_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert o["linear"] == [0.2, 0.0, 0.0] and o["child_frame_id"] == "pelvis"
    c = bt.extract_config(NS(name="imu_offset_json", content='{"imu":[0,2.5,0]}'))
    assert c == {"name": "imu_offset_json", "content": '{"imu":[0,2.5,0]}'}


def test_extractors_never_raise_on_garbage():
    for fn in (bt.extract_lowstate, bt.extract_sport, bt.extract_odom, bt.extract_imu, bt.extract_config):
        assert fn(object()) is None
    nan_imu = imu(p=float("nan"))
    assert bt.extract_imu(nan_imu)["rpy"] is None


def test_wireless_to_body_command_mapping():
    # robot reads [ly, -lx, -rx] -> [vx, vy, omega]
    cmd = bt.wireless_to_body_command([-0.1, 0.2, 0.3, 0.0, 0])
    assert cmd == pytest.approx([0.2, 0.1, -0.3])
    assert bt.wireless_to_body_command(None) is None


def test_loco_command_prefers_published_wireless_state():
    cmd, src = bt.loco_command_from({"published": [0.0, 0.2, 0.0, 0.0, 0]}, (0.5, 0.0, 0.0))
    assert cmd == pytest.approx([0.2, 0.0, 0.0]) and src == "wireless_published"
    cmd, src = bt.loco_command_from({"backend": "rpc"}, (0.1, 0.0, 0.0))
    assert cmd == [0.1, 0.0, 0.0] and src == "dispatched"
    assert bt.loco_command_from(None, None) == (None, "unavailable")


def test_wireless_bus_is_read_passively():
    t = [1.0]
    mon, subs = make_monitor(lambda: t[0])
    assert "rt/wirelesscontroller" in subs
    subs["rt/wirelesscontroller"].handler(NS(lx=0.0, ly=0.4, rx=0.0, ry=0.0, keys=0))
    snap = mon.snapshot(loco_command_source="wireless_published")
    assert snap["wirelesscontroller_bus"]["ly"] == pytest.approx(0.4)
    assert snap["loco_command_source"] == "wireless_published"


def test_derive_flags_uncommanded_motion_and_imu_diff():
    d = bt.derive_balance(
        pelvis_rpy=[0.0, 0.05, 0.0], torso_rpy=[0.01, 0.10, 0.0],
        body_velocity=[0.12, 0.05, 0.0], loco_command=[0.0, 0.0, 0.0])
    assert d["pelvis_minus_torso_rpy"] == pytest.approx([-0.01, -0.05, 0.0])
    assert d["horizontal_speed"] == pytest.approx(math.hypot(0.12, 0.05))
    assert d["command_zero"] is True and d["uncommanded_motion"] is True
    moving = bt.derive_balance(pelvis_rpy=None, torso_rpy=None,
                               body_velocity=[0.3, 0, 0], loco_command=[0.3, 0, 0])
    assert moving["uncommanded_motion"] is False and moving["pelvis_minus_torso_rpy"] is None
    unknown = bt.derive_balance(pelvis_rpy=None, torso_rpy=None, body_velocity=None, loco_command=None)
    assert unknown["uncommanded_motion"] is None


class FakeSub:
    def __init__(self, topic, handler):
        self.topic, self.handler, self.closed = topic, handler, False

    def Close(self):
        self.closed = True


def make_monitor(clock):
    subs = {}

    def factory(topic, type_key, handler):
        subs[topic] = FakeSub(topic, handler)
        return subs[topic]
    mon = bt.BalanceTelemetryMonitor(subscriber_factory=factory, clock=clock, rate_hz=50.0)
    mon.start()
    return mon, subs


def test_monitor_subscribes_passive_topics_and_snapshots_with_ages():
    t = [100.0]
    mon, subs = make_monitor(lambda: t[0])
    assert set(subs) >= {"rt/secondary_imu", "rt/sportmodestate", "rt/odommodestate",
                         "rt/state_estimator/odom_pelvis", "rt/state_estimator/odom_torso",
                         "rt/config_change_status"}
    assert "rt/lowstate" not in subs  # reuses the arm controller's lowstate reader
    mon.on_lowstate(lowstate(pitch=0.05))
    subs["rt/secondary_imu"].handler(imu(p=0.10))
    subs["rt/odommodestate"].handler(sport(vx=0.2))
    subs["rt/config_change_status"].handler(NS(name="imu_offset_json", content="{}"))
    t[0] = 100.05
    snap = mon.snapshot(loco_command=[0.0, 0.0, 0.0], com={"level": "ok", "dx_mm": 30.0, "dy_mm": 1.0})
    assert snap["schema_version"] == bt.SCHEMA_VERSION
    assert snap["sources"]["lowstate"]["age_ms"] == 50
    assert snap["sources"]["secondary_imu"]["age_ms"] == 50
    assert snap["sources"]["sportmodestate"]["age_ms"] is None
    assert snap["imu_pelvis"]["rpy"][1] == pytest.approx(0.05)
    assert snap["imu_torso"]["rpy"][1] == pytest.approx(0.10)
    assert snap["sport"]["odommodestate"]["velocity"][0] == pytest.approx(0.2)
    assert snap["derived"]["uncommanded_motion"] is True
    assert snap["com"]["dx_mm"] == 30.0
    assert snap["config_changes"][0]["name"] == "imu_offset_json"
    assert mon.snapshot()["config_changes"] == []  # events emitted once
    import json
    json.dumps(snap)


def test_monitor_decimates_callbacks_and_counts_errors():
    t = [10.0]
    mon, subs = make_monitor(lambda: t[0])
    for i in range(10):
        subs["rt/secondary_imu"].handler(imu())
        t[0] += 0.001
    assert mon.counters["secondary_imu"]["received"] == 10
    assert mon.counters["secondary_imu"]["stored"] == 1
    subs["rt/sportmodestate"].handler(object())  # garbage stored, fails only at snapshot
    snap = mon.snapshot()
    assert snap["sport"]["sportmodestate"] is None
    assert snap["counters"]["sportmodestate"]["extract_errors"] == 1


def test_monitor_step_events_from_foot_force():
    t = [1.0]
    mon, subs = make_monitor(lambda: t[0])
    subs["rt/odommodestate"].handler(sport(force=(50, 50, 0, 0)))
    first = mon.snapshot()
    assert first["derived"]["foot_contact"] == [True, True, False, False]
    assert first["derived"]["step_events"] == []
    t[0] += 0.1
    subs["rt/odommodestate"].handler(sport(force=(50, 0, 0, 0)))
    snap = mon.snapshot()
    assert snap["derived"]["step_events"] == [{"foot": 1, "contact": False}]


def test_monitor_subscriber_failure_is_counted_not_raised():
    def factory(topic, type_key, handler):
        raise RuntimeError("no dds")
    mon = bt.BalanceTelemetryMonitor(subscriber_factory=factory)
    mon.start()
    assert mon.sub_failures == len(bt.BALANCE_TOPICS)
    assert mon.snapshot()["imu_pelvis"] is None
    mon.close()


def test_callbacks_hold_lock_briefly_and_concurrent_snapshot_is_safe():
    mon, subs = make_monitor(lambda: 5.0)
    stop = threading.Event()

    def spam():
        while not stop.is_set():
            mon.on_lowstate(lowstate())
    th = threading.Thread(target=spam)
    th.start()
    try:
        for _ in range(50):
            mon.snapshot()
    finally:
        stop.set()
        th.join()


def test_create_from_env_is_opt_in():
    assert bt.create_from_env({}, subscriber_factory=lambda *a: None) is None
    mon = bt.create_from_env({"G1_BALANCE_TELEMETRY": "1", "G1_BALANCE_TELEMETRY_HZ": "25"},
                             subscriber_factory=lambda topic, tk, h: FakeSub(topic, h))
    assert mon is not None and mon.rate_hz == 25.0
    mon.close()


def test_snapshot_best_effort_never_raises_and_maps_inputs():
    assert bt.balance_snapshot_best_effort(None) is None

    class Boom:
        def snapshot(self, **kw):
            raise RuntimeError("x")
    warned = []
    assert bt.balance_snapshot_best_effort(Boom(), warn=warned.append) is None and warned
    mon, _ = make_monitor(lambda: 1.0)
    snap = bt.balance_snapshot_best_effort(
        mon, loco_backend={"published": [0.0, 0.0, 0.0, 0.0, 0]}, dispatched=(0, 0, 0),
        loco_raw={"left": [0.0, 0.1]}, com_status=NS(as_dict=lambda: {"level": "ok", "dx_mm": 5.0}),
        arm_commanded_q=[0.1] * 14)
    assert snap["loco_command"] == [0.0, 0.0, 0.0] and snap["com"]["dx_mm"] == 5.0
    assert snap["loco_raw"] == {"left": [0.0, 0.1]}
    import json
    import numpy as np
    snap = bt.balance_snapshot_best_effort(mon, loco_raw={"left": np.array([0.0, 0.3])},
                                           arm_commanded_q=np.zeros(14))
    json.dumps(snap)  # numpy inputs must be JSON-serializable by the worker
    assert snap["loco_raw"]["left"] == [0.0, 0.3]


def test_attach_lowstate_tap_reuses_arm_reader():
    mon, _ = make_monitor(lambda: 1.0)
    arm = NS(lowstate_buffer=object())
    assert bt.attach_lowstate_tap(mon, arm) and arm.lowstate_observer == mon.on_lowstate
    assert not bt.attach_lowstate_tap(mon, NS())
    assert not bt.attach_lowstate_tap(None, arm)


def test_pose_record_carries_balance_block_detached():
    balance = {"schema_version": 1, "derived": {"uncommanded_motion": True}}
    rec = build_pose_record(timestamp=1.0, timestamp_monotonic=2.0, lifecycle="tracking",
                            controller_sample_timestamp=1.9, left_wrist_pose=None, right_wrist_pose=None,
                            measured_arm_q=[0.0] * 14, commanded_arm_q=None, balance=balance)
    balance["derived"]["uncommanded_motion"] = False
    assert rec["balance"]["derived"]["uncommanded_motion"] is True
