"""G1D teach/replay DDS adapter: passive pre-start, single writer, slot parity."""
import ast
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from teleop.robot_control import g1d_teach_replay_io as gio  # noqa: E402
from teleop.utils import teach_replay as tr  # noqa: E402


class FakeSubscriber:
    created = []

    def __init__(self, topic, msg_type):
        self.topic = topic
        self.handler = None
        FakeSubscriber.created.append(self)

    def Init(self, handler=None, queueLen=0):
        self.handler = handler


class FakePublisher:
    created = []

    def __init__(self, topic, msg_type):
        self.topic = topic
        self.inited = False
        self.written = []
        FakePublisher.created.append(self)

    def Init(self):
        self.inited = True

    def Write(self, msg):
        self.written.append(msg)


def _motor(q, dq=0.0):
    return SimpleNamespace(q=q, dq=dq, tau_est=0.0, temperature=[30, 30], mode=1, motorstate=0)


def _lowstate(q=0.1):
    return SimpleNamespace(mode_machine=7, motor_state=[_motor(q + 0.01 * i) for i in range(35)])


def _handstate(q):
    return SimpleNamespace(motor_state=[_motor(q + 0.01 * i) for i in range(7)])


class FakeCrc:
    def Crc(self, msg):
        return 1234


def _msg(n):
    return SimpleNamespace(mode_pr=None, mode_machine=None, crc=None,
                           motor_cmd=[SimpleNamespace(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0)
                                      for _ in range(n)])


def make_io(sides=("left", "right"), clock=time.monotonic):
    FakeSubscriber.created.clear()
    FakePublisher.created.clear()
    return gio.G1DTeachReplayIO(
        hand_sides=sides,
        publisher_cls=FakePublisher,
        subscriber_cls=FakeSubscriber,
        lowcmd_factory=lambda: _msg(35),
        handcmd_factory=lambda: _msg(7),
        crc_factory=FakeCrc,
        lowcmd_type=object, lowstate_type=object, handcmd_type=object, handstate_type=object,
        clock=clock,
    )


def _feed(io, low=0.1, left=0.2, right=-0.2):
    by_topic = {s.topic: s for s in FakeSubscriber.created}
    by_topic["rt/lowstate"].handler(_lowstate(low))
    for side, q in (("left", left), ("right", right)):
        if f"rt/dex3/{side}/state" in by_topic:
            by_topic[f"rt/dex3/{side}/state"].handler(_handstate(q))


def test_construction_is_passive_subscribers_only():
    io = make_io()
    assert FakePublisher.created == []
    topics = sorted(s.topic for s in FakeSubscriber.created)
    assert topics == ["rt/dex3/left/state", "rt/dex3/right/state", "rt/lowstate"]
    assert all(s.handler is not None for s in FakeSubscriber.created), "per-topic callbacks, no blocking Read loop"
    assert io.arm_state() is None and io.hand_state("left") is None


def test_state_snapshots_are_timestamped_per_side_and_independent():
    t = [10.0]
    io = make_io(clock=lambda: t[0])
    by_topic = {s.topic: s for s in FakeSubscriber.created}
    by_topic["rt/lowstate"].handler(_lowstate(0.1))
    t[0] = 11.0
    by_topic["rt/dex3/left/state"].handler(_handstate(0.3))
    arm = io.arm_state()
    assert arm.timestamp == 10.0 and arm.q.shape == (14,)
    np.testing.assert_allclose(arm.q[0], 0.1 + 0.01 * 15)        # slot 15 = left shoulder pitch
    left = io.hand_state("left")
    assert left.timestamp == 11.0 and left.q.shape == (7,)
    assert left.protection["timestamp"] == 11.0 and len(left.protection["q"]) == 7
    assert io.hand_state("right") is None


def test_activate_builds_publishers_only_for_lowcmd_and_enabled_hands():
    io = make_io(sides=("right",))
    _feed(io)
    io.activate(("right",))
    assert sorted(p.topic for p in FakePublisher.created) == ["rt/dex3/right/cmd", "rt/lowcmd"]
    assert all(p.inited for p in FakePublisher.created)
    assert all(p.written == [] for p in FakePublisher.created), "activation alone must not write"
    with pytest.raises(RuntimeError):
        io.write_hand("left", tr.HandCommand(q=np.zeros(7), kp=np.zeros(7), kd=np.zeros(7)))


def test_activate_refuses_without_fresh_lowstate():
    t = [100.0]
    io = make_io(clock=lambda: t[0])
    with pytest.raises(RuntimeError):
        io.activate(("left", "right"))
    _feed(io)
    t[0] = 101.0
    with pytest.raises(RuntimeError):
        io.activate(("left", "right"))
    assert FakePublisher.created == []


def test_write_arm_fills_arm_slots_and_keeps_g1d_passive_slots():
    io = make_io()
    _feed(io, low=0.1)
    io.activate(("left", "right"))
    cmd = tr.ArmCommand(q=np.arange(14) * 0.1, dq=np.zeros(14), kp=np.full(14, 7.0),
                        kd=np.full(14, 0.5), tau=np.arange(14) * 0.01)
    io.write_arm(cmd)
    pub = next(p for p in FakePublisher.created if p.topic == "rt/lowcmd")
    msg = pub.written[-1]
    assert msg.crc == 1234 and msg.mode_pr == 0 and msg.mode_machine == 7
    for i, slot in enumerate(range(15, 29)):
        m = msg.motor_cmd[slot]
        assert (m.q, m.kp, m.kd, m.tau, m.mode) == pytest.approx((0.1 * i, 7.0, 0.5, 0.01 * i, 1))
    for waist in (12, 13, 14):
        assert (msg.motor_cmd[waist].kp, msg.motor_cmd[waist].kd, msg.motor_cmd[waist].tau) == (0.0, 0.0, 0.0)


def test_non_arm_slots_match_the_deployed_g1d_controller_configuration():
    """Same passive/leg slot gains as G1D_ArmController._configure_joint_command."""
    from teleop.robot_control import robot_arm

    io = make_io()
    _feed(io, low=0.1)
    io.activate(("left", "right"))
    mine = io._lowcmd_msg

    shim = SimpleNamespace(msg=_msg(35), command_waist=False, kp_high=300.0, kd_high=3.0,
                           kp_low=80.0, kd_low=3.0, kp_wrist=40.0, kd_wrist=1.5)
    shim._Is_wrist_motor = lambda i: robot_arm.G1_29_ArmController._Is_wrist_motor(shim, i)
    shim._Is_weak_motor = lambda i: robot_arm.G1_29_ArmController._Is_weak_motor(shim, i)
    arm_ids = set(int(m) for m in robot_arm.G1_29_JointArmIndex)
    low = _lowstate(0.1)
    for slot in range(29):
        robot_arm.G1_29_ArmController._configure_joint_command(shim, slot, low.motor_state[slot].q, arm_ids)
        if slot in arm_ids:
            continue
        a, b = mine.motor_cmd[slot], shim.msg.motor_cmd[slot]
        assert (a.mode, a.q, a.kp, a.kd, a.tau) == (b.mode, b.q, b.kp, b.kd, b.tau), slot


def test_tracking_gains_match_the_g1_29_arm_controller_source():
    src = (ROOT / "teleop/robot_control/robot_arm.py").read_text()
    init = src.split("class G1_29_ArmController", 1)[1].split("def activate", 1)[0]
    vals = dict(re.findall(r"self\.(k[pd]_(?:low|wrist)) = ([0-9.]+)", init))
    cfg = tr.TeachReplayConfig()
    expect_kp = [float(vals["kp_low"])] * 4 + [float(vals["kp_wrist"])] * 3
    expect_kd = [float(vals["kd_low"])] * 4 + [float(vals["kd_wrist"])] * 3
    np.testing.assert_allclose(cfg.track_arm_kp, expect_kp * 2)
    np.testing.assert_allclose(cfg.track_arm_kd, expect_kd * 2)


def test_dex3_gains_and_mode_byte_match_the_existing_dex3_controller():
    from teleop.robot_control import robot_hand_unitree as rhu

    cfg = tr.TeachReplayConfig()
    assert cfg.track_hand_kp == rhu.Dex3_Kp and cfg.track_hand_kd == rhu.Dex3_Kd
    for joint in range(7):
        assert gio.dex3_mode_byte(joint) == rhu.Dex3_1_Controller._RIS_Mode(id=joint, status=0x01)._mode_to_uint8()


def test_write_hand_sets_q_kp_kd_and_zero_tau():
    io = make_io()
    _feed(io)
    io.activate(("left", "right"))
    io.write_hand("left", tr.HandCommand(q=np.arange(7) * 0.1, kp=np.zeros(7), kd=np.full(7, 0.05)))
    pub = next(p for p in FakePublisher.created if p.topic == "rt/dex3/left/cmd")
    msg = pub.written[-1]
    for j in range(7):
        m = msg.motor_cmd[j]
        assert (m.q, m.kp, m.kd, m.tau, m.dq) == pytest.approx((0.1 * j, 0.0, 0.05, 0.0, 0.0))
        assert m.mode == gio.dex3_mode_byte(j)


def test_release_stops_all_writes():
    io = make_io()
    _feed(io)
    io.activate(("left", "right"))
    io.release()
    io.write_arm(tr.ArmCommand(q=np.zeros(14), dq=np.zeros(14), kp=np.zeros(14), kd=np.zeros(14), tau=np.zeros(14)))
    io.write_hand("left", tr.HandCommand(q=np.zeros(7), kp=np.zeros(7), kd=np.zeros(7)))
    assert all(p.written == [] for p in FakePublisher.created)


def test_write_rejects_non_finite_or_misshaped_commands():
    io = make_io()
    _feed(io)
    io.activate(("left", "right"))
    bad = np.zeros(14)
    bad[2] = np.nan
    with pytest.raises(ValueError):
        io.write_arm(tr.ArmCommand(q=bad, dq=np.zeros(14), kp=np.zeros(14), kd=np.zeros(14), tau=np.zeros(14)))
    with pytest.raises(ValueError):
        io.write_hand("left", tr.HandCommand(q=np.zeros(6), kp=np.zeros(7), kd=np.zeros(7)))
    assert all(p.written == [] for p in FakePublisher.created)


def test_hand_limits_match_urdf_per_side():
    text = (ROOT / "assets/g1/g1_body29_hand14.urdf").read_text()
    names = {
        "left": ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
        "right": ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"],
    }
    for side, joints in names.items():
        lo, hi = tr.DEX3_LIMITS[side]
        for j, name in enumerate(joints):
            block = text.split(f'<joint name="{side}_hand_{name}_joint"', 1)[1].split("</joint>", 1)[0]
            lower = float(re.search(r'lower="([-0-9.]+)"', block).group(1))
            upper = float(re.search(r'upper="([-0-9.]+)"', block).group(1))
            assert (lo[j], hi[j]) == pytest.approx((lower, upper), abs=1e-6), (side, name)


def test_io_module_does_not_import_the_teleop_entrypoint():
    tree = ast.parse((ROOT / "teleop/robot_control/g1d_teach_replay_io.py").read_text())
    names = {getattr(n, "module", None) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(name and "teleop_hand_and_arm" in name for name in names)


def test_pre_activate_hook_runs_before_any_publisher_and_failure_refuses():
    order = []

    def hook():
        order.append(("hook", len(FakePublisher.created)))

    io = make_io()
    io._pre_activate = hook
    _feed(io)
    io.activate(("left", "right"))
    assert order == [("hook", 0)]

    def failing():
        raise RuntimeError("debug mode not entered")

    io2 = make_io()
    io2._pre_activate = failing
    _feed(io2)
    with pytest.raises(RuntimeError):
        io2.activate(("left", "right"))
    assert FakePublisher.created == []


def test_activation_rechecks_lowstate_after_a_slow_pre_activate_hook():
    t = [50.0]
    io = make_io(clock=lambda: t[0])
    _feed(io)

    def slow():
        t[0] += 2.0

    io._pre_activate = slow
    with pytest.raises(RuntimeError):
        io.activate(("left", "right"))
    assert FakePublisher.created == []


def test_end_to_end_teach_with_real_session_and_real_g1_model(tmp_path):
    pytest.importorskip("pinocchio")
    from teleop import teach_replay_g1d as entry

    gravity_fn, lower, upper, effort = entry._load_model()
    assert lower.shape == upper.shape == effort.shape == (14,)
    assert np.all(lower < upper) and np.all(effort > 0)

    t = [200.0]
    io = make_io(clock=lambda: t[0])
    _feed(io, low=0.05)
    session = tr.TeachReplaySession(io, gravity_fn=gravity_fn, arm_lower=lower, arm_upper=upper,
                                    arm_effort=effort, teach_dir=tmp_path, clock=lambda: t[0])
    session.request_teach()
    t[0] += 0.005
    _feed(io, low=0.05)
    session.tick()
    assert session.mode == tr.Mode.TEACH
    pub = next(p for p in FakePublisher.created if p.topic == "rt/lowcmd")
    msg = pub.written[0]
    q = np.array([msg.motor_cmd[s].q for s in range(15, 29)])
    tau = np.array([msg.motor_cmd[s].tau for s in range(15, 29)])
    np.testing.assert_allclose(q, io.arm_state().q)
    expected = np.clip(gravity_fn(q), -0.5 * effort, 0.5 * effort)
    np.testing.assert_allclose(tau, expected)
    assert np.any(np.abs(tau) > 0.5)
    assert all(msg.motor_cmd[s].kp == 0.0 for s in range(15, 29))
    session.request_quit()
    for _ in range(600):
        t[0] += 0.005
        _feed(io, low=0.05)
        session.tick()
        if session.done:
            break
    assert session.done
    result = session.wait_for_recordings(timeout=5.0)
    assert result and result[-1].status == "aborted_quit"


def test_failed_activation_closes_already_created_publishers_and_stays_inactive():
    class FailingPublisher(FakePublisher):
        closed = []

        def Init(self):
            if self.topic == "rt/dex3/right/cmd":
                raise RuntimeError("init failed")
            self.inited = True

        def Close(self):
            FailingPublisher.closed.append(self.topic)

    FakeSubscriber.created.clear()
    FakePublisher.created.clear()
    io = gio.G1DTeachReplayIO(
        hand_sides=("left", "right"), publisher_cls=FailingPublisher, subscriber_cls=FakeSubscriber,
        lowcmd_factory=lambda: _msg(35), handcmd_factory=lambda: _msg(7), crc_factory=FakeCrc,
        lowcmd_type=object, lowstate_type=object, handcmd_type=object, handstate_type=object)
    _feed(io)
    with pytest.raises(RuntimeError):
        io.activate(("left", "right"))
    assert sorted(FailingPublisher.closed) == ["rt/dex3/left/cmd", "rt/dex3/right/cmd", "rt/lowcmd"]
    io.write_arm(tr.ArmCommand(q=np.zeros(14), dq=np.zeros(14), kp=np.zeros(14), kd=np.zeros(14), tau=np.zeros(14)))
    assert all(p.written == [] for p in FakePublisher.created)


def test_dex3_slot_order_matches_the_reference_joint_index_enums():
    from teleop.robot_control import robot_hand_unitree as rhu

    assert [int(m) for m in rhu.Dex3_1_Left_JointIndex] == list(range(7))
    assert [int(m) for m in rhu.Dex3_1_Right_JointIndex] == list(range(7))
    left = [m.name.replace("kLeftHand", "").lower() for m in rhu.Dex3_1_Left_JointIndex]
    right = [m.name.replace("kRightHand", "").lower() for m in rhu.Dex3_1_Right_JointIndex]
    assert left == [n.replace("_", "") for n in tr.HAND_JOINT_NAMES["left"]]
    assert right == [n.replace("_", "") for n in tr.HAND_JOINT_NAMES["right"]]
