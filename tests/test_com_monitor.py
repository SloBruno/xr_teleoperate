"""CoM monitor: pure maths + fakes. No robot, no DDS, no pinocchio required."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.utils.com_monitor import ArmCoMModel, CoMMonitor, pearson

ROOT = Path(__file__).resolve().parents[1]


def fake_model():
    # delta_x = 0.1 m per rad of left shoulder pitch (q[0]) + right (q[7]), y from roll
    def com(q):
        q = np.asarray(q, float)
        return np.array([0.05 * (q[0] + q[7]), 0.02 * (q[1] - q[8]), 0.0])
    return ArmCoMModel(com_fn=com)


def test_delta_is_relative_to_neutral():
    m = fake_model()
    assert np.allclose(m.delta_com(np.zeros(14)), 0)
    assert m.delta_com(np.eye(14)[0])[0] == pytest.approx(0.05)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        fake_model().delta_com([0.0] * 13)


def test_monitor_warns_once_with_hysteresis_and_rate_limit():
    msgs = []
    t = [0.0]
    mon = CoMMonitor(fake_model(), warn_mm=30.0, clear_mm=20.0, min_interval_s=5.0,
                     emit=msgs.append, clock=lambda: t[0])
    q = np.zeros(14); q[0] = 1.0  # 50 mm
    assert mon.update(q).level == "warn"
    assert len(msgs) == 1 and "CoM" in msgs[0]
    t[0] = 1.0
    mon.update(q)
    assert len(msgs) == 1            # rate limited
    q[0] = 0.5                        # 25 mm: inside hysteresis, still warn, no spam
    t[0] = 10.0
    assert mon.update(q).level == "warn"
    q[0] = 0.2                        # 10 mm: clears
    assert mon.update(q).level == "ok"


def test_monitor_failsafe_disables_on_error():
    msgs = []
    calls = []
    def boom(q):
        if calls:
            raise RuntimeError("x")
        calls.append(1)
        return np.zeros(3)
    mon = CoMMonitor(ArmCoMModel(com_fn=boom), emit=msgs.append, clock=lambda: 0.0)
    s = mon.update(np.zeros(14))
    assert s.level == "disabled" and mon.disabled
    assert mon.update(np.zeros(14)).level == "disabled"  # stays inert, never raises


def test_monitor_nonfinite_input_is_ignored_not_fatal():
    mon = CoMMonitor(fake_model(), emit=lambda m: None, clock=lambda: 0.0)
    q = np.zeros(14); q[0] = np.nan
    assert mon.update(q).level == "invalid"
    assert not mon.disabled


def test_monitor_status_dict_is_json_ready():
    import json
    mon = CoMMonitor(fake_model(), emit=lambda m: None, clock=lambda: 0.0)
    json.dumps(mon.update(np.zeros(14)).as_dict())


def test_pearson_degenerate():
    assert np.isnan(pearson([1, 1, 1], [1, 2, 3]))
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_monitor_has_no_robot_imports():
    src = (ROOT / "teleop/utils/com_monitor.py").read_text()
    for bad in ("unitree_sdk2py", "ChannelPublisher", "LocoClient", "SetBalanceMode", "SetFsmId", "SetVelocity"):
        assert bad not in src


@pytest.mark.skipif(__import__("importlib").util.find_spec("pinocchio") is None, reason="pinocchio absent")
def test_urdf_model_forward_arms_shift_com_forward():
    m = ArmCoMModel.from_urdf(ROOT / "assets/g1/g1_body29_hand14.urdf")
    q = np.zeros(14); q[0] = q[7] = -1.2   # shoulder pitch forward (G1: negative = forward/up)
    d = m.delta_com(q)
    assert abs(d[0]) > 0.003 and np.isfinite(d).all()
    assert np.allclose(m.delta_com(np.zeros(14)), 0, atol=1e-9)


def test_create_from_env_default_off_and_bad_values():
    from teleop.utils.com_monitor import create_from_env
    u = ROOT / "assets/g1/g1_body29_hand14.urdf"
    assert create_from_env({}, u, "G1_29") is None
    assert create_from_env({"G1_COM_MONITOR": "1"}, u, "G1_23") is None
    msgs = []
    assert create_from_env({"G1_COM_MONITOR": "1", "G1_COM_WARN_MM": "nan"}, u, "G1_29", msgs.append) is None


def test_wiring_is_read_only_in_teleop_loop():
    src = (ROOT / "teleop/teleop_hand_and_arm.py").read_text()
    assert "com_monitor.update(current_lr_arm_q)" in src
