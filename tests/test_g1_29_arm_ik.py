import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[1]))


class _FakeSE3:
    def __init__(self, homogeneous=None):
        self.homogeneous = np.eye(4) if homogeneous is None else homogeneous

    @property
    def translation(self):
        return self.homogeneous[:3, 3]

    @property
    def rotation(self):
        return self.homogeneous[:3, :3]


class _FakeData:
    def __init__(self, frame_count):
        self.oMf = [_FakeSE3() for _ in range(frame_count)]


class _FakeModel:
    nq = 14
    nv = 14
    lowerPositionLimit = np.full(14, -1.0)
    upperPositionLimit = np.full(14, 1.0)

    def __init__(self):
        self.frames = [object() for _ in range(106)]
        self._frame_ids = {}

    @property
    def nframes(self):
        return len(self.frames)

    def addFrame(self, frame):
        self.frames.append(frame)
        self._frame_ids[frame.name] = len(self.frames) - 1

    def createData(self):
        return _FakeData(self.nframes)

    def getFrameId(self, name):
        return self._frame_ids[name]

    def getJointId(self, name):
        return 1


class _FakeRobot:
    def __init__(self):
        self.model = _FakeModel()
        self.data = self.model.createData()

    def buildReducedRobot(self, **_kwargs):
        return _FakeRobot()


class _FakeFrame:
    def __init__(self, name, *_args):
        self.name = name


class _FakeValue:
    __array_priority__ = 1000

    def __getitem__(self, _key):
        return self

    def __call__(self, *_args):
        return self

    @property
    def T(self):
        return self

    def __sub__(self, _other):
        return self

    def __rsub__(self, _other):
        return self

    def __add__(self, _other):
        return self

    def __radd__(self, _other):
        return self

    def __mul__(self, _other):
        return self

    def __rmul__(self, _other):
        return self

    def __matmul__(self, _other):
        return self

    def __rmatmul__(self, _other):
        return self


class _FakeOpti:
    def variable(self, *_args):
        return _FakeValue()

    def parameter(self, *_args):
        return _FakeValue()

    def bounded(self, *_args):
        return _FakeValue()

    def subject_to(self, *_args):
        return None

    def minimize(self, *_args):
        return None

    def solver(self, *_args):
        return None


def _import_ik_module(monkeypatch):
    pin = types.ModuleType("pinocchio")
    pin.RobotWrapper = types.SimpleNamespace(BuildFromURDF=lambda *_args: _FakeRobot())
    pin.Frame = _FakeFrame
    pin.SE3 = lambda *_args: _FakeSE3()
    pin.FrameType = types.SimpleNamespace(OP_FRAME=object())
    pin.framesForwardKinematics = lambda _model, data, _q: None

    cpin = types.ModuleType("pinocchio.casadi")
    cpin.Model = lambda model: model
    cpin.log3 = lambda value: value
    cpin.framesForwardKinematics = lambda _model, _data, _q: None
    pin.casadi = cpin

    casadi = types.ModuleType("casadi")
    casadi.SX = types.SimpleNamespace(sym=lambda *_args: _FakeValue())
    casadi.vertcat = lambda *_args: _FakeValue()
    casadi.sumsqr = lambda *_args: _FakeValue()
    casadi.Function = lambda *_args: _FakeValue()
    casadi.Opti = _FakeOpti

    meshcat = types.ModuleType("meshcat")
    meshcat_geometry = types.ModuleType("meshcat.geometry")
    meshcat.geometry = meshcat_geometry
    visualize = types.ModuleType("pinocchio.visualize")
    visualize.MeshcatVisualizer = object
    logging_mp = types.ModuleType("logging_mp")
    logging_mp.getLogger = lambda *_args: types.SimpleNamespace(info=lambda *_a: None)

    for name, module in {
        "pinocchio": pin,
        "pinocchio.casadi": cpin,
        "pinocchio.visualize": visualize,
        "casadi": casadi,
        "meshcat": meshcat,
        "meshcat.geometry": meshcat_geometry,
        "logging_mp": logging_mp,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("teleop.robot_control.robot_arm_ik", None)
    return importlib.import_module("teleop.robot_control.robot_arm_ik")


def test_g1_29_cold_construction_recreates_data_before_first_zero_fk(monkeypatch):
    module = _import_ik_module(monkeypatch)
    monkeypatch.setattr(module.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(module.G1_29_ArmIK, "save_cache", lambda _self: None)

    ik = module.G1_29_ArmIK(Unit_Test=True)
    reduced_robot = ik.reduced_robot

    assert len(reduced_robot.data.oMf) == reduced_robot.model.nframes
    left_pose, right_pose = ik.forward_kinematics(np.zeros(14))
    assert left_pose.shape == (4, 4)
    assert right_pose.shape == (4, 4)


def test_real_pinocchio_cold_start_fk_when_dependency_is_available(monkeypatch):
    pytest.importorskip("pinocchio")
    pytest.importorskip("casadi")
    pytest.importorskip("meshcat.geometry")
    module = importlib.import_module("teleop.robot_control.robot_arm_ik")
    repo_root = Path(__file__).parents[1]
    monkeypatch.chdir(repo_root / "teleop")
    ik = module.G1_29_ArmIK(Unit_Test=False)

    left_pose, right_pose = ik.forward_kinematics(np.zeros(14))

    assert left_pose.shape == (4, 4)
    assert right_pose.shape == (4, 4)
    assert np.isfinite(left_pose).all()
    assert np.isfinite(right_pose).all()
