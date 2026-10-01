"""Inert CoM-shift monitor for the 14 arm joints (read-only, opt-in).

Computes how far the whole-body centre of mass moves (pelvis frame, legs/waist at
neutral) when the arms leave their neutral pose, and warns on the terminal.
It NEVER publishes anything: no DDS, no Loco/Sport calls.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

ARM_DOF = 14
_ARM_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right")
               for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                         "wrist_roll", "wrist_pitch", "wrist_yaw")]


def pearson(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


class ArmCoMModel:
    """CoM(q14) - CoM(0), in the pelvis frame (metres)."""

    def __init__(self, com_fn: Callable[[np.ndarray], np.ndarray]):
        self._com_fn = com_fn
        self._ref = np.asarray(com_fn(np.zeros(ARM_DOF)), float)

    def delta_com(self, q) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1)
        if q.shape != (ARM_DOF,):
            raise ValueError(f"expected {ARM_DOF} arm joints, got {q.shape}")
        return np.asarray(self._com_fn(q), float) - self._ref

    @classmethod
    def from_urdf(cls, urdf_path, extra_hand_mass_kg: float = 0.0):
        import pinocchio as pin  # lazy: only needed for the real model
        model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        if extra_hand_mass_kg:
            for name in ("left_hand_palm_link", "right_hand_palm_link"):
                fid = model.getFrameId(name)
                jid = model.frames[fid].parentJoint
                model.inertias[jid] = model.inertias[jid] + pin.Inertia(
                    extra_hand_mass_kg, model.frames[fid].placement.translation, np.eye(3) * 1e-4)
        idx = [model.joints[model.getJointId(n)].idx_q for n in _ARM_JOINTS]
        data = model.createData()
        base = pin.neutral(model)

        def com(q14):
            q = base.copy()
            q[idx] = q14
            return np.array(pin.centerOfMass(model, data, q))
        return cls(com)


@dataclass
class CoMStatus:
    level: str                       # ok | warn | invalid | disabled
    dx_mm: float = float("nan")
    dy_mm: float = float("nan")

    def as_dict(self):
        return {"level": self.level, "dx_mm": self.dx_mm, "dy_mm": self.dy_mm}


class CoMMonitor:
    """Warn-only monitor with hysteresis, rate limit and fail-safe disable."""

    def __init__(self, model: ArmCoMModel, warn_mm: float = 40.0, clear_mm: Optional[float] = None,
                 min_interval_s: float = 10.0, emit: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.monotonic):
        self.model = model
        self.warn_mm = float(warn_mm)
        self.clear_mm = float(clear_mm if clear_mm is not None else 0.7 * warn_mm)
        self.min_interval_s = float(min_interval_s)
        self._emit = emit
        self._clock = clock
        self.disabled = False
        self._warn = False
        self._last_emit = -math.inf

    def update(self, q14) -> CoMStatus:
        if self.disabled:
            return CoMStatus("disabled")
        try:
            q = np.asarray(q14, float)
            if not np.isfinite(q).all():
                return CoMStatus("invalid")
            d = self.model.delta_com(q) * 1000.0
            dx, dy = float(d[0]), float(d[1])
            if not (math.isfinite(dx) and math.isfinite(dy)):
                return CoMStatus("invalid")
        except Exception as exc:  # fail-safe: never break the control loop
            self.disabled = True
            self._emit(f"[com_monitor] disabled: {exc!r}")
            return CoMStatus("disabled")
        mag = abs(dx)
        if self._warn and mag < self.clear_mm:
            self._warn = False
        elif not self._warn and mag >= self.warn_mm:
            self._warn = True
        if self._warn:
            now = self._clock()
            if now - self._last_emit >= self.min_interval_s:
                self._last_emit = now
                self._emit(f"[com_monitor] CoM {'frente' if dx > 0 else 'tras'} {dx:+.0f} mm "
                           f"(y {dy:+.0f} mm) vs bracos neutros; robo pode compensar andando")
        return CoMStatus("warn" if self._warn else "ok", dx, dy)


def create_from_env(env, urdf_path, arm_profile: str, emit: Callable[[str], None] = print) -> Optional[CoMMonitor]:
    """Opt-in via G1_COM_MONITOR=1 and the G1_29 profile. Any failure -> None (inert)."""
    if str(env.get("G1_COM_MONITOR", "")).strip() not in ("1", "true", "yes"):
        return None
    if arm_profile != "G1_29":
        return None
    try:
        warn = float(env.get("G1_COM_WARN_MM", "40"))
        hand = float(env.get("G1_COM_HAND_KG", "0"))
        if not (math.isfinite(warn) and warn > 0 and 0 <= hand <= 3):
            return None
        return CoMMonitor(ArmCoMModel.from_urdf(urdf_path, extra_hand_mass_kg=hand), warn_mm=warn, emit=emit)
    except Exception as exc:
        emit(f"[com_monitor] not started: {exc!r}")
        return None
