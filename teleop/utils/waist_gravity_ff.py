"""Model-based waist gravity feed-forward for the G1_29 torso lean.

Why: with torso lean on, the waist (motors 13 roll / 14 pitch) is a pure PD
servo on ``rt/arm_sdk`` (kp 300, kd 3, tau 0). The upper body's weight bends
it until ``kp * error`` equals the gravity torque: a constant forward sag of
~1-4 deg was measured (docs/torso_lean.md). The root-cause fix is to add the
static gravity torque of the upper body as feed-forward ``tau``, computed from
the robot model with the MEASURED joints, so the PD only has to correct
residual errors. kp/kd are not changed.

Model: pinocchio ``computeGeneralizedGravity`` on the full fixed-base
``assets/g1/g1_body29_hand14.urdf`` (the URDF the teleop IK uses; root link =
pelvis), with the measured waist (12-14), measured arms (15-28; Dex3 finger
joints at 0) and gravity expressed in the pelvis frame from the pelvis IMU
quaternion of ``rt/lowstate``. Only the roll/pitch components are used; yaw
tau stays 0. ~3 us per call: it is evaluated at 250 Hz inside the arm_sdk
writer from the same lowstate snapshot, so the torque always matches the
current measured posture.

Safety (enforced by :class:`WaistFFShaper` at the final writer):
* only after the torso lean took the waist at ``r`` (configure order);
* per-axis cap ``TAU_CAP_NM`` (model worst case at |lean| <= 20 deg over the
  whole arm range is ~27.5 N.m, see tests) and absolute ceiling
  ``HARD_CEILING_NM`` = 60 % of the 50 N.m URDF waist effort;
* gain ramp 0 -> 1 in ``RAMP_S`` after activation and after any fault, ramp
  1 -> 0 on request (graceful shutdown, before the weight release);
* tau = 0 immediately (no ramp) for stale/invalid lowstate, invalid IMU,
  motor fault or a model error;
* kill switch ``G1_TORSO_LEAN_WAIST_FF=0`` (default 1 when torso lean is on).
"""
from __future__ import annotations

import math
import os
import threading

import numpy as np

ENV_VAR = "G1_TORSO_LEAN_WAIST_FF"

URDF_WAIST_EFFORT_NM = 50.0          # waist_roll/pitch <limit effort> in g1_body29_hand14.urdf
HARD_CEILING_NM = 0.6 * URDF_WAIST_EFFORT_NM   # 30 N.m: absolute, never configurable above
TAU_CAP_NM = 30.0                    # per axis; model max at |lean|<=20 deg, any arm pose ~27.5
RAMP_S = 0.5                         # ramp-in / ramp-out duration
MAX_DT_S = 0.02                      # a writer stall never advances the ramp by more than this
STATE_MAX_AGE_S = 0.1                # lowstate older than this -> tau 0 (lowstate is ~500 Hz)
MAX_PELVIS_TILT_RAD = math.radians(35.0)   # beyond this the robot is falling/fallen -> tau 0

ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)                                    # same order as G1_29_JointArmIndex (motors 15..28)
WAIST_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")   # motors 12, 13, 14

_DEFAULT_URDF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "assets", "g1", "g1_body29_hand14.urdf")


class WaistFFConfigError(ValueError):
    pass


def ff_enabled_from_env(environ):
    """Kill switch. Unset/empty/"1" -> True, "0" -> False, anything else raises."""
    raw = environ.get(ENV_VAR)
    if raw is None or str(raw).strip() == "":
        return True
    value = str(raw).strip()
    if value == "1":
        return True
    if value == "0":
        return False
    raise WaistFFConfigError(f"{ENV_VAR}={raw!r} inválido (use 0|1)")


def quat_wxyz_to_R(quat):
    """Validated unit quaternion (w, x, y, z) -> rotation matrix; raises ValueError."""
    if quat is None:
        raise ValueError("IMU quaternion missing")
    q = np.asarray(quat, dtype=float).reshape(-1)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("IMU quaternion must be 4 finite values")
    n = float(np.linalg.norm(q))
    if not (0.9 < n < 1.1):
        raise ValueError("IMU quaternion is not unit")
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class WaistGravityModel:
    """Static gravity torque on the waist roll/pitch joints (pinocchio, full G1_29 URDF)."""

    def __init__(self, urdf_path=None):
        import pinocchio as pin          # lazy: the pure parts of this module need only numpy
        self._pin = pin
        self.urdf_path = urdf_path or _DEFAULT_URDF
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        jid = self.model.getJointId
        self._waist_q = [self.model.joints[jid(n)].idx_q for n in WAIST_JOINT_NAMES]
        self._arm_q = [self.model.joints[jid(n)].idx_q for n in ARM_JOINT_NAMES]
        self._rp_v = [self.model.joints[jid(n)].idx_v for n in WAIST_JOINT_NAMES[1:]]
        if any(i >= self.model.nq for i in self._waist_q + self._arm_q):
            raise ValueError("URDF does not contain the G1_29 waist/arm joints")
        self._q = np.zeros(self.model.nq)
        self._lock = threading.Lock()

    def waist_tau(self, waist_q, arm_q, pelvis_quat_wxyz):
        """[tau_roll, tau_pitch] (N.m) that holds the measured posture against gravity.

        Raises ValueError for invalid joints/IMU or a pelvis tilted beyond
        ``MAX_PELVIS_TILT_RAD`` (caller must then command tau 0).
        """
        waist = np.asarray(waist_q, dtype=float).reshape(-1)
        arms = np.asarray(arm_q, dtype=float).reshape(-1)
        if waist.shape != (3,) or arms.shape != (14,) or not (np.all(np.isfinite(waist)) and np.all(np.isfinite(arms))):
            raise ValueError("waist gravity model needs 3 finite waist and 14 finite arm joints")
        R = quat_wxyz_to_R(pelvis_quat_wxyz)
        if math.acos(max(-1.0, min(1.0, float(R[2, 2])))) > MAX_PELVIS_TILT_RAD:
            raise ValueError("pelvis tilt beyond the feed-forward envelope")
        with self._lock:
            self._q[self._waist_q] = waist
            self._q[self._arm_q] = arms
            self.model.gravity.linear = R.T @ np.array([0.0, 0.0, -9.81])
            g = self._pin.computeGeneralizedGravity(self.model, self.data, self._q)
            out = np.array([float(g[self._rp_v[0]]), float(g[self._rp_v[1]])])
        if not np.all(np.isfinite(out)):
            raise ValueError("non-finite gravity torque")
        return out


class WaistFFShaper:
    """Final-writer shaping of the model torque: cap, ramp in/out, fail to zero."""

    def __init__(self, cap_nm=TAU_CAP_NM, ramp_s=RAMP_S):
        cap_nm = float(cap_nm)
        ramp_s = float(ramp_s)
        if not (math.isfinite(cap_nm) and 0.0 < cap_nm <= HARD_CEILING_NM):
            raise ValueError(f"waist ff cap must be in (0, {HARD_CEILING_NM}] N.m")
        if not (math.isfinite(ramp_s) and ramp_s > 0.0):
            raise ValueError("waist ff ramp must be > 0 s")
        self.cap = cap_nm
        self.ramp_s = ramp_s
        self.gain = 0.0
        self.ramping_out = False
        self.reason = "ramp_in"
        self.raw = np.zeros(2)
        self.out = np.zeros(2)

    @property
    def finished(self):
        return self.ramping_out and self.gain == 0.0

    def ramp_out(self):
        self.ramping_out = True

    def fault(self, reason):
        self.gain = 0.0
        self.reason = str(reason)
        self.raw = np.zeros(2)
        self.out = np.zeros(2)
        return self.out.copy()

    def step(self, raw, dt, reason=None):
        """One writer frame. ``raw`` = model [roll, pitch] or None (= fault ``reason``)."""
        if raw is None:
            return self.fault(reason or "unavailable")
        try:
            raw = np.asarray(raw, dtype=float).reshape(-1)
            ok = raw.shape == (2,) and bool(np.all(np.isfinite(raw)))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return self.fault("invalid_model_output")
        try:
            dt = float(dt)
        except (TypeError, ValueError):
            dt = float("nan")
        if not (math.isfinite(dt) and dt > 0.0):
            dt = 0.0
        dt = min(dt, MAX_DT_S)
        if self.ramping_out:
            self.gain = max(0.0, self.gain - dt / self.ramp_s)
            self.reason = "ramp_out_done" if self.gain == 0.0 else "ramp_out"
        else:
            self.gain = min(1.0, self.gain + dt / self.ramp_s)
            self.reason = "ok" if self.gain >= 1.0 else "ramp_in"
        self.raw = raw.copy()
        self.out = np.clip(self.gain * raw, -self.cap, self.cap)
        return self.out.copy()


def status_block(torso_status, arm_ctrl, enabled):
    """teleop-status ``torso_lean`` block + ``waist_ff`` (per-axis N.m). Never raises.

    Torso lean off (``configured`` false): returned unchanged.
    """
    if not isinstance(torso_status, dict) or not torso_status.get("configured"):
        return torso_status
    out = dict(torso_status)
    if not enabled:
        out["waist_ff"] = {"enabled": False, "reason": "kill_switch"}
        return out
    try:
        getter = getattr(arm_ctrl, "get_waist_gravity_ff", None)
        st = getter() if getter is not None else {"configured": False}
        if not st.get("configured"):
            out["waist_ff"] = {"enabled": True, "configured": False, "reason": "not_configured"}
            return out
        tau = [float(v) for v in st["tau_nm"]]
        raw = [float(v) for v in st["raw_nm"]]
        out["waist_ff"] = {
            "enabled": True, "configured": True, "gain": float(st["gain"]), "reason": str(st["reason"]),
            "tau_nm": {"yaw": tau[0], "roll": tau[1], "pitch": tau[2]},
            "model_nm": {"roll": raw[1], "pitch": raw[2]},
            "finished": bool(st.get("finished", False)),
        }
    except Exception:
        out["waist_ff"] = {"enabled": True, "reason": "telemetry_error"}
    return out
