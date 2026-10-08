"""Torso lean of the G1_29 commanded by the operator's head DISPLACEMENT.

Opt-in (``G1_TORSO_LEAN=1``; default off). Design and evidence:
``docs/torso_lean.md``. Everything here is pure (numpy only): no DDS, no I/O,
O(1) per control cycle, so it can run inside the teleop loop and be tested
with fakes.

Pipeline per cycle (30 Hz teleop loop)::

    tele_data.head_pose (robot basis: x fwd, y left, z up; Quest world)
      -> validity / freshness / jump gates           (HeadLeanTracker)
      -> neck pivot = head - R_head @ NECK_TO_EYE    (orientation-only head
                                                      motion does not move it)
      -> horizontal displacement in the operator's NEUTRAL yaw frame (r)
      -> deadband + gain + HARD saturation           (lean_from_displacement)
      -> low-pass + rate (+ optional accel) limit + final clamp
                                                     (WaistLeanCommand)
      -> waist q command [yaw, roll, pitch] = neutral(r) + [0, roll, pitch]

Sign convention (URDF ``assets/g1/g1_body29_hand14.urdf``; fixed by
``tests/test_torso_lean.py`` with pinocchio FK):

* ``waist_pitch_joint`` axis +y  -> q > 0 tilts the torso FORWARD (+x).
* ``waist_roll_joint``  axis +x  -> q > 0 tilts the torso to the RIGHT (-y),
  so leaning LEFT is a NEGATIVE roll.
* Motor indices: 12 waist_yaw, 13 waist_roll, 14 waist_pitch
  (``G1_29_JointIndex``; same order as the Unitree g1 arm7 sdk example).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEG = math.pi / 180.0

# ---- hard limits (never configurable above these) --------------------------
HARD_MAX_LEAN_DEG = 10.0          # operator requirement: <= 10 deg fwd/back/sides
HARD_MAX_RATE_DPS = 30.0
HARD_MAX_ACCEL_DPS2 = 200.0
HARD_MAX_GAIN_DEG_PER_M = 200.0
HARD_MAX_DEADBAND_M = 0.20

# ---- defaults ---------------------------------------------------------------
DEFAULT_MAX_LEAN_DEG = 10.0
DEFAULT_DEADBAND_M = 0.03
DEFAULT_GAIN_DEG_PER_M = 10.0 / 0.15     # 15 cm beyond the deadband = 10 deg
DEFAULT_RATE_DPS = 15.0
DEFAULT_ACCEL_DPS2 = 0.0                 # 0 = acceleration limit off
DEFAULT_LOWPASS_TAU_S = 0.3
DEFAULT_STALE_S = 0.2                    # head sample older than this: hold
DEFAULT_DECAY_AFTER_S = 1.0              # loss longer than this: target -> 0
DEFAULT_JUMP_M = 0.25                    # head step between frames = recentre
DEFAULT_JUMP_YAW_DEG = 45.0

# Handover at r: measured waist must be within this of the commanded waist
# (the arm_sdk writer already holds 12-14 at the constructor-time position).
HANDOVER_TOL_RAD = 0.05

# URDF waist limits (g1_body29_hand14.urdf): yaw +-2.618, roll/pitch +-0.52.
URDF_WAIST_LOWER = np.array([-2.618, -0.52, -0.52])
URDF_WAIST_UPPER = np.array([2.618, 0.52, 0.52])
WAIST_MOTOR_INDICES = (12, 13, 14)       # yaw, roll, pitch

# Neck pivot -> eye centre, in the HEAD frame, robot basis (x fwd, z up).
# Oculus/Meta SDK default neck model: eye 0.0805 m forward, 0.075 m up of the
# neck pivot. Using the pivot instead of the headset position makes pure head
# rotation (looking down / sideways) produce no displacement.
NECK_TO_EYE = np.array([0.0805, 0.0, 0.075])

# Operator head mapped onto the robot by TeleVuer (televuer tv_wrapper
# ``transform_IPunitree_Brobot_world_arm_to_head_then_waist``: +0.15 x, +0.45 z).
HEAD_POINT_IN_WAIST = np.array([0.15, 0.0, 0.45])

# TeleVuer fallback head pose (CONST_HEAD_POSE) converted to the robot basis:
# a head_pose equal to this means "no XR head sample" and is rejected.
_T_ROBOT_OPENXR = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
_T_OPENXR_ROBOT = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], float)
_CONST_HEAD_POSE_XR = np.array([[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, -0.2], [0, 0, 0, 1]], float)
FALLBACK_HEAD_POSE = _T_ROBOT_OPENXR @ _CONST_HEAD_POSE_XR @ _T_OPENXR_ROBOT


class TorsoLeanConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TorsoLeanConfig:
    max_deg: float = DEFAULT_MAX_LEAN_DEG
    gain_deg_per_m: float = DEFAULT_GAIN_DEG_PER_M
    deadband_m: float = DEFAULT_DEADBAND_M
    rate_dps: float = DEFAULT_RATE_DPS
    accel_dps2: float = DEFAULT_ACCEL_DPS2
    lowpass_tau_s: float = DEFAULT_LOWPASS_TAU_S
    stale_s: float = DEFAULT_STALE_S
    decay_after_s: float = DEFAULT_DECAY_AFTER_S
    jump_m: float = DEFAULT_JUMP_M
    jump_yaw_deg: float = DEFAULT_JUMP_YAW_DEG

    @property
    def max_rad(self):
        return min(self.max_deg, HARD_MAX_LEAN_DEG) * DEG

    def describe(self):
        return (f"Inclinação do tronco: LIGADA, máx {self.max_deg:g}°, ganho {self.gain_deg_per_m:.1f}°/m, "
                f"zona morta {self.deadband_m * 100:.1f} cm, vel {self.rate_dps:g}°/s")


_TRUE = ("1", "true", "yes", "on")


def _num(env, key, default, lo, hi, *, lo_open=False):
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(str(raw).strip())
    except ValueError:
        raise TorsoLeanConfigError(f"{key}={raw!r} não é número")
    if not math.isfinite(value):
        raise TorsoLeanConfigError(f"{key}={raw!r} não é finito")
    if value > hi or value < lo or (lo_open and value <= lo):
        raise TorsoLeanConfigError(f"{key}={value:g} fora de {'(' if lo_open else '['}{lo:g}, {hi:g}]")
    return value


def config_from_env(environ):
    """``None`` when ``G1_TORSO_LEAN`` is off (default). Raises
    :class:`TorsoLeanConfigError` for any out-of-range value (fail closed:
    the caller must then keep the feature OFF)."""
    if str(environ.get("G1_TORSO_LEAN", "0")).strip().lower() not in _TRUE:
        return None
    return TorsoLeanConfig(
        max_deg=_num(environ, "G1_TORSO_LEAN_MAX_DEG", DEFAULT_MAX_LEAN_DEG, 0.0, HARD_MAX_LEAN_DEG, lo_open=True),
        gain_deg_per_m=_num(environ, "G1_TORSO_LEAN_GAIN_DEG_PER_M", DEFAULT_GAIN_DEG_PER_M, 0.0,
                            HARD_MAX_GAIN_DEG_PER_M, lo_open=True),
        deadband_m=_num(environ, "G1_TORSO_LEAN_DEADBAND_M", DEFAULT_DEADBAND_M, 0.0, HARD_MAX_DEADBAND_M),
        rate_dps=_num(environ, "G1_TORSO_LEAN_RATE_DPS", DEFAULT_RATE_DPS, 0.0, HARD_MAX_RATE_DPS, lo_open=True),
        accel_dps2=_num(environ, "G1_TORSO_LEAN_ACCEL_DPS2", DEFAULT_ACCEL_DPS2, 0.0, HARD_MAX_ACCEL_DPS2),
    )


# ---------------------------------------------------------------- mapping
def _shape(x, deadband, gain_per_m, limit):
    excess = abs(x) - deadband
    if not (excess > 0.0):          # also catches NaN
        return 0.0
    return math.copysign(min(gain_per_m * excess, limit), x)


def lean_from_displacement(forward_m, left_m, cfg):
    """(pitch, roll) in rad from the operator displacement (m).

    forward -> +pitch (torso forward); left -> NEGATIVE roll (torso left).
    Deadband, linear gain and a HARD per-axis saturation at ``cfg.max_rad``.
    """
    gain = cfg.gain_deg_per_m * DEG
    limit = cfg.max_rad
    pitch = _shape(float(forward_m), cfg.deadband_m, gain, limit)
    roll = -_shape(float(left_m), cfg.deadband_m, gain, limit)
    return pitch, roll


def _yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def head_pivot_and_yaw(head_pose):
    """(neck pivot xyz, heading yaw) for a valid rigid head pose, else (None, None)."""
    try:
        T = np.asarray(head_pose, dtype=float)
    except (TypeError, ValueError):
        return None, None
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return None, None
    R = T[:3, :3]
    if np.max(np.abs(R.T @ R - np.eye(3))) > 1e-3 or np.linalg.det(R) <= 0.0:
        return None, None
    if np.allclose(T, FALLBACK_HEAD_POSE, atol=1e-9):
        return None, None
    forward = R[:, 0]
    if math.hypot(forward[0], forward[1]) < 1e-3:   # looking straight up/down: yaw undefined
        yaw = None
    else:
        yaw = math.atan2(forward[1], forward[0])
    return T[:3, 3] - R @ NECK_TO_EYE, yaw


class HeadLeanTracker:
    """Head pose -> raw lean target (pitch, roll) with freshness/jump gates."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.engaged = False
        self.p0 = None
        self.yaw0 = 0.0
        self._c0 = 1.0
        self._s0 = 0.0
        self._last_raw = None
        self._last_new_t = None
        self._last_good_t = None
        self._prev_pivot = None
        self._prev_yaw = None
        self._last_disp = (0.0, 0.0)
        self.target = (0.0, 0.0)
        self.status = "idle"
        self.jumps = 0
        self.rejected = 0

    def engage(self, head_pose, now):
        pivot, yaw = head_pivot_and_yaw(head_pose)
        if pivot is None or yaw is None:
            return False
        self.p0 = pivot.copy()
        self._set_yaw0(yaw)
        self._last_raw = np.array(head_pose, dtype=float)
        self._last_new_t = now
        self._last_good_t = now
        self._prev_pivot = pivot.copy()
        self._prev_yaw = yaw
        self._last_disp = (0.0, 0.0)
        self.target = (0.0, 0.0)
        self.engaged = True
        self.status = "ok"
        return True

    def _set_yaw0(self, yaw):
        self.yaw0 = yaw
        self._c0 = math.cos(yaw)
        self._s0 = math.sin(yaw)

    def _disp(self, pivot):
        dx = pivot[0] - self.p0[0]
        dy = pivot[1] - self.p0[1]
        # rotate the world displacement into the neutral operator frame (Rz(yaw0)^T)
        return self._c0 * dx + self._s0 * dy, -self._s0 * dx + self._c0 * dy

    def update(self, head_pose, now):
        """Return (pitch, roll) raw target. Never raises."""
        if not self.engaged:
            return (0.0, 0.0)
        try:
            pivot, yaw = head_pivot_and_yaw(head_pose)
            new = False
            if pivot is not None:
                raw = np.asarray(head_pose, dtype=float)
                new = self._last_raw is None or not np.array_equal(raw, self._last_raw)
                if new:
                    self._last_raw = raw.copy()
                    self._last_new_t = now
            fresh = pivot is not None and (now - self._last_new_t) <= self.cfg.stale_s
            if pivot is None:
                self.rejected += 1
            if fresh and new:
                yaw_eff = self._prev_yaw if yaw is None else yaw
                step = math.hypot(pivot[0] - self._prev_pivot[0], pivot[1] - self._prev_pivot[1])
                dyaw = abs(_wrap(yaw_eff - self._prev_yaw))
                if step > self.cfg.jump_m or dyaw > self.cfg.jump_yaw_deg * DEG:
                    # Quest recentre / guardian reset: re-anchor the neutral so
                    # the displacement continues from the last accepted value.
                    self.jumps += 1
                    self._set_yaw0(_wrap(self.yaw0 + _wrap(yaw_eff - self._prev_yaw)))
                    f, l = self._last_disp
                    self.p0 = pivot.copy()
                    self.p0[0] -= self._c0 * f - self._s0 * l
                    self.p0[1] -= self._s0 * f + self._c0 * l
                    self.status = "jump"
                else:
                    self._last_disp = self._disp(pivot)
                    self.target = lean_from_displacement(self._last_disp[0], self._last_disp[1], self.cfg)
                    self.status = "ok"
                self._prev_pivot = pivot.copy()
                self._prev_yaw = yaw_eff
                self._last_good_t = now
            elif fresh:
                self._last_good_t = now      # same sample re-read within the freshness window
            else:
                if now - self._last_good_t > self.cfg.decay_after_s:
                    self.target = (0.0, 0.0)
                    self.status = "lost_decay"
                else:
                    self.status = "hold"
        except Exception:
            self.status = "error_hold"
        return self.target


class WaistLeanCommand:
    """Raw lean target -> smooth, limited absolute waist command [yaw, roll, pitch]."""

    def __init__(self, cfg, neutral_q, start_q, now):
        self.cfg = cfg
        self.neutral = np.asarray(neutral_q, dtype=float).reshape(3).copy()
        start = np.asarray(start_q, dtype=float).reshape(3).copy()
        lim = cfg.max_rad
        # Lean box: neutral +- max (roll, pitch), yaw fixed at neutral. The
        # value the servos already hold at r (start, within HANDOVER_TOL of the
        # neutral) is included so the first written frame is continuous; the
        # target itself is always inside neutral +- max. Then URDF limits.
        lo = np.minimum(self.neutral - np.array([0.0, lim, lim]), start)
        hi = np.maximum(self.neutral + np.array([0.0, lim, lim]), start)
        self.lower = np.maximum(lo, URDF_WAIST_LOWER)
        self.upper = np.minimum(hi, URDF_WAIST_UPPER)
        self.cmd = np.clip(start, self.lower, self.upper)
        self.vel = np.zeros(3)
        self.filtered = np.zeros(2)        # (pitch, roll) after the low-pass
        self.last_t = now
        self._rate = cfg.rate_dps * DEG
        self._accel = cfg.accel_dps2 * DEG

    def lean(self):
        """Current commanded lean relative to neutral: (pitch, roll) rad."""
        return float(self.cmd[2] - self.neutral[2]), float(self.cmd[1] - self.neutral[1])

    def step(self, target, now):
        dt = now - self.last_t
        self.last_t = now
        if not (dt > 0.0):
            return self.cmd.copy()
        dt = min(dt, 0.1)
        pitch_t, roll_t = float(target[0]), float(target[1])
        lim = self.cfg.max_rad
        if not (math.isfinite(pitch_t) and math.isfinite(roll_t)):
            pitch_t, roll_t = self.filtered
        pitch_t = min(lim, max(-lim, pitch_t))
        roll_t = min(lim, max(-lim, roll_t))
        alpha = 1.0 - math.exp(-dt / self.cfg.lowpass_tau_s) if self.cfg.lowpass_tau_s > 0 else 1.0
        self.filtered[0] += (pitch_t - self.filtered[0]) * alpha
        self.filtered[1] += (roll_t - self.filtered[1]) * alpha
        desired = self.neutral + np.array([0.0, self.filtered[1], self.filtered[0]])
        desired = np.clip(desired, self.lower, self.upper)
        max_step = self._rate * dt
        if self._accel > 0.0:
            # velocity toward desired, bounded by rate and accel, and able to stop in time
            dist = desired - self.cmd
            v_stop = np.sqrt(2.0 * self._accel * np.abs(dist))
            v_des = np.sign(dist) * np.minimum(np.minimum(np.abs(dist) / dt, self._rate), v_stop)
            dv = np.clip(v_des - self.vel, -self._accel * dt, self._accel * dt)
            self.vel = np.clip(self.vel + dv, -self._rate, self._rate)
            delta = np.clip(self.vel * dt, -max_step, max_step)
        else:
            delta = np.clip(desired - self.cmd, -max_step, max_step)
        self.cmd = np.clip(self.cmd + delta, self.lower, self.upper)
        return self.cmd.copy()


# ------------------------------------------------------------ IK geometry
def waist_rotation(waist_q):
    """Rotation of torso_link w.r.t. pelvis: Rz(yaw) Rx(roll) Ry(pitch) (URDF chain,
    joint origins carry no rotation)."""
    y, r, p = (float(v) for v in waist_q)
    cy, sy, cr, sr, cp, sp = math.cos(y), math.sin(y), math.cos(r), math.sin(r), math.cos(p), math.sin(p)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return Rz @ Rx @ Ry


def retarget_to_torso(target, R_waist, head_point=HEAD_POINT_IN_WAIST):
    """Express a TeleVuer wrist target in the torso (waist-locked IK) frame.

    TeleVuer gives ``target = head_point + v`` with ``v`` the operator's
    hand-minus-head vector in gravity/yaw-aligned axes. Geometrically the
    robot "head point" is fixed to the TORSO, so in the pelvis frame the target
    is ``W(head_point) + v``; mapped into the torso frame (the frame of the
    reduced IK model, waist locked at 0) the translation of W cancels and the
    target is ``head_point + R^T v`` with orientation ``R^T R_target``.
    A rigid operator lean (v rotated by the same R) therefore leaves the IK
    target unchanged: the arms keep their posture relative to the torso.
    """
    T = np.asarray(target, dtype=float)
    out = T.copy()
    Rt = R_waist.T
    out[:3, :3] = Rt @ T[:3, :3]
    out[:3, 3] = head_point + Rt @ (T[:3, 3] - head_point)
    return out


# --------------------------------------------------------------- session
REFUSED = "refused"


class TorsoLeanSession:
    """Glue for teleop_hand_and_arm.py (duck-typed G1_29 arm controller).

    Neutral waist = the command the waist servos are ALREADY holding at ``r``
    (written by the arm_sdk writer since construction), accepted only when the
    measured waist is within ``HANDOVER_TOL_RAD`` of it. Re-sampling the
    measured value as the new hold target would ratchet the gravity sag of the
    torso (kp * error = gravity torque) one step further at every handover.
    """

    def __init__(self, cfg, tracker, command, neutral_measured, log=None):
        self.cfg = cfg
        self.tracker = tracker
        self.command = command
        self.neutral_measured = neutral_measured
        self.log = log
        self.waist_cmd = command.cmd.copy()
        self._R_neutral_T = waist_rotation(command.neutral).T
        self._jumps_logged = 0

    @classmethod
    def try_engage(cls, cfg, arm_ctrl, head_pose, now, log=None):
        """Capture neutral head + waist at r.

        Returns a session; ``None`` = inputs not ready yet (retry next cycle);
        ``REFUSED`` = measured waist far from the held command (feature stays
        OFF for this session; the waist keeps today's behaviour).
        """
        tracker = HeadLeanTracker(cfg)
        if not tracker.engage(head_pose, now):
            return None
        try:
            meas, age = arm_ctrl.get_waist_q_snapshot()
            held = arm_ctrl.get_waist_command_written()
        except Exception:
            return None
        if meas is None or held is None:
            return None
        meas = np.asarray(meas, dtype=float).reshape(-1)
        held = np.asarray(held, dtype=float).reshape(-1)
        if (meas.shape != (3,) or held.shape != (3,) or not np.all(np.isfinite(meas))
                or not np.all(np.isfinite(held)) or not math.isfinite(float(age)) or float(age) > 0.25):
            return None
        if np.max(np.abs(meas - held)) > HANDOVER_TOL_RAD:
            if log is not None:
                log.error(f"[torso_lean] cintura medida {np.round(meas, 3)} longe do comando mantido "
                          f"{np.round(held, 3)} (> {HANDOVER_TOL_RAD} rad); inclinação NÃO ativada nesta sessão")
            return REFUSED
        if np.any(held < URDF_WAIST_LOWER) or np.any(held > URDF_WAIST_UPPER):
            if log is not None:
                log.error(f"[torso_lean] cintura {np.round(held, 3)} fora dos limites do URDF; NÃO ativada")
            return REFUSED
        command = WaistLeanCommand(cfg, held, held, now)
        try:
            arm_ctrl.configure_waist_command(command.lower, command.upper, cfg.rate_dps * DEG, held)
        except Exception as error:
            if log is not None:
                log.error(f"[torso_lean] controlador recusou a configuração da cintura ({error!r}); NÃO ativada")
            return REFUSED
        if log is not None:
            log.info(f"[torso_lean] ativada: cintura neutra {np.round(held, 3)} rad (medida {np.round(meas, 3)}), "
                     f"yaw do operador {tracker.yaw0 / DEG:.0f}°, roll/pitch ±{cfg.max_deg:g}°, {cfg.rate_dps:g}°/s")
        return cls(cfg, tracker, command, meas, log)

    def step(self, head_pose, now):
        """One control cycle: returns the absolute waist command [yaw, roll, pitch]."""
        target = self.tracker.update(head_pose, now)
        self.waist_cmd = self.command.step(target, now)
        if self.log is not None and self.tracker.jumps != self._jumps_logged:
            self._jumps_logged = self.tracker.jumps
            self.log.warning("[torso_lean] salto da cabeça (recentralização do Quest?) ignorado; neutro reancorado")
        return self.waist_cmd

    def torso_rotation(self, waist_q):
        """Torso rotation relative to the neutral torso: R_n^T R(q)."""
        return self._R_neutral_T @ waist_rotation(waist_q)

    def telemetry(self):
        pitch, roll = self.command.lean()
        tp, tr = self.tracker.target
        return {"lean_pitch": pitch, "lean_roll": roll, "target_pitch": tp, "target_roll": tr,
                "waist_cmd": self.waist_cmd, "status": self.tracker.status}


def stream_telemetry(session, arm_ctrl):
    """Lean block for the pose stream (XPS2), or None when the lean is not active.

    Cheap (a few floats + one lowstate snapshot); never raises.
    """
    if not isinstance(session, TorsoLeanSession):
        return None
    try:
        tm = session.telemetry()
        waist_meas, _age = arm_ctrl.get_waist_q_snapshot()
        return {"active": True, "target": (tm["target_pitch"], tm["target_roll"]),
                "cmd": (tm["lean_pitch"], tm["lean_roll"]), "waist_cmd": tm["waist_cmd"], "waist_meas": waist_meas}
    except Exception:
        return None
