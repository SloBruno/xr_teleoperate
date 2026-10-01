"""Pure, I/O-free torque/thermal protection for the Unitree Dex3-1 fingers.

The Dex3 command is a PD law with ``tau_ff = 0``: ``tau ~= kp * (q_cmd - q_meas)
- kd * dq``. The controller never raises gain (kp=1.5, kd=0.2 stay untouched);
this module only *shapes the position target* so the implicit torque cannot
grow when a finger is blocked, and stops commanding a motor that is hot,
stalled, or disabled.

Trigger ownership is unchanged: the input target still comes from the
trigger interpolation (rest = open, proportional closure). Protection can only
move the commanded q toward the measured q / open pose, never beyond the
trigger target.

Evidence (session pose-telemetry-20260930T190939, 287 s, both hands):
  * Free-following joints: error is transient (command steps, finger follows),
    median |err| ~0.01-0.3 rad, |dq| > 0.3 while moving.
  * Right thumb blocked at t=209-251 s: |err| 0.5-1.06 rad (Thumb1) and
    0.7-1.73 rad (Thumb2) held with dq == 0 and |tau_est| ~0.5-1.4 M raw
    (p90 when moving freely is only ~0.05-0.09 M), temperature[1] 39-44 -> 83/90 C;
    Thumb2 went mode 1 -> 0 and motorstate 0 -> 512 at t=244.2 s and stayed.
  * The old command error of 1.75 rad at kp=1.5 is ~2.6 N*m implicit.

EVERY numeric threshold below is a conservative first guess derived from that
single session and MUST BE CALIBRATED IN THE PHYSICAL TEST ("calibrar no teste
fisico"). ``tau_est`` is a raw, uncalibrated integer-like field (values of
1e3-1e6), NOT N*m; the N*m figures here are the *implicit command torque*
``kp * err`` at the motor side (before any gear ratio), which is an estimate.
The documented per-joint torque range of the Dex3 thumb motor (~0.49-3.1 N*m,
third-party source) could not be verified on official Unitree pages.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from teleop.utils.dex3_telemetry import clean_number

NUM_JOINTS = 7
# Slot order: Thumb0, Thumb1, Thumb2, then four finger joints (same on both sides).

# Command PD gain used to convert an error cap to an implicit torque ceiling.
# Must equal ``kp`` in robot_hand_unitree.py (asserted by a test); not changed.
DEX3_KP = 1.5

# --- (1) Torque ceiling by error clamp -------------------------------------
# Implicit torque ceiling (kp*err_max), N*m, for moves toward the closed pose.
# Thumb1/Thumb2 start low (0.75 N*m -> 0.5 rad): in the free-following part of
# the session the thumb tracks a full 1.05/1.75 rad step within ~0.1-0.3 s
# (err transient p90 0.7-1.3 rad, median 0.3), but when blocked the error sat
# at 0.5-1.7 rad for seconds and the motor heated to 83/90 C.
# Calibrar no teste fisico.
# Long fingers (index/middle) raised 1.0 -> 1.4 N*m (= URDF rated effort) because
# the closed pose now sits further than the old 0.67 rad error cap would allow
# when blocked by an object. Thumb ceiling UNCHANGED (0.75): not the reported issue.
# 1.4 -> 1.8 (fix/dex3-grip-no-oscillation): URDF rated effort is 1.4 N*m; no
# documented peak could be verified (third-party ~3.1 N*m for the thumb motor),
# so 1.8 is a modest 29% step. CALIBRAR NO TESTE FISICO (watch temperature).
FINGER_CLOSE_CEILING_NM = 1.8
CLOSE_TORQUE_CEILING_NM = (0.75, 0.75, 0.75) + (FINGER_CLOSE_CEILING_NM,) * 4
# Opening ceiling (toward rest pose). Higher so the hand can still open
# quickly on trigger release, but still below the old 2.6 N*m implicit peak.
# Calibrar no teste fisico.
OPEN_TORQUE_CEILING_NM = (1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5)

# --- (2) Stall detection ----------------------------------------------------
# Stall = closing direction AND |target-q| > STALL_ERR_RAD AND |dq| < STALL_DQ
# AND |tau_est| >= STALL_TAU_RAW, continuously for STALL_TIME_S.
# Blocked thumb: err 0.5-1.7 rad, dq exactly 0.00, tau 0.5-1.4 M raw.
# Free resting joints: tau ~0.01-0.05 M raw; moving p90 ~0.09 M. Calibrar.
STALL_ERR_RAD = 0.35
STALL_DQ = 0.3            # rest noise seen up to ~0.23; moving joints >> 1
STALL_TAU_RAW = 100_000.0  # raw units of tau_est (uncalibrated)
STALL_TIME_S = 0.75
# Grip hold: while stalled, keep pushing toward the trigger target with a
# reduced implicit torque instead of relaxing to q_meas (relaxing lets a held
# object push the finger back and the box slips). Time-limited, fades with the
# thermal derate factor, never while faulted. Calibrar no teste fisico.
# Thumb = 0.0: thumb keeps the old relax-on-stall behaviour (thermal history).
# Hold torque raised 0.8 -> 1.2 N*m (calibrar no teste fisico).
GRIP_HOLD_TORQUE_NM = (0.0, 0.0, 0.0, 1.2, 1.2, 1.2, 1.2)
# Hold time limit 10 -> 30 s; the real defence is the thermal derate (hold torque
# scales with the derate factor and is cut at DERATE_OPEN_C). Calibrar.
GRIP_HOLD_MAX_S = 30.0
# Hold may push the command PAST the trigger target (never past these |q| limits,
# ~0.1 rad inside the URDF stops: joint0 +-1.571, joint1 +-1.745) because with a
# box blocking the finger at q~0.72 the 1.15 rad pose only leaves 0.43 rad of
# error = 0.65 N*m, whatever the ceiling. Per slot; thumb unused.
GRIP_HOLD_CMD_LIMIT_RAD = (0.0, 0.0, 0.0, 1.47, 1.65, 1.47, 1.65)
# Grip latch (hysteresis): once stalled, the grip is held until the TRIGGER target
# magnitude falls by this fraction of its value at entry (0.3 -> 1.0 -> 0.7), and
# only if that persists for STALL_RELEASE_DEBOUNCE_S (one-sample jitter or a
# recoil of the finger never releases). Calibrar.
STALL_RELEASE_FRACTION = 0.3
STALL_RELEASE_DEBOUNCE_S = 0.2
# Terminal warning at this finger temperature (before derate bites hard).
TEMP_WARN_C = 70.0
# --- (3) Thermal derate -----------------------------------------------------
# motor_state.temperature is int16[2] in deg C. In the session temperature[1]
# reacts fast to load (39 -> 55 C in ~1 s, 90 C peak) while temperature[0]
# moves slowly (43 -> 53 C): we protect on max(temperature[0], temperature[1])
# (which element is winding vs. housing is UNVERIFIED). Calibrar.
DERATE_START_C = 65.0     # begin reducing closing target and torque ceiling
DERATE_OPEN_C = 80.0      # derate factor 0: open/relax, latch
DERATE_RESUME_C = 60.0    # hysteresis: leave the hot latch only below this
# Upward (recovery) slew of the derate factor, per second, so closing returns
# gradually after cooling instead of snapping back.
DERATE_RECOVERY_PER_S = 0.25
TEMP_PLAUSIBLE_MAX_C = 150.0

# --- (4) Motor fault ---------------------------------------------------------
# mode == 0 (after having been seen enabled) or motorstate != 0. Debounced;
# latched until controller restart. Never re-enabled automatically.
FAULT_DEBOUNCE_SAMPLES = 3

# --- (6) Freshness / warnings ------------------------------------------------
STATE_STALE_S = 0.5
TORQUE_LIMIT_WARN_S = 0.5      # only warn for sustained torque limiting
WARN_PERIOD_S = 2.0            # per (side, joint, kind)

JOINT_NAMES = ("Thumb0", "Thumb1", "Thumb2", "Finger0", "Finger1", "Finger2", "Finger3")


def derate_factor(temperature_c: float | None) -> float:
    """Stateless ramp: 1 at <=START, 0 at >=OPEN, linear between."""
    if temperature_c is None or not math.isfinite(temperature_c):
        return 1.0
    if temperature_c <= DERATE_START_C:
        return 1.0
    if temperature_c >= DERATE_OPEN_C:
        return 0.0
    return (DERATE_OPEN_C - temperature_c) / (DERATE_OPEN_C - DERATE_START_C)


def extract_protection_state(hand_state: Any, joint_ids: Sequence[int], timestamp: float) -> dict:
    """Pull the fields protection needs from a HandState_; never raises."""
    try:
        motors = hand_state.motor_state
    except Exception:
        motors = None
    q, dq, tau, temp, mode, ms = [], [], [], [], [], []
    for joint_id in joint_ids:
        try:
            motor = motors[int(joint_id)]
        except Exception:
            motor = None

        def get(name, motor=motor):
            try:
                return getattr(motor, name)
            except Exception:
                return None

        q.append(clean_number(get("q")))
        dq.append(clean_number(get("dq")))
        tau.append(clean_number(get("tau_est")))
        mode.append(clean_number(get("mode")))
        ms.append(clean_number(get("motorstate")))
        temps = []
        try:
            temps = [clean_number(v) for v in list(get("temperature"))[:2]]
        except Exception:
            pass
        temps = [t for t in temps if t is not None and 0 <= t <= TEMP_PLAUSIBLE_MAX_C]
        temp.append(max(temps) if temps else None)
    return {"timestamp": float(timestamp), "q": q, "dq": dq, "tau": tau,
            "temp": temp, "mode": mode, "motorstate": ms}


class ProtectionResult:
    __slots__ = ("q_cmd", "enable", "torque_limited", "stall", "derate", "fault",
                 "state_stale", "active", "grip_hold")

    def __init__(self, n=NUM_JOINTS):
        self.q_cmd = np.zeros(n)
        self.enable = [True] * n
        self.torque_limited = [False] * n
        self.stall = [False] * n
        self.derate = [1.0] * n
        self.fault = [False] * n
        self.state_stale = False
        self.grip_hold = [False] * n
        self.active: dict[tuple[str, int], str] = {}

    def flags(self) -> dict:
        """JSON-safe per-joint flags for extended telemetry."""
        return {
            "torque_limited": [bool(v) for v in self.torque_limited],
            "stall": [bool(v) for v in self.stall],
            "derate": [round(float(v), 3) for v in self.derate],
            "fault": [bool(v) for v in self.fault],
            "grip_hold": [bool(v) for v in self.grip_hold],
            "state_stale": bool(self.state_stale),
        }


class Dex3HandProtector:
    """Per-hand deterministic protection state machine. No I/O, no clock reads."""

    def __init__(self, open_pose: Sequence[float], kp: float = DEX3_KP,
                 close_ceiling_nm: Sequence[float] = CLOSE_TORQUE_CEILING_NM,
                 open_ceiling_nm: Sequence[float] = OPEN_TORQUE_CEILING_NM):
        self.open_pose = np.asarray(open_pose, dtype=float)
        if self.open_pose.shape != (NUM_JOINTS,):
            raise ValueError("open_pose must have seven entries")
        self.kp = float(kp)
        self.close_err_max = np.asarray(close_ceiling_nm, dtype=float) / self.kp
        self.open_err_max = np.asarray(open_ceiling_nm, dtype=float) / self.kp
        n = NUM_JOINTS
        self._seen_ok = [False] * n
        self._fault_count = [0] * n
        self._fault = [False] * n
        self._stall_since: list[float | None] = [None] * n
        self._stall = [False] * n
        self._stall_mag = [0.0] * n
        self._stall_start: list[float | None] = [None] * n
        self._release_since: list[float | None] = [None] * n
        self._hot = [False] * n
        self._derate_prev = [1.0] * n
        self._last_temp: list[float | None] = [None] * n
        self._limit_since: list[float | None] = [None] * n
        self._last_now: float | None = None

    def _open_result(self, res: ProtectionResult) -> ProtectionResult:
        res.q_cmd = self.open_pose.copy()
        return res

    def update(self, now: float, target: Sequence[float], state: Mapping | None) -> ProtectionResult:
        res = ProtectionResult()
        open_pose = self.open_pose
        try:
            tgt = np.asarray(target, dtype=float).reshape(NUM_JOINTS).copy()
        except Exception:
            tgt = open_pose.copy()
        tgt = np.where(np.isfinite(tgt), tgt, open_pose)
        dt = 0.0 if self._last_now is None else max(0.0, now - self._last_now)
        self._last_now = now

        ts = None if state is None else state.get("timestamp")
        fresh = (isinstance(ts, (int, float)) and math.isfinite(ts)
                 and 0.0 <= now - ts <= STATE_STALE_S)
        if not fresh:
            # Fail-safe: no trustworthy feedback -> command the open rest pose.
            res.state_stale = True
            res.q_cmd = open_pose.copy()
            res.active[("state_stale", -1)] = "estado do Dex3 ausente/antigo: comando aberto"
            for i in range(NUM_JOINTS):
                res.fault[i] = self._fault[i]
                res.enable[i] = not self._fault[i]
                res.stall[i] = self._stall[i]
                if self._fault[i]:
                    res.active[("fault", i)] = self._fault_msg(i)
            return res

        for i in range(NUM_JOINTS):
            self._update_joint(i, now, dt, tgt[i], state, res)
        return res

    def _fault_msg(self, i):
        return f"{JOINT_NAMES[i]} motor desligado/fault (mode=0 ou motorstate!=0): sem torque, requer power-cycle"

    def _get(self, state, key, i):
        try:
            value = state[key][i]
        except Exception:
            return None
        return clean_number(value)

    def _update_joint(self, i, now, dt, target, state, res):
        open_q = float(self.open_pose[i])
        q = self._get(state, "q", i)
        dq = self._get(state, "dq", i)
        tau = self._get(state, "tau", i)
        mode = self._get(state, "mode", i)
        ms = self._get(state, "motorstate", i)
        temp = self._get(state, "temp", i)

        # --- (4) fault detection (latched; never auto re-enabled) ----------
        if mode is not None and mode != 0 and (ms is None or ms == 0):
            self._seen_ok[i] = True
        bad = (ms is not None and ms != 0) or (mode is not None and mode == 0 and self._seen_ok[i])
        if not self._fault[i]:
            self._fault_count[i] = self._fault_count[i] + 1 if bad else 0
            if self._fault_count[i] >= FAULT_DEBOUNCE_SAMPLES:
                self._fault[i] = True
        if self._fault[i]:
            res.fault[i] = True
            res.enable[i] = False
            res.q_cmd[i] = q if q is not None else open_q
            res.active[("fault", i)] = self._fault_msg(i)
            return

        if q is None:
            res.q_cmd[i] = open_q  # no feedback for this joint: open, fail-safe
            res.active[("state_stale", i)] = f"{JOINT_NAMES[i]} sem leitura de posicao: comando aberto"
            return

        # --- (3) thermal derate -------------------------------------------
        if temp is not None:
            self._last_temp[i] = float(temp)
        t_now = self._last_temp[i]
        if t_now is not None:
            if t_now >= DERATE_OPEN_C:
                self._hot[i] = True
            elif self._hot[i] and t_now < DERATE_RESUME_C:
                self._hot[i] = False
        d = 0.0 if self._hot[i] else derate_factor(t_now)
        if d > self._derate_prev[i]:  # slow recovery only when cooling
            d = min(d, self._derate_prev[i] + DERATE_RECOVERY_PER_S * dt)
        self._derate_prev[i] = d
        res.derate[i] = d
        if d < 1.0:
            res.active[("derate", i)] = (
                f"{JOINT_NAMES[i]} {t_now:.0f}C: derate {d:.2f}" + (" (aberto/relaxado)" if d <= 0.0 else ""))
        t_eff = open_q + d * (target - open_q)
        if (t_now is not None and t_now >= TEMP_WARN_C and GRIP_HOLD_TORQUE_NM[i] > 0.0):
            res.active[("temp_warn", i)] = (
                f"{JOINT_NAMES[i]} {t_now:.0f}C >= {TEMP_WARN_C:.0f}C: aperto em derate, abre a {DERATE_OPEN_C:.0f}C")

        # --- direction ------------------------------------------------------
        sign = np.sign(t_eff - open_q)
        err = t_eff - q
        closing = bool(sign != 0 and sign * err > 0)

        # --- (2) stall ------------------------------------------------------
        if self._stall[i]:
            # Latch: leave only when the trigger itself clearly backs off (or the
            # command is no longer a closing one), sustained for the debounce.
            mag = abs(target - open_q)
            leaving = (not closing) or mag <= self._stall_mag[i] * (1.0 - STALL_RELEASE_FRACTION)
            if leaving:
                if self._release_since[i] is None:
                    self._release_since[i] = now
                if now - self._release_since[i] >= STALL_RELEASE_DEBOUNCE_S:
                    self._stall[i] = False
                    self._stall_since[i] = None
                    self._release_since[i] = None
            else:
                self._release_since[i] = None
        else:
            cond = (closing and abs(err) > STALL_ERR_RAD and dq is not None
                    and abs(dq) < STALL_DQ and (tau is None or abs(tau) >= STALL_TAU_RAW))
            if cond:
                if self._stall_since[i] is None:
                    self._stall_since[i] = now
                if now - self._stall_since[i] >= STALL_TIME_S:
                    self._stall[i] = True
                    self._stall_mag[i] = abs(target - open_q)
                    self._release_since[i] = None
                    self._stall_start[i] = now
            else:
                self._stall_since[i] = None
        if self._stall[i]:
            res.stall[i] = True
            hold_err = GRIP_HOLD_TORQUE_NM[i] / self.kp * d
            held = (self._stall_start[i] is not None
                    and now - self._stall_start[i] <= GRIP_HOLD_MAX_S and hold_err > 1e-9)
            if held:
                # keep a reduced, thermally-faded squeeze toward the target
                step = hold_err
                q_hold = q + sign * step
                lim = GRIP_HOLD_CMD_LIMIT_RAD[i]
                if lim > 0.0:
                    q_hold = float(np.clip(q_hold, open_q - lim, open_q + lim))
                    if sign * (q_hold - q) < 0:  # already past the limit: never pull back
                        q_hold = q
                else:
                    q_hold = q + sign * min(step, abs(t_eff - q))
                step = abs(q_hold - q)
                res.q_cmd[i] = q_hold
                res.grip_hold[i] = True
                res.active[("stall", i)] = (
                    f"{JOINT_NAMES[i]} travado (err {abs(err):.2f} rad): segurando pegada a "
                    f"{step * self.kp:.2f} N*m implicitos (max {GRIP_HOLD_MAX_S:.0f} s)")
            else:
                res.q_cmd[i] = q  # relax: zero implicit torque
                res.active[("stall", i)] = (
                    f"{JOINT_NAMES[i]} travado (err {abs(err):.2f} rad, dq~0, tau alto): aliviando ate soltar/reduzir trigger")
            self._limit_since[i] = None
            return

        # --- (1) torque ceiling as error clamp ---------------------------
        cap = (self.close_err_max[i] * d) if closing else self.open_err_max[i]
        q_cmd = float(np.clip(t_eff, q - cap, q + cap))
        limited = abs(q_cmd - t_eff) > 1e-9
        res.q_cmd[i] = q_cmd
        res.torque_limited[i] = limited
        if limited:
            if self._limit_since[i] is None:
                self._limit_since[i] = now
            if now - self._limit_since[i] >= TORQUE_LIMIT_WARN_S:
                res.active[("torque_limit", i)] = (
                    f"{JOINT_NAMES[i]} limitado a {cap * self.kp:.2f} N*m implicitos (erro {abs(err):.2f} rad)")
        else:
            self._limit_since[i] = None


class ProtectionWarner:
    """Rate-limits active-condition messages per (side, kind, joint)."""

    def __init__(self, period_s: float = WARN_PERIOD_S):
        self.period_s = float(period_s)
        self._last: dict[tuple, float] = {}

    def messages(self, now: float, side: str, active: Mapping[tuple, str]) -> list[str]:
        out = []
        for (kind, joint), text in active.items():
            key = (side, kind, joint)
            last = self._last.get(key)
            if last is None or now - last >= self.period_s:
                self._last[key] = now
                out.append(f"[Dex3 protecao {side}] {text}")
        return out
