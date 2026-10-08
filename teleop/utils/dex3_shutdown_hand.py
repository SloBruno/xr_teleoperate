"""Dex3 hand behaviour at teleop shutdown (terminal q / B / Ctrl+C / SIGTERM / error).

Pure helpers (no I/O, no clock reads); the Dex3 controller owns publication.

Mode (env ``DEX3_SHUTDOWN_HAND``):

* ``close`` (default, operator request "quando apertar encerrar, a mão
  fechar"): trigger authority is revoked, the hand ramps to the SAME pose a
  trigger at 1.0 produces (``Dex3_{Left,Right}_Closed_Pose``: Thumb0 neutral,
  thumb straight-line full grasp, index/middle at the configured grasp), is held
  closed while the arms return home and the ``arm_sdk`` weight ramps 1 -> 0,
  then the command thread stops with the last published command = closed.
* ``open``: previous behaviour, byte-for-byte (open after the arm return,
  before the weight ramp; ``Dex3_1_Controller.open_and_deactivate``).
* ``hold``: like ``close`` but the goal is the last trigger target (frozen).

Unset -> ``close``. Any other value -> ``open`` (previous, no new motion) and a
PT-BR error log; the launcher rejects invalid values before starting.

"Closed" is never a mechanical limit: it is the configured trigger=1.0 pose.
The ramp only shapes the *target*; every cycle still goes through the normal
``Dex3HandProtector`` (same kp/kd, same closing torque ceilings, thermal
derate, stall grip-hold with its fixed hold torque), so closing at shutdown
can never squeeze harder than a full trigger pull during teleoperation.

Safety (unchanged rules win): a side whose DDS state is missing/stale, that has
a latched/reported motor fault, or any joint at >= 80 C (or still in the hot
latch) is NOT closed; it gets the previous open/relax command and the reason is
logged. The check is repeated every cycle and latches to open.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from teleop.utils.dex3_state_grace import state_is_fresh

ENV_VAR = "DEX3_SHUTDOWN_HAND"
MODES = ("close", "open", "hold")
DEFAULT_MODE = "close"
INVALID_FALLBACK_MODE = "open"

# Ramp: smoothstep whose peak joint velocity is <= CLOSE_MAX_JOINT_VELOCITY;
# never shorter than CLOSE_RAMP_MIN_S. Full open->closed (1.75 rad) = 0.9 s.
CLOSE_RAMP_MIN_S = 0.9
CLOSE_MAX_JOINT_VELOCITY = 3.0          # rad/s, per joint, peak
SMOOTHSTEP_PEAK_FACTOR = 1.5
# Same values as dex3_protection (DERATE_OPEN_C / STATE_STALE_S); duplicated
# here only as defaults, asserted equal by a test.
CUTOFF_TEMP_C = 80.0
STATE_STALE_S = 0.5

REASON_TEXT = {
    "state_missing": "estado DDS da mão ausente",
    "state_stale": "estado DDS da mão antigo",
    "fault": "motor em fault",
    "hot": "temperatura >= 80 C",
    "protection_unavailable": "proteção indisponível",
}


def resolve_mode(value: str | None) -> tuple[str, str | None]:
    """Return (mode, warning_pt_br_or_None)."""
    if value is None or str(value).strip() == "":
        return DEFAULT_MODE, None
    mode = str(value).strip().lower()
    if mode in MODES:
        return mode, None
    return INVALID_FALLBACK_MODE, (
        f"{ENV_VAR}='{value}' inválido (use close|open|hold); usando '{INVALID_FALLBACK_MODE}' "
        "(comportamento anterior: mão abre no encerramento)")


def describe_mode(mode: str) -> str:
    return {
        "close": "Dex3 no encerramento: FECHA (rampa suave até a pose do gatilho=1; "
                 "não fecha com fault/estado ausente/>=80 C)",
        "open": "Dex3 no encerramento: ABRE (comportamento anterior)",
        "hold": "Dex3 no encerramento: MANTÉM o último alvo do gatilho (com proteção)",
    }.get(mode, f"Dex3 no encerramento: modo desconhecido '{mode}'")


def _vec(value, n=7):
    try:
        out = np.asarray(value, dtype=float).reshape(n).copy()
    except (TypeError, ValueError):
        return None
    return out if np.all(np.isfinite(out)) else None


def plan_duration(start, goal, max_joint_velocity=CLOSE_MAX_JOINT_VELOCITY, min_duration=CLOSE_RAMP_MIN_S):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    distance = float(np.max(np.abs(goal - start))) if start.size else 0.0
    return max(float(min_duration), SMOOTHSTEP_PEAK_FACTOR * distance / float(max_joint_velocity))


def ramp_target(start, goal, elapsed, duration):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if duration <= 0.0:
        return goal.copy()
    s = min(1.0, max(0.0, float(elapsed) / float(duration)))
    return start + (goal - start) * (s * s * (3.0 - 2.0 * s))


def close_blocker(state: Mapping | None, now: float, *, fault_latched: bool = False,
                  hot_latched: bool = False, cutoff_c: float = CUTOFF_TEMP_C,
                  stale_s: float = STATE_STALE_S) -> str | None:
    """Reason this side must NOT close (None = closing allowed)."""
    if fault_latched:
        return "fault"
    if hot_latched:
        return "hot"
    if not isinstance(state, Mapping) or state.get("timestamp") is None:
        return "state_missing"
    if not state_is_fresh(now, state, stale_s):
        return "state_stale"
    for value in state.get("motorstate") or []:
        if value is not None and math.isfinite(value) and value != 0:
            return "fault"
    for value in state.get("temp") or []:
        if value is not None and math.isfinite(value) and value >= cutoff_c:
            return "hot"
    if any(v is None for v in (state.get("q") or [None])):
        return "state_missing"
    return None


class SideShutdownPlan:
    """Per-side ramp (pure; caller supplies the clock)."""

    def __init__(self, side: str, mode: str, start_target: Sequence[float] | None,
                 closed_pose: Sequence[float], open_pose: Sequence[float], now: float,
                 hold_target: Sequence[float] | None = None):
        """close: start -> closed_pose. hold: start -> hold_target (last trigger
        target; open pose if unknown). Missing/invalid start = open pose."""
        self.side = side
        self.mode = mode
        self.open_pose = np.asarray(open_pose, float).copy()
        start = _vec(start_target)
        self.start = self.open_pose.copy() if start is None else start
        if mode == "close":
            self.goal = np.asarray(closed_pose, float).copy()
        else:
            hold = _vec(hold_target)
            self.goal = self.start.copy() if hold is None else hold
        self.t0 = float(now)
        self.duration = plan_duration(self.start, self.goal)
        self.blocked_reason: str | None = None
        self.done = False

    def block(self, reason: str) -> bool:
        """Latch to open; True only on the first block (log once)."""
        if self.blocked_reason is not None:
            return False
        self.blocked_reason = reason
        self.done = True
        return True

    def target(self, now: float) -> np.ndarray:
        if self.blocked_reason is not None:
            return self.open_pose.copy()
        elapsed = max(0.0, float(now) - self.t0)
        if elapsed >= self.duration:
            self.done = True
        return ramp_target(self.start, self.goal, elapsed, self.duration)


def hand_shutdown_callbacks(hand_ctrl, mode: str | None = None, environ=None, log=None):
    """Return (mode, open_hands, close_hands, release_hands) for the arm shutdown.

    ``mode=None`` reads ``DEX3_SHUTDOWN_HAND`` from ``environ``. ``open`` (or a
    hand controller without the close API, e.g. Inspire/BrainCo) returns the
    previous callbacks exactly: ``open_and_deactivate`` (else ``deactivate``)
    and no close/release.
    """
    if hand_ctrl is None:
        return mode, None, None, None
    open_method = getattr(hand_ctrl, "open_and_deactivate", None)
    open_hands = open_method if open_method is not None else hand_ctrl.deactivate
    if mode is None:
        import os as _os
        mode, warning = resolve_mode((environ if environ is not None else _os.environ).get(ENV_VAR))
        if warning and log is not None:
            try:
                log(f"[Dex3 encerramento] {warning}")
            except BaseException:
                pass
    begin = getattr(hand_ctrl, "begin_shutdown_hand", None)
    finish = getattr(hand_ctrl, "finish_shutdown_hand", None)
    if mode in ("close", "hold") and begin is not None and finish is not None:
        return mode, open_hands, (lambda: begin(mode)), finish
    return mode, open_hands, None, None


def make_sigterm_handler(is_stopping, log=None):
    """SIGTERM -> KeyboardInterrupt (same graceful path as q / Ctrl+C).

    Runs in the main thread between bytecodes, so it must not take locks.
    While a stop is already in progress it only logs: the bounded shutdown
    (Dex3 close, arm return, weight ramp) is not cut by a repeated SIGTERM.
    """
    fired = {"done": False}

    def handler(signum, frame):
        try:
            stopping = bool(is_stopping()) or fired["done"]
        except BaseException:
            stopping = fired["done"]
        if stopping:
            if log is not None:
                try:
                    log("[shutdown] SIGTERM recebido durante o encerramento; continuando o encerramento seguro")
                except BaseException:
                    pass
            return
        if log is not None:
            try:
                log("[shutdown] SIGTERM recebido: encerramento seguro (mesmo caminho do q)")
            except BaseException:
                pass
        fired["done"] = True
        raise KeyboardInterrupt
    return handler
