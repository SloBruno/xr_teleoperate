"""Per-hand Dex3 command retention across DDS state gaps.

Stale or missing DDS state is never treated as new feedback.  While the grip
latch is active, the last command already shaped by fresh protection is held
until feedback recovers or an explicit release/stop takes authority.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Kept for callers that construct Dex3StateGrace(grace_s=...) from older code.
STATE_GRACE_S = 1.5


def state_is_fresh(now: float, state: dict | None, stale_s: float = 0.5) -> bool:
    """Accept only finite, non-future monotonic receive timestamps."""
    try:
        ts = state.get("timestamp") if state is not None else None
        return math.isfinite(float(now)) and math.isfinite(float(ts)) and 0.0 <= now - ts <= stale_s
    except (TypeError, ValueError):
        return False


class Dex3StateGrace:
    """Clock-injected state-gap FSM for one hand; no I/O or protection math."""

    def __init__(self, grace_s: float = STATE_GRACE_S):
        self.grace_s = float(grace_s)  # compatibility only; gaps do not expire open
        self._q_cmd: np.ndarray | None = None
        self._enable: list[bool] | None = None
        self._gap_started_at: float | None = None
        self._gap_count = 0
        self._gap_max_s = 0.0
        self._was_holding = False

    def update(self, now: float, *, fresh: bool, grip_active: bool, safe: bool,
               q_cmd: Sequence[float] | np.ndarray | None = None,
               enable: Sequence[bool] | None = None,
               fallback_q_cmd: Sequence[float] | np.ndarray | None = None,
               fallback_enable: Sequence[bool] | None = None) -> dict:
        """Return fresh/holding metadata without treating a gap as feedback.

        Fresh protection output is cached exactly, including per-joint disabled
        motors.  A state gap while the latch is active replays that cache.  If
        no fresh command was ever cached, callers may provide the already-bounded
        trigger target as an explicit, zero-feedforward fallback; it is not a
        protection recomputation and never escalates a cached command.
        """
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        if fresh:
            was_gap = self._gap_started_at is not None
            self._gap_started_at = None
            self._was_holding = False
            if safe and q_cmd is not None and enable is not None:
                cmd = np.asarray(q_cmd, dtype=float).reshape(7).copy()
                if np.all(np.isfinite(cmd)):
                    self._q_cmd = cmd
                    self._enable = [bool(v) for v in enable]
                else:
                    self._clear_cache()
            else:
                self._clear_cache()
            return self._result("fresh", "fresh_state",
                                "state_gap_recovered" if was_gap else None,
                                q_cmd, enable, 0.0)

        if self._gap_started_at is None:
            self._gap_started_at = now
            self._gap_count += 1
        elapsed = max(0.0, now - self._gap_started_at)
        self._gap_max_s = max(self._gap_max_s, elapsed)

        if grip_active and self._q_cmd is not None and self._enable is not None:
            warning = "state_gap_started" if not self._was_holding else None
            self._was_holding = True
            return self._result("holding_no_feedback", "cached_protected_command", warning,
                                self._q_cmd.copy(), list(self._enable), elapsed)

        if grip_active and fallback_q_cmd is not None:
            fallback = np.asarray(fallback_q_cmd, dtype=float).reshape(7).copy()
            if np.all(np.isfinite(fallback)):
                warning = "state_gap_started" if not self._was_holding else None
                self._was_holding = True
                return self._result("holding_no_feedback", "no_cached_protected_command", warning,
                                    fallback, fallback_enable, elapsed)

        self._was_holding = False
        return self._result("expired", "grip_not_active", None, None, None, elapsed)

    def _clear_cache(self):
        self._q_cmd = None
        self._enable = None

    def _result(self, state, reason, warning, q_cmd, enable, duration):
        held = None if q_cmd is None else np.asarray(q_cmd, dtype=float).tolist()
        return {
            "state": state,
            "reason": reason,
            "warning": warning,
            "q_cmd": None if q_cmd is None else np.asarray(q_cmd, dtype=float).copy(),
            "enable": None if enable is None else [bool(v) for v in enable],
            "held_command": held,
            "hold_duration_s": round(float(duration), 3),
            "gap_count": self._gap_count,
            "gap_max_s": round(self._gap_max_s, 3),
        }
