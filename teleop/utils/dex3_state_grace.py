"""Pure, per-hand output retention for short Dex3 DDS state gaps.

This is deliberately outside ``Dex3HandProtector``: it never treats stale state
as fresh feedback.  It can only replay a command that protection computed from
fresh, safe feedback, for a bounded interval while the grip latch remains
closed.  On expiry the caller must use its normal fail-safe path.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

STATE_GRACE_S = 1.5


def state_is_fresh(now: float, state: dict | None, stale_s: float = 0.5) -> bool:
    """Accept only finite, non-future monotonic receive timestamps."""
    try:
        ts = state.get("timestamp") if state is not None else None
        return math.isfinite(float(now)) and math.isfinite(float(ts)) and 0.0 <= now - ts <= stale_s
    except (TypeError, ValueError):
        return False


class Dex3StateGrace:
    """Clock-injected state-gap FSM for one hand; no I/O and no protection math."""

    def __init__(self, grace_s: float = STATE_GRACE_S):
        self.grace_s = float(grace_s)
        self._q_cmd: np.ndarray | None = None
        self._enable: list[bool] | None = None
        self._safe_at: float | None = None
        self._gap_started_at: float | None = None
        self._gap_count = 0
        self._gap_max_s = 0.0
        self._was_holding = False
        self._was_expired = False

    def update(self, now: float, *, fresh: bool, grip_active: bool, safe: bool,
               q_cmd: Sequence[float] | None = None,
               enable: Sequence[bool] | None = None, gap_eligible: bool = True) -> dict:
        """Return fresh/holding/expired output metadata without mutating commands.

        ``fresh`` must be based only on a monotonic receive timestamp.  A caller
        supplies ``safe`` only after normal protection has evaluated fresh state.
        """
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        warning = None
        if fresh:
            was_gap = self._gap_started_at is not None
            self._gap_started_at = None
            self._was_holding = False
            self._was_expired = False
            if safe and q_cmd is not None and enable is not None:
                cmd = np.asarray(q_cmd, dtype=float).reshape(7).copy()
                if np.all(np.isfinite(cmd)):
                    self._q_cmd = cmd
                    self._enable = [bool(v) for v in enable]
                    self._safe_at = now
                else:
                    self._clear_cache()
            else:
                self._clear_cache()
            if was_gap:
                warning = "state_gap_recovered"
            return self._result("fresh", "fresh_state", warning, q_cmd, enable, 0.0)

        if not gap_eligible:
            self._clear_cache()
            self._was_holding, self._was_expired = False, True
            return self._result("expired", "invalid_state_timestamp", "state_gap_expired", None, None, 0.0)

        if self._gap_started_at is None:
            self._gap_started_at = now
            self._gap_count += 1
        elapsed = max(0.0, now - self._gap_started_at)
        self._gap_max_s = max(self._gap_max_s, elapsed)
        can_hold = (grip_active and self._q_cmd is not None and self._enable is not None
                    and self._safe_at is not None and now - self._safe_at <= self.grace_s
                    and elapsed <= self.grace_s)
        if can_hold:
            if not self._was_holding:
                warning = "state_gap_started"
            self._was_holding, self._was_expired = True, False
            return self._result("holding", "state_gap_short", warning,
                                self._q_cmd.copy(), list(self._enable), elapsed)

        reason = "grip_not_active" if not grip_active else "state_grace_expired"
        if not self._was_expired:
            warning = "state_gap_expired"
        self._was_holding, self._was_expired = False, True
        return self._result("expired", reason, warning, None, None, elapsed)

    def _clear_cache(self):
        self._q_cmd = None
        self._enable = None
        self._safe_at = None

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
