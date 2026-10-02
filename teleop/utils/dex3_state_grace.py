"""Per-joint output retention for short Dex3 DDS state gaps.

This remains outside ``Dex3HandProtector``: stale feedback is never treated as
fresh.  It only replays joints whose last fresh, protected output was safe.
Every other joint is commanded to the caller-provided open pose.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

STATE_GRACE_S = 1.5
# Callback and control threads can sample monotonic time in the opposite order.
# Accept only this bounded scheduling skew; a genuinely future timestamp remains invalid.
STATE_TIMESTAMP_SKEW_S = 0.05
_NUM_JOINTS = 7


def state_is_fresh(now: float, state: dict | None, stale_s: float = 0.5) -> bool:
    """Accept only finite, non-future monotonic receive timestamps."""
    try:
        ts = state.get("timestamp") if state is not None else None
        age_s = float(now) - float(ts)
        return math.isfinite(float(now)) and math.isfinite(float(ts)) and -STATE_TIMESTAMP_SKEW_S <= age_s <= stale_s
    except (TypeError, ValueError):
        return False


class Dex3StateGrace:
    """Clock-injected state-gap FSM; cache eligibility is independent per joint."""

    def __init__(self, grace_s: float = STATE_GRACE_S):
        self.grace_s = float(grace_s)
        self._q_cmd: np.ndarray | None = None
        self._enable: list[bool] | None = None
        self._safe_at: float | None = None
        self._joint_q: list[float | None] = [None] * _NUM_JOINTS
        self._joint_safe_at: list[float | None] = [None] * _NUM_JOINTS
        self._joint_enable: list[bool] = [False] * _NUM_JOINTS
        self._all_normal_enable = False
        self._gap_started_at: float | None = None
        self._gap_count = 0
        self._gap_max_s = 0.0
        self._was_holding = False
        self._was_expired = False

    @staticmethod
    def _joint_values(values, *, default: bool | None = None) -> list[bool] | None:
        if isinstance(values, (bool, np.bool_)):
            return [bool(values)] * _NUM_JOINTS
        try:
            out = [bool(v) for v in values]
        except TypeError:
            return [default] * _NUM_JOINTS if default is not None else None
        return out if len(out) == _NUM_JOINTS else None

    @staticmethod
    def _open_pose(open_q: Sequence[float] | None) -> np.ndarray:
        try:
            pose = np.asarray(np.zeros(_NUM_JOINTS) if open_q is None else open_q, dtype=float).reshape(_NUM_JOINTS)
        except (TypeError, ValueError):
            return np.zeros(_NUM_JOINTS)
        return np.where(np.isfinite(pose), pose, 0.0)

    def update(self, now: float, *, fresh: bool, grip_active: bool, safe: bool | Sequence[bool],
               q_cmd: Sequence[float] | None = None,
               enable: Sequence[bool] | None = None, gap_eligible: bool = True,
               open_q: Sequence[float] | None = None,
               fallback_q: Sequence[float] | None = None) -> dict:
        """Return fresh/holding/expired metadata and an optional hybrid command.

        ``safe`` may be the legacy hand-wide bool or seven per-joint eligibility
        bits. A joint is cached only with a finite protected command and enabled
        fresh feedback. ``open_q`` is used for every uncacheable joint in a gap.
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
            self._cache_fresh(now, safe, q_cmd, enable)
            if was_gap:
                warning = "state_gap_recovered"
            return self._result("fresh", "fresh_state", warning, q_cmd, enable, 0.0, [], [])

        if not gap_eligible:
            self._clear_cache()
            self._was_holding, self._was_expired = False, True
            return self._result("expired", "invalid_state_timestamp", "state_gap_expired", None, None, 0.0, [], list(range(_NUM_JOINTS)))

        if self._gap_started_at is None:
            self._gap_started_at = now
            self._gap_count += 1
        elapsed = max(0.0, now - self._gap_started_at)
        self._gap_max_s = max(self._gap_max_s, elapsed)
        # A quiet state topic is not an actuator safety command.  Retain only
        # outputs already approved by fresh protection while the operator keeps
        # the grip latch active; fresh feedback still supersedes this cache.
        held_joints = self._held_joints() if grip_active else []
        if held_joints:
            if not self._was_holding:
                warning = "state_gap_started"
            self._was_holding, self._was_expired = True, False
            q_out = self._open_pose(open_q)
            for i in held_joints:
                q = self._joint_q[i]
                assert q is not None
                q_out[i] = q
            blocked = [i for i in range(_NUM_JOINTS) if i not in held_joints]
            enable_out = None if self._all_normal_enable and len(held_joints) == _NUM_JOINTS else [i in held_joints for i in range(_NUM_JOINTS)]
            return self._result("holding_no_feedback", "cached_protected_command", warning, q_out, enable_out, elapsed, held_joints, blocked)

        if grip_active and fallback_q is not None:
            fallback = np.asarray(fallback_q, dtype=float).reshape(_NUM_JOINTS)
            if np.all(np.isfinite(fallback)):
                if not self._was_holding:
                    warning = "state_gap_started"
                self._was_holding, self._was_expired = True, False
                return self._result("holding_no_feedback", "no_cached_protected_command", warning,
                                    fallback, None, elapsed, [], [])

        self._was_holding, self._was_expired = False, True
        return self._result("expired", "grip_not_active" if not grip_active else "no_cached_protected_command", None, None, None, elapsed, [], list(range(_NUM_JOINTS)))

    def _cache_fresh(self, now, safe, q_cmd, enable):
        try:
            cmd = np.asarray(q_cmd, dtype=float).reshape(_NUM_JOINTS).copy()
        except (TypeError, ValueError):
            self._clear_cache()
            return
        safe_bits = self._joint_values(safe)
        enable_bits = [True] * _NUM_JOINTS if enable is None else self._joint_values(enable)
        if safe_bits is None or enable_bits is None:
            self._clear_cache()
            return
        for i in range(_NUM_JOINTS):
            if safe_bits[i] and enable_bits[i] and math.isfinite(cmd[i]):
                self._joint_q[i] = float(cmd[i])
                self._joint_safe_at[i] = now
                self._joint_enable[i] = True
            else:
                self._joint_q[i] = None
                self._joint_safe_at[i] = None
                self._joint_enable[i] = False
        self._q_cmd = cmd if all(self._joint_enable) else None
        self._enable = None if enable is None and all(self._joint_enable) else [bool(v) for v in enable_bits]
        self._safe_at = now if any(self._joint_enable) else None
        self._all_normal_enable = enable is None and all(self._joint_enable)

    def _held_joints(self) -> list[int]:
        return [i for i in range(_NUM_JOINTS)
                if self._joint_enable[i] and self._joint_q[i] is not None
                and self._joint_safe_at[i] is not None]

    def _clear_cache(self):
        self._q_cmd = None
        self._enable = None
        self._safe_at = None
        self._joint_q = [None] * _NUM_JOINTS
        self._joint_safe_at = [None] * _NUM_JOINTS
        self._joint_enable = [False] * _NUM_JOINTS
        self._all_normal_enable = False

    def _result(self, state, reason, warning, q_cmd, enable, duration, held_joints, blocked_joints):
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
            "state_grace_joint_hold": list(held_joints),
            "state_grace_blocked_joints": list(blocked_joints),
        }
