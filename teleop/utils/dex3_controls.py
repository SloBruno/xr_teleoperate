"""Pure controller-trigger mapping helpers for the Unitree Dex3 hand."""

import numpy as np


def compose_dex3_targets(base_pose: np.ndarray, trigger: float, closed_pose: np.ndarray) -> np.ndarray:
    """Overlay a raw trigger toward the closed pose onto a retargeted pose.

    Televuer's raw trigger value is expected to be ``0.0`` when released and
    ``1.0`` when fully pressed. Invalid values fail open by preserving the
    retargeted pose; finite values are clamped before interpolation.
    """
    base_pose = np.asarray(base_pose, dtype=float)
    closed_pose = np.asarray(closed_pose, dtype=float)
    if base_pose.shape != (7,) or closed_pose.shape != (7,):
        raise ValueError("Dex3 poses must each contain seven joint targets")
    if not np.isfinite(trigger):
        amount = 0.0
    else:
        amount = float(np.clip(trigger, 0.0, 1.0))
    return base_pose + amount * (closed_pose - base_pose)


def trigger_to_dex3_targets(trigger: float, open_pose: np.ndarray, closed_pose: np.ndarray) -> np.ndarray:
    """Interpolate a seven-joint Dex3 pose from open to closed."""
    return compose_dex3_targets(open_pose, trigger, closed_pose)


# --- Trigger hold on stale controller samples --------------------------------
# The Quest controller feed was measured at ~11-15 Hz with gaps up to 1.25 s, so
# a sample older than 0.25 s used to read as trigger=0 and the hand opened with
# the trigger still closed. Now the LAST VALID trigger is held while the sample
# is at most TRIGGER_HOLD_MAX_AGE_S old (age counted from the sample timestamp);
# beyond that (or with a missing/invalid/future sample) the fail-safe open stays.
# Calibrar no teste fisico.
import math as _math

TRIGGER_FRESH_S = 0.25
TRIGGER_HOLD_MAX_AGE_S = 0.5
TRIGGER_OPEN_THRESHOLD = 0.05  # effective trigger below this = "trigger_low"


def trigger_sample_usable(sample_timestamp: float, now: float) -> bool:
    """True while a sample may still drive the hand (fresh or in hold window)."""
    try:
        if not _math.isfinite(sample_timestamp) or sample_timestamp <= 0.0 or not _math.isfinite(now):
            return False
    except TypeError:
        return False
    return 0.0 <= now - sample_timestamp <= TRIGGER_HOLD_MAX_AGE_S


class TriggerHold:
    """Per-side stale-sample hold. Pure, no I/O, caller supplies the clock."""

    def __init__(self):
        self.last_valid = 0.0
        self.last_valid_ts = 0.0
        self.stale_count = 0
        self.held_count = 0
        self.expired_count = 0
        self.dropouts = 0
        self._was_fresh = True
        self._last_ts = None
        self._gaps = []
        self.gap_max_s = 0.0
        self._first_ts = None
        self._n_updates = 0

    def update(self, trigger: float, sample_ts: float, now: float) -> dict:
        try:
            ts_ok = _math.isfinite(sample_ts) and sample_ts > 0.0
        except TypeError:
            ts_ok = False
        raw = trigger
        trig_ok = isinstance(trigger, (int, float)) and _math.isfinite(trigger)
        age = (now - sample_ts) if ts_ok else None
        if ts_ok and sample_ts != self._last_ts:
            if self._last_ts is not None and sample_ts > self._last_ts:
                gap = sample_ts - self._last_ts
                self._gaps.append(gap)
                del self._gaps[:-50]
                self.gap_max_s = max(self.gap_max_s, gap)
            if self._first_ts is None:
                self._first_ts = sample_ts
            self._n_updates += 1
            self._last_ts = sample_ts
        fresh = ts_ok and trig_ok and 0.0 <= age <= TRIGGER_FRESH_S
        if fresh:
            amount = float(min(max(trigger, 0.0), 1.0))
            self.last_valid, self.last_valid_ts = amount, sample_ts
            state, eff = "fresh", amount
            self._was_fresh = True
        else:
            self.stale_count += 1
            if self._was_fresh:
                self.dropouts += 1
            self._was_fresh = False
            if ts_ok and age is not None and 0.0 <= age <= TRIGGER_HOLD_MAX_AGE_S and self.last_valid_ts > 0.0:
                state, eff = "held", self.last_valid
                self.held_count += 1
            else:
                state, eff = ("missing" if not ts_ok else "expired"), 0.0
                self.expired_count += 1
                if not ts_ok or (age is not None and age < 0.0):
                    self.last_valid, self.last_valid_ts = 0.0, 0.0
        rate = None
        if self._first_ts is not None and self._last_ts is not None and self._last_ts > self._first_ts:
            rate = (self._n_updates - 1) / (self._last_ts - self._first_ts)
        return {
            "trigger_raw": raw if trig_ok else None,
            "trigger_effective": eff,
            "sample_ts": sample_ts if ts_ok else None,
            "age_ms": None if age is None else round(age * 1000.0, 1),
            "stale": state != "fresh",
            "trigger_state": state,
            "stale_count": self.stale_count,
            "held_count": self.held_count,
            "expired_count": self.expired_count,
            "dropouts": self.dropouts,
            "update_hz": None if rate is None else round(rate, 2),
            "gap_max_s": round(self.gap_max_s, 3),
            "gap_recent_max_s": round(max(self._gaps), 3) if self._gaps else None,
        }


# Active-grip latch: safety timeout is deliberately longer than observed 1.25 s
# controller gaps. Fresh release is deliberately debounced, so one low frame
# cannot open a closed grip. Controller loss beyond 2 s stops retained effort.
GRIP_LATCH_EXPIRE_S = 2.0
GRIP_RELEASE_LOW_S = 0.6


class GripLatch:
    """Per-side grip authority state machine; pure and clock-injected.

    A fresh high trigger activates/refreshes the latch. A stale or repeated
    sample holds its last effective value until two seconds from the last fresh
    sample. Releasing requires consecutive fresh low samples for 0.6 s.
    Invalid/future timestamps and explicit stop revoke authority immediately.
    """

    def __init__(self, close_threshold: float = TRIGGER_OPEN_THRESHOLD):
        self.close_threshold = float(close_threshold)
        self.active = False
        self.entered_at = None
        self.last_fresh_at = None
        self.last_effective = 0.0
        self.low_since = None
        self.low_count = 0

    @staticmethod
    def _valid(now, ts, trigger):
        return (isinstance(now, (int, float)) and _math.isfinite(now)
                and isinstance(ts, (int, float)) and _math.isfinite(ts) and ts > 0.0
                and isinstance(trigger, (int, float)) and _math.isfinite(trigger)
                and 0.0 <= now - ts <= TRIGGER_FRESH_S)

    def _info(self, now, raw, state, reason, age=None):
        duration = 0.0 if self.low_since is None else max(0.0, now - self.low_since)
        return {
            "trigger_raw": raw if isinstance(raw, (int, float)) and _math.isfinite(raw) else None,
            "trigger_effective": self.last_effective if self.active else 0.0,
            "sample_ts": None if age is None else now - age,
            "age_ms": None if age is None else round(age * 1000.0, 1),
            "stale": state == "held_stale",
            "trigger_state": state,
            "grip_latch_state": state,
            "latch_entered_at": self.entered_at,
            "latch_age_s": None if self.entered_at is None else round(max(0.0, now - self.entered_at), 3),
            "fresh_low_duration_s": round(duration, 3),
            "fresh_low_count": self.low_count,
            "held_reason": reason if state in ("held_stale", "active") else None,
            "exact_open_reason": reason if state not in ("held_stale", "active") else None,
        }

    def update(self, trigger, sample_ts, now, *, stop=False):
        age = None
        if isinstance(sample_ts, (int, float)) and _math.isfinite(sample_ts):
            age = now - sample_ts
        if stop:
            self.active = False; self.last_effective = 0.0; self.low_since = None; self.low_count = 0
            return self._info(now, trigger, "stopped", "stop", age)
        # A finite past timestamp outside the fresh window is a controller gap:
        # retain an active grip without refreshing its loss timeout.
        if (isinstance(sample_ts, (int, float)) and _math.isfinite(sample_ts) and sample_ts > 0.0
                and isinstance(now, (int, float)) and _math.isfinite(now) and now >= sample_ts
                and not self._valid(now, sample_ts, trigger)):
            return self.hold_stale(trigger, sample_ts, now)
        if not self._valid(now, sample_ts, trigger):
            self.active = False; self.last_effective = 0.0; self.low_since = None; self.low_count = 0
            return self._info(now, trigger, "invalid", "invalid_controller_sample", age)
        raw = float(np.clip(trigger, 0.0, 1.0))
        self.last_fresh_at = now
        if raw > self.close_threshold:
            if not self.active:
                self.entered_at = now
            self.active = True; self.last_effective = raw; self.low_since = None; self.low_count = 0
            return self._info(now, raw, "active", "fresh_trigger_high", age)
        if not self.active:
            return self._info(now, raw, "released", "trigger_low", age)
        if self.low_since is None:
            self.low_since = now; self.low_count = 1
        else:
            self.low_count += 1
        if now - self.low_since >= GRIP_RELEASE_LOW_S:
            self.active = False; self.last_effective = 0.0
            return self._info(now, raw, "released", "fresh_low_sustained", age)
        return self._info(now, raw, "active", "awaiting_fresh_low_debounce", age)

    def hold_stale(self, trigger, sample_ts, now):
        """Retain a latch through a stale/repeated sample; never refreshes it."""
        age = None
        if isinstance(sample_ts, (int, float)) and _math.isfinite(sample_ts):
            age = now - sample_ts
        if not self.active:
            return self._info(now, trigger, "released", "trigger_low", age)
        if self.last_fresh_at is not None and 0.0 <= now - self.last_fresh_at <= GRIP_LATCH_EXPIRE_S:
            self.low_since = None; self.low_count = 0
            return self._info(now, trigger, "held_stale", "controller_gap", age)
        self.active = False; self.last_effective = 0.0; self.low_since = None; self.low_count = 0
        return self._info(now, trigger, "expired", "controller_loss_timeout", age)

    def update_sample(self, trigger, sample_ts, now, *, stop=False):
        """Dispatch fresh samples vs. stale/repeated samples without authority refresh."""
        if stop:
            return self.update(trigger, sample_ts, now, stop=True)
        if self._valid(now, sample_ts, trigger):
            return self.update(trigger, sample_ts, now)
        # Invalid/future samples are invalid signals, not benign stale gaps.
        if not (isinstance(sample_ts, (int, float)) and _math.isfinite(sample_ts) and sample_ts > 0.0
                and isinstance(now, (int, float)) and _math.isfinite(now) and now >= sample_ts):
            return self.update(trigger, sample_ts, now)
        return self.hold_stale(trigger, sample_ts, now)


def open_reasons(trigger_state: str, trigger_effective: float, flags: dict | None,
                 force_open: bool = False, protection_error: bool = False) -> list:
    """Why the commanded hand is open/opening (ordered, empty if none)."""
    out = []
    if force_open:
        out.append("force_open")
    if trigger_state in ("expired", "missing"):
        out.append("stale_expired")
    if protection_error:
        out.append("protection_error")
    if flags:
        if flags.get("state_stale"):
            out.append("state_stale")
        idx = range(3, 7)
        if any(flags.get("fault", [False] * 7)[i] for i in idx):
            out.append("fault")
        stall, hold = flags.get("stall", [False] * 7), flags.get("grip_hold", [False] * 7)
        if any(stall[i] and not hold[i] for i in idx):
            out.append("protection_relax")
        if any(flags.get("derate", [1.0] * 7)[i] < 1.0 for i in idx):
            out.append("derate")
    if trigger_effective < TRIGGER_OPEN_THRESHOLD and trigger_state in ("fresh", "held"):
        out.append("trigger_low")
    return out
