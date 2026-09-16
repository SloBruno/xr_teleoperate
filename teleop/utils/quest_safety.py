"""Fail-safe handling for timestamped Quest controller samples."""

import math
import time


# Controller values are accepted for at most 250 ms after their monotonic sample.
CONTROLLER_SAMPLE_FRESHNESS_LIMIT_S = 0.25


def controller_sample_is_fresh(sample_timestamp: float, now: float | None = None) -> bool:
    """Return whether a controller sample is no older than 0.25 s."""
    if now is None:
        now = time.monotonic()
    if not math.isfinite(sample_timestamp) or sample_timestamp <= 0.0 or not math.isfinite(now):
        return False
    age = now - sample_timestamp
    return 0.0 <= age <= CONTROLLER_SAMPLE_FRESHNESS_LIMIT_S


def fresh_controller_value(value, sample_timestamp: float, now: float | None = None):
    """Return a controller value only while its sample remains fresh."""
    return value if controller_sample_is_fresh(sample_timestamp, now) else 0.0 if isinstance(value, (int, float)) else (0.0, 0.0)
