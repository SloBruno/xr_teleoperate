"""Fail-closed pressure mapping and optional XR haptic transport.

The current local Vuer dependency exposes no verified haptic session method.
HapticTransportAdapter therefore remains deliberately unsupported until one is
verified against the installed session implementation.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np


def pressure_to_haptic(
    pressure: np.ndarray, maximum: float, deadband: float = 0.0
) -> float:
    """Map a finite pressure sample to a bounded contact intensity."""
    try:
        values = np.asarray(pressure, dtype=float)
        maximum = float(maximum)
        deadband = float(deadband)
    except (TypeError, ValueError):
        return 0.0
    if values.size == 0 or not np.all(np.isfinite(values)):
        return 0.0
    if not np.isfinite(maximum) or maximum <= 0.0:
        return 0.0
    if not np.isfinite(deadband) or deadband < 0.0 or deadband >= maximum:
        return 0.0
    peak = float(np.max(values))
    if peak <= deadband:
        return 0.0
    return float(np.clip((peak - deadband) / (maximum - deadband), 0.0, 1.0))


class PressureHapticMapper:
    """Convert pressure samples while bounding change rate and sample age."""

    def __init__(
        self,
        maximum: float,
        deadband: float = 0.0,
        max_rate: float = 4.0,
        max_age: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.maximum = maximum
        self.deadband = deadband
        self.max_rate = max_rate
        self.max_age = max_age
        self.clock = clock
        self._last_time = clock()
        self._last_output = 0.0

    def update(self, pressure: np.ndarray, sample_age: float = 0.0) -> float:
        now = self.clock()
        if (
            not np.isfinite(sample_age)
            or sample_age < 0.0
            or not np.isfinite(self.max_age)
            or sample_age > self.max_age
        ):
            self._last_output = 0.0
            self._last_time = now
            return 0.0
        target = pressure_to_haptic(pressure, self.maximum, self.deadband)
        if not np.isfinite(target) or not np.isfinite(self.max_rate) or self.max_rate < 0.0:
            self._last_output = 0.0
            self._last_time = now
            return 0.0
        elapsed = max(0.0, now - self._last_time)
        limit = self.max_rate * elapsed
        output = float(np.clip(target, self._last_output - limit, self._last_output + limit))
        self._last_output = output
        self._last_time = now
        return output


def extract_dex3_pressure(hand_state: Any) -> float:
    """Return the peak pressure from a verified Dex3 HandState_ side sample."""
    try:
        sensors = hand_state.press_sensor_state
        values = np.concatenate([np.asarray(sensor.pressure, dtype=float) for sensor in sensors])
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if values.size == 0 or not np.all(np.isfinite(values)):
        return 0.0
    return float(np.max(values))


class HapticTransportAdapter:
    """Explicit no-op until a local Vuer haptic session API is verified."""

    supported = False
    MAX_DURATION_MS = 250

    def __init__(self, session: Any = None) -> None:
        self.session = session

    def limit_duration(self, duration_ms: Any) -> int:
        try:
            duration = int(duration_ms)
        except (TypeError, ValueError):
            return 0
        return max(0, min(duration, self.MAX_DURATION_MS))

    def emit(self, side: str, intensity: float, duration_ms: Any = 0) -> bool:
        """Return False and emit nothing because no supported API is known."""
        return False
