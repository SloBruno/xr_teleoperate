"""Fail-closed pressure mapping and Vuer motion-controller haptic transport."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

try:
    from vuer.schemas import MotionControllers
except ImportError:  # Keep pure mapper tests independent of the optional XR dependency.
    MotionControllers = None


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
        try:
            sample_age = float(sample_age)
        except (TypeError, ValueError):
            sample_age = float("nan")
        if (
            not np.isfinite(sample_age)
            or sample_age < 0.0
            or not np.isfinite(self.max_age)
            or sample_age > self.max_age
        ):
            self._last_output = 0.0
            self._last_time = now
            return 0.0
        try:
            values = np.asarray(pressure, dtype=float)
        except (TypeError, ValueError):
            values = np.asarray([], dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            self._last_output = 0.0
            self._last_time = now
            return 0.0
        target = pressure_to_haptic(pressure, self.maximum, self.deadband)
        if not np.isfinite(target) or not np.isfinite(self.max_rate) or self.max_rate < 0.0:
            self._last_output = 0.0
            self._last_time = now
            return 0.0
        if target <= 0.0:
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
    """Map timestamped pressure and emit verified Vuer controller pulses."""

    MAX_DURATION_MS = 250
    DEFAULT_DURATION_MS = 50
    DEFAULT_MIN_INTERVAL = 1.0 / 30.0

    def __init__(self, session: Any = None, maximum: float = 10.0,
                 deadband: float = 1.0, max_rate: float = 4.0,
                 max_age: float = 0.25, duration_ms: int = DEFAULT_DURATION_MS,
                 min_interval: float = DEFAULT_MIN_INTERVAL,
                 clock: Callable[[], float] = time.monotonic,
                 mapper_by_side: dict[str, PressureHapticMapper] | None = None) -> None:
        self.session = session
        self.clock = clock
        self.duration_ms = duration_ms
        self.min_interval = min_interval
        self._sequence = 0
        self._last_emit_time = {"left": -float("inf"), "right": -float("inf")}
        self._mappers = mapper_by_side or {
            side: PressureHapticMapper(maximum, deadband, max_rate, max_age, clock)
            for side in ("left", "right")
        }

    @property
    def supported(self) -> bool:
        return self.session is not None and MotionControllers is not None

    def limit_duration(self, duration_ms: Any) -> int:
        try:
            duration = int(duration_ms)
        except (TypeError, ValueError):
            return 0
        return max(0, min(duration, self.MAX_DURATION_MS))

    def emit(self, side: str, intensity: float, duration_ms: Any = 0) -> bool:
        if side not in self._last_emit_time or self.session is None or MotionControllers is None:
            return False
        try:
            intensity = float(intensity)
            interval = float(self.min_interval)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(intensity) or not 0.0 < intensity <= 1.0:
            return False
        duration = self.limit_duration(duration_ms)
        if duration <= 0 or not np.isfinite(interval) or interval < 0.0:
            return False
        now = self.clock()
        if not np.isfinite(now) or now - self._last_emit_time[side] < interval:
            return False
        self._sequence += 1
        kwargs = {"key": "motionControllers", "left": True, "right": True}
        if side == "left":
            kwargs.update(pulseLeftStrength=intensity, pulseLeftDuration=duration,
                          puseLeftHash=f"dex3-left-{self._sequence}")
        else:
            kwargs.update(pulseRightStrength=intensity, pulseRightDuration=duration,
                          puseRightHash=f"dex3-right-{self._sequence}")
        try:
            self.session.upsert @ MotionControllers(**kwargs)
        except Exception:
            return False
        self._last_emit_time[side] = now
        return True

    def map_pressure(self, side: str, pressure: Any, sample_timestamp: Any) -> float:
        """Return a bounded intensity only for a fresh, finite side sample."""
        if side not in self._mappers:
            return 0.0
        now = self.clock()
        try:
            timestamp = float(sample_timestamp)
        except (TypeError, ValueError):
            self._mappers[side].update(np.array([np.nan]), sample_age=0.0)
            return 0.0
        if not np.isfinite(timestamp) or not np.isfinite(now) or timestamp > now:
            self._mappers[side].update(np.array([np.nan]), sample_age=0.0)
            return 0.0
        return self._mappers[side].update(pressure, sample_age=now - timestamp)

    def emit_pressure(self, side: str, pressure: Any, sample_timestamp: Any,
                      duration_ms: Any = None) -> bool:
        intensity = self.map_pressure(side, pressure, sample_timestamp)
        return self.emit(side, intensity, self.duration_ms if duration_ms is None else duration_ms)
