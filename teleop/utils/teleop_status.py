"""Structured, rate-limited runtime status telemetry for XR teleoperation.

This module only observes and reports. It never changes robot commands or
lifecycle state, so operators can correlate command-path health with robot-side
logs after a fault without introducing a second actuator authority.
"""

import json
import math
from typing import Callable, Mapping, Sequence

from teleop.utils.quest_safety import controller_sample_is_fresh


def camera_frame_is_usable(image: object) -> bool:
    """Return whether an image wrapper contains a usable BGR frame."""
    return image is not None and getattr(image, "bgr", None) is not None


def _age_ms(timestamp: float, now: float) -> int | None:
    """Return non-negative sample age rounded to milliseconds, or ``None``."""
    if not math.isfinite(timestamp) or timestamp <= 0.0 or not math.isfinite(now):
        return None
    age = now - timestamp
    if age < 0.0:
        return None
    return round(age * 1000)


class TeleopStatusMonitor:
    """Emit JSON status heartbeats and controller freshness transitions."""

    def __init__(self, emit: Callable[[str], None], interval_s: float = 1.0):
        if interval_s <= 0.0:
            raise ValueError("status interval must be positive")
        self._emit = emit
        self._interval_s = interval_s
        self._last_status_at: float | None = None
        self._last_controller_fresh: bool | None = None

    def observe(
        self,
        *,
        now: float,
        lifecycle: str,
        controller_sample_timestamp: float,
        motion_enabled: bool = False,
        locomotion: Sequence[float] = (0.0, 0.0, 0.0),
        cameras: Mapping[str, object] | None = None,
        dex3_pressure_timestamps: tuple[float, float] = (0.0, 0.0),
    ) -> dict | None:
        """Observe a control cycle and emit a heartbeat at the configured rate."""
        controller_age_ms = _age_ms(controller_sample_timestamp, now)
        controller_fresh = controller_sample_is_fresh(controller_sample_timestamp, now)
        if self._last_controller_fresh is None:
            self._last_controller_fresh = controller_fresh
        elif controller_fresh != self._last_controller_fresh:
            self._last_controller_fresh = controller_fresh
            self._emit(json.dumps({
                "event": "controller_freshness_changed",
                "fresh": controller_fresh,
                "age_ms": controller_age_ms,
            }, sort_keys=True))

        if self._last_status_at is not None and now - self._last_status_at < self._interval_s:
            return None
        self._last_status_at = now
        left_pressure_timestamp, right_pressure_timestamp = dex3_pressure_timestamps
        status = {
            "event": "teleop_status",
            "lifecycle": lifecycle,
            "controller": {"fresh": controller_fresh, "age_ms": controller_age_ms},
            "locomotion": {
                "enabled": bool(motion_enabled),
                "command": [float(value) for value in locomotion],
            },
            "cameras": {name: bool(available) for name, available in (cameras or {}).items()},
            "dex3_pressure": {
                "left_age_ms": _age_ms(left_pressure_timestamp, now),
                "right_age_ms": _age_ms(right_pressure_timestamp, now),
            },
        }
        self._emit(json.dumps(status, sort_keys=True))
        return status
