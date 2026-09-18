"""Structured, rate-limited runtime status telemetry for XR teleoperation.

This module only observes and reports. It never changes robot commands or
lifecycle state, so operators can correlate command-path health with robot-side
logs after a fault without introducing a second actuator authority.
"""

import json
import math
import os
from queue import Full, Queue
import threading
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


class AsyncStatusFileSink:
    """Append status records from a bounded background writer thread."""

    def __init__(self, path: str, warn: Callable[[str], None], queue_size: int = 64):
        self._path = path
        self._warn = warn
        self._queue = Queue(maxsize=queue_size)
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def emit(self, payload: str) -> None:
        """Queue telemetry without performing filesystem I/O on the control path."""
        if self._closed:
            return
        try:
            self._queue.put_nowait(payload)
        except Full:
            return

    def close(self) -> None:
        """Flush queued records briefly during normal program shutdown."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except Full:
            try:
                self._queue.get_nowait()
            except Exception:
                pass
            try:
                self._queue.put_nowait(None)
            except Full:
                pass
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            directory = os.path.dirname(self._path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as status_log:
                while True:
                    payload = self._queue.get()
                    if payload is None:
                        return
                    status_log.write(payload + "\n")
                    status_log.flush()
        except OSError as error:
            self._warn(f"Could not initialize teleop status log: {error}")


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
