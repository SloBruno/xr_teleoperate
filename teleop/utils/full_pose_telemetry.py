"""Full-pose telemetry with a control-loop-safe producer and JSONL worker."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import uuid
from typing import Callable, Mapping

import numpy as np

from .quest_safety import controller_sample_is_fresh


def _finite_timestamp(value: float) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def _age_ms(timestamp: float | None, now: float) -> int | None:
    timestamp = _finite_timestamp(timestamp)
    if timestamp is None or not math.isfinite(now) or now < timestamp:
        return None
    return round((now - timestamp) * 1000)


def _matrix_or_none(pose: object) -> list[list[float]] | None:
    if pose is None:
        return None
    try:
        matrix = np.asarray(pose, dtype=float).reshape(4, 4)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(matrix).all():
        return None
    return matrix.tolist()


def _vector_or_none(values: object, expected_size: int | None = None) -> list[float] | None:
    if values is None:
        return None
    try:
        vector = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if expected_size is not None and vector.size != expected_size:
        return None
    if not np.isfinite(vector).all():
        return None
    return vector.tolist()


def _split_arm(values: list[float] | None) -> dict[str, list[float] | None]:
    if values is None or len(values) != 14:
        return {"left": None, "right": None}
    return {"left": values[:7], "right": values[7:]}


def _split_dex3(values: list[float] | None) -> dict[str, list[float] | None]:
    if values is None or len(values) != 14:
        return {"left": None, "right": None}
    return {"left": values[:7], "right": values[7:]}


def build_pose_record(
    *,
    timestamp: float,
    lifecycle: str,
    controller_sample_timestamp: float,
    left_wrist_pose: object,
    right_wrist_pose: object,
    measured_arm_q: object,
    commanded_arm_q: object,
    dex3_measured_q: object = None,
    dex3_commanded_q: object = None,
    drop_count: int = 0,
    now: float | None = None,
) -> dict:
    """Build a serializable snapshot without doing JSON or filesystem I/O."""
    sample_timestamp = _finite_timestamp(controller_sample_timestamp)
    freshness_now = timestamp if now is None else now
    dex3_measured = _split_dex3(_vector_or_none(dex3_measured_q, expected_size=14))
    dex3_commanded = _split_dex3(_vector_or_none(dex3_commanded_q, expected_size=14))
    dex3_available = dex3_measured_q is not None or dex3_commanded_q is not None
    return {
        "schema_version": 1,
        "event": "full_pose_telemetry",
        "timestamp": float(timestamp),
        "lifecycle": str(lifecycle),
        "controller": {
            "sample_timestamp": sample_timestamp,
            "fresh": bool(
                sample_timestamp is not None
                and controller_sample_is_fresh(sample_timestamp, now=freshness_now)
            ),
            "age_ms": _age_ms(sample_timestamp, freshness_now),
            "wrist_pose": {
                "left": _matrix_or_none(left_wrist_pose),
                "right": _matrix_or_none(right_wrist_pose),
            },
        },
        "arm": {
            "left": {
                "measured_q": _split_arm(_vector_or_none(measured_arm_q, expected_size=14))["left"],
                "commanded_q": _split_arm(_vector_or_none(commanded_arm_q, expected_size=14))["left"],
            },
            "right": {
                "measured_q": _split_arm(_vector_or_none(measured_arm_q, expected_size=14))["right"],
                "commanded_q": _split_arm(_vector_or_none(commanded_arm_q, expected_size=14))["right"],
            },
        },
        "dex3": {
            "available": dex3_available,
            "reason": None if dex3_available else "dex3_not_configured",
            "left": {
                "measured_q": dex3_measured["left"],
                "commanded_q": dex3_commanded["left"],
            },
            "right": {
                "measured_q": dex3_measured["right"],
                "commanded_q": dex3_commanded["right"],
            },
        },
        "achieved_cartesian_pose": {"left": None, "right": None},
        "achieved_cartesian_pose_reason": "not_available_from_controller_state",
        "drop_count": max(0, int(drop_count)),
    }


class PoseTelemetryJsonlSink:
    """Write validated records from a bounded queue in a daemon worker."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        warn: Callable[[str], None] | None = None,
        queue_size: int = 256,
        close_timeout_s: float = 1.0,
    ):
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if close_timeout_s <= 0:
            raise ValueError("close_timeout_s must be positive")
        self.directory = Path(directory)
        session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = self.directory / f"pose-telemetry-{session}-{uuid.uuid4().hex}.jsonl"
        self._warn = warn or (lambda message: None)
        self._queue: Queue[Mapping[str, object]] = Queue(maxsize=queue_size)
        self._close_timeout_s = close_timeout_s
        self._closed = False
        self._close_requested = threading.Event()
        self._drop_count = 0
        self._drop_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="pose-telemetry-writer", daemon=True)
        self._thread.start()

    @property
    def drop_count(self) -> int:
        with self._drop_lock:
            return self._drop_count

    def emit(self, record: Mapping[str, object]) -> bool:
        """Queue a record without blocking or serializing on the caller thread."""
        if self._closed:
            return False
        try:
            self._queue.put_nowait(record)
            return True
        except Full:
            with self._drop_lock:
                self._drop_count += 1
            return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_requested.set()
        self._thread.join(timeout=self._close_timeout_s)
        if self._thread.is_alive():
            self._warn("Pose telemetry writer did not stop before close timeout")

    def _run(self) -> None:
        try:
            os.makedirs(self.directory, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as telemetry_log:
                while not self._close_requested.is_set() or not self._queue.empty():
                    try:
                        record = self._queue.get(timeout=0.05)
                    except Empty:
                        continue
                    try:
                        telemetry_log.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                        telemetry_log.flush()
                    finally:
                        self._queue.task_done()
        except OSError as error:
            self._warn(f"Could not initialize pose telemetry log: {error}")
        except (TypeError, ValueError) as error:
            self._warn(f"Could not serialize pose telemetry record: {error}")
