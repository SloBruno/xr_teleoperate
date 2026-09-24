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


class _FrozenMappingSnapshot(tuple):
    """Immutable mapping snapshot thawed only by the writer thread."""


def _freeze_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenMappingSnapshot(
            (key, _freeze_payload(item)) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(_freeze_payload(item) for item in value)
    return value


def _thaw_payload(value: object) -> object:
    if isinstance(value, _FrozenMappingSnapshot):
        return {key: _thaw_payload(item) for key, item in value}
    if isinstance(value, tuple):
        return [_thaw_payload(item) for item in value]
    return value


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


def _utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def build_lifecycle_event(
    event: str, *, timestamp: float, timestamp_monotonic: float, cause: str | None = None
) -> dict:
    """Build a lifecycle snapshot without JSON or filesystem work."""
    monotonic_timestamp = _finite_timestamp(timestamp_monotonic)
    if monotonic_timestamp is None:
        raise ValueError("timestamp_monotonic must be a finite positive monotonic timestamp")
    record = {
        "schema_version": 1,
        "event": str(event),
        "timestamp_utc": _utc_timestamp(timestamp),
        "timestamp_monotonic": monotonic_timestamp,
        "clock_domain": {
            "timestamp": "wall_clock_utc",
            "timestamp_monotonic": "monotonic",
        },
    }
    if cause is not None:
        record["cause"] = str(cause)
    return record


def build_pose_record(
    *,
    timestamp: float,
    timestamp_monotonic: float,
    lifecycle: str,
    controller_sample_timestamp: float,
    left_wrist_pose: object,
    right_wrist_pose: object,
    measured_arm_q: object,
    commanded_arm_q: object,
    commanded_arm_q_reason: str | None = None,
    dex3_measured_q: object = None,
    dex3_commanded_q: object = None,
    dex3_configured: bool = False,
    dex3_sample_metadata: Mapping[str, Mapping[str, object]] | None = None,
    drop_count: int = 0,
    now: float | None = None,
) -> dict:
    """Build a serializable snapshot without doing JSON or filesystem I/O."""
    monotonic_timestamp = _finite_timestamp(timestamp_monotonic)
    if monotonic_timestamp is None:
        raise ValueError("timestamp_monotonic must be a finite positive monotonic timestamp")
    sample_timestamp = _finite_timestamp(controller_sample_timestamp)
    freshness_now = monotonic_timestamp if now is None else now
    measured_values = _split_dex3(_vector_or_none(dex3_measured_q, expected_size=14))
    commanded_values = _split_dex3(_vector_or_none(dex3_commanded_q, expected_size=14))
    dex3_sample_metadata = dex3_sample_metadata or {}
    dex3 = {}
    sample_count = 0
    for side in ("left", "right"):
        metadata = dex3_sample_metadata.get(side, {})
        state_valid = bool(metadata.get("state_valid", False)) and _finite_timestamp(
            metadata.get("state_timestamp")
        ) is not None
        action_valid = bool(metadata.get("action_valid", False)) and _finite_timestamp(
            metadata.get("action_timestamp")
        ) is not None
        sample_count += int(state_valid) + int(action_valid)
        dex3[side] = {
            "state_valid": state_valid,
            "state_timestamp": _finite_timestamp(metadata.get("state_timestamp")) if state_valid else None,
            "action_valid": action_valid,
            "action_timestamp": _finite_timestamp(metadata.get("action_timestamp")) if action_valid else None,
            "measured_q": measured_values[side] if state_valid else None,
            "commanded_q": commanded_values[side] if action_valid else None,
        }
    dex3_available = sample_count > 0
    if not dex3_configured:
        dex3_reason = "dex3_not_configured"
    elif sample_count == 0:
        dex3_reason = "dex3_configured_no_sample"
    elif sample_count < 4:
        dex3_reason = "dex3_partially_sampled"
    else:
        dex3_reason = "dex3_sampled"
    arm_measured = _split_arm(_vector_or_none(measured_arm_q, expected_size=14))
    arm_commanded = _split_arm(_vector_or_none(commanded_arm_q, expected_size=14))
    arm_record = {
        "left": {
            "measured_q": arm_measured["left"],
            "commanded_q": arm_commanded["left"],
        },
        "right": {
            "measured_q": arm_measured["right"],
            "commanded_q": arm_commanded["right"],
        },
    }
    if commanded_arm_q_reason is not None:
        for side in arm_record.values():
            side["commanded_q_reason"] = str(commanded_arm_q_reason)
    return {
        "schema_version": 1,
        "event": "full_pose_telemetry",
        "timestamp": float(timestamp),
        "timestamp_utc": _utc_timestamp(timestamp),
        "timestamp_monotonic": float(monotonic_timestamp),
        "clock_domain": {
            "timestamp": "wall_clock_utc",
            "timestamp_monotonic": "monotonic",
            "controller_sample_timestamp": "monotonic",
        },
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
        "arm": arm_record,
        "dex3": {
            "available": dex3_available,
            "reason": dex3_reason,
            "left": dex3["left"],
            "right": dex3["right"],
        },
        "achieved_cartesian_pose": {"left": None, "right": None},
        "achieved_cartesian_pose_reason": "not_available_from_controller_state",
        "drop_count": max(0, int(drop_count)),
    }


def publish_arm_command_for_telemetry(controller, q_target, tauff_target):
    """Return the controller's published q snapshot and an explicit outcome reason."""
    try:
        publication = controller.ctrl_dual_arm(q_target, tauff_target)
    except Exception as error:
        return None, f"arm_command_publication_failed:{type(error).__name__}"
    if publication is None:
        return None, "arm_command_publication_unavailable"
    if isinstance(publication, Mapping):
        published_q = publication.get("published_q")
        reason = publication.get("reason")
    else:
        published_q = getattr(publication, "published_q", None)
        reason = getattr(publication, "reason", None)
    published_q = _vector_or_none(published_q, expected_size=14)
    if published_q is None:
        return None, str(reason or "arm_command_publication_unavailable")
    return np.asarray(published_q, dtype=float), str(reason or "published")


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
        self._queue: Queue[object] = Queue(maxsize=queue_size)
        self._close_timeout_s = close_timeout_s
        self._closed = False
        self._state_lock = threading.Lock()
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
        with self._state_lock:
            if self._closed:
                return False
            try:
                self._queue.put_nowait(_freeze_payload(record))
                return True
            except Full:
                with self._drop_lock:
                    self._drop_count += 1
                return False

    def close(self) -> None:
        with self._state_lock:
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
                        telemetry_log.write(json.dumps(
                            _thaw_payload(record), ensure_ascii=False, separators=(",", ":")
                        ) + "\n")
                        telemetry_log.flush()
                    finally:
                        self._queue.task_done()
        except OSError as error:
            self._warn(f"Could not initialize pose telemetry log: {error}")
        except (TypeError, ValueError) as error:
            self._warn(f"Could not serialize pose telemetry record: {error}")


class _DisabledPoseTelemetrySink:
    drop_count = 0

    def emit(self, record: Mapping[str, object]) -> bool:
        return False

    def close(self) -> None:
        return None


def create_pose_telemetry_sink(
    directory: str | os.PathLike[str], warn: Callable[[str], None] | None = None
):
    """Create telemetry storage without making it a teleoperation prerequisite."""
    warn = warn or (lambda message: None)
    try:
        return PoseTelemetryJsonlSink(directory, warn)
    except Exception as error:
        warn(f"Could not initialize pose telemetry sink: {error}")
        return _DisabledPoseTelemetrySink()
