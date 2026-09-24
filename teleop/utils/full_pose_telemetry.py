"""Full-pose telemetry with a control-loop-safe producer and JSONL worker."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
import time
from pathlib import Path
from collections import OrderedDict, deque
from queue import Empty, Full, Queue
import threading
import uuid
from typing import Callable, Mapping
from dataclasses import dataclass

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


def _warn_best_effort(warn: Callable[[str], None] | None, message: str) -> None:
    try:
        if warn is not None:
            warn(message)
    except Exception:
        pass


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


def _vector_size(values: object) -> int | None:
    if values is None:
        return None
    try:
        return int(np.asarray(values, dtype=float).reshape(-1).size)
    except (TypeError, ValueError):
        return None


def _normalize_arm_joint_split(arm_joint_split: object) -> tuple[int, int]:
    try:
        left_count, right_count = arm_joint_split
        split = (int(left_count), int(right_count))
    except (TypeError, ValueError):
        raise ValueError("arm_joint_split must contain two positive joint counts")
    if split[0] <= 0 or split[1] <= 0 or (left_count, right_count) != split:
        raise ValueError("arm_joint_split must contain two positive joint counts")
    return split


def _split_arm(
    values: list[float] | None, arm_joint_split: tuple[int, int]
) -> dict[str, list[float] | None]:
    left_count, right_count = arm_joint_split
    if values is None or len(values) != left_count + right_count:
        return {"left": None, "right": None}
    return {"left": values[:left_count], "right": values[left_count:]}


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


def emit_lifecycle_event_best_effort(
    sink,
    event: str,
    *,
    warn: Callable[[str], None],
    cause: str | None = None,
) -> bool:
    """Build and enqueue one lifecycle event without affecting control/cleanup."""
    if sink is None:
        return False
    try:
        sink.emit(build_lifecycle_event(
            event,
            cause=cause,
            timestamp=time.time(),
            timestamp_monotonic=time.monotonic(),
        ))
    except Exception as error:
        _warn_best_effort(warn, f"Failed to emit lifecycle telemetry {event}: {type(error).__name__}")
        return False
    return True


def emit_pose_record_best_effort(
    emit: Callable[[Mapping[str, object]], object],
    builder: Callable[..., Mapping[str, object]] | None = None,
    *,
    warn: Callable[[str], None] | None,
    **kwargs,
) -> bool:
    """Build and hand off one pose record without affecting control."""
    try:
        if builder is None:
            builder = build_pose_record
        record = builder(**kwargs)
        return bool(emit(record))
    except Exception as error:
        _warn_best_effort(warn, f"Failed to emit pose telemetry: {type(error).__name__}")
        return False


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
    arm_command_request_id: int | None = None,
    requested_arm_q: object = None,
    selected_arm_q: object = None,
    arm_publication_drop_count: int = 0,
    dex3_measured_q: object = None,
    dex3_commanded_q: object = None,
    dex3_configured: bool = False,
    dex3_sample_metadata: Mapping[str, Mapping[str, object]] | None = None,
    arm_joint_split: tuple[int, int] = (7, 7),
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
    arm_joint_split = _normalize_arm_joint_split(arm_joint_split)
    arm_expected_size = sum(arm_joint_split)
    measured_values = _vector_or_none(measured_arm_q, expected_size=arm_expected_size)
    commanded_values = _vector_or_none(commanded_arm_q, expected_size=arm_expected_size)
    arm_measured = _split_arm(measured_values, arm_joint_split)
    arm_commanded = _split_arm(commanded_values, arm_joint_split)
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
    if any(value is not None for value in (arm_command_request_id, requested_arm_q, selected_arm_q)):
        arm_record["request_id"] = None if arm_command_request_id is None else int(arm_command_request_id)
        for side in ("left", "right"):
            arm_record[side]["requested_q"] = _split_arm(
                _vector_or_none(requested_arm_q, expected_size=arm_expected_size), arm_joint_split
            )[side]
            arm_record[side]["selected_q"] = _split_arm(
                _vector_or_none(selected_arm_q, expected_size=arm_expected_size), arm_joint_split
            )[side]
    if arm_publication_drop_count:
        arm_record["publication_drop_count"] = max(0, int(arm_publication_drop_count))
    if commanded_arm_q is not None and commanded_values is None:
        commanded_arm_q_reason = (
            f"arm_command_dimension_mismatch:expected={arm_expected_size}:"
            f"actual={_vector_size(commanded_arm_q)}"
        )
    if commanded_arm_q_reason is not None:
        for side in (arm_record["left"], arm_record["right"]):
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


@dataclass(frozen=True)
class ArmCommandTelemetry:
    published_q: object
    reason: str
    request_id: int | None = None
    requested_q: object = None
    selected_q: object = None

    def __iter__(self):
        yield self.published_q
        yield self.reason


def publish_arm_command_for_telemetry(controller, q_target, tauff_target):
    """Submit immediately without draining the asynchronous publication receipts."""
    arm_joint_split = _normalize_arm_joint_split(
        getattr(controller, "arm_joint_split", (7, 7))
    )
    expected_size = sum(arm_joint_split)
    target_size = _vector_size(q_target)
    tauff_size = _vector_size(tauff_target)
    if target_size != expected_size or tauff_size != expected_size:
        actual_size = target_size if target_size != expected_size else tauff_size
        return ArmCommandTelemetry(None, f"arm_command_target_dimension_mismatch:expected={expected_size}:actual={actual_size}")
    if _vector_or_none(q_target, expected_size=expected_size) is None:
        return ArmCommandTelemetry(None, "arm_command_target_invalid")
    if _vector_or_none(tauff_target, expected_size=expected_size) is None:
        return ArmCommandTelemetry(None, "arm_command_tauff_target_invalid")
    try:
        publication = controller.ctrl_dual_arm(q_target, tauff_target)
    except Exception as error:
        return ArmCommandTelemetry(None, f"arm_command_publication_failed:{type(error).__name__}")
    if publication is None:
        return ArmCommandTelemetry(None, "arm_command_publication_unavailable")
    requested_q = np.asarray(q_target, dtype=float).copy()
    request_id = None
    if isinstance(publication, Mapping):
        published_q = publication.get("published_q")
        reason = publication.get("reason")
        request_id = publication.get("request_id")
    else:
        published_q = getattr(publication, "published_q", None)
        reason = getattr(publication, "reason", None)
        request_id = publication if isinstance(publication, (int, np.integer)) else getattr(publication, "request_id", None)
    if request_id is not None and published_q is None and reason is None:
        return ArmCommandTelemetry(
            None, "arm_command_publication_pending", int(request_id), requested_q, requested_q.copy()
        )
    published_size = _vector_size(published_q)
    if published_size != expected_size:
        return ArmCommandTelemetry(None, (
            f"arm_command_publication_dimension_mismatch:expected={expected_size}:"
            f"actual={published_size}"
        ), request_id, requested_q, requested_q.copy())
    published_q = _vector_or_none(published_q, expected_size=expected_size)
    if published_q is None:
        publication_reason = str(reason or "arm_command_publication_unavailable")
        if publication_reason == "published":
            publication_reason = "arm_command_publication_invalid"
        return ArmCommandTelemetry(None, publication_reason, request_id, requested_q, requested_q.copy())
    return ArmCommandTelemetry(np.asarray(published_q, dtype=float), str(reason or "published"), request_id, requested_q, requested_q.copy())


def build_arm_publication_event(receipt: object, *, profile: str) -> dict:
    """Build one append-only event from one completed controller receipt."""
    def value(name, default=None):
        return receipt.get(name, default) if isinstance(receipt, Mapping) else getattr(receipt, name, default)

    split = _normalize_arm_joint_split(value("arm_joint_split", (7, 7)))
    raw_published_q = value("published_q")
    published_q = _vector_or_none(raw_published_q, expected_size=sum(split))
    reason = str(value("reason", "arm_command_publication_unavailable"))
    if raw_published_q is not None and published_q is None:
        raise ValueError("arm publication receipt published_q has invalid dimensions or values")
    timestamp_monotonic = _finite_timestamp(value("timestamp_monotonic"))
    if timestamp_monotonic is None:
        raise ValueError("arm publication receipt timestamp must be finite and positive")
    return {
        "schema_version": 1,
        "event": "arm_publication",
        "request_id": int(value("request_id")),
        "published_q": published_q,
        "reason": reason,
        "timestamp_monotonic": timestamp_monotonic,
        "clock_domain": {"timestamp_monotonic": "monotonic"},
        "profile": str(profile),
        "arm_joint_split": list(split),
    }


class ArmPublicationTelemetryBridge:
    """Correlate bounded asynchronous arm receipts with append-only telemetry.

    A request ID enters the bounded LRU only after its publication event is
    accepted by the sink. IDs already pending are treated as duplicates so a
    delayed retry cannot create another pending record. Once an ID is evicted,
    it may be accepted again; this is the intentional wrap/restart contract
    because request IDs have no durable session epoch. A producer restart also
    starts with an empty window, so callers needing cross-session uniqueness
    must provide a session identity in their surrounding telemetry.
    """

    def __init__(
        self,
        sink,
        *,
        profile: str,
        pending_capacity: int = 64,
        dedup_capacity: int = 1024,
        warn: Callable[[str], None] | None = None,
    ):
        if pending_capacity <= 0:
            raise ValueError("pending_capacity must be positive")
        if dedup_capacity <= 0:
            raise ValueError("dedup_capacity must be positive")
        self._sink = sink
        self._profile = str(profile)
        self._pending = deque()
        self._pending_capacity = int(pending_capacity)
        self._dedup_capacity = int(dedup_capacity)
        self._seen_request_ids = OrderedDict()
        self._pending_request_ids = set()
        self._pending_buffer_drop_count = 0
        self._telemetry_sink_overflow_count = 0
        self._last_controller_receipt_drop_count = 0
        self._invalid_receipt_count = 0
        self._receipt_drain_failure_count = 0
        self._duplicate_receipt_count = 0
        self._warn = warn

    @property
    def pending_buffer_drop_count(self) -> int:
        return self._pending_buffer_drop_count

    @property
    def telemetry_sink_overflow_count(self) -> int:
        return self._telemetry_sink_overflow_count

    @property
    def invalid_receipt_count(self) -> int:
        return self._invalid_receipt_count

    @property
    def receipt_drain_failure_count(self) -> int:
        return self._receipt_drain_failure_count

    @property
    def duplicate_receipt_count(self) -> int:
        return self._duplicate_receipt_count

    @property
    def seen_request_ids(self) -> tuple[int, ...]:
        return tuple(self._seen_request_ids)

    def _try_emit(self, record: Mapping[str, object]) -> bool:
        try:
            accepted = bool(self._sink.emit(record))
        except Exception as error:
            _warn_best_effort(self._warn, f"Failed to emit arm telemetry: {type(error).__name__}")
            accepted = False
        if not accepted:
            self._telemetry_sink_overflow_count += 1
        return accepted

    def _mark_seen(self, request_id: int) -> None:
        self._pending_request_ids.discard(request_id)
        self._seen_request_ids[request_id] = None
        self._seen_request_ids.move_to_end(request_id)
        while len(self._seen_request_ids) > self._dedup_capacity:
            self._seen_request_ids.popitem(last=False)

    def _retain_or_drop(self, record: Mapping[str, object]) -> None:
        try:
            if len(self._pending) >= self._pending_capacity:
                self._pending_buffer_drop_count += 1
                _warn_best_effort(self._warn, "Dropped arm telemetry from full pending buffer")
                return
            self._pending.append(record)
            self._pending_request_ids.add(int(record["request_id"]))
        except Exception as error:
            self._pending_buffer_drop_count += 1
            _warn_best_effort(self._warn, f"Failed to retain arm telemetry: {type(error).__name__}")

    def _flush_pending(self) -> None:
        try:
            while self._pending:
                record = self._pending[0]
                if not self._try_emit(record):
                    return
                self._pending.popleft()
                self._mark_seen(int(record["request_id"]))
        except Exception as error:
            _warn_best_effort(self._warn, f"Failed to flush pending arm telemetry: {type(error).__name__}")

    def _invalid_receipt_event(self, receipt: object, error: Exception) -> dict:
        try:
            request_id = receipt.get("request_id") if isinstance(receipt, Mapping) else getattr(receipt, "request_id", None)
        except Exception:
            request_id = None
        return {
            "schema_version": 1,
            "event": "arm_publication_invalid",
            "request_id": request_id,
            "reason": f"invalid_receipt:{type(error).__name__}",
            "profile": self._profile,
        }

    def _emit_receipt(self, receipt: object) -> None:
        try:
            event = build_arm_publication_event(receipt, profile=self._profile)
        except Exception as error:
            self._invalid_receipt_count += 1
            _warn_best_effort(self._warn, f"Dropped invalid arm publication receipt: {type(error).__name__}")
            self._try_emit(self._invalid_receipt_event(receipt, error))
            return
        request_id = int(event["request_id"])
        if request_id in self._seen_request_ids or request_id in self._pending_request_ids:
            self._duplicate_receipt_count += 1
            self._try_emit({
                "schema_version": 1,
                "event": "arm_publication_duplicate",
                "request_id": request_id,
                "duplicate_count": self._duplicate_receipt_count,
                "profile": self._profile,
            })
            return
        if not self._try_emit(event):
            self._retain_or_drop(event)
        else:
            self._mark_seen(request_id)

    def emit_cycle(self, pose_record: dict, controller) -> bool:
        """Flush retries, drain all completed receipts, then enqueue this cycle's pose."""
        try:
            self._flush_pending()
        except Exception as error:
            _warn_best_effort(self._warn, f"Failed to flush arm telemetry: {type(error).__name__}")
        try:
            receipts = controller.drain_arm_publication_receipts()
            for receipt in receipts:
                self._emit_receipt(receipt)
        except Exception as error:
            self._receipt_drain_failure_count += 1
            _warn_best_effort(self._warn, f"Failed to drain arm publication receipts: {type(error).__name__}")
        try:
            controller_drop_count = int(getattr(controller, "publication_receipt_drop_count", 0))
            pose_record["telemetry_overflow"] = {
                "controller_receipt_drop_count": controller_drop_count,
                "pending_buffer_drop_count": self._pending_buffer_drop_count,
                "telemetry_sink_overflow_count": self._telemetry_sink_overflow_count,
                "invalid_receipt_count": self._invalid_receipt_count,
                "receipt_drain_failure_count": self._receipt_drain_failure_count,
                "duplicate_receipt_count": self._duplicate_receipt_count,
            }
            self._last_controller_receipt_drop_count = controller_drop_count
        except Exception as error:
            _warn_best_effort(self._warn, f"Failed to annotate arm telemetry: {type(error).__name__}")
        return self._try_emit(pose_record)


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

    def _warn_best_effort(self, message: str) -> None:
        try:
            self._warn(message)
        except BaseException:
            pass

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
            self._warn_best_effort("Pose telemetry writer did not stop before close timeout")

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
            self._warn_best_effort(f"Could not initialize pose telemetry log: {error}")
        except (TypeError, ValueError) as error:
            self._warn_best_effort(f"Could not serialize pose telemetry record: {error}")


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
        try:
            warn(f"Could not initialize pose telemetry sink: {error}")
        except BaseException:
            pass
        return _DisabledPoseTelemetrySink()
