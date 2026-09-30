"""Pure, I/O-free helpers for extended Dex3-1 telemetry.

Real ``HandState_`` fields (unitree_hg IDL): ``motor_state[i]`` with ``mode, q,
dq, ddq, tau_est, temperature[2], vol, sensor[2], motorstate, reserve[4]``;
``press_sensor_state[j]`` with ``pressure[12], temperature[12], lost, reserve``;
``imu_state``; ``power_v, power_a, system_v, device_v``; ``error[2]``;
``reserve[2]``.

Nothing here touches the filesystem, JSON or sleeps. Every function tolerates
missing/garbled fields and maps them (and non-finite numbers) to ``None``.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Iterable, Mapping

EXTENDED_SCHEMA_VERSION = 1
MAX_PRESSURE_SENSORS = 16
MAX_LIST_LEN = 16

# Fields that change slowly; emitted on change or every ``slow_every`` records.
SLOW_JOINT_FIELDS = ("temperature", "vol", "sensor", "mode", "motorstate", "reserve")
SLOW_HAND_FIELDS = (
    "power_v", "power_a", "system_v", "device_v", "error", "reserve",
    "pressure_lost", "pressure_reserve", "pressure_temperature_max",
)


def clean_number(value: Any) -> float | int | None:
    """Return a JSON-safe number; bool/None/non-finite/garbage -> None."""
    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    try:
        if isinstance(value, int) or (hasattr(value, "dtype") and getattr(value.dtype, "kind", "") in "iu"):
            return int(value)
    except (TypeError, ValueError):
        pass
    return number


def clean_list(values: Any, limit: int = MAX_LIST_LEN) -> list | None:
    try:
        items = list(values)[:limit]
    except TypeError:
        return None
    return [clean_number(item) for item in items]


def sanitize(value: Any) -> Any:
    """Recursively make a payload JSON-safe (non-finite -> None)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        try:
            items = value.tolist() if hasattr(value, "tolist") else value
            if not isinstance(items, (list, tuple)):
                return clean_number(items)
            return [sanitize(item) for item in items]  # type: ignore[union-attr]
        except Exception:
            return None
    return clean_number(value)


def _attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return None


def extract_hand_snapshot(hand_state: Any, joint_ids: Iterable[int]) -> dict:
    """Extract per-joint and per-hand extended fields from a HandState_."""
    motors = _attr(hand_state, "motor_state")
    joints = []
    for joint_id in joint_ids:
        try:
            motor = motors[int(joint_id)]
        except Exception:
            motor = None
        joints.append({
            "q": clean_number(_attr(motor, "q")),
            "dq": clean_number(_attr(motor, "dq")),
            "tau_est": clean_number(_attr(motor, "tau_est")),
            "temperature": clean_list(_attr(motor, "temperature"), 2),
            "vol": clean_number(_attr(motor, "vol")),
            "sensor": clean_list(_attr(motor, "sensor"), 2),
            "mode": clean_number(_attr(motor, "mode")),
            "motorstate": clean_number(_attr(motor, "motorstate")),
            "reserve": clean_list(_attr(motor, "reserve"), 4),
        })
    sensors = _attr(hand_state, "press_sensor_state")
    pressure_max, pressure_lost, pressure_reserve, pressure_temp_max = [], [], [], []
    try:
        sensor_list = list(sensors)[:MAX_PRESSURE_SENSORS] if sensors is not None else []
    except TypeError:
        sensor_list = []
    for sensor in sensor_list:
        values = clean_list(_attr(sensor, "pressure"), 12) or []
        finite = [v for v in values if v is not None]
        pressure_max.append(max(finite) if finite else None)
        temps = [v for v in (clean_list(_attr(sensor, "temperature"), 12) or []) if v is not None]
        pressure_temp_max.append(max(temps) if temps else None)
        pressure_lost.append(clean_number(_attr(sensor, "lost")))
        pressure_reserve.append(clean_number(_attr(sensor, "reserve")))
    return {
        "joints": joints,
        "hand": {
            "power_v": clean_number(_attr(hand_state, "power_v")),
            "power_a": clean_number(_attr(hand_state, "power_a")),
            "system_v": clean_number(_attr(hand_state, "system_v")),
            "device_v": clean_number(_attr(hand_state, "device_v")),
            "error": clean_list(_attr(hand_state, "error"), 2),
            "reserve": clean_list(_attr(hand_state, "reserve"), 2),
            "pressure_max": pressure_max,
            "pressure_lost": pressure_lost,
            "pressure_reserve": pressure_reserve,
            "pressure_temperature_max": pressure_temp_max,
        },
    }


def extract_published_command(motor_cmds: Any, joint_ids: Iterable[int]) -> dict:
    """Snapshot the command actually written (q, dq, tau, kp, kd, mode)."""
    out = {key: [] for key in ("q", "dq", "tau", "kp", "kd", "mode")}
    for joint_id in joint_ids:
        try:
            cmd = motor_cmds[int(joint_id)]
        except Exception:
            cmd = None
        for key in out:
            out[key].append(clean_number(_attr(cmd, key)))
    return out


class RateEstimator:
    """Receive-rate estimate over the last ``window`` timestamps."""

    def __init__(self, window: int = 50):
        self._stamps: deque = deque(maxlen=max(2, int(window)))
        self.count = 0

    def add(self, timestamp: float) -> None:
        self.count += 1
        self._stamps.append(float(timestamp))

    def rate_hz(self) -> float | None:
        if len(self._stamps) < 2:
            return None
        span = self._stamps[-1] - self._stamps[0]
        if span <= 0:
            return None
        return (len(self._stamps) - 1) / span


class Dex3SlowFieldGate:
    """Drop unchanged slow fields except every ``slow_every``-th record.

    Fast fields (q, dq, tau_est, command, age, rate) are always kept. Slow
    fields (temperature, voltage, mode, error/lost, power...) are emitted when
    they changed or periodically, bounding JSONL size. Pure state machine.
    """

    def __init__(self, slow_every: int = 10):
        self.slow_every = max(1, int(slow_every))
        self._counter: dict[str, int] = {}
        self._last: dict[str, Any] = {}

    def apply(self, side: str, extended: Mapping[str, Any] | None) -> dict | None:
        if not isinstance(extended, Mapping):
            return None
        data = sanitize(extended)
        slow = {
            "joints": [{k: j.get(k) for k in SLOW_JOINT_FIELDS} for j in data.get("joints") or [] if isinstance(j, dict)],
            "hand": {k: (data.get("hand") or {}).get(k) for k in SLOW_HAND_FIELDS},
        }
        n = self._counter.get(side, 0)
        include_slow = n % self.slow_every == 0 or self._last.get(side) != slow
        self._counter[side] = n + 1
        self._last[side] = slow
        if include_slow:
            data["slow_included"] = True
            return data
        for joint in data.get("joints") or []:
            if isinstance(joint, dict):
                for key in SLOW_JOINT_FIELDS:
                    joint.pop(key, None)
        hand = data.get("hand")
        if isinstance(hand, dict):
            for key in SLOW_HAND_FIELDS:
                hand.pop(key, None)
        data["slow_included"] = False
        return data


def build_extended_payload(samples: Any, gate: Dex3SlowFieldGate | None) -> dict | None:
    """Apply slow-field decimation to controller samples; never raises."""
    if not isinstance(samples, Mapping):
        return None
    out: dict = {}
    for side in ("left", "right"):
        side_sample = samples.get(side)
        if not isinstance(side_sample, Mapping):
            out[side] = None
            continue
        item = dict(side_sample)
        try:
            state = item.get("state")
            if gate is not None and state is not None:
                item["state"] = gate.apply(side, state)
        except Exception:
            item["state"] = None
        out[side] = item
    return out


def collect_extended_payload(controller: Any, gate: Dex3SlowFieldGate | None, warn=None) -> dict | None:
    """Best-effort in-memory read of controller telemetry for the record."""
    try:
        return build_extended_payload(controller.get_extended_samples(), gate)
    except Exception as error:
        try:
            if warn is not None:
                warn(f"Dex3 extended telemetry unavailable: {type(error).__name__}")
        except Exception:
            pass
        return None
