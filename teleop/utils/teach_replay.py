"""G1D teach-and-replay core: recording format, background writer, session.

Hardware-independent. The session owns the whole r/s/p/q lifecycle and is the
ONLY producer of arm/Dex3 commands; it never touches DDS directly but drives
an injected IO adapter (see ``teleop/robot_control/g1d_teach_replay_io.py``)
from a single ticking thread:

* pre-start: nothing is activated or written (passive subscribers only);
* ``r``  TEACH: arms compliant (kp=0, small kd) + model gravity feed-forward at
  the measured q, Dex3 kp=0 / small kd; measured joints are sampled at a fixed
  rate and handed to a bounded background writer (no I/O on this thread);
* ``s``  stop teaching, close the recording, hold the measured pose with the
  normal tracking gains + gravity feed-forward;
* ``p``  REPLAY: bounded smoothstep ramp from the measured pose to the first
  sample, then time-interpolated playback with per-joint step and joint-limit
  clamps; holds the final pose (never goes home);
* ``q``  stop wins over everything: hold the measured pose (gravity ff) for a
  short bounded time, then release all outputs and finish.

Stale robot state while active => HOLD (never keep streaming a trajectory).
Every numeric default below is a first guess for the first supported physical
trial and must be confirmed on hardware ("calibrar no teste fisico").
"""

from __future__ import annotations

import enum
import json
import math
import numbers
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

RECORDING_SCHEMA = "xr_teleoperate.g1d_teach.v1"
HAND_SIDES = ("left", "right")
ARM_DOF = 14
HAND_DOF = 7
ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
# Dex3 slot order of rt/dex3/<side>/cmd (Dex3_1_{Left,Right}_JointIndex).
HAND_JOINT_NAMES = {
    "left": ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"),
    "right": ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"),
}
# Per-side URDF limits (assets/g1/g1_body29_hand14.urdf); mirrored, not copied.
DEX3_LIMITS = {
    "left": (
        np.array([-1.04719755, -0.72431163, 0.0, -1.57079632, -1.74532925, -1.57079632, -1.74532925]),
        np.array([1.04719755, 1.04719755, 1.74532925, 0.0, 0.0, 0.0, 0.0]),
    ),
    "right": (
        np.array([-1.04719755, -1.04719755, -1.74532925, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.04719755, 0.72431163, 0.0, 1.57079632, 1.74532925, 1.57079632, 1.74532925]),
    ),
}
SMOOTHSTEP_PEAK_FACTOR = 1.5   # max d/ds of 3s^2 - 2s^3
# Dex3 encoders read up to ~0.06 rad past the URDF limits at the mechanical
# stops (measured on G1D recordings). Accept that sensor overshoot when loading;
# commands are still clamped strictly inside the limits.
HAND_LIMIT_MEASUREMENT_TOL_RAD = 0.10
# Same idea for the arm encoders at their mechanical stops (measured up to
# ~0.009 rad on G1D); kept small because arm commands carry more energy.
ARM_LIMIT_MEASUREMENT_TOL_RAD = 0.05
# A DDS callback may stamp a sample just after the tick took its time
# reference; accept that small monotonic skew instead of a false "stale".
STATE_FUTURE_SKEW_S = 0.01
MAX_REPLAY_REPEAT = 100


def validate_replay_repeat(repeat) -> int:
    """Return a finite replay count, rejecting coercive/non-integral inputs."""
    if isinstance(repeat, bool) or not isinstance(repeat, numbers.Integral):
        raise ValueError("invalid replay repeat")
    repeat = int(repeat)
    if not 1 <= repeat <= MAX_REPLAY_REPEAT:
        raise ValueError("invalid replay repeat")
    return repeat


class RecordingError(ValueError):
    """Invalid, missing or unusable teach recording."""


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def default_teach_dir(home: Path | None = None) -> Path:
    base = Path.home() if home is None else Path(home)
    return base / ".local" / "state" / "xr_teleoperate" / "teach"


def utc_recording_name(now: datetime | None = None) -> str:
    now = datetime.now(timezone.utc) if now is None else now
    return now.strftime("%Y%m%dT%H%M%S_%fZ")


def latest_recording(directory) -> Path | None:
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = sorted(
        p for p in directory.glob("*.jsonl")
        if not p.name.endswith(".aborted.jsonl") and p.is_file()
    )
    return candidates[-1] if candidates else None


def _recording_base(filename: str) -> str:
    for suffix in (".aborted.jsonl", ".jsonl.part", ".jsonl"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def resolve_replay_path(directory, name: str | None) -> Path:
    directory = Path(directory)
    if name is None:
        path = latest_recording(directory)
        if path is None:
            raise RecordingError(f"no saved recording in {directory}")
        return path
    candidates = [Path(name).expanduser(), directory / name, directory / f"{name}.jsonl"]
    for path in candidates:
        if path.is_file():
            return path
    raise RecordingError(f"recording not found: {name}")


# --------------------------------------------------------------------------
# Recording format + background writer
# --------------------------------------------------------------------------
def build_header(*, hand_sides: Sequence[str], rate_hz: float, extra: Mapping | None = None) -> dict:
    header = {
        "type": "header",
        "schema": RECORDING_SCHEMA,
        "profile": "G1D",
        "robot": "G1D",
        "rate_hz": float(rate_hz),
        "clock": "monotonic_s",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm_joint_names": list(ARM_JOINT_NAMES),
        "hand_sides": [side for side in HAND_SIDES if side in tuple(hand_sides)],
        "hand_joint_names": {side: list(HAND_JOINT_NAMES[side]) for side in HAND_SIDES},
    }
    if extra:
        header.update(dict(extra))
    header["profile"] = "G1D"
    return header


@dataclass(frozen=True)
class RecordingResult:
    path: Path | None
    samples: int
    dropped: int
    status: str
    error: str | None


_CLOSE = object()


class RecordingWriter:
    """Bounded, non-blocking JSONL writer; all file I/O on its own thread.

    ``emit`` copies the sample and uses ``put_nowait``; a full queue drops and
    counts. ``request_close`` and ``emit`` are linearized by one lock so no
    sample can land behind the shutdown sentinel. The file is written as
    ``<name>.jsonl.part`` and renamed at the end: ``<name>.jsonl`` when saved,
    ``<name>.aborted.jsonl`` otherwise (never selected as "latest").
    """

    def __init__(self, directory, header: Mapping, *, name: str | None = None,
                 maxsize: int = 4096, open_file: Callable = open):
        self.directory = Path(directory)
        self.name = name or utc_recording_name()
        self._header = json.loads(json.dumps(dict(header)))
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(maxsize)))
        self._lock = threading.Lock()
        self._closing = False
        self._status = "saved"
        self._dropped = 0
        self._done = threading.Event()
        self._result: RecordingResult | None = None
        self._first_t = None
        self._last_t = None
        self._first_t_mono = None
        self._last_t_mono = None
        self._open = open_file
        self._thread = threading.Thread(target=self._run, name="teach-recorder", daemon=True)
        self._thread.start()

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

    def emit(self, sample: Mapping) -> bool:
        try:
            t = float(sample["t"])
            t_mono = float(sample["t_mono"])
        except (KeyError, TypeError, ValueError):
            return False
        if (isinstance(sample.get("t"), bool) or isinstance(sample.get("t_mono"), bool)
                or not math.isfinite(t) or not math.isfinite(t_mono)):
            return False
        record = {"type": "sample"}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, (list, tuple)):
                value = [float(v) for v in value]
            record[key] = value
        with self._lock:
            if self._closing:
                return False
            if ((self._last_t is not None and t <= self._last_t)
                    or (self._last_t_mono is not None and t_mono <= self._last_t_mono)):
                return False
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._dropped += 1
                return False
            self._last_t = t
            self._last_t_mono = t_mono
            return True

    def request_close(self, status: str = "saved") -> None:
        with self._lock:
            if self._closing:
                return
            self._closing = True
            self._status = str(status)
            while True:
                try:
                    self._queue.put_nowait(_CLOSE)
                    break
                except queue.Full:
                    # Make room for the sentinel: drop the oldest sample.
                    try:
                        self._queue.get_nowait()
                        self._dropped += 1
                    except queue.Empty:
                        pass

    def wait(self, timeout: float | None = None) -> RecordingResult | None:
        if not self._done.wait(timeout if timeout is None else max(0.0, timeout)):
            return None
        return self._result

    def close(self, status: str = "saved", timeout: float | None = 5.0) -> RecordingResult:
        self.request_close(status)
        result = self.wait(timeout)
        if result is None:
            return RecordingResult(None, 0, self.dropped_count, status, "writer close timeout")
        return result

    def _run(self):
        written = 0
        handle = None
        part = None
        error = None
        try:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                fd, private_name = tempfile.mkstemp(prefix=f".{self.name}-", suffix=".jsonl.part",
                                                     dir=self.directory)
                os.close(fd)
                part = Path(private_name)
                handle = self._open(part, "w")
                handle.write(json.dumps(self._header) + "\n")
            except Exception as exc:  # storage failure: keep draining, report
                error = f"{type(exc).__name__}: {exc}"
                handle = None
            while True:
                item = self._queue.get()
                if item is _CLOSE:
                    break
                if handle is None:
                    continue
                try:
                    handle.write(json.dumps(item) + "\n")
                    written += 1
                    t = float(item["t"])
                    self._first_t = t if self._first_t is None else self._first_t
                    self._last_t = t
                except Exception as exc:
                    error = error or f"{type(exc).__name__}: {exc}"
            with self._lock:
                status, dropped = self._status, self._dropped
            final = None
            if handle is not None:
                try:
                    duration = (self._last_t - self._first_t) if written else 0.0
                    handle.write(json.dumps({"type": "footer", "status": status,
                                             "samples": written, "dropped": dropped,
                                             "duration_s": duration}) + "\n")
                    handle.flush()
                    try:
                        os.fsync(handle.fileno())
                    except Exception:
                        pass
                    handle.close()
                    handle = None
                    suffix = ".jsonl" if status == "saved" else ".aborted.jsonl"
                    base = self.name
                    index = 0
                    while True:
                        candidate = self.directory / f"{base}{suffix}" if index == 0 else \
                            self.directory / f"{base}_{index}{suffix}"
                        try:
                            os.link(part, candidate)
                            part.unlink()
                            final = candidate
                            break
                        except FileExistsError:
                            index += 1
                except Exception as exc:
                    error = error or f"{type(exc).__name__}: {exc}"
            self._result = RecordingResult(final, written, dropped, status, error)
        except BaseException as exc:  # never leave waiters hanging
            self._result = RecordingResult(None, written, self._dropped, self._status,
                                           f"{type(exc).__name__}: {exc}")
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
            if part is not None:
                try:
                    part.unlink()
                except OSError:
                    pass
            self._done.set()


@dataclass
class Recording:
    path: Path
    header: dict
    t: np.ndarray
    arm_q: np.ndarray
    hand_q: dict
    hand_sides: tuple

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])


def _finite_matrix(rows, width, label):
    try:
        array = np.asarray(rows, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RecordingError(f"{label}: not numeric") from exc
    if array.ndim != 2 or array.shape[1] != width:
        raise RecordingError(f"{label}: expected {width} values per sample")
    if not np.all(np.isfinite(array)):
        raise RecordingError(f"{label}: non-finite value")
    return array


def load_recording(path, *, max_gap_s: float = 0.25, min_samples: int = 2,
                   arm_lower=None, arm_upper=None, hand_limits=None) -> Recording:
    path = Path(path)
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = [json.loads(line) for line in lines]
    except (OSError, ValueError) as exc:
        raise RecordingError(f"cannot read recording {path}: {exc}") from exc
    if (not records or any(not isinstance(record, dict) for record in records)
            or records[0].get("type") != "header" or records[0].get("schema") != RECORDING_SCHEMA):
        raise RecordingError("missing or unknown header schema")
    header = records[0]
    if header.get("clock") != "monotonic_s":
        raise RecordingError("clock must be monotonic_s")
    # Recordings saved before the explicit ``profile`` field (same v1 schema)
    # carry only ``robot``; a present profile must still be G1D.
    if header.get("robot") != "G1D" or header.get("profile", "G1D") != "G1D":
        raise RecordingError("unsupported robot/profile")
    try:
        rate_hz = float(header["rate_hz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RecordingError("invalid recording rate") from exc
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise RecordingError("invalid recording rate")
    if tuple(header.get("arm_joint_names", ())) != ARM_JOINT_NAMES:
        raise RecordingError("arm joint schema does not match G1D")
    declared_sides = header.get("hand_sides")
    if (not isinstance(declared_sides, list) or len(declared_sides) != len(set(declared_sides))
            or declared_sides != [s for s in HAND_SIDES if s in declared_sides]):
        raise RecordingError("invalid hand sides")
    hand_names = header.get("hand_joint_names")
    if not isinstance(hand_names, dict) or any(tuple(hand_names.get(s, ())) != HAND_JOINT_NAMES[s]
                                               for s in HAND_SIDES):
        raise RecordingError("hand joint schema does not match G1D")
    footer = records[-1]
    if footer.get("type") != "footer":
        raise RecordingError("missing footer (incomplete recording)")
    if footer.get("status") != "saved":
        raise RecordingError(f"footer status is {footer.get('status')!r}, not 'saved'")
    if any(r.get("type") != "sample" for r in records[1:-1]):
        raise RecordingError("unexpected record between header and footer")
    samples = records[1:-1]
    if (isinstance(footer.get("samples"), bool) or not isinstance(footer.get("samples"), int)
            or footer["samples"] < 0 or footer["samples"] != len(samples)):
        raise RecordingError("invalid footer samples")
    if (isinstance(footer.get("dropped"), bool) or not isinstance(footer.get("dropped"), int)
            or footer["dropped"] < 0):
        raise RecordingError("invalid footer dropped")
    if len(samples) < min_samples:
        raise RecordingError(f"too few samples ({len(samples)})")
    sides = tuple(s for s in HAND_SIDES if s in header.get("hand_sides", []))
    try:
        t = np.asarray([float(s["t"]) for s in samples], dtype=float)
        t_mono = np.asarray([float(s["t_mono"]) for s in samples], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise RecordingError("sample time missing") from exc
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        raise RecordingError("sample time must be finite and strictly increasing")
    if not np.all(np.isfinite(t_mono)) or np.any(np.diff(t_mono) <= 0.0):
        raise RecordingError("sample time must be finite and strictly increasing")
    # Older G1D writers did not record footer.duration_s; then the duration is
    # the (already validated, strictly increasing) sample time span.
    if "duration_s" not in footer:
        duration = float(t[-1] - t[0])
    else:
        try:
            duration = float(footer["duration_s"])
        except (TypeError, ValueError) as exc:
            raise RecordingError("invalid footer duration") from exc
    if not math.isfinite(duration) or duration <= 0.0 or not math.isclose(duration, t[-1] - t[0], rel_tol=1e-9, abs_tol=1e-9):
        raise RecordingError("invalid footer duration")
    if np.max(np.diff(t)) > max_gap_s:
        raise RecordingError(f"time gap {np.max(np.diff(t)):.3f}s exceeds {max_gap_s}s")
    arm = _finite_matrix([s.get("arm_q") for s in samples], ARM_DOF, "arm_q")
    hands = {side: _finite_matrix([s.get(f"{side}_hand_q") for s in samples], HAND_DOF, f"{side}_hand_q")
             for side in sides}
    _finite_matrix([s.get("arm_dq") for s in samples], ARM_DOF, "arm_dq")
    for side in sides:
        _finite_matrix([s.get(f"{side}_hand_dq") for s in samples], HAND_DOF, f"{side}_hand_dq")
    if arm_lower is not None or arm_upper is not None:
        lo, hi = _finite_matrix([arm_lower], ARM_DOF, "arm_lower")[0], _finite_matrix([arm_upper], ARM_DOF, "arm_upper")[0]
        # Arms pushed onto their mechanical stop read slightly past the model
        # limits; replay commands are clamped inside them (_clamp_arm).
        tol = ARM_LIMIT_MEASUREMENT_TOL_RAD
        if np.any(lo >= hi) or np.any(arm < lo - tol) or np.any(arm > hi + tol):
            raise RecordingError("arm joint limit violation")
    if hand_limits is not None:
        for side, values in hands.items():
            try:
                lo, hi = hand_limits[side]
            except (KeyError, TypeError) as exc:
                raise RecordingError(f"missing {side} hand limits") from exc
            lo = _finite_matrix([lo], HAND_DOF, f"{side} lower limit")[0]
            hi = _finite_matrix([hi], HAND_DOF, f"{side} upper limit")[0]
            # Measured Dex3 q reads slightly past the URDF limits at the
            # mechanical stops; replay commands are clamped inside the limits
            # anyway (_clamp_hand), so only reject data clearly out of range.
            tol = HAND_LIMIT_MEASUREMENT_TOL_RAD
            if np.any(lo >= hi) or np.any(values < lo - tol) or np.any(values > hi + tol):
                raise RecordingError(f"{side} hand joint limit violation")
    return Recording(path=path, header=header, t=t - t[0], arm_q=arm, hand_q=hands, hand_sides=sides)


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmState:
    q: np.ndarray
    dq: np.ndarray
    timestamp: float


@dataclass(frozen=True)
class HandState:
    q: np.ndarray
    dq: np.ndarray
    timestamp: float
    protection: Mapping | None = None


@dataclass(frozen=True)
class ArmCommand:
    q: np.ndarray
    dq: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray


@dataclass(frozen=True)
class HandCommand:
    q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


def _per_arm(shoulder_elbow, wrist):
    side = [float(shoulder_elbow)] * 4 + [float(wrist)] * 3
    return tuple(side * 2)


@dataclass
class TeachReplayConfig:
    hand_sides: tuple = HAND_SIDES
    # freshness (s)
    arm_state_max_age: float = 0.1
    # Same freshness rule as dex3_protection.STATE_STALE_S (one shared
    # predicate); stale Dex3 => HOLD, never open/release.
    hand_state_max_age: float = 0.5
    # TEACH (compliant). Position term disabled: q_cmd = measured, kp = 0.
    teach_arm_kp: tuple = _per_arm(0.0, 0.0)
    teach_arm_kd: tuple = _per_arm(1.0, 0.5)
    teach_arm_kp_max: float = 5.0          # validation ceiling for teach kp
    teach_hand_kp: float = 0.0
    teach_hand_kd: float = 0.05
    # Gravity ff is applied in full from the first write: with kp=0 any ramp
    # would let the arms sag. Set > 0 only for a deliberate bench experiment.
    gravity_ramp_s: float = 0.0
    gravity_effort_fraction: float = 0.5   # |tau_ff| <= fraction * URDF effort
    record_rate_hz: float = 50.0
    # tracking gains = G1_29/G1D arm controller (kp_low/kd_low, kp_wrist/kd_wrist)
    track_arm_kp: tuple = _per_arm(80.0, 40.0)
    track_arm_kd: tuple = _per_arm(3.0, 1.5)
    # Dex3 controller gains (robot_hand_unitree.Dex3_Kp/Kd)
    track_hand_kp: float = 1.5
    track_hand_kd: float = 0.2
    # REPLAY
    ramp_arm_vmax: float = 0.5             # rad/s peak during the ramp-in
    ramp_hand_vmax: float = 2.0
    ramp_min_s: float = 1.0
    ramp_max_s: float = 15.0               # refuse longer ramps
    replay_arm_vmax: float = 0.8           # rad/s: fast taught segments are time-stretched to this
    replay_hand_vmax: float = 4.0
    limit_margin: float = 0.05             # rad inside model arm limits
    hand_limit_margin: float = 0.0
    max_tracking_error: float = 0.35       # rad |last cmd - measured| => hold
    # QUIT
    quit_hold_s: float = 2.0               # hold measured pose (gravity ff), then release


class Mode(enum.Enum):
    IDLE = "idle"
    TEACH = "teach"
    HOLD = "hold"
    REPLAY_RAMP = "replay_ramp"
    REPLAY_PLAY = "replay_play"
    QUITTING = "quitting"
    DONE = "done"


def time_warp(t, arm_q, hand_q, arm_vmax, hand_vmax):
    """Stretch each recording segment so linear playback never exceeds the caps.

    Keeps the taught path; only the timing of fast segments is slowed down.
    """
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    need = np.max(np.abs(np.diff(arm_q, axis=0)), axis=1) / float(arm_vmax)
    for q in hand_q.values():
        need = np.maximum(need, np.max(np.abs(np.diff(q, axis=0)), axis=1) / float(hand_vmax))
    return np.concatenate([[0.0], np.cumsum(np.maximum(dt, need))])


def smoothstep(s):
    s = min(1.0, max(0.0, float(s)))
    return s * s * (3.0 - 2.0 * s)


def _vec(value, size):
    try:
        array = np.asarray(value, dtype=float).reshape(-1).copy()
    except (TypeError, ValueError):
        return None
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        return None
    return array


class TeachReplaySession:
    """Single-threaded lifecycle + command generator (see module docstring)."""

    def __init__(self, io, *, gravity_fn, arm_lower, arm_upper, arm_effort,
                 config: TeachReplayConfig | None = None, teach_dir=None,
                 clock=time.monotonic, writer_factory=RecordingWriter,
                 on_message: Callable[[str], None] | None = None):
        self.io = io
        self.gravity_fn = gravity_fn
        self.config = config or TeachReplayConfig()
        self.config.hand_sides = tuple(s for s in HAND_SIDES if s in tuple(self.config.hand_sides))
        self.arm_lower = _vec(arm_lower, ARM_DOF)
        self.arm_upper = _vec(arm_upper, ARM_DOF)
        self.arm_effort = _vec(arm_effort, ARM_DOF)
        if self.arm_lower is None or self.arm_upper is None or self.arm_effort is None:
            raise ValueError("arm limits/effort must be 14 finite values")
        if np.any(self.arm_upper - self.arm_lower <= 2 * self.config.limit_margin):
            raise ValueError("arm joint limits are empty after the safety margin")
        if np.any(self.arm_effort <= 0.0):
            raise ValueError("arm effort limits must be positive")
        if np.any(np.asarray(self.config.teach_arm_kp) > self.config.teach_arm_kp_max):
            raise ValueError("teach arm kp exceeds teach_arm_kp_max")
        self.teach_dir = Path(teach_dir) if teach_dir is not None else default_teach_dir()
        self.clock = clock
        self.writer_factory = writer_factory
        self.on_message = on_message
        self.lock = threading.Lock()
        self._quit = threading.Event()
        self._pending = None              # ("teach",) | ("save",) | ("replay", Recording)
        self.mode = Mode.IDLE
        self.last_message = ""
        self.messages = []
        self.active = False
        self._activated_at = None
        self._writer = None
        self._closing_writers = []
        self._recording_results = []
        self._teach_t0 = None
        self._next_sample = None
        self._last_arm_cmd: ArmCommand | None = None
        self._last_hand_cmd: dict = {}   # rate-limiter state (trajectory point)
        self._written_hand: dict = {}    # exact command last written per side
        self._hand_disabled: dict = {}   # per-joint mask latched by Dex3 protection
        self._last_gravity = np.zeros(ARM_DOF)
        self._hold_arm: ArmCommand | None = None
        self._hold_hand: dict = {}
        self._recording: Recording | None = None
        self._ramp = None
        self.play_start_time = None
        self._quit_deadline = None
        self._protectors = {}
        # Web panel (optional): which input started the current replay.
        self._replay_origin = None
        self._writer_name = None

    # ---------------------------------------------------------------- input
    @property
    def done(self) -> bool:
        return self.mode == Mode.DONE

    @property
    def quit_requested(self) -> bool:
        return self._quit.is_set()

    def request_quit(self) -> bool:
        """Lock-free and idempotent: the next tick honours it before any write."""
        self._quit.set()
        return True

    def request_teach(self) -> bool:
        with self.lock:
            if self._quit.is_set() or self.mode not in (Mode.IDLE, Mode.HOLD) or self._pending:
                return self._refuse_locked("r ignored in mode " + self.mode.value)
            self._pending = ("teach",)
            return True

    def request_save(self) -> bool:
        with self.lock:
            if self._quit.is_set() or self.mode != Mode.TEACH or self._pending:
                return self._refuse_locked("s ignored: not teaching")
            self._pending = ("save",)
            return True

    def request_replay(self, name: str | None, *, origin: str = "terminal",
                       repeat: int = 1, continuous: bool = False) -> bool:
        try:
            repeat = validate_replay_repeat(repeat)
        except ValueError:
            return False
        if repeat < 1 or (continuous and repeat != 1):
            return False
        with self.lock:
            if self._quit.is_set() or self.mode not in (Mode.IDLE, Mode.HOLD) or self._pending:
                return self._refuse_locked("p ignored in mode " + self.mode.value)
        # File I/O outside the lifecycle lock (never stalls the control tick).
        try:
            path = resolve_replay_path(self.teach_dir, name)
            recording = load_recording(path, arm_lower=self.arm_lower, arm_upper=self.arm_upper,
                                       hand_limits=DEX3_LIMITS)
        except RecordingError as exc:
            with self.lock:
                return self._refuse_locked(f"REPLAY refused: {exc}")
        with self.lock:
            if self._quit.is_set() or self.mode not in (Mode.IDLE, Mode.HOLD) or self._pending:
                return self._refuse_locked("p ignored in mode " + self.mode.value)
            self._pending = ("replay", recording, origin, repeat, bool(continuous))
            return True

    # -------------------------------------------------------------- helpers
    def _say(self, text):
        self.last_message = text
        self.messages.append(text)
        del self.messages[:-200]
        if self.on_message is not None:
            try:
                self.on_message(text)
            except Exception:
                pass

    def _refuse_locked(self, text):
        self._say(text)
        return False

    def _fresh_arm(self, now):
        try:
            state = self.io.arm_state()
        except Exception:
            return None, "arm state read failed"
        if state is None:
            return None, "arm state missing (no rt/lowstate sample)"
        q, dq = _vec(state.q, ARM_DOF), _vec(state.dq, ARM_DOF)
        if q is None or dq is None:
            return None, "arm state invalid"
        age = now - float(state.timestamp)
        if not (-STATE_FUTURE_SKEW_S <= age <= self.config.arm_state_max_age + 1e-9):
            return None, f"arm state stale ({age:.3f}s)"
        return ArmState(q=q, dq=dq, timestamp=float(state.timestamp)), None

    def _fresh_hand(self, side, now):
        try:
            state = self.io.hand_state(side)
        except Exception:
            return None, f"{side} Dex3 state read failed"
        if state is None:
            return None, f"{side} Dex3 state missing (no rt/dex3/{side}/state sample)"
        q, dq = _vec(state.q, HAND_DOF), _vec(state.dq, HAND_DOF)
        if q is None or dq is None:
            return None, f"{side} Dex3 state invalid"
        age = now - float(state.timestamp)
        if not (-STATE_FUTURE_SKEW_S <= age <= self.config.hand_state_max_age + 1e-9):
            return None, f"{side} Dex3 state stale ({age:.3f}s)"
        return HandState(q=q, dq=dq, timestamp=float(state.timestamp), protection=state.protection), None

    def _fresh_all(self, now):
        arm, reason = self._fresh_arm(now)
        if reason:
            return None, None, reason
        hands = {}
        for side in self.config.hand_sides:
            hands[side], reason = self._fresh_hand(side, now)
            if reason:
                return None, None, reason
        return arm, hands, None

    def _gravity(self, q):
        """Model gravity ff clipped to a fraction of the URDF effort, or None."""
        try:
            tau = _vec(self.gravity_fn(q), ARM_DOF)
        except Exception:
            return None
        if tau is None:
            return None
        cap = self.config.gravity_effort_fraction * self.arm_effort
        tau = np.clip(tau, -cap, cap)
        return tau if np.all(np.isfinite(tau)) else None

    def _gravity_scale(self, now):
        if self._activated_at is None or self.config.gravity_ramp_s <= 0:
            return 1.0
        return min(1.0, max(0.0, (now - self._activated_at) / self.config.gravity_ramp_s))

    def _clamp_arm(self, q):
        m = self.config.limit_margin
        return np.clip(q, self.arm_lower + m, self.arm_upper - m)

    def _clamp_hand(self, side, q):
        lo, hi = DEX3_LIMITS[side]
        m = self.config.hand_limit_margin
        return np.clip(q, lo + m, hi - m)

    def _activate(self, now):
        """One-way output activation (publishers built, nothing written)."""
        if self.active:
            return
        self.io.activate(self.config.hand_sides)
        self.active = True
        self._activated_at = now

    def _write_arm(self, cmd: ArmCommand):
        self.io.write_arm(cmd)
        self._last_arm_cmd = cmd

    def _write_hand(self, side, cmd: HandCommand):
        self.io.write_hand(side, cmd)
        self._last_hand_cmd[side] = cmd
        self._written_hand[side] = cmd

    def _track_hand(self, q):
        return HandCommand(q=np.asarray(q, dtype=float).copy(),
                           kp=np.full(HAND_DOF, self.config.track_hand_kp),
                           kd=np.full(HAND_DOF, self.config.track_hand_kd))

    def _track_arm(self, q, tau):
        return ArmCommand(q=np.asarray(q, dtype=float).copy(), dq=np.zeros(ARM_DOF),
                          kp=np.asarray(self.config.track_arm_kp, dtype=float).copy(),
                          kd=np.asarray(self.config.track_arm_kd, dtype=float).copy(),
                          tau=np.asarray(tau, dtype=float).copy())

    # ---------------------------------------------------------- transitions
    def _enter_hold(self, now, reason, arm_q=None, hand_q=None):
        """Constant hold: given/last-commanded q, tracking gains, gravity ff."""
        if arm_q is None:
            arm_q = self._last_arm_cmd.q if self._last_arm_cmd is not None else None
        if arm_q is not None:
            g = self._gravity(arm_q)
            tau = g * self._gravity_scale(now) if g is not None else self._last_gravity
            self._hold_arm = self._track_arm(arm_q, tau)
        self._hold_hand = {}
        for side in self.config.hand_sides:
            q = (hand_q or {}).get(side)
            written = self._written_hand.get(side)
            if q is None and written is not None and np.any(written.kp > 0):
                # Retain the exact validated (protected) command; never
                # recompute or raise it.
                self._hold_hand[side] = written
                continue
            if q is None and written is not None:
                q = written.q
            if q is not None:
                cmd = self._track_hand(q)
                disabled = self._hand_disabled.get(side)
                if disabled is not None and np.any(disabled):
                    # A protection-latched joint (fault/thermal) stays without torque.
                    cmd = HandCommand(q=cmd.q, kp=np.where(disabled, 0.0, cmd.kp),
                                      kd=np.where(disabled, 0.0, cmd.kd))
                self._hold_hand[side] = cmd
        self.mode = Mode.HOLD
        self._ramp = None
        self._recording = None
        self._say(f"HOLD: {reason}")

    def _close_writer(self, status):
        if self._writer is None:
            return
        self._writer.request_close(status)
        self._closing_writers.append(self._writer)
        self._writer = None

    def _poll_writers(self):
        still = []
        for writer in self._closing_writers:
            result = writer.wait(0)
            if result is None:
                still.append(writer)
                continue
            self._recording_results.append(result)
            if result.error:
                self._say(f"RECORDING ERROR ({result.status}): {result.error}")
            elif result.status == "saved":
                self._say(f"Recording saved: {result.path} ({result.samples} samples, "
                          f"{result.dropped} dropped)")
            else:
                self._say(f"Recording kept as {result.status}: {result.path}")
        self._closing_writers = still

    def _start_teach(self, now):
        arm, hands, reason = self._fresh_all(now)
        if reason:
            self._say(f"TEACH refused: {reason}")
            return
        if self._gravity(arm.q) is None:
            self._say("TEACH refused: gravity model unavailable/non-finite (fail closed)")
            return
        try:
            self._activate(now)
        except Exception as exc:
            self._say(f"TEACH refused: output activation failed: {exc}")
            return
        # Activation may block (debug-mode RPC): re-seed from a fresh sample.
        now = self.clock()
        arm, hands, reason = self._fresh_all(now)
        if reason:
            self._say(f"TEACH refused after activation: {reason}")
            return
        self._activated_at = now
        if not self._protectors:
            self._protectors = self._make_protectors()
        header = build_header(hand_sides=self.config.hand_sides, rate_hz=self.config.record_rate_hz,
                              extra={"teach_arm_kd": list(self.config.teach_arm_kd)})
        name = utc_recording_name()
        try:
            self._writer = self.writer_factory(self.teach_dir, header, name=name)
            self._writer_name = name
        except Exception as exc:
            self._writer = None
            self._say(f"TEACH: recorder unavailable ({exc}); holding instead")
            self._enter_hold(now, "recorder unavailable", arm_q=arm.q,
                             hand_q={s: h.q for s, h in hands.items()})
            return
        self._teach_t0 = now
        self._next_sample = now
        self.mode = Mode.TEACH
        self._say("TEACH: arms + Dex3 compliant, recording. Move by hand; 's' saves, 'q' quits.")
        self._teach_step(now, arm, hands)

    def _teach_step(self, now, arm, hands):
        g = self._gravity(arm.q)
        if g is None:
            self._close_writer("aborted_gravity")
            self._enter_hold(now, "gravity model failed during teach")
            return
        self._last_gravity = g
        cfg = self.config
        self._write_arm(ArmCommand(q=arm.q.copy(), dq=np.zeros(ARM_DOF),
                                   kp=np.asarray(cfg.teach_arm_kp, dtype=float).copy(),
                                   kd=np.asarray(cfg.teach_arm_kd, dtype=float).copy(),
                                   tau=g * self._gravity_scale(now)))
        for side, hand in hands.items():
            kd = np.full(HAND_DOF, cfg.teach_hand_kd)
            _, enable = self._protect_hand(side, now, hand.q, hand)
            if enable is not None:
                self._latch_disabled(side, enable)
            disabled = self._hand_disabled.get(side)
            if disabled is not None:
                kd = np.where(disabled, 0.0, kd)
            self._write_hand(side, HandCommand(q=hand.q.copy(), kp=np.full(HAND_DOF, cfg.teach_hand_kp), kd=kd))
        period = 1.0 / cfg.record_rate_hz
        if now + 1e-9 >= self._next_sample:
            sample = {"t": now - self._teach_t0, "t_mono": now, "arm_q": arm.q, "arm_dq": arm.dq}
            for side in HAND_SIDES:
                hand = hands.get(side)
                sample[f"{side}_hand_q"] = None if hand is None else hand.q
                sample[f"{side}_hand_dq"] = None if hand is None else hand.dq
            self._writer.emit(sample)
            self._next_sample += period
            if self._next_sample < now:
                self._next_sample = now + period

    def _save(self, now):
        self._close_writer("saved")
        q = self._last_arm_cmd.q if self._last_arm_cmd is not None else None
        hand_q = {s: c.q for s, c in self._last_hand_cmd.items()}
        self._enter_hold(now, "teach stopped; saving recording, holding pose", arm_q=q, hand_q=hand_q)

    def _plan_ramp(self, now, arm, hands, recording):
        cfg = self.config
        arm_goal = self._clamp_arm(recording.arm_q[0])
        hand_goal = {s: self._clamp_hand(s, recording.hand_q[s][0]) for s in cfg.hand_sides}
        d_arm = float(np.max(np.abs(arm_goal - arm.q)))
        d_hand = max([float(np.max(np.abs(hand_goal[s] - hands[s].q))) for s in hand_goal] or [0.0])
        duration = max(cfg.ramp_min_s,
                       SMOOTHSTEP_PEAK_FACTOR * d_arm / cfg.ramp_arm_vmax,
                       SMOOTHSTEP_PEAK_FACTOR * d_hand / cfg.ramp_hand_vmax)
        if duration > cfg.ramp_max_s:
            return None, (f"ramp-in would take {duration:.1f}s (> {cfg.ramp_max_s:.0f}s); "
                          "move the arms closer to the recording start")
        return {
            "t0": now, "duration": duration,
            "arm_start": arm.q.copy(), "arm_goal": arm_goal,
            "hand_start": {s: hands[s].q.copy() for s in hand_goal}, "hand_goal": hand_goal,
        }, None

    def _start_replay(self, now, recording: Recording, origin: str = "terminal",
                      repeat: int = 1, continuous: bool = False):
        missing = [s for s in self.config.hand_sides if s not in recording.hand_sides]
        if missing:
            self._say(f"REPLAY refused: recording has no data for hand(s) {missing}")
            return
        arm, hands, reason = self._fresh_all(now)
        if reason is None and self._gravity(arm.q) is None:
            reason = "gravity model unavailable/non-finite (fail closed)"
        if reason is None:
            _, reason = self._plan_ramp(now, arm, hands, recording)
        if reason:
            self._say(f"REPLAY refused: {reason}")
            return
        if not self.active:
            try:
                self._activate(now)
            except Exception as exc:
                self._say(f"REPLAY refused: output activation failed: {exc}")
                return
            # Activation may block (debug-mode RPC): re-plan from a fresh sample.
            now = self.clock()
            self._activated_at = now
            arm, hands, reason = self._fresh_all(now)
            if reason is None:
                ramp, reason = self._plan_ramp(now, arm, hands, recording)
            if reason:
                self._say(f"REPLAY refused after activation: {reason}")
                return
        else:
            ramp, _ = self._plan_ramp(now, arm, hands, recording)
        self._ramp = ramp
        self._recording = recording
        self._replay_origin = origin
        self._replay_iteration = 1
        self._replay_total = None if continuous else int(repeat)
        self._replay_continuous = bool(continuous)
        self._play_t = time_warp(recording.t, recording.arm_q,
                                 {s: recording.hand_q[s] for s in self.config.hand_sides},
                                 self.config.replay_arm_vmax, self.config.replay_hand_vmax)
        if not self._protectors:
            self._protectors = self._make_protectors()
        self.mode = Mode.REPLAY_RAMP
        total_text = "contínuo" if continuous else f"1/{repeat}"
        self._say(f"REPLAY: {recording.path.name} ({recording.duration:.1f}s taught, "
                  f"{self._play_t[-1]:.1f}s at <={self.config.replay_arm_vmax} rad/s); ramp-in "
                  f"{ramp['duration']:.1f}s at <={self.config.ramp_arm_vmax} rad/s; ciclo {total_text}. 'q' stops.")
        self._replay_step(now, arm, hands)

    def _make_protectors(self):
        try:
            from teleop.utils.dex3_protection import Dex3HandProtector
        except Exception:
            return {}
        return {side: Dex3HandProtector(np.zeros(HAND_DOF)) for side in self.config.hand_sides}

    def _latch_disabled(self, side, enable):
        mask = ~np.asarray(enable, dtype=bool).reshape(HAND_DOF)
        previous = self._hand_disabled.get(side, np.zeros(HAND_DOF, dtype=bool))
        self._hand_disabled[side] = previous | mask

    def _protect_hand(self, side, now, target, hand):
        """Existing Dex3 torque/thermal protection on fresh state (q, enable)."""
        protector = self._protectors.get(side)
        if protector is None or hand.protection is None:
            return target, None
        try:
            result = protector.update(now, target, hand.protection)
        except Exception:
            return target, None
        if result.state_stale:
            return target, None   # stale handled by the session (hold, never open)
        return np.asarray(result.q_cmd, dtype=float), list(result.enable)

    def _replay_step(self, now, arm, hands):
        cfg = self.config
        not_first = self.mode == Mode.REPLAY_PLAY or now > self._ramp["t0"]
        if not_first and self._last_arm_cmd is not None:
            err = float(np.max(np.abs(self._last_arm_cmd.q - arm.q)))
            if err > cfg.max_tracking_error:
                self._enter_hold(now, f"tracking error {err:.2f} rad > {cfg.max_tracking_error} rad "
                                      "(blocked/obstructed?)", arm_q=arm.q)
                return
        if self.mode == Mode.REPLAY_RAMP:
            ramp = self._ramp
            s = smoothstep((now - ramp["t0"]) / ramp["duration"])
            arm_q = ramp["arm_start"] + (ramp["arm_goal"] - ramp["arm_start"]) * s
            hand_q = {side: ramp["hand_start"][side] + (ramp["hand_goal"][side] - ramp["hand_start"][side]) * s
                      for side in ramp["hand_goal"]}
            finished = now - ramp["t0"] >= ramp["duration"]
        else:
            rec = self._recording
            play_t = self._play_t
            tp = now - self.play_start_time
            dt = max(0.0, now - self._last_tick) if self._last_tick is not None else 0.0
            target = np.array([np.interp(tp, play_t, rec.arm_q[:, j]) for j in range(ARM_DOF)])
            target = self._clamp_arm(target)
            prev = self._last_arm_cmd.q
            step = cfg.replay_arm_vmax * dt
            arm_q = prev + np.clip(target - prev, -step, step)
            hand_q = {}
            for side in cfg.hand_sides:
                ht = np.array([np.interp(tp, play_t, rec.hand_q[side][:, j]) for j in range(HAND_DOF)])
                ht = self._clamp_hand(side, ht)
                hprev = self._last_hand_cmd[side].q
                hstep = cfg.replay_hand_vmax * dt
                hand_q[side] = hprev + np.clip(ht - hprev, -hstep, hstep)
            final_arm = self._clamp_arm(rec.arm_q[-1])
            final_hand = {s: self._clamp_hand(s, rec.hand_q[s][-1]) for s in cfg.hand_sides}
            finished = (tp >= play_t[-1]
                        and np.allclose(arm_q, final_arm, atol=1e-9, rtol=0)
                        and all(np.allclose(hand_q[s], final_hand[s], atol=1e-9, rtol=0) for s in cfg.hand_sides))
        g = self._gravity(arm_q)
        if g is None:
            self._enter_hold(now, "gravity model failed during replay", arm_q=arm.q)
            return
        self._last_gravity = g
        self._write_arm(self._track_arm(arm_q, g * self._gravity_scale(now)))
        for side, q in hand_q.items():
            q_cmd, enable = self._protect_hand(side, now, q, hands[side])
            cmd = self._track_hand(q_cmd)
            if enable is not None:
                mask = np.asarray(enable, dtype=bool)
                self._latch_disabled(side, enable)
            else:
                mask = np.ones(HAND_DOF, dtype=bool)
            disabled = self._hand_disabled.get(side)
            if disabled is not None:
                mask = mask & ~np.asarray(disabled, dtype=bool)
            cmd = HandCommand(q=cmd.q, kp=np.where(mask, cmd.kp, 0.0), kd=np.where(mask, cmd.kd, 0.0))
            self._write_hand(side, cmd)
            # rate limiting continues from the unprotected trajectory point
            self._last_hand_cmd[side] = HandCommand(q=np.asarray(q, dtype=float).copy(), kp=cmd.kp, kd=cmd.kd)
        if finished:
            if self.mode == Mode.REPLAY_RAMP:
                self.mode = Mode.REPLAY_PLAY
                self.play_start_time = now
                self._say("REPLAY: playing recording")
            elif self._replay_continuous or self._replay_iteration < self._replay_total:
                self._replay_iteration += 1
                ramp, reason = self._plan_ramp(now, arm, hands, self._recording)
                if reason:
                    self._enter_hold(now, f"replay loop ramp refused: {reason}", arm_q=arm_q)
                    return
                self._ramp = ramp
                self.mode = Mode.REPLAY_RAMP
                self.play_start_time = None
                total_text = "contínuo" if self._replay_continuous else str(self._replay_total)
                self._say(f"REPLAY: ciclo {self._replay_iteration}/{total_text}; rampa segura")
            else:
                self._enter_hold(now, "replay finished; holding final pose (no go-home)", arm_q=arm_q)

    def _write_hold(self):
        if self._hold_arm is not None:
            self._write_arm(self._hold_arm)
        for side, cmd in self._hold_hand.items():
            self._write_hand(side, cmd)

    def _begin_quit(self, now):
        self.mode = Mode.QUITTING
        if self._writer is not None:
            self._close_writer("aborted_quit")
        self._pending = None
        if not self.active:
            self._finish()
            return
        if self._last_arm_cmd is None:
            # Activated but nothing ever written: do not add authority now.
            self._release()
            self._finish()
            return
        arm, _ = self._fresh_arm(now)
        arm_q = arm.q if arm is not None else self._last_arm_cmd.q
        hand_q = {}
        for side in self.config.hand_sides:
            hand, _ = self._fresh_hand(side, now)
            if hand is not None:
                hand_q[side] = hand.q
        self._enter_hold(now, "quit requested", arm_q=arm_q, hand_q=hand_q)
        self.mode = Mode.QUITTING
        self._quit_deadline = now + self.config.quit_hold_s
        self._say(f"QUIT: holding measured pose for {self.config.quit_hold_s:.1f}s, then releasing outputs")
        self._write_hold()

    def _release(self):
        try:
            self.io.release()
        finally:
            self.active = False

    def _finish(self):
        self.mode = Mode.DONE
        self._say("DONE: outputs released" if self._activated_at is not None else "DONE: no output was ever activated")

    # ----------------------------------------------------------------- tick
    _last_tick = None

    def tick(self):
        """One control cycle; the only place that writes commands."""
        with self.lock:
            now = self.clock()
            try:
                self._tick_locked(now)
            except Exception as exc:
                self._fault(now, exc)
            finally:
                self._last_tick = now
                try:
                    self._poll_writers()
                except Exception:
                    pass

    def _fault(self, now, exc):
        """Unexpected error inside a cycle: stop wins (hold briefly, release)."""
        self._say(f"ERROR in control cycle: {type(exc).__name__}: {exc}; stopping")
        if self.mode in (Mode.QUITTING, Mode.DONE) or not self.active:
            # Failure while already stopping (or before any output): release now.
            try:
                if self.active:
                    self._release()
            except Exception:
                pass
            if self._writer is not None:
                self._close_writer("aborted_error")
            self.mode = Mode.DONE
            return
        if self._writer is not None:
            self._close_writer("aborted_error")
        self._quit.set()

    def _tick_locked(self, now):
        if self.mode == Mode.DONE:
            return
        if self._quit.is_set() and self.mode != Mode.QUITTING:
            self._begin_quit(now)
            return
        if self.mode == Mode.QUITTING:
            if now >= self._quit_deadline:
                self._release()
                self._finish()
            else:
                self._write_hold()
            return

        pending, self._pending = self._pending, None
        if pending is not None:
            kind = pending[0]
            if kind == "teach" and self.mode in (Mode.IDLE, Mode.HOLD):
                self._start_teach(now)
                if self.mode == Mode.TEACH:
                    return
            elif kind == "save" and self.mode == Mode.TEACH:
                self._save(now)
            elif kind == "replay" and self.mode in (Mode.IDLE, Mode.HOLD):
                self._start_replay(now, pending[1], pending[2] if len(pending) > 2 else "terminal",
                                   pending[3] if len(pending) > 3 else 1,
                                   pending[4] if len(pending) > 4 else False)
                if self.mode == Mode.REPLAY_RAMP:
                    return

        if self.mode == Mode.IDLE:
            return
        if self.mode == Mode.HOLD:
            self._write_hold()
            return
        arm, hands, reason = self._fresh_all(now)
        if reason:
            if self.mode == Mode.TEACH:
                self._close_writer("aborted_stale")
            # Hold the last validated command; never keep streaming.
            self._enter_hold(now, f"stale robot state: {reason}")
            self._write_hold()
            return
        if self.mode == Mode.TEACH:
            self._teach_step(now, arm, hands)
        elif self.mode in (Mode.REPLAY_RAMP, Mode.REPLAY_PLAY):
            self._replay_step(now, arm, hands)

    # --------------------------------------------------------------- status
    def status_view(self) -> dict:
        """Lock-free display snapshot for the web panel (never waits for a tick).

        Reads only references that the tick replaces atomically; values may be
        one cycle old, which is fine for display. Never used for control.
        """
        now = self.clock()
        mode = self.mode
        view = {"mode": mode.value, "message": self.last_message,
                "messages": list(self.messages[-8:]), "replay_origin": None,
                "recording": None, "replay": None, "quit_requested": self._quit.is_set()}
        busy = []
        writer, t0, name = self._writer, self._teach_t0, self._writer_name
        if mode == Mode.TEACH and writer is not None and name:
            view["recording"] = {"name": name, "elapsed_s": max(0.0, now - (t0 or now))}
            busy.append(name)
        for closing in list(self._closing_writers):
            if getattr(closing, "name", None):
                busy.append(closing.name)
        recording, ramp, play_t, play_start = (self._recording, self._ramp,
                                               getattr(self, "_play_t", None), self.play_start_time)
        if mode in (Mode.REPLAY_RAMP, Mode.REPLAY_PLAY) and recording is not None:
            view["replay_origin"] = self._replay_origin
            info = {"name": recording.path.name, "phase": "ramp" if mode == Mode.REPLAY_RAMP else "play",
                    "progress": 0.0, "iteration": getattr(self, "_replay_iteration", 1),
                    "total": getattr(self, "_replay_total", 1)}
            try:
                if mode == Mode.REPLAY_RAMP and ramp:
                    info["progress"] = min(1.0, max(0.0, (now - ramp["t0"]) / ramp["duration"]))
                elif play_t is not None and play_start is not None and play_t[-1] > 0:
                    info["progress"] = min(1.0, max(0.0, (now - play_start) / float(play_t[-1])))
            except Exception:
                pass
            view["replay"] = info
            busy.append(_recording_base(recording.path.name))
        pending = self._pending
        if pending and pending[0] == "replay":
            busy.append(_recording_base(pending[1].path.name))
        view["busy_recordings"] = sorted(set(busy))
        return view

    # -------------------------------------------------------------- results
    def wait_for_recordings(self, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        for writer in list(self._closing_writers):
            result = writer.wait(max(0.0, deadline - time.monotonic()))
            if result is not None:
                self._recording_results.append(result)
        self._closing_writers = []
        self._poll_writers()
        return list(self._recording_results)
