"""Fake DDS transport standing in for rt/waistpitchcmd / rt/waistpitchstate.

NO real DDS, NO sockets, NO robot. This only lets the keyboard tool and its
safety logic be built and exercised end-to-end before any real Cyclone DDS
transport is written and reviewed. A real concurrent-publisher check against
the actual DDS domain (0) still needs cyclonedds discovery APIs; this fake
models writer presence/heartbeats so the arming logic has something real to
test against, but it proves nothing about the physical network.
"""
from dataclasses import dataclass, field
import threading
import time


@dataclass
class MotorCmd_:
    """Mirrors unitree_hg::msg::dds_::MotorCmd_ field layout (fake only)."""
    mode: int = 1
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    reserve: tuple = ()


@dataclass
class MotorState_:
    """Mirrors unitree_hg::msg::dds_::MotorState_ field layout (fake only)."""
    mode: int = 1
    q: float = 0.0
    dq: float = 0.0
    ddq: float = 0.0
    tau_est: float = 0.0
    temperature: tuple = (25, 25)
    vol: float = 48.0
    sensor: tuple = (0, 0)
    motorstate: int = 0
    reserve: tuple = ()


class PublisherRegistry:
    """Fake DDS-writer discovery for one topic within one fake domain."""

    def __init__(self):
        self._lock = threading.Lock()
        self._writers = {}  # writer_id -> last_seen (monotonic)

    def heartbeat(self, writer_id, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            self._writers[writer_id] = now

    def leave(self, writer_id):
        with self._lock:
            self._writers.pop(writer_id, None)

    def other_writers(self, self_id, now=None, stale_after=2.0):
        """Writer ids other than self_id seen within stale_after seconds."""
        now = time.monotonic() if now is None else now
        with self._lock:
            return sorted(
                w for w, t in self._writers.items()
                if w != self_id and 0 <= now - t <= stale_after
            )


class FakeBus:
    """In-process stand-in for the rt/waistpitchcmd + rt/waistpitchstate topics."""

    def __init__(self):
        self.cmd_registry = PublisherRegistry()
        self.state_registry = PublisherRegistry()
        self._lock = threading.Lock()
        self._state = MotorState_(mode=1, q=0.0, dq=0.0, motorstate=0)
        self._state_stamp = 0.0

    def publish_cmd(self, writer_id, cmd: MotorCmd_, now=None):
        self.cmd_registry.heartbeat(writer_id, now)
        with self._lock:
            self._last_cmd = cmd

    def read_state(self):
        """Returns (MotorState_, stamp). Never talks to real hardware."""
        with self._lock:
            return self._state, self._state_stamp

    def inject_state(self, state: MotorState_, stamp):
        """Test/sim-only hook; no equivalent exists against the real robot."""
        with self._lock:
            self._state = state
            self._state_stamp = stamp

    def inject_other_publisher(self, writer_id, now=None):
        """Simulates a concurrent, independent command writer for testing."""
        self.cmd_registry.heartbeat(writer_id, now)
