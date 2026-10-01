"""Opt-in ``--loco-backend wirelesscontroller``: walk by publishing the simulated
joystick state on ``rt/wirelesscontroller`` (unitree_go WirelessController_:
lx, ly, rx, ry, keys), the way unitree_webrtc_connect does, instead of the
timed SetVelocity RPC.

State, not command: published CONTINUOUSLY at 20 Hz from a dedicated thread
(zeros included).  Client ramp RAMP_UP/RAMP_DOWN per tick (reference client
values) is applied on top of whatever the control loop submits; safety zero
(stale, q, dead-man, emergency, shutdown) is immediate and unramped.  Never
sends Damp.  Nothing here touches DDS unless a real writer is injected.

Robot mapping (reference teleop.py): robot reads [ly, -lx, -rx] -> [vx, vy,
omega], scaled by its own limit (1.0 m/s forward by default), so
ly = vx / 1.0, lx = -vy / 1.0, rx = -omega / ROBOT_MAX_YAW_RADPS.
ROBOT_MAX_YAW_RADPS is an ASSUMPTION (1.0) pending physical validation.
"""
import math
import threading
import time

PUBLISH_HZ = 20.0
RAMP_UP = 0.15     # per tick @20 Hz, normalized units
RAMP_DOWN = 0.30
ROBOT_MAX_LINEAR_MPS = 1.0
ROBOT_MAX_YAW_RADPS = 1.0   # assumption, unverified
DEFAULT_STALE_S = 0.3
ZERO = (0.0, 0.0, 0.0)


def _finite(v):
    try:
        v = float(v)
    except Exception:
        return 0.0
    return v if math.isfinite(v) else 0.0


def _clamp(v, lim):
    return max(-lim, min(lim, v))


def normalize_command(vx, vy, omega, walk_cap, turn_cap):
    """(vx m/s, vy m/s, omega rad/s) -> (lx, ly, rx) normalized, capped by the
    configured caps expressed as a fraction of the robot's own limits."""
    lin_lim = walk_cap / ROBOT_MAX_LINEAR_MPS
    yaw_lim = turn_cap / ROBOT_MAX_YAW_RADPS
    ly = _clamp(_finite(vx) / ROBOT_MAX_LINEAR_MPS, lin_lim)
    lx = _clamp(-_finite(vy) / ROBOT_MAX_LINEAR_MPS, lin_lim)
    rx = _clamp(-_finite(omega) / ROBOT_MAX_YAW_RADPS, yaw_lim)
    return (0.0 if lx == 0 else lx, 0.0 if ly == 0 else ly, 0.0 if rx == 0 else rx)


def _approach(cur, target):
    rate = RAMP_UP if abs(target) > abs(cur) else RAMP_DOWN
    if cur < target:
        return min(cur + rate, target)
    if cur > target:
        return max(cur - rate, target)
    return cur


class WirelessControllerPublisher:
    def __init__(self, writer, walk_cap=0.3, turn_cap=0.3, clock=time.monotonic,
                 stale_s=DEFAULT_STALE_S):
        self._write = writer          # (lx, ly, rx, ry, keys) -> truthy on success
        self.walk_cap = float(walk_cap)
        self.turn_cap = float(turn_cap)
        self._clock = clock
        self.stale_s = float(stale_s)
        self._lock = threading.Lock()
        self._target = ZERO           # normalized (lx, ly, rx)
        self._cur = ZERO
        self._target_t = None
        self._thread = None
        self._stop = threading.Event()
        self.sent = 0
        self.write_failures = 0
        self.zero_events = 0
        self.last_reason = "idle"
        self.last_published = (0.0, 0.0, 0.0, 0.0, 0)
        self._first_t = None
        self._last_t = None

    # -- control loop side: pure, in-memory ---------------------------------
    def set_command(self, vx, vy, omega):
        lx, ly, rx = normalize_command(vx, vy, omega, self.walk_cap, self.turn_cap)
        with self._lock:
            self._target = (lx, ly, rx)
            self._target_t = self._clock()

    def zero_now(self, reason="zero"):
        """Immediate, unramped zero: clears target and ramp, publishes now."""
        with self._lock:
            self._target = ZERO
            self._cur = ZERO
            self._target_t = None
            self.last_reason = reason
            self.zero_events += 1
        return self._publish(0.0, 0.0, 0.0)

    # -- publisher thread ---------------------------------------------------
    def tick(self):
        now = self._clock()
        with self._lock:
            if any(self._target) and (self._target_t is None or now - self._target_t > self.stale_s):
                self._target = ZERO
                self._cur = ZERO       # stale: immediate zero, never ramped
                self.last_reason = "stale"
                self.zero_events += 1
            else:
                self._cur = tuple(_approach(c, t) for c, t in zip(self._cur, self._target))
                self.last_reason = "active" if any(self._cur) or any(self._target) else "idle"
            lx, ly, rx = self._cur
        return self._publish(lx, ly, rx)

    def _publish(self, lx, ly, rx):
        lx, ly, rx = (0.0 if v == 0 else float(v) for v in (lx, ly, rx))
        ok = False
        try:
            ok = bool(self._write(lx, ly, rx, 0.0, 0))
        except BaseException:
            ok = False
        with self._lock:
            if ok:
                self.sent += 1
                self.last_published = (lx, ly, rx, 0.0, 0)
                t = self._clock()
                if self._first_t is None:
                    self._first_t = t
                self._last_t = t
            else:
                self.write_failures += 1
        return ok

    def _loop(self):
        period = 1.0 / PUBLISH_HZ
        nxt = time.monotonic()
        while not self._stop.is_set():
            self.tick()
            nxt += period
            delay = nxt - time.monotonic()
            if delay < -period:       # overrun: do not burst to catch up
                nxt = time.monotonic()
                delay = 0.0
            self._stop.wait(max(0.0, delay))

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="loco-wireless", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
        self._thread = None
        self.zero_now("shutdown")

    def telemetry(self):
        with self._lock:
            hz = None
            if self.sent > 1 and self._last_t and self._first_t and self._last_t > self._first_t:
                hz = (self.sent - 1) / (self._last_t - self._first_t)
            return {
                "backend": "wirelesscontroller",
                "published": list(self.last_published),
                "target": list(self._target),
                "reason": self.last_reason,
                "sent": self.sent,
                "write_failures": self.write_failures,
                "zero_events": self.zero_events,
                "actual_hz": hz,
                "walk_cap": self.walk_cap,
                "turn_cap": self.turn_cap,
            }


def make_dds_writer(topic="rt/wirelesscontroller"):
    """Real writer (needs ChannelFactoryInitialize already done). Not used in tests."""
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
    pub = ChannelPublisher(topic, WirelessController_)
    pub.Init()

    def write(lx, ly, rx, ry, keys):
        return pub.Write(WirelessController_(lx=lx, ly=ly, rx=rx, ry=ry, keys=keys))
    return write
