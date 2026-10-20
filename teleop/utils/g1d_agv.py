"""Opt-in Quest control of the G1-D AGV base and lift column (``--g1d-agv``).

Wire contract (unitree_sdk2 C++ ``g1/agv`` headers; server replied API
version 1.0.0.1 on the robot):
  service "agv", API 1001 Move {"vx", "vy", "vyaw"}: vx [-1.5, 1.5] m/s
  (+forward), vy unsupported (always 0.0), vyaw [-0.6, 0.6] rad/s (+CCW).
  API 1002 HeightAdjust {"data": vz}: vz normalized [-1, 1] -> column speed
  vz * 76.5 mm/s (+up).  Velocity only.  Errors 9101/9102/9103.
  Height feedback: rt/hispeed_state (geometry_msgs Point32_), height = y [m].
Wheels/column are not in rt/lowcmd, so this never conflicts with the arm writer.

Stick signs reuse the validated G1 walking convention (quest_controls.
joystick_to_locomotion): Quest reports y < 0 for stick forward, x < 0 for stick
left.  vx = -left_y, vyaw = -left_x (left = CCW), column vz = -right_y
(forward = rise).  Raw sticks are logged beside the command so the sign can be
verified physically.  No software height limits: the column has hardware
end-stops (operator decision).

Threading: the 30 Hz control loop only posts the latest validated target with
its controller-sample timestamp (``submit``; in-memory, never RPC).  A dedicated
20 Hz sender thread re-checks arming and sample freshness itself and performs
all RPCs.  Nothing (client, subscriber, thread) is constructed until the
launcher calls start_feedback()/activate(), which teleop does only after the
Vuer fork and DDS initialization.
"""
import json
import math
import threading
import time

from teleop.utils.quest_controls import _shape_stick_value
from teleop.utils.quest_safety import controller_sample_is_fresh

AGV_SERVICE_NAME = "agv"
AGV_API_VERSION = "1.0.0.1"
AGV_API_MOVE = 1001
AGV_API_HEIGHT_ADJUST = 1002
AGV_NATIVE_MAX_VX_MPS = 1.5
AGV_HARD_MAX_VX_MPS = AGV_NATIVE_MAX_VX_MPS
AGV_DEFAULT_MAX_VX_MPS = 0.6
AGV_MAX_VYAW_RADPS = 0.6          # native limit
AGV_COLUMN_SPEED_MMPS_PER_UNIT = 76.5
HEIGHT_TOPIC = "rt/hispeed_state"
HEIGHT_STALE_S = 0.5
SEND_HZ = 20.0
RPC_TIMEOUT_S = 0.2
MAX_LOGGED_RPC_ERRORS = 5
SHUTDOWN_ZERO_BURST = 3


def _finite_or_zero(value):
    try:
        value = float(value)
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _clamp(value, limit):
    value = _finite_or_zero(value)
    value = max(-limit, min(limit, value))
    return 0.0 if value == 0 else value


def _axis(value):
    try:
        return _shape_stick_value(value)
    except Exception:
        return 0.0


def _pair(xy):
    try:
        if isinstance(xy, (str, bytes)) or len(xy) != 2:
            return None
        return xy[0], xy[1]
    except Exception:
        return None


def map_sticks(left_xy, right_xy, max_vx, max_vyaw=AGV_MAX_VYAW_RADPS):
    """(left stick, right stick) -> (vx m/s, vyaw rad/s, column vz in [-1, 1]).

    Deadzone + cubic shaping from quest_controls; non-finite or malformed
    input zeroes only the affected axis.
    """
    left = _pair(left_xy)
    right = _pair(right_xy)
    lx, ly = (_axis(left[0]), _axis(left[1])) if left else (0.0, 0.0)
    ry = _axis(right[1]) if right else 0.0
    return (_clamp(-ly * max_vx, max_vx),
            _clamp(-lx * max_vyaw, max_vyaw),
            _clamp(-ry, 1.0))


def resolve_agv_max_vx(value):
    """None -> 0.6 m/s default; otherwise a finite value in (0, 1.5]."""
    if value is None:
        return AGV_DEFAULT_MAX_VX_MPS
    try:
        cap = float(value)
    except Exception:
        raise ValueError(f"G1D_AGV_MAX_VX={value!r} is not a number") from None
    if not math.isfinite(cap) or cap <= 0.0 or cap > AGV_HARD_MAX_VX_MPS:
        raise ValueError(f"G1D_AGV_MAX_VX={value!r} must be in (0, {AGV_HARD_MAX_VX_MPS}]")
    return cap


def _raw_xy(xy):
    pair = _pair(xy)
    if pair is None:
        return None
    out = []
    for v in pair:
        try:
            v = float(v)
            out.append(v if math.isfinite(v) else None)
        except Exception:
            out.append(None)
    return out


# ---------------------------------------------------------------- RPC client
def build_agv_client_class(base):
    """AgvClient over unitree_sdk2py's rpc Client (``base``; injectable for tests)."""

    class AgvClient(base):
        def __init__(self):
            super().__init__(AGV_SERVICE_NAME, False)

        def Init(self):
            self._SetApiVerson(AGV_API_VERSION)
            self._RegistApi(AGV_API_MOVE, 0)
            self._RegistApi(AGV_API_HEIGHT_ADJUST, 0)

        def Move(self, vx, vy, vyaw):
            # vy is unsupported by the G1-D base: always 0.0.
            p = {"vx": _clamp(vx, AGV_NATIVE_MAX_VX_MPS), "vy": 0.0,
                 "vyaw": _clamp(vyaw, AGV_MAX_VYAW_RADPS)}
            code, _ = self._Call(AGV_API_MOVE, json.dumps(p))
            return code

        def HeightAdjust(self, vz):
            code, _ = self._Call(AGV_API_HEIGHT_ADJUST, json.dumps({"data": _clamp(vz, 1.0)}))
            return code

    return AgvClient


def make_agv_client(timeout_s=RPC_TIMEOUT_S):
    """Real client (needs ChannelFactoryInitialize already done)."""
    from unitree_sdk2py.rpc.client import Client
    client = build_agv_client_class(Client)()
    client.SetTimeout(timeout_s)
    client.Init()
    return client


# ---------------------------------------------------------------- height feedback
def _default_height_subscriber(topic, handler):
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_
    sub = ChannelSubscriber(topic, Point32_)
    sub.Init(handler, 1)
    return sub


class HeightMonitor:
    """Passive rt/hispeed_state subscriber: latest column height (y, metres)."""

    def __init__(self, subscriber_factory=None, clock=time.monotonic, stale_s=HEIGHT_STALE_S):
        self._factory = subscriber_factory or _default_height_subscriber
        self._clock = clock
        self.stale_s = float(stale_s)
        self._lock = threading.Lock()
        self._height = None
        self._t = None
        self.samples = 0
        self._sub = None

    def start(self):
        if self._sub is None:
            self._sub = self._factory(HEIGHT_TOPIC, self.on_sample)

    def on_sample(self, msg):
        try:
            y = float(getattr(msg, "y"))
        except Exception:
            return
        if not math.isfinite(y):
            return
        t = self._clock()
        with self._lock:
            self._height, self._t = y, t
            self.samples += 1

    def is_fresh(self, now=None):
        now = self._clock() if now is None else now
        with self._lock:
            t = self._t
        return t is not None and 0.0 <= now - t <= self.stale_s

    def snapshot(self, now=None):
        now = self._clock() if now is None else now
        with self._lock:
            h, t, n = self._height, self._t, self.samples
        age = None if t is None else max(0.0, now - t)
        return {"height_m": h, "age_s": age, "fresh": t is not None and 0.0 <= now - t <= self.stale_s,
                "samples": n}

    def close(self):
        sub, self._sub = self._sub, None
        close = getattr(sub, "Close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


# ---------------------------------------------------------------- controller
class G1dAgvController:
    def __init__(self, max_vx, armed_fn, client_factory=make_agv_client, height_factory=HeightMonitor,
                 clock=time.monotonic, warn=None, max_vyaw=AGV_MAX_VYAW_RADPS, send_hz=SEND_HZ):
        self.max_vx = resolve_agv_max_vx(max_vx)
        self.max_vyaw = min(float(max_vyaw), AGV_MAX_VYAW_RADPS)
        self._armed_fn = armed_fn
        self._client_factory = client_factory
        self._height_factory = height_factory
        self._clock = clock
        self._warn = warn or (lambda message: None)
        self._period = 1.0 / float(send_hz)
        self._lock = threading.Lock()        # target/telemetry only; never held across RPC
        self._rpc_lock = threading.Lock()    # serializes RPCs with the shutdown decision
        self._stop = threading.Event()
        self._thread = None
        self._client = None
        self.height = None
        self._active = False
        self._closed = False
        self._target = (0.0, 0.0, 0.0)
        self._sample_t = 0.0
        self._submit_reason = "inactive"
        self._raw_left = None
        self._raw_right = None
        self._sent = [0.0, 0.0, 0.0, 0.0]
        self._reason = "inactive"
        self._column_reason = "inactive"
        self.ticks = 0
        self.move_code = None
        self.height_code = None
        self.nonzero_move_codes = 0
        self.nonzero_height_codes = 0
        self.rpc_exceptions = 0
        self._logged_errors = 0
        self.shutdown_reason = None

    # ---------------------------------------------------- lifecycle
    def start_feedback(self):
        """Passive height subscriber only (after DDS init). No command path."""
        if self.height is None and not self._closed:
            self.height = self._height_factory()
            self.height.start()

    def activate(self, start_thread=True):
        """Construct the RPC client and start the 20 Hz sender (post-r)."""
        if self._closed or self._active:
            return
        self._client = self._client_factory()
        self._active = True
        if start_thread:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="g1d-agv-sender", daemon=True)
            self._thread.start()

    def sender_alive(self):
        return self._thread is not None and self._thread.is_alive()

    # ---------------------------------------------------- control loop side
    def submit(self, left_xy, right_xy, sample_timestamp):
        """Post the latest target (in-memory, never blocks on RPC)."""
        now = self._clock()
        raw_left, raw_right = _raw_xy(left_xy), _raw_xy(right_xy)
        ts = _finite_or_zero(sample_timestamp)
        if not self._active or self._closed:
            target, reason = (0.0, 0.0, 0.0), "inactive"
        elif not self._armed():
            target, reason = (0.0, 0.0, 0.0), "not_armed"
        elif not controller_sample_is_fresh(ts, now):
            target, reason = (0.0, 0.0, 0.0), "controller_stale"
        else:
            target = map_sticks(left_xy, right_xy, self.max_vx, self.max_vyaw)
            reason = "active" if any(target) else "idle"
        with self._lock:
            if self._active and not self._closed:
                self._target, self._sample_t, self._submit_reason = target, ts, reason
            self._raw_left, self._raw_right = raw_left, raw_right
        snap = self.snapshot(now)
        snap["target"] = list(target)
        snap["submit_reason"] = reason
        if reason not in ("active", "idle"):
            snap["reason"] = reason
        return snap

    def _armed(self):
        try:
            return bool(self._armed_fn())
        except Exception:
            return False

    # ---------------------------------------------------- sender side
    def tick(self):
        now = self._clock()
        with self._lock:
            target, ts, closed = self._target, self._sample_t, self._closed
        if closed:
            return
        if not self._armed():
            reason = "not_armed"
        elif not controller_sample_is_fresh(ts, now):
            reason = "controller_stale"
        else:
            reason = "active" if any(target) else "idle"
        ok = reason in ("active", "idle")
        vx, vyaw, vz = target if ok else (0.0, 0.0, 0.0)
        column_reason = reason
        if ok and vz != 0.0:
            height = self.height
            if height is None or not height.is_fresh(now):
                vz, column_reason = 0.0, "height_stale"
        with self._rpc_lock:
            with self._lock:
                if self._closed:
                    return
            self._send(vx, vyaw, vz)
        with self._lock:
            self._sent = [vx, 0.0, vyaw, vz]
            self._reason, self._column_reason = reason, column_reason
            self.ticks += 1

    def _send(self, vx, vyaw, vz):
        client = self._client
        if client is None:
            return
        try:
            code = client.Move(vx, 0.0, vyaw)
            self._record("move", code)
        except Exception as error:
            self._record_exception("Move", error)
        try:
            code = client.HeightAdjust(vz)
            self._record("height", code)
        except Exception as error:
            self._record_exception("HeightAdjust", error)

    def _record(self, kind, code):
        with self._lock:
            setattr(self, f"{kind}_code", code)
            if code != 0:
                setattr(self, f"nonzero_{kind}_codes", getattr(self, f"nonzero_{kind}_codes") + 1)
                log = self._logged_errors < MAX_LOGGED_RPC_ERRORS
                self._logged_errors += log
            else:
                log = False
        if log:
            self._safe_warn(f"[agv] {'Move' if kind == 'move' else 'HeightAdjust'} returned code {code}")

    def _record_exception(self, name, error):
        with self._lock:
            self.rpc_exceptions += 1
            log = self._logged_errors < MAX_LOGGED_RPC_ERRORS
            self._logged_errors += log
        if log:
            self._safe_warn(f"[agv] {name} raised {type(error).__name__}: {error}")

    def _safe_warn(self, message):
        try:
            self._warn(message)
        except BaseException:
            pass

    def _loop(self):
        nxt = time.monotonic()
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as error:
                self._record_exception("tick", error)
            nxt += self._period
            delay = nxt - time.monotonic()
            if delay < -self._period:     # overrun: do not burst to catch up
                nxt = time.monotonic()
                delay = 0.0
            self._stop.wait(max(0.0, delay))

    # ---------------------------------------------------- shutdown
    def shutdown(self, reason="shutdown", join_timeout_s=0.5, burst_gap_s=0.05):
        """Zero immediately, stop the sender (bounded), zero burst. Never raises."""
        try:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._target = (0.0, 0.0, 0.0)
                self.shutdown_reason = reason
            self._stop.set()
            got = self._rpc_lock.acquire(timeout=max(0.0, join_timeout_s))
            try:
                for i in range(SHUTDOWN_ZERO_BURST if self._client is not None else 0):
                    self._send(0.0, 0.0, 0.0)
                    if i < SHUTDOWN_ZERO_BURST - 1 and burst_gap_s:
                        time.sleep(burst_gap_s)
            finally:
                if got:
                    self._rpc_lock.release()
            if self._thread is not None:
                self._thread.join(max(0.0, join_timeout_s))
            with self._lock:
                self._sent = [0.0, 0.0, 0.0, 0.0]
                self._reason = self._column_reason = "shutdown"
            if self.height is not None:
                self.height.close()
        except BaseException as error:
            self._safe_warn(f"[agv] shutdown error: {type(error).__name__}: {error}")

    # ---------------------------------------------------- telemetry
    def snapshot(self, now=None):
        now = self._clock() if now is None else now
        height = None
        if self.height is not None:
            try:
                height = self.height.snapshot(now)
            except Exception:
                height = None
        with self._lock:
            return {
                "enabled": True,
                "active": self._active and not self._closed,
                "raw_left_xy": self._raw_left,
                "raw_right_xy": self._raw_right,
                "target": list(self._target),
                "sent": list(self._sent),             # [vx, vy(=0), vyaw, column vz]
                "reason": self._reason,
                "column_reason": self._column_reason,
                "move_code": self.move_code,
                "height_code": self.height_code,
                "nonzero_move_codes": self.nonzero_move_codes,
                "nonzero_height_codes": self.nonzero_height_codes,
                "rpc_exceptions": self.rpc_exceptions,
                "ticks": self.ticks,
                "height": height,
                "max_vx": self.max_vx,
                "max_vyaw": self.max_vyaw,
                "sender_alive": self.sender_alive(),
                "shutdown_reason": self.shutdown_reason,
            }
