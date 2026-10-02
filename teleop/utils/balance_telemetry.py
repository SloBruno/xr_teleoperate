"""Passive balance/IMU side channel for diagnosing uncommanded G1 walking.

Opt-in with ``G1_BALANCE_TELEMETRY=1`` (rate ``G1_BALANCE_TELEMETRY_HZ``,
default 30).  Read-only: only DDS *readers* are created here, never writers,
RPC clients or actuator paths.

* DDS callbacks only store the received sample reference + monotonic time
  under a short lock (decimated); no extraction, JSON or I/O.
* ``snapshot()`` is pure in-memory (lock held only to copy references) and is
  what the control loop calls; the result goes into the pose-telemetry record,
  which the existing bounded JSONL worker serializes (``put_nowait``).
* ``rt/lowstate`` is NOT subscribed again: the arm controller's existing
  reader thread forwards each message to ``on_lowstate`` (pelvis IMU, legs,
  waist, arms q/dq/tau_est).
* Every failure becomes a counter; nothing raises into the caller.

Field names follow the SDK IDLs (unitree_sdk2py/idl): unitree_hg IMUState_,
unitree_hg LowState_/MotorState_, unitree_go SportModeState_, nav_msgs
Odometry_, unitree_go ConfigChangeStatus_ {name, content}.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

SCHEMA_VERSION = 1

# name -> (topic, type key understood by the default subscriber factory)
BALANCE_TOPICS = {
    "secondary_imu": ("rt/secondary_imu", "unitree_hg.IMUState_"),
    "sportmodestate": ("rt/sportmodestate", "unitree_go.SportModeState_"),
    "odommodestate": ("rt/odommodestate", "unitree_go.SportModeState_"),
    "odom_pelvis": ("rt/state_estimator/odom_pelvis", "nav_msgs.Odometry_"),
    "odom_torso": ("rt/state_estimator/odom_torso", "nav_msgs.Odometry_"),
    "fusion_odom": ("rt/state_estimator/fusion_odom", "nav_msgs.Odometry_"),
    "config_change_status": ("rt/config_change_status", "unitree_go.ConfigChangeStatus_"),
    # Passive reader of the joystick bus: our own publications AND any other
    # publisher (app/physical-remote bridge) show up here.
    "wirelesscontroller_bus": ("rt/wirelesscontroller", "unitree_go.WirelessController_"),
}
SOURCES = ("lowstate",) + tuple(BALANCE_TOPICS)
_EVENT_SOURCES = ("config_change_status",)

LEG_IDX = tuple(range(0, 12))
WAIST_IDX = (12, 13, 14)
ARM_IDX = tuple(range(15, 29))

UNCOMMANDED_SPEED_MPS = 0.05
COMMAND_ZERO_EPS = 1e-3
FOOT_CONTACT_THRESHOLD = 20.0


# ---------------------------------------------------------------- extraction
def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _vec(values, n=None):
    try:
        out = [float(v) for v in list(values)]
    except (TypeError, ValueError):
        return None
    if n is not None and len(out) != n:
        return None
    return out if all(math.isfinite(v) for v in out) else None


def extract_imu(imu):
    """unitree_hg/unitree_go IMUState_ -> dict; rpy rad, gyro rad/s, acc m/s^2."""
    try:
        return {
            "rpy": _vec(imu.rpy, 3),
            "quaternion_wxyz": _vec(imu.quaternion, 4),
            "gyroscope": _vec(imu.gyroscope, 3),
            "accelerometer": _vec(imu.accelerometer, 3),
            "temperature": _f(imu.temperature),
        }
    except Exception:
        return None


def _motor_block(motors, idx):
    return {
        "q": [_f(motors[i].q) for i in idx],
        "dq": [_f(motors[i].dq) for i in idx],
        "tau_est": [_f(motors[i].tau_est) for i in idx],
        "temperature": [list(motors[i].temperature)[:2] for i in idx],
    }


def extract_lowstate(msg):
    try:
        motors = msg.motor_state
        return {
            "tick": int(msg.tick),
            # physical remote raw bytes (unitree_hg LowState_.wireless_remote[40])
            "wireless_remote_hex": bytes(int(b) & 0xFF for b in list(msg.wireless_remote)).hex(),
            "mode_machine": int(msg.mode_machine),
            "imu_pelvis": extract_imu(msg.imu_state),
            "legs": _motor_block(motors, LEG_IDX),
            "waist": _motor_block(motors, WAIST_IDX),
            "arms": {k: v for k, v in _motor_block(motors, ARM_IDX).items() if k != "temperature"},
        }
    except Exception:
        return None


def extract_sport(msg):
    """unitree_go SportModeState_ (the only SportModeState_ in the Python SDK)."""
    try:
        return {
            "mode": int(msg.mode),
            "gait_type": int(msg.gait_type),
            "progress": _f(msg.progress),
            "error_code": int(msg.error_code),
            "body_height": _f(msg.body_height),
            "position": _vec(msg.position, 3),
            "velocity": _vec(msg.velocity, 3),
            "yaw_speed": _f(msg.yaw_speed),
            "foot_force": [int(v) for v in list(msg.foot_force)],
            "foot_position_body": _vec(msg.foot_position_body),
            "foot_speed_body": _vec(msg.foot_speed_body),
            "imu": extract_imu(msg.imu_state),
        }
    except Exception:
        return None


def extract_odom(msg):
    """nav_msgs Odometry_ -> position, orientation (w,x,y,z), twist."""
    try:
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
        stamp = msg.header.stamp
        return {
            "stamp_s": _f(int(stamp.sec) + int(stamp.nanosec) * 1e-9),
            "frame_id": str(msg.header.frame_id),
            "child_frame_id": str(msg.child_frame_id),
            "position": _vec((p.x, p.y, p.z), 3),
            "orientation_wxyz": _vec((o.w, o.x, o.y, o.z), 4),
            "linear": _vec((lin.x, lin.y, lin.z), 3),
            "angular": _vec((ang.x, ang.y, ang.z), 3),
        }
    except Exception:
        return None


def extract_wireless(msg):
    try:
        return {"lx": _f(msg.lx), "ly": _f(msg.ly), "rx": _f(msg.rx), "ry": _f(msg.ry),
                "keys": int(msg.keys)}
    except Exception:
        return None


def extract_config(msg):
    try:
        return {"name": str(msg.name), "content": str(msg.content)[:512]}
    except Exception:
        return None


_EXTRACT = {
    "lowstate": extract_lowstate,
    "secondary_imu": extract_imu,
    "sportmodestate": extract_sport,
    "odommodestate": extract_sport,
    "odom_pelvis": extract_odom,
    "odom_torso": extract_odom,
    "fusion_odom": extract_odom,
    "config_change_status": extract_config,
    "wirelesscontroller_bus": extract_wireless,
}


# ------------------------------------------------------------------- derived
def wireless_to_body_command(published):
    """Published WirelessController (lx, ly, rx, ...) -> robot [vx, vy, omega]
    in normalized units (robot reads [ly, -lx, -rx]; see loco_wireless)."""
    if published is None:
        return None
    try:
        lx, ly, rx = (float(v) for v in list(published)[:3])
    except (TypeError, ValueError):
        return None
    return [ly + 0.0, -lx + 0.0, -rx + 0.0]


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def derive_balance(*, pelvis_rpy, torso_rpy, body_velocity, loco_command):
    diff = None
    if pelvis_rpy and torso_rpy and len(pelvis_rpy) == 3 and len(torso_rpy) == 3:
        diff = [_wrap(a - b) for a, b in zip(pelvis_rpy, torso_rpy)]
    speed = None
    if body_velocity and len(body_velocity) >= 2 and None not in body_velocity[:2]:
        speed = math.hypot(body_velocity[0], body_velocity[1])
    command_zero = None
    if loco_command is not None:
        try:
            command_zero = all(abs(float(v)) < COMMAND_ZERO_EPS for v in loco_command)
        except (TypeError, ValueError):
            command_zero = None
    uncommanded = None
    if speed is not None and command_zero is not None:
        uncommanded = bool(command_zero and speed > UNCOMMANDED_SPEED_MPS)
    return {
        "pelvis_minus_torso_rpy": diff,
        "body_velocity": list(body_velocity) if body_velocity else None,
        "horizontal_speed": speed,
        "command_zero": command_zero,
        "uncommanded_motion": uncommanded,
    }


# ------------------------------------------------------------------- monitor
def _default_subscriber_factory(topic, type_key, handler):
    from unitree_sdk2py.core.channel import ChannelSubscriber
    if type_key == "unitree_hg.IMUState_":
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_ as cls
    elif type_key == "unitree_go.SportModeState_":
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as cls
    elif type_key == "nav_msgs.Odometry_":
        from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as cls
    elif type_key == "unitree_go.WirelessController_":
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_ as cls
    elif type_key == "unitree_go.ConfigChangeStatus_":
        from teleop.utils.config_change_status_idl import ConfigChangeStatus_ as cls
    else:
        raise ValueError(type_key)
    sub = ChannelSubscriber(topic, cls)
    sub.Init(handler, 0)  # listener callback: store-only handler
    return sub


class BalanceTelemetryMonitor:
    def __init__(self, subscriber_factory=None, clock=time.monotonic, rate_hz=30.0,
                 topics=None, foot_contact_threshold=FOOT_CONTACT_THRESHOLD):
        self._factory = subscriber_factory or _default_subscriber_factory
        self._clock = clock
        self.rate_hz = max(1.0, min(200.0, float(rate_hz)))
        self._store_interval = 0.5 / self.rate_hz
        self._topics = dict(BALANCE_TOPICS if topics is None else topics)
        self._foot_threshold = float(foot_contact_threshold)
        self._lock = threading.Lock()
        self._latest = {name: None for name in SOURCES}  # name -> (t, raw msg)
        self._events = deque(maxlen=32)
        self.counters = {name: {"received": 0, "stored": 0, "extract_errors": 0} for name in SOURCES}
        self.events_dropped = 0
        self.sub_failures = 0
        self.snapshot_errors = 0
        self._subs = []
        self._prev_contact = None
        self._closed = False

    # -- writers: DDS/arm-reader threads; store reference only ----------------
    def _store(self, name, msg):
        t = self._clock()
        with self._lock:
            c = self.counters[name]
            c["received"] += 1
            if name in _EVENT_SOURCES:
                if len(self._events) == self._events.maxlen:
                    self.events_dropped += 1
                self._events.append((t, msg))
                c["stored"] += 1
                return
            last = self._latest[name]
            if last is not None and t - last[0] < self._store_interval:
                return
            self._latest[name] = (t, msg)
            c["stored"] += 1

    def on_lowstate(self, msg):
        try:
            self._store("lowstate", msg)
        except Exception:
            pass

    def _handler(self, name):
        def handle(msg):
            try:
                self._store(name, msg)
            except Exception:
                pass
        return handle

    def start(self):
        for name, (topic, type_key) in self._topics.items():
            try:
                sub = self._factory(topic, type_key, self._handler(name))
                if sub is not None:
                    self._subs.append(sub)
            except BaseException:
                self.sub_failures += 1
        return self

    def close(self):
        if self._closed:
            return
        self._closed = True
        for sub in self._subs:
            try:
                sub.Close()
            except BaseException:
                pass
        self._subs = []

    # -- reader: control loop; pure in-memory ---------------------------------
    def _extract(self, name, raw):
        out = _EXTRACT[name](raw)
        if out is None:
            with self._lock:
                self.counters[name]["extract_errors"] += 1
        return out

    def snapshot(self, now=None, *, loco_command=None, loco_raw=None, com=None,
                 arm_commanded_q=None, loco_command_source=None):
        try:
            out = self._snapshot(now, loco_command, loco_raw, com, arm_commanded_q)
            out["loco_command_source"] = loco_command_source
            return out
        except Exception as error:
            self.snapshot_errors += 1
            return {"schema_version": SCHEMA_VERSION, "error": type(error).__name__,
                    "snapshot_errors": self.snapshot_errors}

    def _snapshot(self, now, loco_command, loco_raw, com, arm_commanded_q):
        t = self._clock() if now is None else now
        with self._lock:
            latest = dict(self._latest)
            events = list(self._events)
            self._events.clear()
        sources, data = {}, {}
        for name in SOURCES:
            item = latest.get(name)
            sources[name] = {
                "t_monotonic": item[0] if item else None,
                "age_ms": round((t - item[0]) * 1000.0) if item and t >= item[0] else None,
            }
            data[name] = self._extract(name, item[1]) if item else None
        with self._lock:
            counters = {k: dict(v) for k, v in self.counters.items()}
        low = data["lowstate"] or {}
        pelvis = low.get("imu_pelvis")
        torso = data["secondary_imu"]
        sport = data["odommodestate"] or data["sportmodestate"]
        velocity = (sport or {}).get("velocity")
        if velocity is None and data["odom_pelvis"]:
            velocity = data["odom_pelvis"].get("linear")
        derived = derive_balance(
            pelvis_rpy=(pelvis or {}).get("rpy"), torso_rpy=(torso or {}).get("rpy"),
            body_velocity=velocity, loco_command=loco_command)
        contact, steps = None, []
        force = (sport or {}).get("foot_force")
        if force:
            contact = [abs(v) >= self._foot_threshold for v in force]
            if self._prev_contact is not None and len(self._prev_contact) == len(contact):
                steps = [{"foot": i, "contact": c} for i, (p, c) in
                         enumerate(zip(self._prev_contact, contact)) if p != c]
            self._prev_contact = contact
        derived["foot_contact"] = contact
        derived["step_events"] = steps
        config_changes = []
        for et, raw in events:
            item = self._extract("config_change_status", raw)
            if item is not None:
                item["t_monotonic"] = et
                config_changes.append(item)
        arms = low.get("arms")
        if arms is not None:
            arms = dict(arms)
            arms["commanded_q"] = _vec(arm_commanded_q, 14) if arm_commanded_q is not None else None
        return {
            "schema_version": SCHEMA_VERSION,
            "timestamp_monotonic": t,
            "rate_hz": self.rate_hz,
            "sources": sources,
            "imu_pelvis": pelvis,
            "imu_torso": torso,
            "lowstate_tick": low.get("tick"),
            "wireless_remote_hex": low.get("wireless_remote_hex"),
            "wirelesscontroller_bus": data["wirelesscontroller_bus"],
            "mode_machine": low.get("mode_machine"),
            "legs": low.get("legs"),
            "waist": low.get("waist"),
            "arms": arms,
            "sport": {"sportmodestate": data["sportmodestate"], "odommodestate": data["odommodestate"]},
            "odom": {"pelvis": data["odom_pelvis"], "torso": data["odom_torso"],
                     "fusion": data["fusion_odom"]},
            "loco_command": list(loco_command) if loco_command is not None else None,
            "loco_raw": ({str(k): _vec(v) for k, v in loco_raw.items()} if isinstance(loco_raw, dict)
                         else _vec(loco_raw) if loco_raw is not None else None),
            "com": dict(com) if com is not None else None,
            "derived": derived,
            "config_changes": config_changes,
            "counters": counters,
            "events_dropped": self.events_dropped,
            "sub_failures": self.sub_failures,
            "snapshot_errors": self.snapshot_errors,
        }


def loco_command_from(backend_telemetry, dispatched):
    """Command effectively sent: the WirelessController publisher's last
    published state mapped to [vx, vy, omega] (normalized), else the
    dispatched (vx, vy, omega) for RPC backends."""
    try:
        if isinstance(backend_telemetry, dict) and backend_telemetry.get("published") is not None:
            return wireless_to_body_command(backend_telemetry["published"]), "wireless_published"
        if dispatched is not None:
            return [float(v) for v in dispatched][:3], "dispatched"
    except Exception:
        pass
    return None, "unavailable"


def balance_snapshot_best_effort(monitor, *, loco_backend=None, dispatched=None, loco_raw=None,
                                 com_status=None, arm_commanded_q=None, warn=None):
    """Control-loop entry point: in-memory snapshot or None; never raises."""
    if monitor is None:
        return None
    try:
        command, source = loco_command_from(loco_backend, dispatched)
        com = com_status.as_dict() if hasattr(com_status, "as_dict") else com_status
        return monitor.snapshot(loco_command=command, loco_raw=loco_raw, com=com,
                                arm_commanded_q=arm_commanded_q, loco_command_source=source)
    except Exception as error:
        try:
            if warn is not None:
                warn(f"[balance_telemetry] snapshot failed: {type(error).__name__}")
        except Exception:
            pass
        return None


def attach_lowstate_tap(monitor, arm_ctrl):
    """Reuse the arm controller's rt/lowstate reader (no new subscriber)."""
    if monitor is None or arm_ctrl is None or not hasattr(arm_ctrl, "lowstate_buffer"):
        return False
    try:
        arm_ctrl.lowstate_observer = monitor.on_lowstate
        return True
    except Exception:
        return False


def create_from_env(env, subscriber_factory=None, warn=None):
    """Opt-in via G1_BALANCE_TELEMETRY=1; any failure -> None (inert)."""
    if str(env.get("G1_BALANCE_TELEMETRY", "")).strip().lower() not in ("1", "true", "yes"):
        return None
    try:
        hz = float(env.get("G1_BALANCE_TELEMETRY_HZ", "30"))
        if not math.isfinite(hz) or hz <= 0:
            hz = 30.0
        return BalanceTelemetryMonitor(subscriber_factory=subscriber_factory, rate_hz=hz).start()
    except Exception as error:
        try:
            if warn is not None:
                warn(f"[balance_telemetry] not started: {error!r}")
        except Exception:
            pass
        return None
