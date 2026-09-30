"""Non-blocking robot-state side channel (sportmodestate + periodic FSM id).

Nothing here runs in the control loop except ``snapshot()`` (pure, in-memory,
lock-guarded).  DDS callbacks and the FSM poll thread write the state; every
failure becomes a counter and never raises into the caller.
"""
import threading
import time

SPORT_TOPICS = ("rt/sportmodestate", "rt/odommodestate")


def _default_subscriber_factory(topic, handler):
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    sub = ChannelSubscriber(topic, SportModeState_)
    sub.Init(handler, 1)
    return sub


class RobotStateMonitor:
    def __init__(self, subscriber_factory=None, fsm_reader=None, fsm_period_s=1.0,
                 topics=SPORT_TOPICS):
        self._factory = subscriber_factory
        self._fsm_reader = fsm_reader
        self._fsm_period_s = max(0.05, float(fsm_period_s)) if fsm_period_s else 1.0
        self._topics = tuple(topics)
        self._lock = threading.Lock()
        self._subs = []
        self._thread = None
        self._stop = threading.Event()
        self._closed = False
        self._sport = None  # (t, topic, mode, gait, velocity)
        self._fsm = None  # (t, id)
        self.sport_samples = 0
        self.sport_errors = 0
        self.sub_failures = 0
        self.fsm_errors = 0

    # -- writers (callback / poll thread) ----------------------------------
    def on_sport_state(self, msg, now=None, topic=None):
        try:
            t = time.monotonic() if now is None else now
            vel = [float(v) for v in list(msg.velocity)[:3]]
            rec = (t, topic, int(msg.mode), int(msg.gait_type), vel)
            with self._lock:
                self._sport = rec
                self.sport_samples += 1
        except Exception:
            with self._lock:
                self.sport_errors += 1

    def start(self):
        for topic in self._topics:
            try:
                factory = self._factory
                handler = (lambda m, _t=topic: self.on_sport_state(m, topic=_t))
                sub = (factory or _default_subscriber_factory)(topic, handler)
                if sub is not None:
                    self._subs.append(sub)
            except BaseException:
                with self._lock:
                    self.sub_failures += 1
        if self._fsm_reader is not None:
            self._thread = threading.Thread(target=self._poll, name="robot-fsm-poll", daemon=True)
            self._thread.start()

    def _poll(self):
        while not self._stop.is_set():
            try:
                value = self._fsm_reader()
                if value is not None:
                    with self._lock:
                        self._fsm = (time.monotonic(), int(value))
            except BaseException:
                with self._lock:
                    self.fsm_errors += 1
            self._stop.wait(self._fsm_period_s)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        for sub in self._subs:
            try:
                sub.Close()
            except BaseException:
                pass
        self._subs = []

    # -- reader (control loop): pure, in-memory -----------------------------
    def snapshot(self, now=None):
        t = time.monotonic() if now is None else now
        with self._lock:
            sport, fsm = self._sport, self._fsm
            out = {
                "sport_samples": self.sport_samples,
                "sport_errors": self.sport_errors,
                "sub_failures": self.sub_failures,
                "fsm_errors": self.fsm_errors,
                "sport_topic": sport[1] if sport else None,
                "sport_mode": sport[2] if sport else None,
                "sport_gait_type": sport[3] if sport else None,
                "sport_velocity": list(sport[4]) if sport else None,
                "sport_age_ms": round((t - sport[0]) * 1000.0, 1) if sport else None,
                "fsm_id": fsm[1] if fsm else None,
                "fsm_age_ms": round((t - fsm[0]) * 1000.0, 1) if fsm else None,
            }
        return out


def stop_locomotion_best_effort(loco_wrapper, reason, timeout=0.3, attempts=3):
    """Blocking-but-bounded StopMove for cleanup paths; never raises."""
    if loco_wrapper is None:
        return None
    try:
        return loco_wrapper.StopMove(reason, timeout=timeout, attempts=attempts)
    except BaseException:
        return None
