"""BotBrain-style Regular-mode (FSM 500/501) walking preflight, watchdog and
latest-only Move sender.  SetFsmId is sent ONLY when explicitly requested
(--loco-request-fsm 500), once, in the preflight; RUN (801) is not supported."""
import logging
import threading
import time

ACCEPTED_FSM_IDS = frozenset({500, 501})  # Regular mode (R1+X) only
SPEED_MODE_ATTEMPTS = 3
STICK_TIMEOUT_S = 0.2
FSM_CONFIRM_TIMEOUT_S = 3.0
FSM_POLL_INTERVAL_S = 0.1
logger = logging.getLogger(__name__)


def parse_request_fsm(value):
    """None/'none' -> None; only 500 is accepted (801/501/anything else -> ValueError)."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
        return None
    try:
        v = int(str(value).strip())
    except ValueError:
        v = None
    if v != 500:
        raise ValueError(f"--loco-request-fsm só aceita 'none' ou 500 (recebido {value!r}); 801/501/outros são proibidos")
    return v


def _request_fsm_500(wrapper, res, fsm, sleep, clock):
    """fsm is 500 or 501. Returns True when locomotion may proceed."""
    res["fsm_requested"] = 500
    if fsm == 500:
        res.update(fsm_after=500, fsm_confirm_s=0.0, fsm_banner="FSM solicitado: 500 (já estava em 500, nada enviado)")
        return True
    t0 = clock()
    rc = wrapper.set_fsm_id(500)   # exactly one SetFsmId, never retried
    res["set_fsm_rc"] = rc
    after = None
    reason = None
    if rc != 0:
        after = wrapper.read_fsm_id(timeout=0.3)
        reason = f"fsm_request_failed:rc={rc}"
    else:
        while True:
            after = wrapper.read_fsm_id(timeout=0.3)
            if after == 500:
                break
            if clock() - t0 >= FSM_CONFIRM_TIMEOUT_S:
                reason = f"fsm_request_failed:timeout_fsm={after}"
                break
            sleep(FSM_POLL_INTERVAL_S)
    res["fsm_after"] = after
    res["fsm_confirm_s"] = round(clock() - t0, 3) if reason is None else None
    if reason:
        res["refusal_reason"] = reason
        res["fsm_banner"] = f"FSM solicitado: {fsm} -> 500 FALHOU ({reason}); locomoção DESABILITADA"
        res["message"] = res["fsm_banner"]
        return False
    res["fsm_banner"] = f"FSM solicitado: {fsm} -> 500 (confirmado)"
    return True


def run_loco_preflight(wrapper, sleep=time.sleep, attempts=SPEED_MODE_ATTEMPTS, request_fsm=None, clock=time.monotonic):
    """Read-only FSM check, then best-effort SetSpeedMode(0) w/ retries, best-effort ContinuousGait(false),
    checked zero Move.  Returns dict with loco_enabled and refusal_reason."""
    res = {"loco_enabled": False, "refusal_reason": None, "message": "", "fsm_id": None,
           "set_speed_mode_rc": None, "set_speed_mode_ok": None, "preflight_ok": False,
           "fsm_before": None, "fsm_requested": None, "fsm_after": None, "set_fsm_rc": None,
           "fsm_confirm_s": None, "fsm_banner": None}
    request = parse_request_fsm(request_fsm)   # ValueError before anything is read/sent
    res["backend"] = getattr(wrapper, "backend", "setvelocity")
    berr = getattr(wrapper, "backend_error", None)
    if berr:   # fail-safe: no silent fallback to another backend
        res["refusal_reason"] = f"backend_unavailable:{berr}"
        res["message"] = (f"Backend {res['backend']} indisponível ({berr}); locomoção DESABILITADA. "
                          "Use G1_LOCO_BACKEND=setvelocity para forçar o backend antigo.")
        return res
    fsm = wrapper.read_fsm_id(timeout=0.3)
    res["fsm_id"] = res["fsm_before"] = fsm
    if fsm is None:
        res["refusal_reason"] = "fsm_unreadable"
        res["message"] = "FSM unreadable; locomotion disabled. Put the robot in Regular mode (R1+X on the R3 remote)."
        return res
    if fsm not in ACCEPTED_FSM_IDS:
        res["refusal_reason"] = f"fsm_not_walk:{fsm}"
        res["message"] = (f"FSM id {fsm} is not Regular walk (500/501); locomotion disabled. "
                          "Enter Regular mode with R1+X on the R3 remote and restart.")
        return res
    if request == 500 and not _request_fsm_500(wrapper, res, fsm, sleep, clock):
        return res
    if res["fsm_after"] is not None:
        fsm = res["fsm_after"]
    rc = None
    if res["backend"] == "wirelesscontroller":
        # SetSpeedMode/ContinuousGait are RPCs of the SetVelocity path; the
        # joystick-state backend does not need them (not called, cannot fail).
        zrc = wrapper.checked_zero()
        if zrc != 0:
            res["refusal_reason"] = f"zero_publish_failed:{zrc}"
            res["message"] = f"Zero publish on rt/wirelesscontroller failed (rc={zrc}); locomotion disabled."
            return res
        res.update(loco_enabled=True, preflight_ok=True, message=f"FSM {fsm} Regular walk; preflight ok (wirelesscontroller).")
        return res
    for i in range(max(1, attempts)):
        rc = wrapper.set_speed_mode(0)
        if rc == 0:
            break
        sleep(0.25 * (i + 1))
    res["set_speed_mode_rc"] = rc
    res["set_speed_mode_ok"] = rc == 0
    if rc != 0:  # best effort: firmware may not accept RPC 7107; keep robot default profile
        logger.warning("SetSpeedMode não aceito pelo firmware (rc=%s); seguindo com o perfil padrão do robô", rc)
    try:
        wrapper.set_balance_mode(0)  # ContinuousGait(false), as BotBrain; best effort
    except Exception as e:
        logger.warning("ContinuousGait(false) falhou (%s); seguindo", e)
    zrc = wrapper.checked_zero()
    if zrc != 0:
        res["refusal_reason"] = f"zero_move_failed:{zrc}"
        res["message"] = f"Zero Move not acknowledged (rc={zrc}); locomotion disabled."
        return res
    res.update(loco_enabled=True, preflight_ok=True, message=f"FSM {fsm} Regular walk; preflight ok.")
    return res


class LocoWatchdog:
    """Sends StopMove if a non-zero command is older than timeout_s; retries
    until acked; re-arms on the next non-zero command."""

    def __init__(self, stop_fn, timeout_s=STICK_TIMEOUT_S):
        self._stop = stop_fn
        self.timeout_s = timeout_s
        self._last_nonzero = None
        self.armed_stop = False
        self.trips = 0
        self.stop_failures = 0
        self.last_rc = None
        self._counted = False

    def feed(self, nonzero, now):
        if nonzero:
            self._last_nonzero = now
            self.armed_stop = True
        else:
            self._last_nonzero = None
            self.armed_stop = False

    def check(self, now):
        if not self.armed_stop or self._last_nonzero is None:
            return False
        if now - self._last_nonzero <= self.timeout_s:
            return False
        try:
            rc = self._stop("watchdog_timeout")
        except BaseException:
            self.stop_failures += 1
            rc = None
        self.last_rc = rc
        if rc == 0:
            self.armed_stop = False
        if not self._counted:
            self.trips += 1
            self._counted = True
        if rc == 0:
            self._counted = False
        return True


class LatestMoveSender:
    """Daemon thread sending Move over a client with a short blocking timeout.
    One-slot mailbox: newer commands replace unsent older ones, so the control
    loop never blocks and a zero is never queued behind stale motion. Idle ticks
    run the watchdog, so a stalled control loop still stops the robot."""

    def __init__(self, send, stop_fn=None, watchdog_timeout_s=STICK_TIMEOUT_S, tick_s=0.05):
        self._send = send
        self._cv = threading.Condition()
        self._slot = None
        self._running = False
        self._thread = None
        self._tick = tick_s
        self.last_rc = None
        self.sent = 0
        self.nonzero_rc = 0
        self.watchdog = LocoWatchdog(stop_fn, watchdog_timeout_s) if stop_fn else None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="loco-move", daemon=True)
        self._thread.start()

    def stop(self):
        with self._cv:
            self._running = False
            self._cv.notify_all()
        if self._thread:
            self._thread.join(1.0)

    def submit(self, cmd):
        with self._cv:
            self._slot = tuple(cmd)
            if self.watchdog:
                self.watchdog.feed(any(cmd), time.monotonic())
            self._cv.notify()

    def _loop(self):
        while True:
            with self._cv:
                if self._slot is None and self._running:
                    self._cv.wait(self._tick)
                if not self._running:
                    return
                cmd, self._slot = self._slot, None
            if cmd is not None:
                try:
                    rc = self._send(*cmd)
                except BaseException:
                    rc = None
                self.last_rc = rc
                self.sent += 1
                if rc != 0:
                    self.nonzero_rc += 1
            elif self.watchdog:
                self.watchdog.check(time.monotonic())
