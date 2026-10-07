"""Graceful session shutdown for the dev-inspire teleop (G1_29 + Inspire).

Ported from the main line (SloBruno/xr_teleoperate, commit edfd901,
``teleop/utils/arm_graceful_shutdown.py`` + ``graceful_g1_29_shutdown``) and
adapted to this branch, where the arm writer and the Inspire command process
already publish from construction (before ``r``).

Order on ``q`` / Ctrl+C / SIGTERM / SIGHUP / exception (each step bounded,
exceptions never skip later steps, idempotent):

1. Stop the Inspire hand command process (stop event -> join -> terminate ->
   kill). No open/close command is sent: the hand stays at its last commanded
   position (operator preference: no automatic hand motion at session edges).
2. G1_29: velocity-limited return of both arms to the all-zero pose
   (<= 0.5 rad/s per joint), then ramp the rt/arm_sdk weight
   (kNotUsedJoint0.q) 1 -> 0 over 2 s so the Unitree motion controller takes
   the arms back, confirm a published weight-0 frame, deactivate the writer.
   Other arm profiles keep the legacy ``ctrl_dual_arm_go_home``.
3. TeleVuer: ``close()``, then make sure the Vuer process is gone
   (terminate -> kill, bounded joins) so no orphan is left behind.
4. Reap every remaining multiprocessing child (except the logging listener)
   and their descendants.

The RS-485 driver is NOT touched here: the launcher stops the driver it
started after this process exits.
"""

import multiprocessing
import os
import signal
import threading
import time

from teleop.utils.arm_graceful_shutdown import run_graceful_arm_shutdown

LOG_LISTENER_NAMES = ("LogListenerProcess",)


class ShutdownSignal(KeyboardInterrupt):
    """Raised in the main process on SIGTERM/SIGHUP to run the finally block."""


def install_shutdown_signal_handlers(signals=(signal.SIGTERM, signal.SIGHUP)):
    """SIGTERM/SIGHUP in the main process behave like Ctrl+C (graceful).

    Forked children inherit the handler; there it restores the default action
    and re-raises the signal, so ``process.terminate()`` still kills them.
    """
    main_pid = os.getpid()
    fired = {"n": 0}

    def handler(signum, _frame):
        if os.getpid() != main_pid:
            # Forked child (Vuer, hand process): behave exactly as before
            # (default action = die), so terminate() keeps working.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        fired["n"] += 1
        if fired["n"] == 1:
            raise ShutdownSignal(f"signal {signum}")
        # A repeated signal while the graceful shutdown runs is ignored here:
        # the shutdown is bounded and always releases/deactivates.

    installed = []
    for sig in signals:
        try:
            signal.signal(sig, handler)
            installed.append(sig)
        except (ValueError, OSError):
            pass
    return installed


def _log(log, level, message):
    if log is None:
        return
    try:
        getattr(log, level)(message)
    except BaseException:
        pass


def stop_process(process, *, join_timeout=1.0, kill_timeout=0.5):
    """join -> terminate -> kill a multiprocessing.Process; bounded, never raises."""
    if process is None:
        return True
    try:
        if process.is_alive() and join_timeout > 0:
            process.join(timeout=join_timeout)
        if process.is_alive():
            process.terminate()
            process.join(timeout=kill_timeout)
        if process.is_alive():
            process.kill()
            process.join(timeout=kill_timeout)
        return not process.is_alive()
    except BaseException:
        return False


def _descendants(pid):
    if pid is None:
        return []
    try:
        import psutil
        return psutil.Process(pid).children(recursive=True)
    except BaseException:
        return []


def _kill_all(procs, timeout):
    for proc in procs:
        try:
            if proc.is_running():
                proc.kill()
                proc.wait(timeout=timeout)
        except BaseException:
            pass


def close_televuer(tv_wrapper, *, timeout=1.0, log=None):
    """Close TeleVuer and guarantee its Vuer process is gone (bounded)."""
    if tv_wrapper is None:
        return True
    error = []

    def _close():
        try:
            tv_wrapper.close()
        except BaseException as exc:  # pragma: no cover - logged below
            error.append(exc)

    closer = threading.Thread(target=_close, name="televuer-close", daemon=True)
    closer.start()
    closer.join(timeout=timeout)
    if closer.is_alive():
        _log(log, "warning", f"[shutdown] TeleVuer close() exceeded {timeout:.1f}s; forcing.")
    if error:
        _log(log, "warning", f"[shutdown] TeleVuer close() failed: {error[0]!r}")
    process = getattr(getattr(tv_wrapper, "tvuer", None), "process", None)
    # Collect the Vuer process's own children BEFORE it dies; afterwards they
    # are reparented to init and become the orphan seen blocked in pipe_read.
    descendants = _descendants(getattr(process, "pid", None))
    gone = stop_process(process, join_timeout=0.0, kill_timeout=timeout)
    _kill_all(descendants, timeout)
    _log(log, "info" if gone else "error", f"[shutdown] TeleVuer process stopped={gone}")
    return gone


def reap_child_processes(*, timeout=1.0, exclude_names=LOG_LISTENER_NAMES, log=None,
                         active_children=multiprocessing.active_children):
    """Terminate/kill remaining multiprocessing children and their descendants."""
    try:
        children = [p for p in active_children() if p.name not in exclude_names]
    except BaseException:
        return []
    descendants = []
    for child in children:  # grandchildren would be orphaned once the child dies
        descendants.extend(_descendants(child.pid))
    leftovers = []
    for child in children:
        if not stop_process(child, join_timeout=0.0, kill_timeout=timeout):
            leftovers.append(child.pid)
    _kill_all(descendants, timeout)
    if children:
        _log(log, "info", f"[shutdown] reaped {len(children)} child process(es); leftovers={leftovers}")
    return leftovers


def run_session_shutdown(*, arm_kind, arm_ctrl=None, arm_ik=None, hand_ctrl=None,
                         tv_wrapper=None, log=None, clock=time.monotonic, sleep=time.sleep,
                         hand_join_timeout=1.0, televuer_timeout=1.0, reap=True):
    """Bounded, idempotent session teardown. Returns the list of steps run."""
    steps = []

    # 1. Hand first: signal + bounded join; no hand command is sent.
    if hand_ctrl is not None and not getattr(hand_ctrl, "_xr_session_shutdown_done", False):
        try:
            hand_ctrl._xr_session_shutdown_done = True
        except BaseException:
            pass
        try:
            deactivate = getattr(hand_ctrl, "deactivate", None)
            if deactivate is not None:
                deactivate(join_timeout=hand_join_timeout)
            steps.append("hand_stopped")
        except BaseException as error:
            _log(log, "error", f"[shutdown] hand stop failed: {error!r}")
            steps.append("hand_stop_failed")

    # 2. Arm.
    if arm_ctrl is not None:
        if arm_kind == "G1_29":
            def emit(name, detail):
                _log(log, "info", f"[shutdown] {name} {detail}")

            gravity = getattr(arm_ik, "gravity_tauff", None) if arm_ik is not None else None
            try:
                result = run_graceful_arm_shutdown(arm_ctrl, clock=clock, sleep=sleep, emit=emit,
                                                   gravity_tauff=gravity)
            except BaseException as error:
                _log(log, "error", f"[shutdown] graceful arm shutdown failed: {error!r}")
                result = None
            if result is None or not result.deactivated:
                try:
                    arm_ctrl.deactivate()
                except BaseException as error:
                    _log(log, "error", f"[shutdown] arm deactivate failed: {error!r}")
            steps.append("arm_released")
        elif not getattr(arm_ctrl, "_xr_session_shutdown_done", False):
            try:
                arm_ctrl._xr_session_shutdown_done = True
                arm_ctrl.ctrl_dual_arm_go_home()
            except BaseException as error:
                _log(log, "error", f"[shutdown] ctrl_dual_arm_go_home failed: {error!r}")
            steps.append("arm_go_home")

    # 3. TeleVuer without orphans.
    if tv_wrapper is not None and not getattr(tv_wrapper, "_xr_session_shutdown_done", False):
        try:
            tv_wrapper._xr_session_shutdown_done = True
        except BaseException:
            pass
        close_televuer(tv_wrapper, timeout=televuer_timeout, log=log)
        steps.append("televuer_closed")

    # 4. Anything left (hand/vuer children that ignored the request).
    if reap:
        reap_child_processes(timeout=televuer_timeout, log=log)
        steps.append("children_reaped")
    return steps
