"""G1D teach-and-replay operator entrypoint (arms + Dex3, terminal keys only).

Keys (interactive terminal, sshkeyboard):
  r  start TEACH: arms + Dex3 compliant (gravity-compensated), recording
  s  stop teaching and save the recording
  p  REPLAY the latest saved recording (or --replay NAME)
  q  stop everything immediately (hold measured pose briefly, release), exit

Optional ``--web [PORT]`` (default 8095): offline browser control panel with the
same r/s/p/q actions (token-protected URL printed below); the keys keep working.

Separate from teleop_hand_and_arm.py: no XR, no IK, no video. Before the first
accepted r/p only DDS subscribers exist. Launch via run_g1d_teach_replay.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import secrets
import sys
import threading
import time
from contextlib import contextmanager
from urllib.request import Request, urlopen
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teleop.utils.teach_replay import (  # noqa: E402
    TeachReplayConfig,
    TeachReplaySession,
    default_teach_dir,
)

CONTROL_HZ = 200.0
STATE_WAIT_S = 3.0


def hand_sides_from_arg(value: str) -> tuple:
    return {"both": ("left", "right"), "left": ("left",), "right": ("right",), "none": ()}[value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network-interface", default="enP2p1s0", help="robot-bus DDS interface")
    parser.add_argument("--hands", choices=("both", "left", "right", "none"), default="both",
                        help="Dex3 hands to require/command (default both; 'none' = arms only, explicit opt-in)")
    parser.add_argument("--replay", default=None, metavar="NAME",
                        help="recording used by p (file name in the teach dir or path); default latest saved")
    parser.add_argument("--teach-dir", default=None, help="recording directory "
                        "(default ~/.local/state/xr_teleoperate/teach)")
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--web", nargs="?", type=int, const=8095, default=None, metavar="PORT",
                        help="also serve the offline web control panel on 0.0.0.0:PORT (default 8095)")
    return parser


def make_key_handler(session, replay_name=None):
    def on_press(key):
        if key == "q":
            session.request_quit()        # lock-free: never waits for a tick
        elif key == "r":
            session.request_teach()
        elif key == "s":
            session.request_save()
        elif key == "p":
            session.request_replay(replay_name)
    return on_press


class AsyncPrinter:
    """Terminal output off the control thread (bounded queue, drops if full)."""

    def __init__(self, print_fn=None, maxsize=256):
        self._print = print_fn or (lambda text: print(text, flush=True))
        self._queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._run, name="teach-printer", daemon=True)
        self._thread.start()

    def __call__(self, text):
        try:
            self._queue.put_nowait(str(text))
        except queue.Full:
            pass

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._print(item)
            except Exception:
                pass

    def close(self, timeout=1.0):
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            return
        self._thread.join(timeout)


def run_control_loop(session, *, period_s, sleep=time.sleep, clock=time.monotonic, max_quit_ticks=2000):
    """Tick until the session is done. Ctrl+C == q. Bounded after quit."""
    quit_ticks = 0
    next_t = clock()
    while not session.done:
        try:
            session.tick()
            if session.quit_requested:
                quit_ticks += 1
                if quit_ticks > max_quit_ticks:
                    break
            next_t += period_s
            delay = next_t - clock()
            if delay < -0.1:
                next_t = clock()
            sleep(max(0.0, delay))
        except KeyboardInterrupt:
            session.request_quit()


def enter_debug_mode_bounded(timeout_s=10.0, cancelled=lambda: False):
    """Release the vendor motion mode (same call as the G1D teleop) or raise."""
    from teleop.utils.motion_switcher import MotionSwitcher

    result = {}

    def worker():
        try:
            result["value"] = MotionSwitcher().Enter_Debug_Mode()
        except Exception as exc:  # pragma: no cover - hardware path
            result["error"] = exc

    thread = threading.Thread(target=worker, name="teach-debug-mode", daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_s
    while thread.is_alive():
        if cancelled():
            raise RuntimeError("cancelled by q while entering debug mode")
        if time.monotonic() > deadline:
            raise RuntimeError("debug mode (motion-mode release) timed out")
        time.sleep(0.05)
    status = (result.get("value") or (None, None))[0]
    if "error" in result or status is None:
        raise RuntimeError("could not enter debug mode (motion-mode release failed)")


CONFLICTING_SCRIPTS = ("teleop_hand_and_arm.py", "teach_replay_g1d.py")


def find_conflicting_processes(proc_root="/proc", own_pid=None):
    """Other python processes running a script that may write rt/lowcmd or Dex3 cmd.

    Exact argv matching (argv[0] basename starts with 'python', a later argv
    element's basename equals a conflicting script); never a grep pattern.
    """
    own_pid = os.getpid() if own_pid is None else own_pid
    found = []
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return found
    for entry in entries:
        if not entry.isdigit() or int(entry) == own_pid:
            continue
        try:
            with open(os.path.join(proc_root, entry, "cmdline"), "rb") as handle:
                argv = [a.decode(errors="replace") for a in handle.read().split(b"\0") if a]
        except OSError:
            continue
        if not argv or not os.path.basename(argv[0]).startswith("python"):
            continue
        if any(os.path.basename(a) in CONFLICTING_SCRIPTS for a in argv[1:]):
            found.append((int(entry), " ".join(argv)))
    return found


DEFAULT_LOCK_PATH = Path.home() / ".local" / "state" / "xr_teleoperate" / "teach_replay.lock"
DEFAULT_STATE_PATH = Path.home() / ".local" / "state" / "xr_teleoperate" / "teach_replay.session.json"
NO_WEB_SESSION = "__G1D_TEACH_REPLAY_LIVE_NO_WEB__"


def proc_start_time(pid, proc_root="/proc"):
    text = Path(proc_root, str(int(pid)), "stat").read_text(encoding="utf-8")
    return int(text.rsplit(")", 1)[1].split()[19])


def current_process_argv(proc_root="/proc", pid=None):
    pid = os.getpid() if pid is None else pid
    return [arg.decode(errors="replace") for arg in Path(proc_root, str(pid), "cmdline").read_bytes().split(b"\0") if arg]


def write_session_state(path, *, pid, proc_start_time, argv, port, token, full_url,
                        web_enabled, session_id):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": int(pid), "proc_start_time": int(proc_start_time), "argv": list(argv),
               "port": int(port), "token": str(token), "full_url": str(full_url),
               "web_enabled": bool(web_enabled), "session_id": str(session_id),
               "started_at": time.time()}
    with _state_file_lock(path):
        temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temp, 0o600)
        os.replace(temp, path)


@contextmanager
def _state_file_lock(path):
    import fcntl
    lock_path = Path(path).with_name(Path(path).name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def remove_session_state(path, session_id):
    path = Path(path)
    with _state_file_lock(path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("pid") == os.getpid() and data.get("session_id") == session_id:
                path.unlink()
        except (OSError, ValueError):
            pass


def _probe_live_panel(data):
    request = Request(f"http://127.0.0.1:{int(data['port'])}/api/session",
                      headers={"X-Token": str(data["token"])})
    with urlopen(request, timeout=0.5) as response:
        if response.status != 200:
            raise ValueError("panel session probe failed")
        body = json.loads(response.read().decode("utf-8"))
    if body != {"ok": True, "session_id": str(data["session_id"])}:
        raise ValueError("panel session mismatch")


def recover_existing_session(path=DEFAULT_STATE_PATH, *, proc_root="/proc", own_pid=None):
    path = Path(path)
    with _state_file_lock(path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            pid = int(data["pid"])
            web_enabled = bool(data.get("web_enabled"))
            if web_enabled:
                parsed = urlsplit(str(data["full_url"]))
                if parsed.scheme != "http" or parsed.port != int(data["port"]):
                    raise ValueError("panel URL/port mismatch")
                if parse_qs(parsed.query).get("t", [None])[0] != str(data["token"]):
                    raise ValueError("panel URL/token mismatch")
            if not data.get("session_id") or not isinstance(data.get("started_at"), (int, float)):
                raise ValueError("invalid session freshness")
            if pid <= 0 or proc_start_time(pid, proc_root) != int(data["proc_start_time"]):
                raise ValueError("stale process identity")
            if current_process_argv(proc_root, pid) != list(data["argv"]):
                raise ValueError("process argv changed")
            if web_enabled:
                if int(data["port"]) <= 0:
                    raise ValueError("invalid session state")
                _probe_live_panel(data)
            return data["full_url"] if web_enabled else NO_WEB_SESSION
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            try:
                path.unlink()
            except OSError:
                pass
            return None


def acquire_session_lock(path=DEFAULT_LOCK_PATH):
    """Exclusive non-blocking process-lifetime lock; None if already held."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    os.set_inheritable(handle.fileno(), False)
    return handle


def _load_model():
    """Arm gravity + limits from the same model the G1D teleop uses."""
    import numpy as np
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

    cwd = os.getcwd()
    os.chdir(Path(__file__).resolve().parent)       # IK uses ../assets relative paths
    try:
        ik = G1_29_ArmIK()
    finally:
        os.chdir(cwd)
    model = ik.reduced_robot.model
    lower = np.asarray(model.lowerPositionLimit, dtype=float).copy()
    upper = np.asarray(model.upperPositionLimit, dtype=float).copy()
    effort = np.asarray(model.effortLimit, dtype=float).copy()
    return ik.gravity_tauff, lower, upper, effort


def make_freshness_fn(io, sides, clock=time.monotonic):
    """State ages (s) for the web panel; None when missing/not configured."""
    def ages():
        out = {"arm": None, "left": None, "right": None}
        now = clock()
        try:
            arm = io.arm_state()
            out["arm"] = None if arm is None else max(0.0, now - float(arm.timestamp))
        except Exception:
            pass
        for side in sides:
            try:
                hand = io.hand_state(side)
                out[side] = None if hand is None else max(0.0, now - float(hand.timestamp))
            except Exception:
                pass
        return out
    return ages


def start_web_panel(session, io, sides, teach_dir, port, *, session_id=None, print_fn=print):
    """Start the panel; on failure keep the terminal keys only (never abort)."""
    from teleop.utils.teach_web import WebPanel, local_ipv4_addresses

    try:
        panel = WebPanel(session, teach_dir, host="0.0.0.0", port=port,
                         session_id=session_id,
                         freshness_fn=make_freshness_fn(io, sides, getattr(io, "clock", time.monotonic)),
                         arm_lower=getattr(session, "arm_lower", None),
                         arm_upper=getattr(session, "arm_upper", None))
        panel.start()
    except Exception as exc:
        print_fn(f"[teach] WEB panel NOT started ({type(exc).__name__}: {exc}); terminal keys only.")
        return None
    addresses = local_ipv4_addresses() or [("loopback", "127.0.0.1")]
    panel.canonical_url = panel.url(addresses[0][1])
    print_fn("[teach] WEB panel (same r/s/p/q; PARAR = q). Open on the phone/Quest/PC:")
    for name, ip in addresses:
        print_fn(f"[teach]   {name:>10}: {panel.url(ip)}")
    return panel


def _wait_for_state(io, sides, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if io.arm_state() is not None and all(io.hand_state(s) is not None for s in sides):
            return
        time.sleep(0.05)


def main(argv=None):
    args = build_parser().parse_args(argv)
    recovered = recover_existing_session()
    if recovered == NO_WEB_SESSION:
        print("[teach] existing session is live without a web panel", flush=True)
        return 0
    if recovered is not None:
        print(recovered, flush=True)
        return 0
    sides = hand_sides_from_arg(args.hands)
    teach_dir = Path(args.teach_dir).expanduser() if args.teach_dir else default_teach_dir()
    printer = AsyncPrinter()
    print("=" * 72, flush=True)
    print("G1D TEACH/REPLAY  (arms + Dex3: " + (", ".join(sides) if sides else "NONE - arms only") + ")",
          flush=True)
    print(f"DDS interface: {args.network_interface}   recordings: {teach_dir}", flush=True)
    print("Robot SUPPORTED, area clear, R3/e-stop in reach.  Keys: r=teach  s=save  p=replay  q=quit",
          flush=True)
    print("=" * 72, flush=True)

    session_lock = acquire_session_lock()
    if session_lock is None:
        print(f"[teach] REFUSED: another teach/replay session holds {DEFAULT_LOCK_PATH}", flush=True)
        printer.close()
        return 3
    conflicts = find_conflicting_processes()
    if conflicts:
        for pid, cmdline in conflicts:
            print(f"[teach] REFUSED: another arm/Dex3 writer may be running: pid {pid}: {cmdline}", flush=True)
        printer.close()
        return 3

    gravity_fn, lower, upper, effort = _load_model()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from teleop.robot_control.g1d_teach_replay_io import G1DTeachReplayIO

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)
    holder = {}
    io = G1DTeachReplayIO(
        hand_sides=sides,
        pre_activate=lambda: enter_debug_mode_bounded(
            cancelled=lambda: holder["session"].quit_requested),
    )
    _wait_for_state(io, sides, STATE_WAIT_S)
    session = TeachReplaySession(io, gravity_fn=gravity_fn, arm_lower=lower, arm_upper=upper,
                                 arm_effort=effort, config=TeachReplayConfig(hand_sides=sides),
                                 teach_dir=teach_dir, on_message=lambda m: printer(f"[teach] {m}"))
    holder["session"] = session
    state_line = "arm=" + ("ok" if io.arm_state() is not None else "MISSING")
    for side in sides:
        state_line += f" {side}=" + ("ok" if io.hand_state(side) is not None else "MISSING")
    printer(f"[teach] passive (no command publisher). state: {state_line}. Press r, p or q.")

    session_id = secrets.token_urlsafe(18)
    panel = None
    if args.web is not None:
        panel = start_web_panel(session, io, sides, teach_dir, args.web, session_id=session_id,
                                print_fn=lambda m: print(m, flush=True))
    write_session_state(DEFAULT_STATE_PATH, pid=os.getpid(), proc_start_time=proc_start_time(os.getpid()),
                        argv=current_process_argv(), port=panel.port if panel is not None else 0,
                        token=panel.token if panel is not None else "disabled",
                        full_url=getattr(panel, "canonical_url", "") if panel is not None else "",
                        web_enabled=panel is not None, session_id=session_id)

    from sshkeyboard import listen_keyboard, stop_listening
    keyboard = threading.Thread(
        target=listen_keyboard,
        kwargs={"on_press": make_key_handler(session, args.replay), "until": None, "sequential": False},
        daemon=True)
    keyboard.start()
    try:
        run_control_loop(session, period_s=1.0 / args.control_hz)
    finally:
        session.request_quit()
        bounded = 0
        while not session.done and bounded < int(5 * args.control_hz):
            session.tick()
            time.sleep(1.0 / args.control_hz)
            bounded += 1
        if not session.done:
            try:
                io.release()
            except Exception:
                pass
        for result in session.wait_for_recordings(timeout=5.0):
            print(f"[teach] recording {result.status}: {result.path} "
                  f"samples={result.samples} dropped={result.dropped} error={result.error}", flush=True)
        if panel is not None:
            try:
                panel.stop()
            except Exception:
                pass
        remove_session_state(DEFAULT_STATE_PATH, session_id)
        try:
            stop_listening()
        except Exception:
            pass
        printer.close()
        print("[teach] exited.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
