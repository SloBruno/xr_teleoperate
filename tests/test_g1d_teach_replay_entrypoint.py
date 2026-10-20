"""Teach/replay entrypoint + launcher: key mapping, stop precedence, isolation."""
import ast
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
ENTRY = ROOT / "teleop" / "teach_replay_g1d.py"
LAUNCHER = ROOT / "teleop" / "run_g1d_teach_replay.sh"
G1D_TELEOP_LAUNCHER = ROOT / "teleop" / "run_g1d_quest_dex3.sh"


def _entry():
    import importlib
    return importlib.import_module("teleop.teach_replay_g1d")


class FakeSession:
    def __init__(self):
        self.calls = []
        self.done = False
        self.ticks = 0

    def request_teach(self):
        self.calls.append("r"); return True

    def request_save(self):
        self.calls.append("s"); return True

    def request_replay(self, name):
        self.calls.append(("p", name)); return True

    def request_quit(self):
        self.calls.append("q"); return True

    @property
    def quit_requested(self):
        return "q" in self.calls

    def tick(self):
        self.ticks += 1
        if "q" in self.calls and self.ticks > 3:
            self.done = True


def test_entrypoint_imports_without_dds_and_never_imports_the_teleop_entrypoint():
    module = _entry()
    assert callable(module.main)
    tree = ast.parse(ENTRY.read_text())
    mods = {getattr(n, "module", None) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and "teleop_hand_and_arm" in m for m in mods)
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("unitree_sdk2py" in (getattr(n, "module", "") or "") for n in top_level)


def test_key_mapping_routes_r_s_p_q_and_ignores_others():
    module = _entry()
    session = FakeSession()
    handler = module.make_key_handler(session, replay_name="demo")
    for key in ("r", "s", "p", "x", "q"):
        handler(key)
    assert session.calls == ["r", "s", ("p", "demo"), "q"]


def test_parser_defaults_require_both_hands_and_g1d_interface():
    module = _entry()
    args = module.build_parser().parse_args([])
    assert args.hands == "both"
    assert args.network_interface == "enP2p1s0"
    assert module.hand_sides_from_arg("both") == ("left", "right")
    assert module.hand_sides_from_arg("left") == ("left",)
    assert module.hand_sides_from_arg("none") == ()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--hands", "bogus"])


def test_control_loop_turns_ctrl_c_into_quit_and_keeps_ticking_until_done():
    module = _entry()
    session = FakeSession()
    state = {"n": 0}

    def sleep(_dt):
        state["n"] += 1
        if state["n"] == 2:
            raise KeyboardInterrupt

    module.run_control_loop(session, period_s=0.004, sleep=sleep, max_quit_ticks=100)
    assert "q" in session.calls and session.done


def test_control_loop_is_bounded_even_if_session_never_finishes():
    module = _entry()

    class Stuck(FakeSession):
        def tick(self):
            self.ticks += 1

    session = Stuck()
    session.request_quit()
    module.run_control_loop(session, period_s=0.0, sleep=lambda _dt: None, max_quit_ticks=10)
    assert session.ticks <= 12


def test_quit_is_not_blocked_by_a_held_session_lock(tmp_path):
    from teleop.utils import teach_replay as tr
    import numpy as np

    class IO:
        def arm_state(self): return None
        def hand_state(self, side): return None

    session = tr.TeachReplaySession(IO(), gravity_fn=lambda q: np.zeros(14), arm_lower=-np.ones(14),
                                    arm_upper=np.ones(14), arm_effort=np.ones(14) * 25, teach_dir=tmp_path)
    module = _entry()
    handler = module.make_key_handler(session, replay_name=None)
    with session.lock:
        thread = threading.Thread(target=handler, args=("q",))
        thread.start()
        thread.join(1.0)
        assert not thread.is_alive(), "q must not wait for the control tick"
    session.tick()
    assert session.done


def test_message_printer_is_asynchronous():
    module = _entry()
    printed = []
    gate = threading.Event()

    def slow_print(text):
        gate.wait(2.0)
        printed.append(text)

    printer = module.AsyncPrinter(print_fn=slow_print)
    printer("hello")          # must return immediately even while print blocks
    gate.set()
    printer.close(timeout=2.0)
    assert printed == ["hello"]


# ----------------------------------------------------------------- launcher
def test_launcher_is_separate_and_syntax_valid():
    assert LAUNCHER.exists() and LAUNCHER != G1D_TELEOP_LAUNCHER
    result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_launcher_reuses_the_g1d_dds_binding_and_env():
    src = LAUNCHER.read_text()
    teleop = G1D_TELEOP_LAUNCHER.read_text()
    for line in (
        "G1D_DDS_INTERFACE=${G1D_DDS_INTERFACE:-enP2p1s0}",
        "export PYTHONNOUSERSITE=1",
        "TELEIMAGER_PYTHON:-/home/unitree/.local/envs/xr-g1d/bin/python",
    ):
        assert line in src and line in teleop
    assert "printf -v CYCLONEDDS_URI '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"%s\"/>" in src
    assert "export CYCLONEDDS_URI" in src
    assert "teach_replay_g1d.py" in src
    assert '--network-interface "$g1d_dds_interface"' in src
    for forbidden in ("teleop_hand_and_arm.py", "televuer", "--motion", "xr_net_announce"):
        assert forbidden not in src


def test_launcher_dry_run_prints_command_without_starting_python(tmp_path):
    env = dict(os.environ, G1D_TEACH_SKIP_RUN="1", TELEIMAGER_PYTHON="/nonexistent/python")
    result = subprocess.run(["bash", str(LAUNCHER), "--hands", "left"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "teach_replay_g1d.py" in result.stdout
    assert "--network-interface enP2p1s0" in result.stdout
    assert "--hands left" in result.stdout


@pytest.mark.parametrize("arg", ["--network-interface", "--network-interface=eth0"])
def test_launcher_refuses_interface_override_argument(arg):
    env = dict(os.environ, G1D_TEACH_SKIP_RUN="1")
    result = subprocess.run(["bash", str(LAUNCHER), arg], capture_output=True, text=True, env=env)
    assert result.returncode == 2


def test_conflicting_writer_detection_uses_exact_argv(tmp_path):
    module = _entry()

    def proc(pid, *argv):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")

    proc(10, "/env/bin/python", "-s", "teleop_hand_and_arm.py", "--arm", "G1D")
    proc(11, "bash", "-c", "grep teleop_hand_and_arm.py")          # not python
    proc(12, "/env/bin/python3", "teach_replay_g1d.py")             # own pid
    proc(13, "/env/bin/python", "-c", "print('teach_replay_g1d.py')")
    (tmp_path / "self").mkdir()
    found = module.find_conflicting_processes(proc_root=str(tmp_path), own_pid=12)
    assert [pid for pid, _ in found] == [10]


def test_exclusive_session_lock_blocks_a_second_instance(tmp_path):
    module = _entry()
    first = module.acquire_session_lock(tmp_path / "teach.lock")
    assert first is not None
    code = (
        "import sys; sys.path.insert(0, %r); from teleop import teach_replay_g1d as m; "
        "print(m.acquire_session_lock(%r) is None)" % (str(ROOT), str(tmp_path / "teach.lock"))
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert out.stdout.strip() == "True", out.stderr
    first.close()
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert out.stdout.strip() == "False", out.stderr


def test_launcher_state_recovery_returns_only_existing_full_web_url(tmp_path, monkeypatch):
    module = _entry()
    from teleop.utils.teach_web import WebPanel
    class Session:
        session_id = "s1"
    panel = WebPanel(Session(), tmp_path, host="127.0.0.1", port=0, token="secret")
    panel.start()
    state = tmp_path / "session.json"
    try:
        me = os.getpid()
        module.write_session_state(state, pid=me, proc_start_time=module.proc_start_time(me),
                                   argv=module.current_process_argv(), port=panel.port, token="secret",
                                   full_url=panel.url("127.0.0.1"), web_enabled=True, session_id="s1")
        assert module.recover_existing_session(state, proc_root="/proc", own_pid=me) == panel.url("127.0.0.1")
        assert state.exists()
    finally:
        panel.stop()


def test_recovery_requires_live_exact_session_and_token_http_service(tmp_path):
    module = _entry()
    from teleop.utils.teach_web import WebPanel
    class Session:
        session_id = "s-live"
    panel = WebPanel(Session(), tmp_path, host="127.0.0.1", port=0, token="secret")
    panel.start()
    state = tmp_path / "session.json"
    try:
        me = os.getpid(); start = module.proc_start_time(me); argv = module.current_process_argv()
        module.write_session_state(state, pid=me, proc_start_time=start, argv=argv,
                                   port=panel.port, token="secret", full_url=panel.url("127.0.0.1"),
                                   web_enabled=True, session_id="s-live")
        assert module.recover_existing_session(state, proc_root="/proc", own_pid=me) == panel.url("127.0.0.1")
        panel.session_id = "wrong"
        assert module.recover_existing_session(state, proc_root="/proc", own_pid=me) is None
        assert not state.exists()
    finally:
        panel.stop()


def test_launcher_state_recovery_cleans_stale_identity(tmp_path):
    module = _entry()
    state = tmp_path / "session.json"
    module.write_session_state(state, pid=999999, proc_start_time=1, argv=["python", "teach_replay_g1d.py"],
                               port=1234, token="secret", full_url="http://bad", web_enabled=True,
                               session_id="stale")
    assert module.recover_existing_session(state, proc_root="/proc", own_pid=os.getpid()) is None
    assert not state.exists()
