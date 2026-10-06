"""Inspire DFQ RS-485 wrapper + launcher driver management (fakes only: no serial,
no DDS, no robot, no hand command)."""
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
WRAPPER = ROOT / "teleop" / "robot_control" / "inspire_dfq_485_driver.py"
LAUNCHER = ROOT / "teleop" / "run_g1_quest_inspire.sh"


def load_wrapper():
    spec = importlib.util.spec_from_file_location("inspire_dfq_485_driver_t", WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


drv = load_wrapper()


class FakeRsp:
    def __init__(self, regs, err=False):
        self.registers = regs
        self._err = err

    def isError(self):
        return self._err


class RecordingClient:
    """Fake Modbus client: records every call; any write is a test failure."""

    def __init__(self, calls, port, baudrate):
        self.calls = calls
        self.calls.append(("init", port, baudrate))

    def connect(self):
        self.calls.append(("connect",))
        return True

    def read_holding_registers(self, addr, count, dev_id):
        self.calls.append(("read", addr, count, dev_id))
        return FakeRsp([0, 1000, 0xFFFF, 3, 4, 5])

    def close(self):
        self.calls.append(("close",))

    def __getattr__(self, name):
        if name.startswith("write"):
            def _w(*a, **k):
                self.calls.append((name,) + a)
            return _w
        raise AttributeError(name)


class WrapperArgsTest(unittest.TestCase):
    def test_defaults_match_sdk_example(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("INSPIRE_")}
        old = os.environ.copy()
        try:
            os.environ.clear(); os.environ.update(env)
            a = drv.build_parser().parse_args([])
        finally:
            os.environ.clear(); os.environ.update(old)
        self.assertEqual((a.left_port, a.right_port, a.baudrate, a.left_id, a.right_id, a.iface),
                         ("/dev/ttyUSB1", "/dev/ttyUSB2", 115200, 1, 1, "enP8p1s0"))

    def test_env_and_args(self):
        old = os.environ.copy()
        try:
            os.environ.update(INSPIRE_LEFT_PORT="/dev/serial/by-id/usb-L", INSPIRE_RIGHT_PORT="/dev/serial/by-path/R",
                              INSPIRE_BAUDRATE="57600", INSPIRE_LEFT_ID="2", INSPIRE_RIGHT_ID="3",
                              INSPIRE_DDS_IFACE="eth9")
            a = drv.build_parser().parse_args([])
            self.assertEqual((a.left_port, a.right_port, a.baudrate, a.left_id, a.right_id, a.iface),
                             ("/dev/serial/by-id/usb-L", "/dev/serial/by-path/R", 57600, 2, 3, "eth9"))
            a = drv.build_parser().parse_args(["--left-port", "/x", "--iface", "lo"])
            self.assertEqual((a.left_port, a.iface), ("/x", "lo"))
        finally:
            os.environ.clear(); os.environ.update(old)

    def test_missing_port_clear_error(self):
        with self.assertRaises(drv.PortError) as cm:
            drv.check_port("/dev/serial/by-id/nao-existe", "porta ESQUERDA")
        self.assertIn("não existe", str(cm.exception))
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaises(drv.PortError):
                drv.check_port(f.name)  # regular file, not a tty
        self.assertEqual(drv.check_port("/dev/null"), "/dev/null")

    def test_cli_missing_port_exit3_without_sdk(self):
        r = subprocess.run([sys.executable, str(WRAPPER), "--left-port", "/dev/nao_ttyUSB1",
                            "--right-port", "/dev/nao_ttyUSB2"], capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, PYTHONPATH="/nonexistent"))
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("não existe", r.stderr)

    def test_help(self):
        r = subprocess.run([sys.executable, str(WRAPPER), "--help"], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0)
        self.assertIn("--probe", r.stdout)


class ProbeTest(unittest.TestCase):
    def test_probe_reads_only(self):
        calls = []
        args = drv.build_parser().parse_args(["--probe", "--left-port", "/dev/null", "--right-port", "/dev/zero",
                                              "--left-id", "1", "--right-id", "2", "/dev/nao_existe"])
        lines = []
        rc = drv.run_probe(args, client_factory=lambda p, b: RecordingClient(calls, p, b), out=lines.append)
        self.assertEqual(rc, 0)
        self.assertFalse([c for c in calls if c[0].startswith("write")], calls)
        reads = [c for c in calls if c[0] == "read"]
        self.assertEqual(reads, [("read", 1546, 6, 1), ("read", 1546, 6, 2)] * 2)
        out = "\n".join(lines)
        self.assertIn("[0, 1000, -1, 3, 4, 5]", out)
        self.assertIn("/dev/nao_existe", out)
        self.assertIn("não existe", out)

    def test_probe_no_port_rc3(self):
        args = drv.build_parser().parse_args(["--probe", "--left-port", "/dev/nx1", "--right-port", "/dev/nx2"])
        self.assertEqual(drv.run_probe(args, client_factory=None, out=lambda *_: None), 3)


class FakeHandler:
    instances = []

    def __init__(self, **kw):
        self.kw = kw
        self.reads = 0
        self.client = SimpleNamespace(closed=False)
        self.client.close = lambda: setattr(self.client, "closed", True)
        FakeHandler.instances.append(self)

    def read(self):
        self.reads += 1
        return {"states": {"ANGLE_ACT": [0] * 6}}


class DriverLoopTest(unittest.TestCase):
    def test_driver_uses_config_and_stops(self):
        FakeHandler.instances = []
        stop = drv.StopFlag()
        init = []
        args = drv.build_parser().parse_args(["--left-port", "/dev/null", "--right-port", "/dev/zero",
                                              "--baudrate", "57600", "--right-id", "2", "--iface", "eth7"])
        n = {"t": 0}

        def clock():
            n["t"] += 1
            if n["t"] > 20:
                stop.handler(2, None)
            return n["t"] * 0.01
        rc = drv.run_driver(args, stop, sdk=SimpleNamespace(ModbusDataHandler=FakeHandler),
                            channel_init=lambda *a: init.append(a), out=lambda *_: None, clock=clock)
        self.assertEqual(rc, 0)
        self.assertEqual(init, [(0, "eth7")])
        l, r = FakeHandler.instances
        self.assertEqual((l.kw["LR"], l.kw["serial_port"], l.kw["device_id"], l.kw["baudrate"], l.kw["initDDS"]),
                         ("l", "/dev/null", 1, 57600, False))
        self.assertEqual((r.kw["LR"], r.kw["serial_port"], r.kw["device_id"]), ("r", "/dev/zero", 2))
        self.assertEqual(l.kw["states_structure"], [("angle_act", 1546, 6, "short")])
        self.assertTrue(l.client.closed and r.client.closed)
        self.assertGreater(l.reads, 0)

    def test_same_device_rejected(self):
        args = drv.build_parser().parse_args(["--left-port", "/dev/null", "--right-port", "/dev/null"])
        with self.assertRaises(drv.PortError):
            drv.run_driver(args, drv.StopFlag(), sdk=SimpleNamespace(ModbusDataHandler=FakeHandler),
                           channel_init=lambda *a: None, out=lambda *_: None)


# ------------------------------------------------------------------ launcher
FAKE_PY = r'''#!/usr/bin/env bash
echo "$*" >>"$FAKE_LOG"
case "$*" in
  *--detect*) echo "head 243122072230 both"; exit 0;;
  "-s - "*) exit 0;;
  *teleop_hand_and_arm*) exit "${FAKE_TELEOP_RC:-0}";;
  *--health-check*) [[ "${FAKE_HEALTH:-ok}" == ok ]] && exit 0; exit 4;;
  *inspire_dfq_485_driver.py*) exec python3 "$FAKE_DRV" "$@";;
esac
'''

FAKE_DRV = r'''import os, signal, sys, time
log = os.environ["FAKE_LOG"]
def h(sig, _f):
    open(log, "a").write(f"driver-stopped-{sig}\n"); sys.exit(0)
signal.signal(signal.SIGINT, h)
signal.signal(signal.SIGTERM, h)
open(log, "a").write("driver-running\n")
while True:
    time.sleep(0.05)
'''


class LauncherDriverTest(unittest.TestCase):
    def _run(self, args=(), **extra):
        tmp = Path(tempfile.mkdtemp())
        self.tmp = tmp
        state = tmp / "state"; state.mkdir()
        log = tmp / "log"; log.write_text("")
        fake = tmp / "fakepy"; fake.write_text(FAKE_PY); fake.chmod(0o755)
        (tmp / "fakedrv.py").write_text(FAKE_DRV)
        bindir = tmp / "bin"; bindir.mkdir()
        (bindir / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n"); (bindir / "pgrep").chmod(0o755)
        proc = tmp / "proc"; proc.mkdir()
        env = dict(os.environ, TELEIMAGER_STATE_DIR=str(state), TELEIMAGER_PYTHON=str(fake),
                   G1_LAUNCHER_SKIP_NET="1", FAKE_LOG=str(log), FAKE_DRV=str(tmp / "fakedrv.py"),
                   PATH=f"{bindir}:{os.environ['PATH']}", TELEIMAGER_TIMEOUT_S="5",
                   TELEIMAGER_LOCK_TIMEOUT_S="2", INSPIRE_STATE_DIR=str(tmp / "istate"),
                   INSPIRE_PROC_ROOT=str(proc), INSPIRE_LEFT_PORT="/dev/null", INSPIRE_RIGHT_PORT="/dev/zero",
                   INSPIRE_DRIVER_TIMEOUT_S="3", INSPIRE_SDK_DIR=str(tmp / "nosdk"))
        env.pop("INSPIRE_DRIVER", None)
        env.update(extra)
        r = subprocess.run(["bash", str(LAUNCHER), *args], env=env, capture_output=True, text=True, timeout=90)
        time.sleep(0.3)
        leftover = subprocess.run(["pgrep", "-f", str(tmp / "fakedrv.py")], capture_output=True, text=True).stdout
        subprocess.run(["pkill", "-f", str(tmp)], check=False)
        return r, log.read_text(), leftover.strip()

    def test_missing_ports_abort_without_start(self):
        r, log, left = self._run(INSPIRE_LEFT_PORT="/dev/nao_ttyUSB1", INSPIRE_RIGHT_PORT="/dev/nao_ttyUSB2")
        self.assertEqual(r.returncode, 6, r.stderr)
        self.assertIn("adaptador RS-485 da Inspire não encontrado", r.stderr)
        self.assertIn("/dev/serial/by-id", r.stderr)
        self.assertNotIn("inspire_dfq_485_driver", log)
        self.assertNotIn("teleop_hand_and_arm", log)
        self.assertEqual(left, "")

    def test_preflight_readonly(self):
        r, log, _ = self._run(args=["--inspire-preflight"], INSPIRE_LEFT_PORT="/dev/nao1")
        self.assertEqual(r.returncode, 6)
        self.assertIn("não encontrado", r.stderr)
        self.assertEqual(log, "")
        r, log, _ = self._run(args=["--inspire-preflight"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("nada foi iniciado", r.stdout)
        self.assertIn("AVISO", r.stderr)
        self.assertEqual(log, "")

    def test_started_driver_stopped_on_exit(self):
        r, log, left = self._run()
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        lines = log.splitlines()
        drv_line = [l for l in lines if "inspire_dfq_485_driver.py" in l and "--health-check" not in l][0]
        for want in ("--left-port /dev/null", "--right-port /dev/zero", "--baudrate 115200", "--iface enP8p1s0"):
            self.assertIn(want, drv_line)
        self.assertIn("--iface enP8p1s0", [l for l in lines if "--health-check" in l][0])
        i_tel = next(i for i, l in enumerate(lines) if "teleop_hand_and_arm" in l)
        i_stop = next(i for i, l in enumerate(lines) if l.startswith("driver-stopped"))
        self.assertLess(i_tel, i_stop)
        self.assertEqual(lines[i_stop], "driver-stopped-2")  # SIGINT first
        self.assertEqual(left, "")
        self.assertFalse((self.tmp / "istate" / "inspire_driver.pid").exists())

    def test_teleop_error_still_stops_driver(self):
        r, log, left = self._run(FAKE_TELEOP_RC="3")
        self.assertEqual(r.returncode, 3)
        self.assertIn("driver-stopped-2", log)
        self.assertEqual(left, "")

    def test_health_timeout_aborts_and_stops(self):
        r, log, left = self._run(FAKE_HEALTH="fail")
        self.assertEqual(r.returncode, 6, r.stderr)
        self.assertIn("abortando", r.stderr)
        self.assertIn("últimas linhas", r.stderr)
        self.assertIn("driver-stopped-2", log)
        self.assertNotIn("teleop_hand_and_arm", log)
        self.assertEqual(left, "")

    def test_existing_driver_reused_not_stopped(self):
        holder = subprocess.Popen(["sleep", "30"])
        try:
            proc = Path(tempfile.mkdtemp())
            (proc / str(holder.pid)).mkdir()
            (proc / str(holder.pid) / "cmdline").write_bytes(
                b"/home/unitree/miniconda3/envs/tv/bin/python\0Headless_driver_485_double.py\0")
            (proc / "999").mkdir()  # a probe is not a driver
            (proc / "999" / "cmdline").write_bytes(b"python\0inspire_dfq_485_driver.py\0--probe\0")
            r, log, _ = self._run(INSPIRE_PROC_ROOT=str(proc), INSPIRE_LEFT_PORT="/dev/nao1")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"reaproveitando driver já em execução (PID {holder.pid})", r.stdout)
            self.assertNotIn("999", r.stdout)
            self.assertIn("--health-check", log)
            self.assertNotIn("driver-running", log)
            self.assertIsNone(holder.poll())
        finally:
            holder.kill(); holder.wait()

    def test_skip_and_external(self):
        r, log, _ = self._run(INSPIRE_DRIVER="skip", INSPIRE_LEFT_PORT="/dev/nao1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("inspire_dfq_485_driver", log)
        r, log, _ = self._run(INSPIRE_DRIVER="external", INSPIRE_LEFT_PORT="/dev/nao1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--health-check", log)
        self.assertNotIn("driver-running", log)
        r, log, _ = self._run(INSPIRE_DRIVER="external", FAKE_HEALTH="fail")
        self.assertEqual(r.returncode, 6)
        self.assertNotIn("teleop_hand_and_arm", log)
        r, _, _ = self._run(INSPIRE_DRIVER="bogus")
        self.assertEqual(r.returncode, 2)

    def test_lock_fd_not_leaked_to_driver(self):
        r, log, _ = self._run(FAKE_TELEOP_RC="0")
        self.assertEqual(r.returncode, 0)
        # Re-run acquires the lock immediately (it would block 10 s if leaked).
        t = time.monotonic()
        r, _, _ = self._run()
        self.assertEqual(r.returncode, 0)
        self.assertLess(time.monotonic() - t, 30)


if __name__ == "__main__":
    unittest.main()
