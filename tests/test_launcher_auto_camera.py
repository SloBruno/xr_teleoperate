"""Behavioural tests of the launcher's automatic camera-mode detection (fakes only)."""
import os
import signal
import stat
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).parents[1] / "teleop" / "run_g1_quest_dex3.sh"

FAKE_PY = r'''#!/usr/bin/env bash
echo "$*" >>"$FAKE_LOG"
case "$*" in
  *--detect*)
    [[ -n "${FAKE_DETECT_SLEEP:-}" ]] && sleep "$FAKE_DETECT_SLEEP"
    echo "${FAKE_DETECT_OUT:-}"; exit "${FAKE_DETECT_RC:-0}";;
  "-s - "*) echo "PROBE $*" >>"$FAKE_LOG"
    pid=$(cat "$TELEIMAGER_STATE_DIR/teleimager.pid" 2>/dev/null) || exit 2
    kill -0 "$pid" 2>/dev/null || exit 2; exit 0;;
  *teleimager_head_only_server*|*teleimager.image_server*) sleep 8 & wait;;
  *teleop_hand_and_arm*) exit 0;;
esac
'''


class AutoCameraTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.state = self.tmp / "state"
        self.state.mkdir()
        self.log = self.tmp / "log"
        self.log.write_text("")
        fake = self.tmp / "fakepy"
        fake.write_text(FAKE_PY)
        fake.chmod(0o755)
        bindir = self.tmp / "bin"
        bindir.mkdir()
        pg = bindir / "pgrep"
        pg.write_text('#!/usr/bin/env bash\n[[ "${FAKE_TELEOP:-0}" == 1 ]]\n')
        pg.chmod(0o755)
        self.fake = fake
        self.bindir = bindir
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
        subprocess.run(["pkill", "-f", str(self.tmp)], check=False)

    def old_server(self, mode, source=None):
        p = subprocess.Popen(["bash", "-c", "sleep 30; :", "teleimager_fake"], start_new_session=True)
        self.procs.append(p)
        threading.Thread(target=p.wait, daemon=True).start()  # reap so kill -0 sees it die
        (self.state / "teleimager.pid").write_text(f"{p.pid}\n")
        (self.state / "teleimager.mode").write_text(mode + "\n")
        if source:
            (self.state / "teleimager.source").write_text(source + "\n")
        return p

    def run_launcher(self, detect=None, rc=0, skip=True, **extra):
        env = dict(os.environ)
        env.pop("TELEIMAGER_CAMERA_MODE", None)
        env.update({
            "TELEIMAGER_STATE_DIR": str(self.state), "TELEIMAGER_PYTHON": str(self.fake),
            "FAKE_LOG": str(self.log), "PATH": f"{self.bindir}:{env['PATH']}",
            "TELEIMAGER_TIMEOUT_S": "5", "TELEIMAGER_LOCK_TIMEOUT_S": "2",
            "FAKE_DETECT_RC": str(rc),
        })
        if detect is not None:
            env["FAKE_DETECT_OUT"] = detect
        if skip:
            env["G1_LAUNCHER_SKIP_TELEOP"] = "1"
        env.update(extra)
        r = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=60)
        return r, self.log.read_text()

    def mode(self):
        return (self.state / "teleimager.mode").read_text().strip()

    def test_two_cameras_auto_selects_both_vertical(self):
        r, log = self.run_launcher("head 243122072230 both", skip=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.mode(), "both")
        self.assertIn("DUAS CÂMERAS", r.stdout)
        self.assertIn("--camera-layout vertical", log)
        self.assertIn("teleimager.image_server", log)
        self.assertNotIn("CÂMERA ÚNICA", r.stdout)

    def test_two_cameras_probe_requires_both_mode(self):
        r, log = self.run_launcher("head 243122072230 both")
        self.assertIn("PROBE -s - 127.0.0.1 both", log)

    def test_one_camera_head_selects_any_head_layout(self):
        r, log = self.run_launcher("head 243122072230 single", skip=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.mode(), "any")
        self.assertIn("CÂMERA ÚNICA (cabeça", r.stdout)
        self.assertIn("--camera-layout head", log)
        self.assertIn("PROBE -s - 127.0.0.1 any", log)
        self.assertNotIn("AVISO", r.stdout)

    def test_one_camera_wrist_warns(self):
        r, log = self.run_launcher("left_wrist 233622070789 single")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("CÂMERA ÚNICA (pulso esquerdo", r.stdout)
        self.assertIn("PULSO", r.stdout)
        self.assertEqual((self.state / "teleimager.source").read_text().strip(), "left_wrist 233622070789")

    def test_zero_cameras_fails_clearly_without_starting(self):
        r, log = self.run_launcher("", rc=3)
        self.assertEqual(r.returncode, 3)
        self.assertIn("nenhuma câmera", r.stderr)
        self.assertNotIn("teleimager_head_only_server --rs", log)
        self.assertNotIn("image_server", log)
        self.assertFalse((self.state / "teleimager.pid").exists())

    def test_detection_error_fails_clearly(self):
        r, log = self.run_launcher("", rc=4)
        self.assertEqual(r.returncode, 4)
        self.assertIn("falha ao detectar", r.stderr)
        self.assertFalse((self.state / "teleimager.pid").exists())

    def test_detection_timeout_fails_clearly(self):
        r, log = self.run_launcher("head 243122072230 both", TELEIMAGER_DETECT_TIMEOUT_S="1", FAKE_DETECT_SLEEP="5")
        self.assertEqual(r.returncode, 4)
        self.assertIn("falha ao detectar", r.stderr)

    def test_garbage_detection_output_fails(self):
        r, _ = self.run_launcher("")
        self.assertEqual(r.returncode, 4)

    def test_reuses_healthy_same_mode_server_without_restart(self):
        p = self.old_server("any", "left_wrist 233622070789")
        r, log = self.run_launcher("left_wrist 233622070789 single")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Reusing healthy", r.stdout)
        self.assertIsNone(p.poll())
        self.assertNotIn("--rs", log)

    def test_reuses_legacy_server_without_source_file(self):
        p = self.old_server("any")
        r, _ = self.run_launcher("left_wrist 233622070789 single")
        self.assertIn("Reusing healthy", r.stdout)
        self.assertIsNone(p.poll())

    def test_restarts_when_camera_count_changed_and_no_teleop(self):
        p = self.old_server("any", "left_wrist 233622070789")
        r, log = self.run_launcher("head 243122072230 both")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("SIGTERM", r.stdout)
        p.wait(timeout=5)
        self.assertEqual(self.mode(), "both")
        self.assertIn("--rs", log)
        new = int((self.state / "teleimager.pid").read_text())
        self.assertNotEqual(new, p.pid)

    def test_restarts_when_single_camera_source_changed(self):
        p = self.old_server("any", "left_wrist 233622070789")
        r, _ = self.run_launcher("head 243122072230 single")
        self.assertEqual(r.returncode, 0, r.stderr)
        p.wait(timeout=5)
        self.assertEqual((self.state / "teleimager.source").read_text().strip(), "head 243122072230")

    def test_refuses_restart_when_teleop_running(self):
        p = self.old_server("any", "left_wrist 233622070789")
        r, log = self.run_launcher("head 243122072230 both", FAKE_TELEOP="1")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("teleop_hand_and_arm.py em execução", r.stderr)
        self.assertIsNone(p.poll())
        self.assertEqual(self.mode(), "any")
        self.assertNotIn("--rs", log)

    def test_refuses_to_kill_pid_that_is_not_teleimager(self):
        p = subprocess.Popen(["sleep", "30"], start_new_session=True)
        self.procs.append(p)
        (self.state / "teleimager.pid").write_text(f"{p.pid}\n")
        (self.state / "teleimager.mode").write_text("any\n")
        r, _ = self.run_launcher("head 243122072230 both")
        self.assertNotEqual(r.returncode, 0)
        self.assertIsNone(p.poll())

    def test_explicit_override_both_skips_detection(self):
        r, log = self.run_launcher("", rc=3, TELEIMAGER_CAMERA_MODE="both")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("--detect", log)
        self.assertEqual(self.mode(), "both")

    def test_explicit_head_override(self):
        r, log = self.run_launcher("", rc=3, TELEIMAGER_CAMERA_MODE="head", skip=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--camera-layout head", log)
        self.assertNotIn("--detect", log)

    def test_explicit_any_with_two_uses_head_with_warning(self):
        r, _ = self.run_launcher("head 243122072230 both", TELEIMAGER_CAMERA_MODE="any")
        self.assertIn("usa apenas a cabeça", r.stdout)
        self.assertEqual(self.mode(), "any")

    def test_lock_fd_not_leaked_to_children(self):
        self.run_launcher("head 243122072230 single")
        time.sleep(0.2)
        out = subprocess.run(["fuser", str(self.state / "teleimager.lock")], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
