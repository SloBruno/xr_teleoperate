"""Dex3 launcher: G1_TORSO_LEAN* / XR_POSE_WEB options (behavioural, fakes only).

Off (default): the teleop argv/env is the same as before (exec path, no
XR_POSE_STREAM, G1_TORSO_LEAN=0). On: validated values exported and the state
printed; XR_POSE_WEB=1 starts tools/pose_compare_web.py and stops it at exit.
"""
import os
import signal
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).parents[1] / "teleop" / "run_g1_quest_dex3.sh"

FAKE_PY = r'''#!/usr/bin/env bash
echo "ARGV $*" >>"$FAKE_LOG"
case "$*" in
  *--detect*) echo "head 243122072230 both"; exit 0;;
  "-s - "*) pid=$(cat "$TELEIMAGER_STATE_DIR/teleimager.pid" 2>/dev/null) || exit 2
    kill -0 "$pid" 2>/dev/null || exit 2; exit 0;;
  *teleimager_head_only_server*|*teleimager.image_server*) sleep 8 & wait;;
  *pose_compare_web*) echo "WEBSTART $$" >>"$FAKE_LOG"; trap 'echo WEBSTOP >>"$FAKE_LOG"; exit 0' INT; sleep 30 & wait; echo WEBEND >>"$FAKE_LOG";;
  *teleop_hand_and_arm*)
    echo "ENV G1_TORSO_LEAN=${G1_TORSO_LEAN:-unset} MAX=${G1_TORSO_LEAN_MAX_DEG:-unset} XR_POSE_STREAM=${XR_POSE_STREAM:-unset}" >>"$FAKE_LOG"
    exit 0;;
esac
'''


class Dex3LauncherTorsoPoseWebTest(unittest.TestCase):
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
        pg.write_text("#!/usr/bin/env bash\nexit 1\n")
        pg.chmod(0o755)
        self.fake, self.bindir, self.procs = fake, bindir, []
        # A healthy running Teleimager (reused; no restart).
        p = subprocess.Popen(["bash", "-c", "sleep 60; :", "teleimager_fake"], start_new_session=True)
        self.procs.append(p)
        threading.Thread(target=p.wait, daemon=True).start()
        (self.state / "teleimager.pid").write_text(f"{p.pid}\n")
        (self.state / "teleimager.mode").write_text("both\n")
        (self.state / "teleimager.realsense_profile").write_text("usb2\n")

    def tearDown(self):
        for p in self.procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
        subprocess.run(["pkill", "-f", str(self.tmp)], check=False)

    def run_launcher(self, **extra):
        env = dict(os.environ)
        for k in list(env):
            if k.startswith(("G1_TORSO_LEAN", "XR_POSE", "POSE_WEB", "TELEIMAGER_CAMERA_MODE")):
                env.pop(k)
        env.update({
            "TELEIMAGER_STATE_DIR": str(self.state), "TELEIMAGER_PYTHON": str(self.fake),
            "G1_LAUNCHER_SKIP_NET": "1", "FAKE_LOG": str(self.log), "PATH": f"{self.bindir}:{env['PATH']}",
            "TELEIMAGER_TIMEOUT_S": "5", "TELEIMAGER_LOCK_TIMEOUT_S": "2", "POSE_WEB_STOP_TIMEOUT_S": "3",
        })
        env.update(extra)
        r = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=60)
        return r, self.log.read_text()

    def teleop_line(self, log):
        return next(l for l in log.splitlines() if "teleop_hand_and_arm.py" in l)

    def test_default_off_prints_state_and_teleop_args_unchanged(self):
        r, log = self.run_launcher()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Inclinação do tronco: DESLIGADA", r.stdout)
        self.assertIn("Página de comparação: DESLIGADA", r.stdout)
        self.assertIn("ENV G1_TORSO_LEAN=0 MAX=unset XR_POSE_STREAM=unset", log)
        self.assertNotIn("pose_compare_web", log)
        self.assertEqual(
            self.teleop_line(log),
            "ARGV -s teleop_hand_and_arm.py --arm G1_29 --ee dex3 --input-mode hand --motion "
            "--camera-layout vertical --walk-speed-cap 0.3 --turn-rate-cap 0.3 --loco-backend wirelesscontroller "
            "--loco-request-fsm none --video-plane-height auto")

    def test_launcher_installs_no_shell_trap(self):
        # The persistent Teleimager must never be stopped by the launcher.
        self.assertNotIn("trap", LAUNCHER.read_text(encoding="utf-8"))

    def test_launcher_saves_terminal_output_without_redirecting_stdin(self):
        r, _ = self.run_launcher()
        self.assertEqual(r.returncode, 0, r.stderr)
        logs = list(self.state.glob("launcher-*.log"))
        self.assertEqual(len(logs), 1)
        saved = logs[0].read_text(encoding="utf-8")
        self.assertIn("Inclinação do tronco: DESLIGADA", saved)
        launcher = LAUNCHER.read_text(encoding="utf-8")
        log_block = launcher[launcher.index("launcher_log="):launcher.index("# ---- pose compare web")]
        self.assertIn("exec > >(", log_block)
        self.assertNotIn("exec </dev/null", log_block)

    def test_lean_on_exports_and_prints(self):
        r, log = self.run_launcher(G1_TORSO_LEAN="1", G1_TORSO_LEAN_MAX_DEG="3")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Inclinação do tronco: LIGADA, máx 3° pitch/roll (yaw fixo)", r.stdout)
        self.assertIn("ENV G1_TORSO_LEAN=1 MAX=3", log)

    def test_lean_max_above_10_or_bad_value_refused_before_anything_starts(self):
        for env in ({"G1_TORSO_LEAN": "1", "G1_TORSO_LEAN_MAX_DEG": "15"},
                    {"G1_TORSO_LEAN": "1", "G1_TORSO_LEAN_MAX_DEG": "abc"},
                    {"G1_TORSO_LEAN": "yes"}, {"XR_POSE_WEB": "2"}):
            self.log.write_text("")
            r, log = self.run_launcher(**env)
            self.assertEqual(r.returncode, 2, (env, r.stderr))
            self.assertNotIn("teleop_hand_and_arm", log)

    def test_pose_web_on_starts_web_exports_stream_and_stops_it(self):
        r, log = self.run_launcher(XR_POSE_WEB="1", G1_TORSO_LEAN="1", G1_TORSO_LEAN_MAX_DEG="3")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Página de comparação (XR_POSE_WEB=1): LIGADA, porta 8093", r.stdout)
        self.assertIn("tools/pose_compare_web.py --port 8093 --exit-with-pid", log)
        self.assertIn("XR_POSE_STREAM=1", log)
        self.assertIn("POSE WEB: parado.", r.stderr)
        self.assertIn("WEBSTOP", log)


if __name__ == "__main__":
    unittest.main()
