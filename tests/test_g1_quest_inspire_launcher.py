"""Static + fake-run checks of the Inspire launcher (no robot, no DDS)."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).parents[1] / "teleop" / "run_g1_quest_inspire.sh"

FAKE_PY = r'''#!/usr/bin/env bash
echo "$*" >>"$FAKE_LOG"
case "$*" in
  *--detect*) echo "head 243122072230 both"; exit 0;;
  "-s - "*) exit 0;;
  *teleop_hand_and_arm*) echo "XR_POSE_STREAM=${XR_POSE_STREAM:-unset}" >>"$FAKE_LOG"; echo "LEAN=${G1_TORSO_LEAN:-unset} MAX=${G1_TORSO_LEAN_MAX_DEG:-unset} RATE=${G1_TORSO_LEAN_RATE_DPS:-unset}" >>"$FAKE_LOG"; exit 0;;
  *pose_compare_web.py*) echo "web-started $$" >>"$FAKE_LOG"; trap 'echo web-stopped >>"$FAKE_LOG"; exit 0' INT TERM; while :; do sleep 0.1; done;;
esac
'''


class InspireLauncherTest(unittest.TestCase):
    def setUp(self):
        self.src = LAUNCHER.read_text(encoding="utf-8")

    def test_pins_cyclonedds_and_hand_tracking(self):
        self.assertIn('NetworkInterface name="enP8p1s0"', self.src)
        self.assertIn("--input-mode hand", self.src)
        self.assertIn('--ee "$G1_EE"', self.src)
        self.assertIn("G1_EE=${G1_EE:-inspire_dfx}", self.src)

    def test_no_dex3_no_quest_locomotion_flags(self):
        for bad in ("--ee dex3", "--walk-speed-cap", "--turn-rate-cap", "--loco-backend",
                    "--loco-request-fsm", "--arm-pose-source", "controller"):
            self.assertNotIn(bad, self.src.split("# for local terminal r/q.")[1], bad)

    def test_usb2_and_auto_plane_defaults(self):
        self.assertIn("XR_REALSENSE_PROFILE=${XR_REALSENSE_PROFILE:-usb2}", self.src)
        self.assertIn("XR_VIDEO_PLANE_HEIGHT=${XR_VIDEO_PLANE_HEIGHT:-auto}", self.src)

    def _run(self, **extra):
        tmp = Path(tempfile.mkdtemp())
        state = tmp / "state"
        state.mkdir()
        log = tmp / "log"
        log.write_text("")
        fake = tmp / "fakepy"
        fake.write_text(FAKE_PY)
        fake.chmod(0o755)
        bindir = tmp / "bin"
        bindir.mkdir()
        (bindir / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n")
        (bindir / "pgrep").chmod(0o755)
        env = dict(os.environ, TELEIMAGER_STATE_DIR=str(state), TELEIMAGER_PYTHON=str(fake),
                   G1_LAUNCHER_SKIP_NET="1", FAKE_LOG=str(log), PATH=f"{bindir}:{os.environ['PATH']}",
                   TELEIMAGER_TIMEOUT_S="5", TELEIMAGER_LOCK_TIMEOUT_S="2", INSPIRE_DRIVER="skip")
        env.update(extra)
        r = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=60)
        subprocess.run(["pkill", "-f", str(tmp)], check=False)
        return r, log.read_text()

    def test_fake_run_passes_inspire_args(self):
        r, log = self._run()
        self.assertEqual(r.returncode, 0, r.stderr)
        line = [l for l in log.splitlines() if "teleop_hand_and_arm.py" in l][-1]
        for want in ("--arm G1_29", "--ee inspire_dfx", "--input-mode hand", "--motion",
                     "--camera-layout vertical", "--video-plane-height auto"):
            self.assertIn(want, line)
        self.assertNotIn("dex3", line)
        self.assertNotIn("loco", line)

    def test_motion_off_and_ftp(self):
        r, log = self._run(G1_MOTION="0", G1_EE="inspire_ftp")
        self.assertEqual(r.returncode, 0, r.stderr)
        line = [l for l in log.splitlines() if "teleop_hand_and_arm.py" in l][-1]
        self.assertIn("--ee inspire_ftp", line)
        self.assertNotIn("--motion", line)

    def test_rejects_dex3(self):
        r, _ = self._run(G1_EE="dex3")
        self.assertEqual(r.returncode, 2)

    def test_pose_web_off_by_default(self):
        r, log = self._run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("XR_POSE_STREAM=unset", log)
        self.assertNotIn("pose_compare_web", log)

    def test_pose_web_on_starts_and_stops_child(self):
        r, log = self._run(XR_POSE_WEB="1", POSE_WEB_STOP_TIMEOUT_S="3")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("XR_POSE_STREAM=1", log)
        self.assertIn("web-started", log)
        self.assertIn("web-stopped", log)
        self.assertIn("8093", r.stdout + r.stderr)
        lines = log.splitlines()
        self.assertLess(lines.index(next(l for l in lines if "web-started" in l)),
                        lines.index(next(l for l in lines if "teleop_hand_and_arm" in l)))

    def _clean_lean_env(self):
        return {k: "" for k in os.environ if k.startswith("G1_TORSO_LEAN")}

    def test_torso_lean_off_by_default(self):
        r, log = self._run(**self._clean_lean_env())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Inclinação do tronco: DESLIGADA", r.stdout)
        self.assertIn("LEAN=0 ", log)

    def test_torso_lean_on_passes_env_and_prints_state(self):
        env = self._clean_lean_env()
        env.update(G1_TORSO_LEAN="1", G1_TORSO_LEAN_MAX_DEG="3", G1_TORSO_LEAN_RATE_DPS="10")
        r, log = self._run(**env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Inclinação do tronco: LIGADA, máx 3°", r.stdout)
        self.assertIn("LEAN=1 MAX=3 RATE=10", log)

    def test_torso_lean_max_above_10_rejected_before_anything_starts(self):
        for bad in ("10.5", "20", "0", "abc"):
            env = self._clean_lean_env()
            env.update(G1_TORSO_LEAN="1", G1_TORSO_LEAN_MAX_DEG=bad)
            r, log = self._run(**env)
            self.assertEqual(r.returncode, 2, bad)
            self.assertIn("rejeitado", r.stderr)
            self.assertNotIn("teleop_hand_and_arm", log)
            self.assertNotIn("--detect", log)

    def test_torso_lean_requires_motion(self):
        env = self._clean_lean_env()
        env.update(G1_TORSO_LEAN="1", G1_MOTION="0")
        r, _ = self._run(**env)
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
