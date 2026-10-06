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
  *teleop_hand_and_arm*) exit 0;;
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


if __name__ == "__main__":
    unittest.main()
