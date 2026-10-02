from pathlib import Path
import subprocess
import unittest


LAUNCHER = Path(__file__).parents[1] / "teleop" / "run_g1_quest_dex3.sh"


class G1QuestDex3LauncherTest(unittest.TestCase):
    def test_pins_cyclonedds_to_the_g1_internal_bus_interface(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("CYCLONEDDS_URI", source)
        self.assertIn('NetworkInterface name="enP8p1s0"', source)

    def test_hybrid_mode_keeps_dex3_available_while_arms_use_controllers(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("--input-mode hand", source)
        self.assertIn("--ee dex3", source)

    def test_uses_launcher_checkout_for_teleimager_and_starts_before_teleop(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)', source)
        self.assertIn('teleimager_dir="$repo/teleop/teleimager"', source)
        self.assertNotIn("/home/unitree/xr_teleoperate_slo", source)
        self.assertLess(source.index("if ! ensure_teleimager; then"), source.index('exec "$teleimager_python"'))

    def test_teleimager_start_is_persistent_and_uses_state_files(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("setsid", source)
        self.assertIn("nohup", source)
        self.assertIn("teleimager.pid", source)
        self.assertIn("teleimager.log", source)
        self.assertIn("/home/unitree/.local/state/xr_teleoperate", source)
        self.assertNotIn("trap", source)

    def test_teleimager_child_does_not_inherit_launcher_lock_fd(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('2>&1 < /dev/null 9>&- &', source)

    def test_health_check_requires_expected_ports_and_real_bgr_frames(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        for port in ("60000", "55555", "55556"):
            self.assertIn(port, source)
        self.assertIn("get_head_frame", source)
        self.assertIn("get_left_wrist_frame", source)
        self.assertIn("(720, 1280, 3)", source)
        self.assertIn("nbytes", source)
        self.assertIn("deadline = time.monotonic()", source)
        self.assertIn("time.sleep(", source)
        self.assertIn("os._exit(0)", source)

    def test_start_failure_is_bounded_and_happens_before_actuator_python(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("TELEIMAGER_TIMEOUT_S", source)
        self.assertIn("teleimager did not become healthy", source)
        self.assertLess(source.index("if ! ensure_teleimager; then"), source.index('exec "$teleimager_python"'))


class HeadOnlyModeLauncherTest(unittest.TestCase):
    def setUp(self):
        self.source = LAUNCHER.read_text(encoding="utf-8")

    def test_mode_defaults_to_auto_detection(self):
        self.assertIn('TELEIMAGER_CAMERA_MODE=${TELEIMAGER_CAMERA_MODE:-auto}', self.source)
        self.assertIn('both|head|any|single', self.source)
        self.assertIn("unsupported TELEIMAGER_CAMERA_MODE", self.source)

    def test_visible_banner_for_head_only(self):
        self.assertIn("TELEIMAGER: modo SOMENTE CABEÇA (pulso esquerdo desativado)", self.source)

    def test_head_probe_does_not_require_wrist_port_or_frame(self):
        self.assertIn('ports = (60000, 55555) if head_only else (60000, 55555, 55556)', self.source)
        self.assertIn('frames = (head,) if head_only else (head, left_wrist)', self.source)
        self.assertIn('head_only = mode == "head" or mode == "any"', self.source)

    def test_default_probe_still_requires_three_ports_and_both_frames(self):
        self.assertIn("55556", self.source)
        self.assertIn("client.get_left_wrist_frame()", self.source)

    def test_head_mode_starts_server_through_head_only_wrapper(self):
        self.assertIn("teleop.utils.teleimager_head_only_server", self.source)
        self.assertIn("server_module=teleimager.image_server", self.source)
        self.assertIn("teleimager.mode", self.source)
        self.assertIn('TELEIMAGER_HEAD_ONLY_CONFIG="$teleimager_state_dir/', self.source)

    def test_teleop_layout_follows_mode(self):
        self.assertIn("--camera-layout \"$teleop_camera_layout\"", self.source)
        self.assertIn("teleop_camera_layout=head", self.source)
        self.assertIn("teleop_camera_layout=vertical", self.source)
        self.assertNotIn("--camera-layout vertical", self.source)

    def test_refuses_duplicate_when_running_server_has_other_mode(self):
        self.assertIn("running in mode", self.source)
        self.assertIn("refusing a duplicate start", self.source)


class AnyModeLauncherTest(unittest.TestCase):
    def setUp(self):
        self.source = LAUNCHER.read_text(encoding="utf-8")

    def test_any_and_single_accepted_single_normalized(self):
        self.assertIn("both|head|any|single", self.source)
        self.assertIn('TELEIMAGER_CAMERA_MODE=any', self.source)

    def test_any_detects_camera_and_shows_banner_and_wrist_warning(self):
        self.assertIn("--detect", self.source)
        self.assertIn("TELEIMAGER: modo CÂMERA ÚNICA", self.source)
        self.assertIn("AVISO", self.source)
        self.assertIn("TELEIMAGER_CAMERA_SOURCE", self.source)

    def test_any_uses_head_probe_and_head_layout(self):
        self.assertIn('mode == "head" or mode == "any"', self.source)
        self.assertIn("teleop_camera_layout=head", self.source)

    def test_any_starts_through_wrapper_and_fails_without_camera(self):
        self.assertIn('== any', self.source)
        self.assertIn("teleimager_head_only_server", self.source)


if __name__ == "__main__":
    unittest.main()


class LauncherWalkCapTest(unittest.TestCase):
    source = LAUNCHER.read_text(encoding="utf-8")

    def test_defaults_match_botbrain_frontend_with_env_override(self):
        self.assertIn("walk_speed_cap=${G1_WALK_SPEED_CAP:-0.3}", self.source)
        self.assertIn("turn_rate_cap=${G1_TURN_RATE_CAP:-0.3}", self.source)
        self.assertIn("g1-r1.ts", self.source)

    def test_caps_are_passed_explicitly_to_teleop(self):
        self.assertIn('--walk-speed-cap "$walk_speed_cap"', self.source)
        self.assertIn('--turn-rate-cap "$turn_rate_cap"', self.source)
        self.assertIn("Teto de caminhada", self.source)

    def test_shell_expansion_defaults_and_override(self):
        lines = [l for l in self.source.splitlines() if l.startswith(("walk_speed_cap=", "turn_rate_cap="))]
        script = "\n".join(lines) + '\necho "$walk_speed_cap $turn_rate_cap"'
        def run(env):
            return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(run({"PATH": "/usr/bin:/bin"}), "0.3 0.3")
        self.assertEqual(run({"PATH": "/usr/bin:/bin", "G1_WALK_SPEED_CAP": "0.4", "G1_TURN_RATE_CAP": "0.2"}), "0.4 0.2")

    def test_python_hard_limits_unchanged(self):
        quest = (LAUNCHER.parents[1] / "teleop" / "utils" / "quest_controls.py").read_text(encoding="utf-8")
        self.assertIn("0.6", quest)
        self.assertIn("1.0", quest)


class RealSenseUsb2ProfileLauncherTest(unittest.TestCase):
    source = LAUNCHER.read_text(encoding="utf-8")

    def test_profile_is_opt_in_and_tracks_running_profile(self):
        self.assertIn('XR_REALSENSE_PROFILE=${XR_REALSENSE_PROFILE:-normal}', self.source)
        self.assertIn("normal|usb2", self.source)
        self.assertIn("low-bandwidth", self.source)
        self.assertIn("teleimager.realsense_profile", self.source)
        self.assertIn("perfil RealSense mudou", self.source)

    def test_usb2_health_probe_expects_the_low_bandwidth_shape(self):
        self.assertIn('expected_shape = (480, 640, 3) if profile == "usb2" else (720, 1280, 3)', self.source)
