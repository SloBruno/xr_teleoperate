import os
from pathlib import Path
import subprocess
import tempfile
import unittest


LAUNCHER = Path(__file__).parents[1] / "teleop" / "run_g1_quest_dex3.sh"
VUER_CONNECTION = Path(__file__).parents[1] / "teleop" / "lib" / "vuer_connection.sh"


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
        self.assertLess(source.index("if ! ensure_teleimager; then"), source.rindex('exec "$teleimager_python"'))

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
        self.assertLess(source.index("if ! ensure_teleimager; then"), source.rindex('exec "$teleimager_python"'))


class VuerConnectionTest(unittest.TestCase):
    def _select_ip(self, ip_output, *, override=None, tailscale_output=""):
        with tempfile.TemporaryDirectory() as tempdir:
            fake_ip = Path(tempdir) / "ip"
            fake_ip.write_text(
                "#!/usr/bin/env bash\n"
                "case \"$*\" in\n"
                "  '-4 route show default') printf '%s\\n' \"${FAKE_DEFAULT_ROUTE:-}\" ;;\n"
                "  '-4 -o addr show dev wlfxc23cd929ddc scope global') printf '%s\\n' \"${FAKE_WIFI_ADDR:-}\" ;;\n"
                "  '-4 -o addr show dev tailscale0 scope global') printf '%s\\n' \"${FAKE_TAILSCALE_ADDR:-}\" ;;\n"
                "  *) exit 1 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            fake_ip.chmod(0o755)
            env = os.environ | {
                "PATH": f"{tempdir}:{os.environ['PATH']}",
                "FAKE_DEFAULT_ROUTE": ip_output,
                "FAKE_WIFI_ADDR": "2: wlfxc23cd929ddc    inet 10.22.16.110/20 brd 10.22.31.255 scope global dynamic wlfxc23cd929ddc",
                "FAKE_TAILSCALE_ADDR": tailscale_output,
            }
            if override is not None:
                env["XR_TELEOP_VUER_IP"] = override
            return subprocess.run(
                [
                    "bash",
                    "-c",
                    f"set -e; source {VUER_CONNECTION}; xr_teleop_select_vuer_ip; printf '%s' \"$XR_TELEOP_VUER_IP\"",
                ],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

    def test_uses_current_wifi_ip_from_default_route(self):
        result = self._select_ip("default via 10.22.16.1 dev wlfxc23cd929ddc proto dhcp src 10.22.16.110 metric 600")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "10.22.16.110")

    def test_explicit_override_takes_precedence(self):
        result = self._select_ip("default via 10.22.16.1 dev wlfxc23cd929ddc", override="10.22.16.111")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "10.22.16.111")

    def test_invalid_override_fails_closed(self):
        result = self._select_ip("default via 10.22.16.1 dev wlfxc23cd929ddc", override="not-an-ip")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("XR_TELEOP_VUER_IP must be a valid IPv4 address", result.stderr)

    def test_uses_tailscale_address_when_no_wifi_route_is_available(self):
        result = self._select_ip(
            "",
            tailscale_output="7: tailscale0    inet 100.126.188.19/32 scope global tailscale0",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "100.126.188.19")

    def test_fails_with_clear_message_when_no_connection_ip_is_available(self):
        result = self._select_ip("")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No valid Wi-Fi or Tailscale IPv4 address found", result.stderr)

    def test_builds_the_exact_quest_url(self):
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {VUER_CONNECTION}; xr_teleop_quest_url 10.22.16.110",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "https://vuer.ai?ws=wss://10.22.16.110:8012&grid=False")

    def test_certificate_san_must_cover_announced_ip(self):
        with tempfile.TemporaryDirectory() as tempdir:
            cert = Path(tempdir) / "cert.pem"
            key = Path(tempdir) / "key.pem"
            subprocess.run(
                [
                    "openssl", "req", "-x509", "-nodes", "-newkey", "rsa:2048",
                    "-keyout", str(key), "-out", str(cert), "-days", "1",
                    "-subj", "/CN=xr-teleoperate",
                    "-addext", "subjectAltName=IP:10.22.16.110,IP:100.126.188.19,IP:192.168.123.164",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            covered = subprocess.run(
                ["bash", "-c", f"source {VUER_CONNECTION}; xr_teleop_verify_cert_san {cert} 10.22.16.110"],
                capture_output=True,
                text=True,
                check=False,
            )
            missing = subprocess.run(
                ["bash", "-c", f"source {VUER_CONNECTION}; xr_teleop_verify_cert_san {cert} 10.22.16.111"],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(covered.returncode, 0, covered.stderr)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("does not contain IP SAN 10.22.16.111", missing.stderr)


if __name__ == "__main__":
    unittest.main()
