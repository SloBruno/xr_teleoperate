"""Launcher network announcement: Wi-Fi IP, Quest URL and TLS SAN renewal.

`ip` and `openssl` are replaced by PATH stubs; nothing touches the network.
"""
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
LIB = ROOT / "teleop" / "lib" / "vuer_network.sh"
LAUNCHER = ROOT / "teleop" / "run_g1_quest_dex3.sh"

IP_STUB = r"""#!/usr/bin/env bash
# args: -4 route get 1.1.1.1 | -4 -o addr show
if [[ "$*" == *"route get"* ]]; then
    [[ -n "${FAKE_ROUTE:-}" ]] && echo "$FAKE_ROUTE"
    exit 0
fi
if [[ "$*" == *"addr show"* ]]; then
    printf '%b' "$FAKE_ADDRS"
    exit 0
fi
exit 1
"""
OPENSSL_STUB = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
case "$1" in
x509)
    [[ -r "$FAKE_SAN_FILE" ]] || exit 1
    echo "X509v3 Subject Alternative Name: "
    cat "$FAKE_SAN_FILE"
    ;;
req)
    [[ "${FAKE_REQ_FAIL:-0}" == 1 ]] && exit 1
    while (($#)); do
        case "$1" in
        -out) echo NEWCERT > "$2"; shift ;;
        -keyout) echo NEWKEY > "$2"; shift ;;
        -addext) echo "$2" >> "$FAKE_LOG"; shift ;;
        esac
        shift
    done
    ;;
esac
"""
ADDRS = (
    "1: lo    inet 127.0.0.1/8 scope host lo\\n"
    "2: enP8p1s0    inet 192.168.123.164/24 brd 192.168.123.255 scope global enP8p1s0\\n"
    "3: wlxfc23cd929ddc    inet 10.22.16.175/20 brd 10.22.31.255 scope global dynamic wlxfc23cd929ddc\\n"
    "4: tailscale0    inet 100.126.188.19/32 scope global tailscale0\\n"
    "5: docker0    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0\\n"
)
ROUTE = "1.1.1.1 via 10.22.16.1 dev wlxfc23cd929ddc src 10.22.16.175 uid 1000"
SAN_WITH_WIFI = "    DNS:unitree-g1-nx, IP Address:10.22.16.175, IP Address:100.126.188.19, IP Address:10.22.16.110\n"
SAN_OLD = "    DNS:unitree-g1-nx, IP Address:100.126.188.19, IP Address:10.22.16.110\n"


class ShowIpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        for name, body in (("ip", IP_STUB), ("openssl", OPENSSL_STUB)):
            p = self.bin / name
            p.write_text(body)
            p.chmod(p.stat().st_mode | stat.S_IXUSR)
        self.cert = self.tmp / "cfg" / "cert.pem"
        self.key = self.tmp / "cfg" / "key.pem"
        self.cert.parent.mkdir()
        self.cert.write_text("OLDCERT")
        self.key.write_text("OLDKEY")
        self.log = self.tmp / "log"
        self.sanfile = self.tmp / "san"
        self.state = self.tmp / "state"

    def run_announce(self, san=SAN_OLD, route=ROUTE, addrs=ADDRS, env=None, req_fail=False):
        self.sanfile.write_text(san)
        e = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "FAKE_ROUTE": route,
            "FAKE_ADDRS": addrs,
            "FAKE_LOG": str(self.log),
            "FAKE_SAN_FILE": str(self.sanfile),
            "FAKE_REQ_FAIL": "1" if req_fail else "0",
            "XR_TELEOP_CERT": str(self.cert),
            "XR_TELEOP_KEY": str(self.key),
            "XR_TELEOP_STATE_DIR": str(self.state),
        }
        e.update(env or {})
        r = subprocess.run(
            ["bash", "-c", f'source "{LIB}"; xr_net_announce; echo "rc=$?"'],
            env=e, capture_output=True, text=True, timeout=30,
        )
        return r.stdout + r.stderr

    def log_text(self):
        return self.log.read_text() if self.log.exists() else ""

    def test_wifi_present_and_in_cert(self):
        out = self.run_announce(san=SAN_WITH_WIFI)
        self.assertIn("IP do robô na rede (Wi-Fi): 10.22.16.175", out)
        self.assertIn("URL recomendada: https://vuer.ai?ws=wss://10.22.16.175:8012&grid=False", out)
        self.assertIn("Opção local (não recomendada): https://10.22.16.175:8012?grid=False", out)
        self.assertIn("100.126.188.19", out)
        self.assertIn("192.168.123.164", out)
        self.assertNotIn("172.17.0.1", out)
        self.assertNotIn("127.0.0.1", out)
        self.assertNotIn("req ", self.log_text())
        self.assertEqual(self.cert.read_text(), "OLDCERT")
        self.assertIn("rc=0", out)
        self.assertIn("wss://10.22.16.175:8012", (self.state / "quest_url").read_text())

    def test_wifi_missing_from_cert_regenerates_with_backup(self):
        out = self.run_announce(san=SAN_OLD)
        self.assertIn("certificado regenerado", out)
        self.assertIn("https://10.22.16.175:8012", out)
        log = self.log_text()
        self.assertIn("req -x509 -newkey rsa:2048 -nodes -days 825", log)
        for want in ("IP:10.22.16.175", "IP:10.22.16.110", "IP:100.126.188.19",
                     "IP:192.168.123.164", "DNS:unitree-g1-nx"):
            self.assertIn(want, log)
        self.assertEqual(self.cert.read_text().strip(), "NEWCERT")
        self.assertEqual(stat.S_IMODE(self.key.stat().st_mode), 0o600)
        baks = list(self.cert.parent.glob("cert.pem.bak.*"))
        self.assertEqual(len(baks), 1)
        self.assertEqual(baks[0].read_text(), "OLDCERT")
        self.assertIn("rc=0", out)

    def test_regeneration_failure_warns_and_does_not_block(self):
        out = self.run_announce(san=SAN_OLD, req_fail=True)
        self.assertIn("FALHA ao regenerar", out)
        self.assertEqual(self.cert.read_text(), "OLDCERT")
        self.assertEqual(self.key.read_text(), "OLDKEY")
        self.assertIn("rc=0", out)

    def test_no_default_route_warns_clearly(self):
        out = self.run_announce(route="", san=SAN_WITH_WIFI)
        self.assertIn("SEM rota default", out)
        self.assertIn("rc=0", out)

    def test_no_route_falls_back_to_wireless_name(self):
        out = self.run_announce(route="", san=SAN_WITH_WIFI)
        self.assertIn("100.126.188.19", out)  # still listed as secondary

    def test_env_override_wins(self):
        out = self.run_announce(san=SAN_WITH_WIFI, env={"XR_TELEOP_VUER_IP": "10.22.17.58"})
        self.assertIn("wss://10.22.17.58:8012", out)
        self.assertIn("XR_TELEOP_VUER_IP", out)
        self.assertIn("certificado regenerado", out)  # .58 not in SAN_WITH_WIFI

    def test_invalid_override_warns_without_blocking(self):
        out = self.run_announce(env={"XR_TELEOP_VUER_IP": "not-an-ip"})
        self.assertIn("inválido", out)
        self.assertIn("rc=0", out)


class LauncherWiringTest(unittest.TestCase):
    def test_launcher_announces_before_teleimager_and_teleop(self):
        src = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("vuer_network.sh", src)
        self.assertIn("xr_net_announce || true", src)
        self.assertIn("G1_LAUNCHER_SKIP_NET", src)
        self.assertLess(src.index("xr_net_announce || true"), src.index("detect_cameras\n    if"))
        self.assertLess(src.index("xr_net_announce || true"), src.index('exec "$teleimager_python" -s teleop_hand_and_arm.py'))
        self.assertIn("XR_TELEOP_CERT:-", src)

    def test_syntax(self):
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        subprocess.run(["bash", "-n", str(LIB)], check=True)


if __name__ == "__main__":
    unittest.main()
