"""Side-channel UDP pose stream (teleop -> pose compare web): pack/unpack,
non-blocking send, decimation, opt-in env."""
import math
import socket
import time
import unittest

import numpy as np

from teleop.utils import pose_stream as ps


def _pose(x, y, z):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


class PackTest(unittest.TestCase):
    def test_roundtrip(self):
        q_cmd = np.arange(14) * 0.1
        q_meas = np.arange(14) * -0.05
        pkt = ps.pack_sample(7, 12.5, 1700000000.25, True, True,
                             (0.1, 0.2, 0.3), (0.4, -0.5, 0.6), q_cmd, q_meas)
        self.assertEqual(len(pkt), ps.PACKET_SIZE)
        s = ps.unpack_sample(pkt)
        self.assertEqual(s["seq"], 7)
        self.assertAlmostEqual(s["t_mono"], 12.5)
        self.assertAlmostEqual(s["t_unix"], 1700000000.25)
        self.assertTrue(s["tracking"])
        self.assertTrue(s["fresh"])
        np.testing.assert_allclose(s["hand_l"], (0.1, 0.2, 0.3), atol=1e-6)
        np.testing.assert_allclose(s["hand_r"], (0.4, -0.5, 0.6), atol=1e-6)
        np.testing.assert_allclose(s["q_cmd"], q_cmd, atol=1e-6)
        np.testing.assert_allclose(s["q_meas"], q_meas, atol=1e-6)

    def test_missing_q_cmd_is_nan(self):
        pkt = ps.pack_sample(1, 1.0, 2.0, False, False, (0, 0, 0), (0, 0, 0), None, np.zeros(14))
        s = ps.unpack_sample(pkt)
        self.assertFalse(s["tracking"])
        self.assertTrue(all(math.isnan(v) for v in s["q_cmd"]))

    def test_rejects_garbage(self):
        self.assertIsNone(ps.unpack_sample(b"hello"))
        self.assertIsNone(ps.unpack_sample(b"X" * ps.PACKET_SIZE))


class SenderTest(unittest.TestCase):
    def test_from_env_opt_in(self):
        self.assertIsNone(ps.PoseStreamSender.from_env({}))
        self.assertIsNone(ps.PoseStreamSender.from_env({"XR_POSE_STREAM": "0"}))
        s = ps.PoseStreamSender.from_env({"XR_POSE_STREAM": "1", "XR_POSE_STREAM_PORT": "47999",
                                          "XR_POSE_STREAM_HZ": "25"})
        self.assertIsNotNone(s)
        self.assertEqual(s.addr, ("127.0.0.1", 47999))
        self.assertAlmostEqual(s.min_period, 1 / 25)
        s.close()

    def test_send_without_listener_never_raises(self):
        # Pick a port nobody listens on: bind, read port, close.
        tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tmp.bind(("127.0.0.1", 0))
        port = tmp.getsockname()[1]
        tmp.close()
        s = ps.PoseStreamSender(port=port, rate_hz=1e6)
        t0 = time.perf_counter()
        for i in range(200):
            s.maybe_send(True, _pose(0, 0, 0), _pose(1, 1, 1), np.zeros(14), np.zeros(14), now=i * 1.0)
        dt = time.perf_counter() - t0
        self.assertLess(dt, 0.5)
        self.assertEqual(s.sent + s.errors, 200)
        s.close()

    def test_bad_inputs_are_counted_not_raised(self):
        s = ps.PoseStreamSender(port=47998, rate_hz=1e6)
        self.assertFalse(s.maybe_send(True, None, "x", [1, 2], object(), now=1.0))
        self.assertEqual(s.errors, 1)
        s.close()

    def test_decimation_and_delivery(self):
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(1.0)
        port = rx.getsockname()[1]
        s = ps.PoseStreamSender(port=port, rate_hz=50)
        sent = [s.maybe_send(True, _pose(0.1, 0, 0), _pose(0, 0.2, 0), np.zeros(14), np.ones(14), now=t)
                for t in (0.0, 0.005, 0.019, 0.020, 0.030, 0.041)]
        self.assertEqual(sent, [True, False, False, True, False, True])
        got = [ps.unpack_sample(rx.recv(2048)) for _ in range(3)]
        self.assertEqual([g["seq"] for g in got], [0, 1, 2])
        np.testing.assert_allclose(got[0]["hand_l"], (0.1, 0, 0), atol=1e-6)
        np.testing.assert_allclose(got[0]["q_meas"], np.ones(14), atol=1e-6)
        # fresh: first sample True, unchanged target afterwards False
        self.assertTrue(got[0]["fresh"])
        self.assertFalse(got[1]["fresh"])
        self.assertTrue(s.due(now=1.0))
        s.close()
        rx.close()


class TeleopIntegrationStaticTest(unittest.TestCase):
    """Teleop without XR_POSE_STREAM: every use is guarded by `pose_stream is not None`."""

    def test_guarded(self):
        from pathlib import Path
        src = (Path(__file__).parents[1] / "teleop" / "teleop_hand_and_arm.py").read_text()
        self.assertIn("pose_stream = PoseStreamSender.from_env()", src)
        src_lines = src.splitlines()
        uses = [i for i, l in enumerate(src_lines) if l.strip().startswith("pose_stream.")]
        self.assertGreaterEqual(len(uses), 2)  # pre-start + main loop sends (+ close)
        for i in uses:
            prev = [p for p in src_lines[max(0, i - 4):i] if p.strip()][-3:]
            self.assertTrue(any("if pose_stream is not None" in p for p in prev), src_lines[i])


if __name__ == "__main__":
    unittest.main()
