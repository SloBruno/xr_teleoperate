"""tools/pose_compare_web.py: recorder (duration/manual), hub, HTTP endpoints
with an ephemeral server and a fake UDP source. FK is faked (pinocchio-free)."""
import csv
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "tools"))
import pose_compare_web as web  # noqa: E402
from teleop.utils import pose_stream as ps  # noqa: E402


def fake_fk(q):
    q = np.asarray(q, float)
    if not np.all(np.isfinite(q)):
        return None, None
    return q[:3] * 0.1, q[7:10] * 0.1


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _pkt(seq, t, tracking=True, hand=(0.3, 0.2, 0.1)):
    q = np.arange(14, dtype=float)
    return ps.pack_sample(seq, t, 1.7e9 + t, tracking, True, hand, (0.3, -0.2, 0.1), q, q * 2)


class RecorderTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.clock = Clock()
        self.rec = web.Recorder(self.dir, version="abc", clock=self.clock)
        self.hub = web.PoseHub(fk=fake_fk, recorder=self.rec, clock=self.clock)

    def _feed(self, n, dt=0.02, start_seq=0):
        for i in range(n):
            self.clock.t += dt
            self.hub.ingest(_pkt(start_seq + i, self.clock.t))

    def test_duration_with_countdown(self):
        self.rec.start("pegar caixa!", "duration", duration=1.0, countdown=3)
        self.assertEqual(self.rec.status()["state"], "countdown")
        self._feed(100)  # 2 s -> still countdown, nothing kept
        self.assertEqual(self.rec.status()["n_samples"], 0)
        self._feed(110, start_seq=100)  # crosses t=3 s then the 1 s recording window
        st = self.rec.status()
        self.assertEqual(st["state"], "idle")
        res = self.rec.last_result
        self.assertTrue(res["csv"].endswith("_pegar_caixa.csv"))
        rows = list(csv.DictReader(open(res["csv"])))
        self.assertTrue(48 <= len(rows) <= 52, len(rows))
        self.assertEqual(list(rows[0].keys()), web.CSV_COLUMNS)
        self.assertAlmostEqual(float(rows[0]["hand_L_x"]), 0.3, places=5)
        self.assertAlmostEqual(float(rows[0]["robot_cmd_L_y"]), 0.1, places=5)   # q[1]*0.1
        self.assertAlmostEqual(float(rows[0]["robot_meas_R_x"]), 1.4, places=5)  # 2*q[7]*0.1
        self.assertEqual(rows[0]["q_meas_13"], "26.000000")
        self.assertTrue(rows[0]["t_utc"].endswith("Z"))
        meta = json.load(open(res["json"]))
        self.assertEqual(meta["name"], "pegar_caixa")
        self.assertEqual(meta["mode"], "duration")
        self.assertEqual(meta["n_samples"], len(rows))
        self.assertAlmostEqual(meta["rate_hz"], 50.0, delta=1.0)
        self.assertEqual(meta["git_version"], "abc")
        self.assertIn("frame", meta)
        self.assertEqual(meta["tracking_fraction"], 1.0)

    def test_manual_and_tick_end(self):
        self.rec.start("m", "manual")
        self._feed(25)
        res = self.rec.stop()
        self.assertEqual(res["n_samples"], 25)
        self.assertEqual(self.rec.status()["state"], "idle")
        # duration ends by tick() even without samples
        self.rec.start("d", "duration", duration=0.5)
        self.clock.t += 0.6
        self.assertIsNotNone(self.rec.tick())
        self.assertEqual(self.rec.last_result["n_samples"], 0)
        self.assertEqual(len(web.list_tasks(self.dir)), 2)

    def test_rejects(self):
        self.rec.start("a")
        with self.assertRaises(RuntimeError):
            self.rec.start("b")
        self.rec.stop()
        with self.assertRaises(ValueError):
            self.rec.start("c", "duration", duration=-1)
        with self.assertRaises(ValueError):
            self.rec.start("c", "weird")

    def test_hub_buffer_lost_and_nan(self):
        self.hub.ingest(b"junk")
        self._feed(3)
        self.hub.ingest(_pkt(10, self.clock.t + 0.02))
        st = self.hub.status()
        self.assertEqual(st["n_bad"], 1)
        self.assertEqual(st["lost"], 7)
        # q_cmd NaN (before r) -> robot cmd null, hand still there
        self.hub.ingest(ps.pack_sample(11, 200.0, 1.7e9, False, False, (1, 2, 3), (4, 5, 6), None, np.zeros(14)))
        s = self.hub.samples_since(0)[-1]
        self.assertEqual(s["cl"], [None, None, None])
        self.assertEqual(s["hl"], [1.0, 2.0, 3.0])
        self.assertEqual(s["tr"], 0)


def fake_skel(q):
    q = np.asarray(q, float)
    if not np.all(np.isfinite(q)):
        return None
    arm = lambda sgn: [[0.0, sgn * 0.1, 0.3]] + [[0.01 * i, sgn * 0.15, 0.3 - 0.03 * i] for i in range(1, 8)]
    return {"l": arm(1), "r": arm(-1), "b": [[0, 0, 0], [0, 0, 0.04]]}


class SkeletonHubTest(unittest.TestCase):
    def test_decimated_and_absent(self):
        clock = Clock()
        hub = web.PoseHub(fk=fake_fk, clock=clock, skeleton=fake_skel, skeleton_hz=20.0)
        for i in range(100):  # 2 s @ 50 Hz
            clock.t += 0.02
            hub.ingest(_pkt(i, clock.t))
        smp = hub.samples_since(0)
        sk = [s for s in smp if "sk" in s]
        self.assertTrue(38 <= len(sk) <= 51, len(sk))
        self.assertEqual(len(sk[0]["sk"]["l"]), 8)
        self.assertEqual(sk[0]["sk"]["r"][0], [0.0, -0.1, 0.3])
        st = hub.status()
        self.assertTrue(st["skeleton"])
        self.assertEqual(st["n_skel"], len(sk))
        # no skeleton (e.g. --no-fk): field absent, status says so
        hub2 = web.PoseHub(fk=None, clock=clock)
        hub2.ingest(_pkt(0, clock.t))
        self.assertNotIn("sk", hub2.samples_since(0)[0])
        self.assertFalse(hub2.status()["skeleton"])
        # skeleton errors do not break ingest
        hub3 = web.PoseHub(fk=fake_fk, clock=clock, skeleton=lambda q: 1 / 0)
        self.assertIsNotNone(hub3.ingest(_pkt(0, clock.t)))
        self.assertEqual(hub3.status()["fk_errors"], 1)


class HttpTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.rec = web.Recorder(self.dir)
        self.hub = web.PoseHub(fk=fake_fk, recorder=self.rec, skeleton=fake_skel)
        self.udp_port = self.hub.bind("127.0.0.1", 0)
        threading.Thread(target=self.hub.run_udp, daemon=True).start()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(self.hub, self.rec, self.dir,
                                                                             token="s3", info={"frame": "f"}))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.stop = threading.Event()
        threading.Thread(target=web.fake_source, args=(self.udp_port, self.stop), daemon=True).start()

    def tearDown(self):
        self.stop.set()
        self.httpd.shutdown()
        self.hub.stop()

    def get(self, path, token="s3"):
        sep = "&" if "?" in path else "?"
        with urllib.request.urlopen(self.base + path + (f"{sep}token={token}" if token else ""), timeout=5) as r:
            return r.status, r.headers, r.read()

    def post(self, path, obj):
        req = urllib.request.Request(self.base + path, data=json.dumps(obj).encode(),
                                     headers={"X-Token": "s3", "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def test_endpoints(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/api/status", token=None)
        self.assertEqual(cm.exception.code, 403)
        code, hdr, body = self.get("/")
        self.assertEqual(code, 200)
        html = body.decode()
        self.assertIn("Salvar tarefa", html)
        self.assertNotIn("cdn", html.lower())
        self.assertNotIn("<script src", html.lower())
        self.assertIn('id="c3d"', html)       # 3D panel
        for v in ("Frente", "Lado", "Topo", "Isométrica", "Recentrar"):
            self.assertIn(v, html)
        time.sleep(0.6)
        st = json.loads(self.get("/api/status")[2])
        self.assertGreater(st["stream"]["n_rx"], 5)
        self.assertGreater(st["stream"]["rate_hz"], 20)
        self.assertEqual(st["frame"], "f")
        smp = json.loads(self.get("/api/samples?since=0")[2])["samples"]
        self.assertTrue(smp and all(len(s["ml"]) == 3 for s in smp))
        sks = [s["sk"] for s in smp if "sk" in s]
        self.assertTrue(sks and len(sks) < len(smp))  # decimated skeleton in the stream
        self.assertEqual(len(sks[-1]["l"]), 8)
        last = smp[-1]["id"]
        newer = json.loads(self.get(f"/api/samples?since={last}")[2])["samples"]
        self.assertTrue(all(s["id"] > last for s in newer))
        # record manual 0.4 s
        r = self.post("/api/record/start", {"name": "teste http", "mode": "manual"})
        self.assertEqual(r["record"]["state"], "recording")
        time.sleep(0.4)
        r = self.post("/api/record/stop", {})
        self.assertGreater(r["result"]["n_samples"], 5)
        tasks = json.loads(self.get("/api/tasks")[2])["tasks"]
        self.assertEqual(len(tasks), 1)
        code, hdr, body = self.get("/tasks/" + tasks[0]["csv"])
        self.assertIn("text/csv", hdr["Content-Type"])
        self.assertTrue(body.decode().startswith("t_mono_s,"))
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/tasks/..%2F..%2Fetc%2Fpasswd")
        self.assertEqual(cm.exception.code, 404)
        # duration via HTTP
        r = self.post("/api/record/start", {"name": "dur", "mode": "duration", "duration": 0.3, "countdown": 0})
        time.sleep(0.7)
        st = json.loads(self.get("/api/status")[2])["record"]
        self.assertEqual(st["state"], "idle")
        self.assertEqual(len(json.loads(self.get("/api/tasks")[2])["tasks"]), 2)


class MainSmokeTest(unittest.TestCase):
    def test_main_no_fk_runs_and_exits(self):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        d = tempfile.mkdtemp()
        rc = web.main(["--host", "127.0.0.1", "--port", str(port), "--udp-port", "0", "--no-fk",
                       "--fake-source", "--run-seconds", "0.5", "--task-dir", d])
        self.assertEqual(rc, 0)

    def test_main_no_fk_http_has_no_skeleton(self):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        d = tempfile.mkdtemp()
        out = {}

        def probe():
            time.sleep(0.6)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/samples?since=0", timeout=3) as r:
                out["j"] = json.loads(r.read())

        th = threading.Thread(target=probe); th.start()
        rc = web.main(["--host", "127.0.0.1", "--port", str(port), "--udp-port", "0", "--no-fk",
                       "--fake-source", "--run-seconds", "1.0", "--task-dir", d])
        th.join()
        self.assertEqual(rc, 0)
        j = out["j"]
        self.assertFalse(j["stream"]["skeleton"])
        self.assertTrue(j["samples"])
        self.assertTrue(all("sk" not in s and s["ml"] == [None, None, None] for s in j["samples"]))


if __name__ == "__main__":
    unittest.main()
