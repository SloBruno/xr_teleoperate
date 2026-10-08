"""tools/arm_path_web.py + arm_path_metrics + arm_path_report: path length on known
paths, per-arm separation, hand-centre offset in the wrist frame, recorder
files, re-open, HTTP endpoints (ephemeral server, fake source) and a mocked
DDS subscriber (no real DDS)."""
import csv
import json
import math
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))
import arm_path_metrics as apm  # noqa: E402
import arm_path_report as rpt  # noqa: E402
import arm_path_web as web  # noqa: E402

try:
    import pinocchio  # noqa: F401
    HAVE_PIN = True
except Exception:  # pragma: no cover
    HAVE_PIN = False


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def circle(r=0.1, n=1000, dur=10.0, c=(0.3, 0.2, 0.1)):
    t = np.linspace(0, dur, n)
    a = 2 * np.pi * t / dur
    return t, np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(n, c[2])], axis=1)


class MetricsTest(unittest.TestCase):
    def test_straight_line(self):
        t = np.linspace(0, 2, 201)
        P = np.stack([0.1 + 0.15 * t, np.zeros_like(t), 0.05 * t], axis=1)
        m = apm.path_metrics(t, P)
        L = math.hypot(0.3, 0.1)
        self.assertAlmostEqual(m["length_raw_m"], L, places=9)
        self.assertAlmostEqual(m["length_filtered_m"], L, places=3)  # MA shortens only the ends slightly
        self.assertAlmostEqual(m["net_displacement_m"], L, places=3)
        self.assertAlmostEqual(m["length_axis_raw_m"][0], 0.3, places=9)
        self.assertAlmostEqual(m["length_axis_raw_m"][2], 0.1, places=9)
        self.assertAlmostEqual(m["duration_s"], 2.0)
        self.assertAlmostEqual(m["mean_speed_m_s"], L / 2, places=3)
        self.assertAlmostEqual(m["max_speed_m_s"], L / 2, places=3)

    def test_circle_2pi_r(self):
        t, P = circle(0.1, 1001)
        m = apm.path_metrics(t, P)
        self.assertAlmostEqual(m["length_raw_m"], 2 * math.pi * 0.1, delta=1e-3)
        self.assertAlmostEqual(m["length_filtered_m"], 2 * math.pi * 0.1, delta=2e-3)
        self.assertLess(m["net_displacement_m"], 1e-3)

    def test_noise_inflates_raw_filtered_close(self):
        t, P = circle(0.1, 1001)  # 100 Hz
        rng = np.random.default_rng(1)
        Pn = P + rng.normal(0, 0.001, P.shape)  # 1 mm white noise per axis
        true = 2 * math.pi * 0.1
        m = apm.path_metrics(t, Pn)
        self.assertGreater(m["length_raw_m"], 3 * true)          # noise inflates the raw sum a lot (x3.7)
        self.assertLess(abs(m["length_filtered_m"] - true) / true, 0.12)  # default 50 ms + 1 mm
        m2 = apm.path_metrics(t, Pn, window_s=0.05, min_step_m=0.003)
        self.assertLess(abs(m2["length_filtered_m"] - true) / true, 0.05)
        Ps = P + rng.normal(0, 0.0003, P.shape)  # 0.3 mm (closer to encoder noise)
        self.assertLess(abs(apm.path_metrics(t, Ps)["length_filtered_m"] - true) / true, 0.02)

    def test_still_arm_with_noise(self):
        t = np.linspace(0, 10, 1001)
        P = np.full((1001, 3), 0.3) + np.random.default_rng(2).normal(0, 0.0005, (1001, 3))
        m = apm.path_metrics(t, P, min_step_m=0.003)
        self.assertGreater(m["length_raw_m"], 0.5)
        self.assertLess(m["length_filtered_m"], 0.01)

    def test_smooth_zero_phase_and_edges(self):
        t = np.linspace(0, 1, 101)
        P = np.stack([t, t ** 2, np.zeros_like(t)], axis=1)
        F = apm.smooth(t, P, 0.05)
        np.testing.assert_allclose(F[:, 0], t, atol=1e-12)    # linear signal unchanged (symmetric)
        np.testing.assert_allclose(F[0], P[0]); np.testing.assert_allclose(F[-1], P[-1])

    def test_short(self):
        m = apm.path_metrics([0.0], [[1, 2, 3]])
        self.assertEqual(m["length_raw_m"], 0.0)
        self.assertEqual(apm.path_metrics([], np.zeros((0, 3)))["n_samples"], 0)


def fake_points(q, ol, orr):
    """Pinocchio-free stand-in: left = q[0:3] + offset, right = q[7:10] + offset."""
    q = np.asarray(q, float)
    if not np.all(np.isfinite(q)):
        return None, None
    return q[0:3] + np.asarray(ol), q[7:10] + np.asarray(orr)


class RecorderSamplerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.clock = Clock()
        self.qs = web.QState(clock=self.clock)
        self.pc = web.PointConfig("wrist")
        self.rec = web.Recorder(self.dir, version="abc", clock=self.clock)
        self.smp = web.Sampler(self.qs, fake_points, self.pc, self.rec, rate_hz=100, clock=self.clock)

    def _run(self, n, fl, fr, dt=0.01):
        for i in range(n):
            self.clock.t += dt
            q = np.zeros(14)
            q[0:3] = fl(i * dt)
            q[7:10] = fr(i * dt)
            self.qs.set(q)
            self.smp.step(t_unix=1.7e9 + self.clock.t)

    def test_start_stop_per_arm_files(self):
        self._run(50, lambda t: (0, 0, 0), lambda t: (0, 0, 0))  # pre-roll not recorded
        self.rec.start("pegar caixa!", meta_extra={"point": self.pc.describe()})
        # left: straight 0.2 m in x over 1 s; right: circle r=0.05 (one turn in 1 s)
        self._run(101, lambda t: (0.2 * t, 0.0, 0.0),
                  lambda t: (0.05 * math.cos(2 * math.pi * t), 0.05 * math.sin(2 * math.pi * t), 0.1))
        st = self.rec.status()
        self.assertEqual(st["state"], "recording")
        self.assertAlmostEqual(st["live"]["left"]["length_raw_m"], 0.2, delta=1e-6)
        res = self.rec.stop()
        d = Path(res["dir"])
        self.assertTrue(d.name.endswith("_pegar_caixa"))
        self.assertEqual(sorted(p.name for p in d.iterdir()), ["left.csv", "right.csv", "summary.json"])
        L = list(csv.DictReader(open(d / "left.csv")))
        R = list(csv.DictReader(open(d / "right.csv")))
        self.assertEqual(len(L), 101); self.assertEqual(len(R), 101)
        self.assertEqual(list(L[0].keys()), web.CSV_COLUMNS)
        self.assertAlmostEqual(float(L[-1]["x_raw"]) - float(L[0]["x_raw"]), 0.2, places=5)
        self.assertAlmostEqual(float(L[-1]["dist_cum_raw_m"]), 0.2, places=5)
        self.assertAlmostEqual(float(R[0]["z_raw"]), 0.1 + 0.0, places=5)
        self.assertAlmostEqual(float(R[3]["q_shoulder_pitch"]), float(R[3]["x_raw"]) - 0.05, places=5)  # right q = q[7:14]
        self.assertTrue(L[0]["t_utc"].endswith("Z"))
        s = json.load(open(d / "summary.json"))
        self.assertEqual(s["n_samples"], 101)
        self.assertAlmostEqual(s["arms"]["left"]["length_raw_m"], 0.2, places=6)
        self.assertAlmostEqual(s["arms"]["right"]["length_raw_m"], 2 * math.pi * 0.05, delta=2e-3)
        self.assertAlmostEqual(s["arms"]["right"]["length_filtered_m"], 2 * math.pi * 0.05, delta=0.01)
        self.assertLess(s["arms"]["right"]["net_displacement_m"], 1e-3)
        self.assertEqual(s["point"]["kind"], "wrist")
        self.assertEqual(s["filter"]["window_s"], apm.DEFAULT_WINDOW_S)
        self.assertAlmostEqual(s["rate_hz_measured"], 100.0, delta=0.5)
        # reopen
        tasks = web.list_tasks(self.dir)
        self.assertEqual(tasks[0]["id"], d.name)
        self.assertAlmostEqual(tasks[0]["arms"]["left"]["length_raw_m"], 0.2, places=6)
        t = web.load_task(self.dir, d.name)
        self.assertEqual(len(t["left"]["t"]), 101)
        self.assertAlmostEqual(t["left"]["cum_raw"][-1], 0.2, places=4)
        self.assertEqual(t["summary"]["name"], "pegar_caixa")
        with self.assertRaises(FileNotFoundError):
            web.load_task(self.dir, "../etc")
        # offline report recomputes the same numbers
        rep, _ = rpt.compute(str(d))
        self.assertAlmostEqual(rep["arms"]["left"]["length_raw_m"], 0.2, places=5)
        self.assertEqual(rpt.main([str(d), "--no-png"]), 0)
        self.assertTrue((d / "report.json").exists())

    def test_duration_countdown_and_stale(self):
        self.rec.start("fixa", duration=0.5, countdown=1)
        self._run(90, lambda t: (t, 0, 0), lambda t: (0, 0, 0))
        self.assertEqual(self.rec.status()["state"], "countdown")
        self._run(80, lambda t: (t, 0, 0), lambda t: (0, 0, 0))
        self.assertEqual(self.rec.status()["state"], "idle")
        self.assertTrue(48 <= self.rec.last_result["n_samples"] <= 52, self.rec.last_result)
        # stale lowstate -> no sample
        self.clock.t += 1.0
        self.assertIsNone(self.smp.step())
        self.assertGreaterEqual(self.smp.n_stale, 1)

    def test_hand_offset_switch(self):
        self.pc.kind = "hand"
        self._run(1, lambda t: (0, 0, 0), lambda t: (0, 0, 0))
        s = self.smp.samples_since(0)[-1]
        self.assertAlmostEqual(s["l"][0], web.HAND_CENTER_OFFSET[0])
        self.assertEqual(self.pc.describe()["offset_left_m"], list(web.HAND_CENTER_OFFSET))


@unittest.skipUnless(HAVE_PIN, "pinocchio ausente")
class FKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from teleop.utils.arm_fk import G1_29_WristFK
        cls.fk = G1_29_WristFK()

    def test_wrist_point_equals_arm_fk(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            q = rng.uniform(-1, 1, 14)
            l, r = self.fk.wrist_xyz(q)
            pl, pr = self.fk.points_xyz(q, *web.PointConfig("wrist").offsets())
            np.testing.assert_allclose(pl, l, atol=1e-12); np.testing.assert_allclose(pr, r, atol=1e-12)

    def test_hand_offset_rotates_with_wrist(self):
        pin = self.fk._pin
        rng = np.random.default_rng(4)
        off = np.array([0.11, 0.01, -0.02])
        jl, jr = self.fk.arm_joint_ids[0][-1], self.fk.arm_joint_ids[1][-1]
        for _ in range(10):
            q = rng.uniform(-1, 1, 14)
            pl, pr = self.fk.points_xyz(q, off, off)
            pin.forwardKinematics(self.fk.model, self.fk.data, q)
            for p, j in ((pl, jl), (pr, jr)):
                M = self.fk.data.oMi[j]
                np.testing.assert_allclose(p, M.translation + M.rotation @ off, atol=1e-12)
            # moving only the wrist yaw changes the hand point but not the wrist joint origin
            q2 = q.copy(); q2[6] += 0.5
            pl2, _ = self.fk.points_xyz(q2, off, off)
            pl0, _ = self.fk.points_xyz(q, (0, 0, 0), (0, 0, 0))
            pl02, _ = self.fk.points_xyz(q2, (0, 0, 0), (0, 0, 0))
            np.testing.assert_allclose(pl0, pl02, atol=1e-12)
            self.assertGreater(np.linalg.norm(pl2 - pl), 1e-3)
            self.assertAlmostEqual(np.linalg.norm(pl - pl0), np.linalg.norm(off), places=12)

    def test_zero_pose_hand_is_forward_of_wrist(self):
        q = np.zeros(14)
        w, _ = self.fk.points_xyz(q, (0, 0, 0), (0, 0, 0))
        h, _ = self.fk.points_xyz(q, web.HAND_CENTER_OFFSET, web.HAND_CENTER_OFFSET)
        self.assertAlmostEqual(float(np.linalg.norm(h - w)), 0.110, places=6)


class Dex3HandCentreTest(unittest.TestCase):
    """Dex3 line: 'centro da mão' defaults to the Dex3-1 profile derived from the
    repo URDF (marked ESTIMADO); the Inspire value stays selectable."""

    def test_default_profile_is_dex3_and_inspire_selectable(self):
        self.assertEqual(web.DEFAULT_HAND, "dex3")
        l, r, src = web.HAND_PROFILES["dex3"]
        self.assertEqual((l[0], l[1]), (0.080, 0.004))
        self.assertEqual((r[0], r[1]), (0.080, -0.004))
        self.assertIn("ESTIMADO", src)
        self.assertIn("Não medido", src)                                  # page shows ESTIMADO badge
        self.assertEqual(web.HAND_PROFILES["inspire"][0], web.HAND_CENTER_OFFSET)

    def test_dex3_offset_matches_urdf_palm_and_finger_bases(self):
        import re
        urdf = (REPO / "assets" / "g1" / "g1_body29_hand14.urdf").read_text()

        def origin(joint):
            m = re.search(r'<joint name="%s"[^>]*>\s*<origin xyz="([^"]+)"' % joint, urdf)
            return np.array([float(v) for v in m.group(1).split()])
        for side, sign in (("left", 1.0), ("right", -1.0)):
            palm = origin(f"{side}_hand_palm_joint")
            idx = palm + origin(f"{side}_hand_index_0_joint")
            mid = palm + origin(f"{side}_hand_middle_0_joint")
            centre = (palm + (idx + mid) / 2.0) / 2.0
            prof = np.array(web.HAND_PROFILES["dex3"][0 if side == "left" else 1])
            np.testing.assert_allclose(prof, centre, atol=0.002)
            self.assertAlmostEqual(prof[1] * sign, 0.004, places=6)


class FakeMotor:
    def __init__(self, q):
        self.q = q


class FakeDDSTest(unittest.TestCase):
    def test_subscriber_only_and_arm_indices(self):
        calls = {"init": [], "pub": 0}

        class Sub:
            def __init__(self, topic, typ):
                calls["topic"] = topic

            def Init(self, cb, depth):
                calls["cb"] = cb

            def Close(self):
                calls["closed"] = True

        class Pub:
            def __init__(self, *a, **k):
                calls["pub"] += 1

        ch = types.ModuleType("unitree_sdk2py.core.channel")
        ch.ChannelFactoryInitialize = lambda d, i=None: calls["init"].append((d, i))
        ch.ChannelSubscriber = Sub
        ch.ChannelPublisher = Pub
        idl = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
        idl.LowState_ = object
        mods = {"unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
                "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
                "unitree_sdk2py.core.channel": ch,
                "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
                "unitree_sdk2py.idl.unitree_hg": types.ModuleType("unitree_sdk2py.idl.unitree_hg"),
                "unitree_sdk2py.idl.unitree_hg.msg": types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg"),
                "unitree_sdk2py.idl.unitree_hg.msg.dds_": idl}
        saved = {k: sys.modules.get(k) for k in mods}
        sys.modules.update(mods)
        try:
            qs = web.QState()
            src = web.LowStateSource(qs, "enP8p1s0").start()
            self.assertEqual(calls["init"], [(0, "enP8p1s0")])
            self.assertEqual(calls["topic"], "rt/lowstate")
            self.assertEqual(calls["pub"], 0)
            msg = types.SimpleNamespace(motor_state=[FakeMotor(float(i)) for i in range(35)])
            calls["cb"](msg)
            q, t, n = qs.get()
            np.testing.assert_array_equal(q, np.arange(15, 29, dtype=float))
            self.assertEqual(n, 1)
            calls["cb"](types.SimpleNamespace(motor_state=[]))  # malformed -> counted, no crash
            self.assertEqual(src.n_bad, 1)
            src.stop()
            self.assertTrue(calls.get("closed"))
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_source_has_no_publisher_code(self):
        src = (REPO / "tools" / "arm_path_web.py").read_text()
        self.assertNotIn("ChannelPublisher", src)
        self.assertNotIn(".Write(", src)


class HTTPTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.qs = web.QState()
        self.src = web.FakeSource(self.qs, rate=500).start()
        self.pc = web.PointConfig("wrist")
        self.rec = web.Recorder(self.dir, version="t")
        self.smp = web.Sampler(self.qs, fake_points, self.pc, self.rec, rate_hz=100).start()
        info = {"version": "t", "frame": "f", "source": self.src.describe(), "fake": True,
                "filter": apm.filter_desc(), "buffer_s": 60}
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(self.smp, self.rec, self.dir, self.pc,
                                                                             token="tok", info=info))
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown(); self.smp.stop(); self.src.stop()

    def get(self, p, token=True):
        url = self.base + p + (("&" if "?" in p else "?") + "token=tok" if token else "")
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()

    def post(self, p, body):
        req = urllib.request.Request(self.base + p + "?token=tok", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_endpoints(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/api/status", token=False)
        self.assertEqual(cm.exception.code, 403)
        st, body = self.get("/")
        self.assertEqual(st, 200); self.assertIn(b"path_plot.js", body)
        st, js = self.get("/path_plot.js", token=False)
        self.assertEqual(st, 200); self.assertIn(b"PathPlot", js)
        time.sleep(0.4)
        st, b = self.get("/api/samples?since=0")
        j = json.loads(b)
        self.assertGreater(len(j["samples"]), 10)
        self.assertGreater(j["stream"]["lowstate_rate_hz"], 100)
        self.assertGreater(j["stream"]["rate_hz"], 50)
        self.assertEqual(j["point"]["kind"], "wrist")
        code, r = self.post("/api/record/start", {"name": "teste http"})
        self.assertEqual(code, 200)
        self.assertEqual(self.post("/api/record/start", {"name": "x"})[0], 409)
        self.assertEqual(self.post("/api/config", {"point": "hand"})[0], 409)
        time.sleep(0.6)
        st, b = self.get("/api/record/path")
        self.assertGreater(len(json.loads(b)["path"]), 20)
        code, r = self.post("/api/record/stop", {})
        tid = r["result"]["task_id"]
        self.assertTrue(tid.endswith("_teste_http"))
        st, b = self.get("/api/tasks")
        self.assertEqual(json.loads(b)["tasks"][0]["id"], tid)
        st, b = self.get("/api/task?id=" + tid)
        t = json.loads(b)
        self.assertGreater(len(t["right"]["t"]), 20)
        self.assertIn("arms", t["summary"])
        for f in ("left.csv", "right.csv", "summary.json"):
            st, b = self.get(f"/tasks/{tid}/{f}")
            self.assertEqual(st, 200); self.assertGreater(len(b), 50)
        with self.assertRaises(urllib.error.HTTPError):
            self.get(f"/tasks/{tid}/../../x.csv")
        with self.assertRaises(urllib.error.HTTPError):
            self.get("/api/task?id=nope")
        code, r = self.post("/api/config", {"point": "hand"})
        self.assertEqual(code, 200); self.assertEqual(r["point"]["kind"], "hand")
        self.assertEqual(self.post("/api/config", {"point": "foo"})[0], 400)


@unittest.skipUnless(HAVE_PIN, "pinocchio ausente")
class MainSmokeTest(unittest.TestCase):
    def test_main_fake_source_runs(self):
        d = tempfile.mkdtemp()
        rc = web.main(["--fake-source", "--host", "127.0.0.1", "--port", "0", "--out-dir", d, "--run-seconds", "1.0",
                       "--point", "hand", "--hand-center-offset", "0.1,0,0"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
