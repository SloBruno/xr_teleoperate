"""Manual calibration markers + walk-trim analyzer on synthetic JSONL."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import analyze_walk_calibration as awc
from tools import mark_calibration_segments as mcs

T0 = 1_800_000_000.0  # wall clock epoch base
HZ = 20.0


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def wall(self):
        return T0 + self.t

    def mono(self):
        return 500.0 + self.t


def _session(path, clock):
    return mcs.MarkerSession(path, wall_clock=clock.wall, monotonic_clock=clock.mono)


# ------------------------------------------------------------------ markers
def test_marker_session_writes_append_only_and_undo(tmp_path):
    clock = FakeClock()
    out = tmp_path / "m.jsonl"
    s = _session(out, clock)
    assert "sem caixa" in s.handle("s")
    s.handle("t 2")
    clock.t = 1.0
    s.handle("r")
    assert s.state()["open_segment"]["kind"] == "reta"
    clock.t = 2.0
    s.handle("e")                       # auto-closes reta
    assert s.state()["open_segment"]["kind"] == "giro_esquerda"
    msg = s.handle("u")                 # undo the giro -> reta open again
    assert "desfeito" in msg
    assert s.state()["open_segment"]["kind"] == "reta"
    clock.t = 3.0
    s.handle("f")
    assert s.state()["open_segment"] is None
    s.handle("x pe escorregou")
    assert "desconhecid" in s.handle("zz")   # unknown key: no write
    s.handle("q")
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    types = [l["type"] for l in lines]
    assert types[0] == "session_start" and types[-1] == "session_end"
    assert "undo" in types and "zz" not in json.dumps(lines)
    seg = [l for l in lines if l["type"] == "segment_start"]
    assert seg[0]["condition"] == "sem_caixa" and seg[0]["attempt"] == 2
    for l in lines:
        assert l["event"] == "calib_marker"
        assert l["timestamp_utc"].endswith("Z")
        assert isinstance(l["timestamp"], float) and isinstance(l["timestamp_monotonic"], float)
    segs = mcs.segments_from_markers(lines)
    assert len(segs) == 1
    assert segs[0]["kind"] == "reta" and segs[0]["start"] == T0 + 1.0 and segs[0]["end"] == T0 + 3.0
    assert segs[0]["notes"] == []


def test_marker_help_and_main_help(capsys):
    s = mcs.MarkerSession(None, wall_clock=lambda: T0, monotonic_clock=lambda: 1.0)
    assert "giro" in s.handle("?")
    assert "automatic" in s.handle("p")       # no manual stop key
    with pytest.raises(SystemExit) as e:
        mcs.main(["--help"])
    assert e.value.code == 0


# ---------------------------------------------------------------- telemetry
def rec(t, cmd, pos, yaw, vel, yaw_speed, balance=True, raw=None):
    r = {"event": "full_pose_telemetry", "timestamp": T0 + t, "timestamp_monotonic": 500.0 + t,
         "timestamp_utc": "x"}
    if not balance:
        return r
    sport = {"position": [pos[0], pos[1], 0.75], "velocity": [vel[0], vel[1], 0.0],
             "yaw_speed": yaw_speed, "imu": {"rpy": [0.01, 0.02, math.remainder(yaw, 2 * math.pi)]}}
    r["balance"] = {
        "loco_command": list(cmd),
        "loco_raw": raw or {"left": [-cmd[1], cmd[0]], "right": [-cmd[2], 0.0]},
        "sport": {"odommodestate": sport, "sportmodestate": None},
        "odom": {"pelvis": None, "torso": None, "fusion": None},
        "imu_pelvis": {"rpy": [0.01, 0.03, 0.0]}, "imu_torso": {"rpy": [0.0, 0.08, 0.0]},
        "arms": {"tau_est": [2.0] * 14, "q": [0.0] * 14},
        "com": {"dx_mm": 10.0, "dy_mm": -3.0},
        "derived": {"horizontal_speed": math.hypot(*vel), "command_zero": max(map(abs, cmd)) < 1e-3},
    }
    return r


def attempt(t_start, vy_trim, om_trim, yaw0=0.2, x0=0.0, balance=True):
    """reta 10 s + parada 3 s + giro esq 7 s + giro dir 7 s starting at t_start."""
    rows, n = [], int(HZ)
    x, y, yaw = x0, 0.0, yaw0
    for i in range(10 * n):                        # straight, 0.5 m/s along heading yaw0
        t = t_start + i / HZ
        rows.append(rec(t, (0.5, vy_trim, om_trim), (x, y), yaw,
                        (0.5 * math.cos(yaw0), 0.5 * math.sin(yaw0)), 0.0, balance))
        x += 0.5 * math.cos(yaw0) / HZ
        y += 0.5 * math.sin(yaw0) / HZ
    lat = (-math.sin(yaw0), math.cos(yaw0))         # left of heading
    for i in range(3 * n):                         # stop: drift 0.1 m left for 1 s
        t = t_start + 10 + i / HZ
        v = 0.1 if i < n else 0.0
        rows.append(rec(t, (0.0, 0.0, 0.0), (x, y), yaw, (v * lat[0], v * lat[1]), 0.0, balance))
        x += v * lat[0] / HZ
        y += v * lat[1] / HZ
    for sign, base in ((1, 13), (-1, 20)):         # +/-360 deg over 7 s (crosses +/-pi)
        rate = sign * 2 * math.pi / 7.0
        for i in range(7 * n):
            t = t_start + base + i / HZ
            rows.append(rec(t, (0.0, 0.0, sign * 0.6), (x, y), yaw, (0.0, 0.0), rate, balance))
            yaw += rate / HZ
    return rows


def markers_for(cond, att, t_start, with_mistake=False, explicit_end=False):
    ev, seq = [], [0]

    def m(t, typ, **kw):
        seq[0] += 1
        d = {"event": "calib_marker", "seq": seq[0], "type": typ, "timestamp": T0 + t,
             "timestamp_monotonic": 500.0 + t, "timestamp_utc": "x", "condition": cond,
             "attempt": att, "kind": None, "note": None}
        d.update(kw)
        ev.append(d)
        return d

    m(t_start - 1, "condition")
    m(t_start - 0.5, "attempt")
    m(t_start, "segment_start", kind="reta")
    if with_mistake:
        bad = m(t_start + 5, "segment_start", kind="giro_direita")
        m(t_start + 5.2, "undo", undo_of=bad["seq"])
    if explicit_end:                     # 'f' pressed before the release
        m(t_start + 9.5, "segment_end")
    m(t_start + 13, "segment_start", kind="giro_esquerda")
    m(t_start + 20, "segment_start", kind="giro_direita")
    m(t_start + 27, "segment_end")
    return ev


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.fixture
def dataset(tmp_path):
    tel, mk = [], []
    plan = [("sem_caixa", 1, 0.0, 0.04, -0.10), ("sem_caixa", 2, 40.0, 0.06, -0.10),
            ("com_caixa", 1, 80.0, 0.10, -0.20)]
    for cond, att, ts, vy, om in plan:
        tel += attempt(ts, vy, om, yaw0=3.0 if att == 2 else 0.2)
        mk += markers_for(cond, att, ts, with_mistake=(att == 1 and cond == "sem_caixa"),
                          explicit_end=(att == 2))
    for i, e in enumerate(mk):
        e["seq"] = i + 1
    # undo references must follow renumbering
    for i, e in enumerate(mk):
        if e["type"] == "undo":
            e["undo_of"] = mk[i - 1]["seq"]
    half = len(tel) // 2
    t1 = write_jsonl(tmp_path / "pose-a.jsonl", tel[:half])
    t2 = write_jsonl(tmp_path / "pose-b.jsonl", tel[half:] + [{"broken": 1}])
    (tmp_path / "pose-b.jsonl").write_text((tmp_path / "pose-b.jsonl").read_text() + "{bad\n")
    m = write_jsonl(tmp_path / "calib.jsonl", mk)
    return [t1, t2], m


def _seg(rep, cond, att, kind):
    return next(s for s in rep["segments"]
                if s["condition"] == cond and s["attempt"] == att and s["kind"] == kind)


def test_straight_trim_and_geometry(dataset):
    tels, m = dataset
    rep = awc.analyze(tels, m)
    # 3 manual retas + 3 automatic paradas + 6 giros; undo removed the mistaken marker
    assert len(rep["segments"]) == 12
    assert sorted(x["kind"] for x in rep["segments"] if x.get("auto")) == ["parada"] * 3
    s = _seg(rep, "sem_caixa", 1, "reta")
    assert abs(s["duration_s"] - 10.0) < 0.1
    assert abs(s["trim_sugerido"]["vy"] - 0.04) < 1e-9
    assert abs(s["trim_sugerido"]["omega"] + 0.10) < 1e-9
    assert abs(s["cmd"]["vx"]["mean"] - 0.5) < 1e-9 and s["cmd"]["vy"]["std"] < 1e-9
    assert abs(s["stick"]["left_y"]["mean"] - 0.5) < 1e-9
    assert abs(s["distancia_m"] - 5.0) < 0.1
    assert abs(s["deriva_lateral_m"]) < 0.02 and abs(s["deriva_heading_deg"]) < 0.5
    assert abs(s["velocidade_media_mps"] - 0.5) < 0.02
    assert s["missing"] == []
    assert abs(s["soltou_joystick_s"] - 10.0) < 0.1
    s2 = _seg(rep, "sem_caixa", 2, "reta")       # heading 3.0 rad, 'f' at 9.5 s
    assert abs(s2["distancia_m"] - 4.75) < 0.1 and abs(s2["deriva_lateral_m"]) < 0.02


def test_stop_is_detected_automatically(dataset):
    tels, m = dataset
    rep = awc.analyze(tels, m)
    s2 = _seg(rep, "sem_caixa", 2, "parada")     # release after an explicit 'f'
    assert s2["auto"] and abs(s2["start"] - (T0 + 50.0)) < 0.1
    assert abs(s2["deslocamento_m"] - 0.1) < 0.02
    s = _seg(rep, "sem_caixa", 1, "parada")
    assert s["auto"] and abs(s["start"] - (T0 + 10.0)) < 0.1
    assert abs(s["end"] - (T0 + 11.5)) < 0.15          # still for 0.5 s -> end
    assert abs(s["deslocamento_m"] - 0.1) < 0.02
    assert abs(s["deriva_lateral_m"] - 0.1) < 0.02     # drifted to the left
    assert 0.8 < s["tempo_ate_parar_s"] < 1.3
    assert s["cmd_zero_frac"] == 1.0


def test_turns_with_yaw_wrap(dataset):
    tels, m = dataset
    rep = awc.analyze(tels, m)
    for att in (1, 2):
        left = _seg(rep, "sem_caixa", att, "giro_esquerda")
        right = _seg(rep, "sem_caixa", att, "giro_direita")
        assert abs(left["angulo_deg"] - 360.0) < 8.0, left["angulo_deg"]
        assert abs(right["angulo_deg"] + 360.0) < 8.0, right["angulo_deg"]
        assert abs(left["cmd"]["omega"]["mean"] - 0.6) < 1e-9
        assert left["translacao_m"] < 0.01 and left["sentido_ok"] and right["sentido_ok"]


def test_aggregate_by_condition_and_difference(dataset):
    tels, m = dataset
    rep = awc.analyze(tels, m)
    agg = rep["aggregate"]["sem_caixa"]["reta"]
    assert agg["n"] == 2
    assert abs(agg["trim_vy"]["mean"] - 0.05) < 1e-9
    assert abs(agg["trim_vy"]["std"] - math.sqrt(2) * 0.01) < 1e-6
    diff = rep["diferenca_com_menos_sem"]["reta"]
    assert abs(diff["trim_vy"] - 0.05) < 1e-9
    assert abs(diff["trim_omega"] + 0.10) < 1e-9


def test_missing_balance_is_reported(tmp_path):
    tel = write_jsonl(tmp_path / "p.jsonl", attempt(0.0, 0.04, -0.1, balance=False))
    mk = write_jsonl(tmp_path / "m.jsonl", markers_for("sem_caixa", 1, 0.0))
    rep = awc.analyze([tel], mk)
    s = _seg(rep, "sem_caixa", 1, "reta")
    assert "balance" in s["missing"] and s["trim_sugerido"]["vy"] is None
    assert not [x for x in rep["segments"] if x["kind"] == "parada"]
    assert any("parada" in w for w in rep["warnings"])


def test_missing_odom_is_reported(tmp_path):
    rows = attempt(0.0, 0.04, -0.1)
    for r in rows:
        r["balance"]["sport"] = {"odommodestate": None, "sportmodestate": None}
    rep = awc.analyze([write_jsonl(tmp_path / "p.jsonl", rows)],
                      write_jsonl(tmp_path / "m.jsonl", markers_for("sem_caixa", 1, 0.0)))
    s = _seg(rep, "sem_caixa", 1, "reta")
    assert "odom" in s["missing"] and s["distancia_m"] is None
    assert abs(s["trim_sugerido"]["vy"] - 0.04) < 1e-9


def test_cli_text_json_csv(dataset, tmp_path, capsys):
    tels, m = dataset
    csv_path = tmp_path / "o.csv"
    assert awc.main([str(tels[0]), str(tels[1]), "--markers", str(m), "--csv", str(csv_path)]) == 0
    out = capsys.readouterr().out
    assert "trim sugerido" in out.lower() and "com_caixa" in out
    assert len(csv_path.read_text().splitlines()) == 13
    assert awc.main([str(tels[0]), str(tels[1]), "--markers", str(m), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["segments"]) == 12


# ---------------------------------------------------------------- IMU offset
def test_marker_offset_key_is_persistent(tmp_path):
    clock = FakeClock()
    out = tmp_path / "m.jsonl"
    s = _session(out, clock)
    assert s.state()["imu_offset"] is None
    assert "ERRO" in s.handle("o 1 2")                 # needs 3 numbers; nothing written
    assert "offset" in s.handle("o 0.5 -1,0 0").lower() # comma decimal accepted
    assert s.state()["imu_offset"] == [0.5, -1.0, 0.0]
    s.handle("c")
    clock.t = 1.0
    s.handle("r")
    clock.t = 2.0
    s.handle("f")
    s.handle("q")
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    seg = [l for l in lines if l["type"] == "segment_start"][0]
    assert seg["imu_offset"] == [0.5, -1.0, 0.0]
    assert mcs.segments_from_markers(lines)[0]["imu_offset"] == [0.5, -1.0, 0.0]


def _offset_dataset(tmp_path, conflict=False):
    """com_caixa, roll in {-0.5,0,0.5,1.0}: trim_vy = 0.02 + 0.04*roll;
    pitch in {-0.5,0,0.5}: stop forward drift = 0.05 - 0.1*pitch."""
    tel, mk = [], []
    plan = [(-0.5, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, -0.5), (0.0, 0.5)]
    for i, (roll, pitch) in enumerate(plan):
        ts = i * 40.0
        rows = attempt(ts, 0.02 + 0.04 * roll, 0.0, yaw0=0.0)
        drift = 0.05 - 0.1 * pitch            # forward drift during stop (m), over 1 s
        x_end = None
        for r in rows:
            t = r["timestamp"] - T0 - ts
            if 10.0 <= t < 27.0:
                r["balance"]["sport"]["odommodestate"]["position"][1] -= 0.1 * min(t - 10.0, 1.0)
                r["balance"]["sport"]["odommodestate"]["position"][0] += drift * min(t - 10.0, 1.0)
        tel += rows
        ev = markers_for("com_caixa", i + 1, ts)
        ev.insert(0, {"event": "calib_marker", "seq": 0, "type": "imu_offset",
                      "timestamp": T0 + ts - 2, "timestamp_utc": "x", "timestamp_monotonic": 1.0,
                      "condition": None, "attempt": None, "imu_offset": [roll, pitch, 0.0]})
        mk += ev
    if conflict:
        tel[5]["balance"]["config_changes"] = [
            {"name": "imu_offset_json", "content": json.dumps({"imu_offset_json": [9.0, 9.0, 0.0]})}]
    for i, e in enumerate(mk):
        e["seq"] = i + 1
    return [write_jsonl(tmp_path / "p.jsonl", tel)], write_jsonl(tmp_path / "m.jsonl", mk)


def test_fit_offset_roll_and_pitch(tmp_path):
    tels, m = _offset_dataset(tmp_path)
    rep = awc.analyze(tels, m, fit_offset=True)
    assert "com_caixa @ roll=+0.50 pitch=+0.00 yaw=+0.00" in rep["aggregate_by_offset"]
    roll = rep["fit_offset"]["com_caixa"]["roll_vs_trim_vy"]
    assert roll["n"] == 6 and roll["niveis"] == 4
    assert abs(roll["inclinacao"] - 0.04) < 1e-6 and abs(roll["intercepto"] - 0.02) < 1e-6
    assert abs(roll["r2"] - 1.0) < 1e-6
    assert abs(roll["offset_zero"] + 0.5) < 1e-6 and roll["avisos"] == []
    pitch = rep["fit_offset"]["com_caixa"]["pitch_vs_deriva_frente_parada"]
    assert pitch["niveis"] == 3
    assert abs(pitch["inclinacao"] + 0.1) < 0.02
    assert abs(pitch["offset_zero"] - 0.5) < 0.1
    assert "pitch_vs_erro_vx" in rep["fit_offset"]["com_caixa"]
    assert rep["fit_offset"]["com_caixa"]["yaw"]["recomendacao"] is None


def test_linear_fit_warnings():
    f = awc.linear_fit([0.0, 0.0, 1.0], [1.0, 1.1, 2.0])
    assert any("niveis" in w for w in f["avisos"])
    g = awc.linear_fit([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])        # zero at -1: extrapolated
    assert abs(g["offset_zero"] + 1.0) < 1e-9 and any("extrapol" in w for w in g["avisos"])
    assert awc.linear_fit([1.0], [2.0])["inclinacao"] is None


def test_offset_conflict_with_config_changes(tmp_path):
    tels, m = _offset_dataset(tmp_path, conflict=True)
    rep = awc.analyze(tels, m)
    assert any("conflito" in w for w in rep["warnings"])
    seg = rep["segments"][0]
    assert seg["imu_offset"] == [-0.5, 0.0, 0.0] and seg["imu_offset_fonte"] == "marcador"
    assert rep["segments"][3]["imu_offset_telemetria"] == [9.0, 9.0, 0.0]
