"""Real teleop main-path orchestration for the opt-in G1-D waist pitch
(in-process runpy harness of tests/test_g1d_agv_orchestration.py; recording
fake supervisor; no DDS, no robot)."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tests.test_g1d_agv_orchestration as harness  # noqa: E402
from teleop.utils import g1d_waist_pitch as wp  # noqa: E402

PITCH_ARGV = harness.BASE_ARGV + ["--g1d-waist-pitch"]
_LAST = {}          # events/supervisors of the latest _run, readable after it raised


class RecordingSupervisor:
    def __init__(self, log, homing_polls=3):
        self.log = log
        self.state = wp.IDLE
        self.submits = []
        self._thread = None
        self._polls = homing_polls
        self.home_reasons = []

    def request_arm(self, start_thread=True):
        self.log.append("pitch_arm")
        self.state = wp.ACTIVE
        return True

    def submit(self, xy, pressed, ts):
        self.submits.append((tuple(float(v) for v in xy), pressed, ts))

    def request_home_and_release(self, reason):
        self.home_reasons.append(reason)
        if "pitch_home" not in self.log:
            self.log.append("pitch_home")

    def step(self, now=None):
        if self.state == wp.IDLE:
            self.state = wp.DONE

    @property
    def finished(self):
        if self.state == wp.ACTIVE and self.home_reasons:
            self._polls -= 1
            if self._polls <= 0:
                self.state = wp.RELEASED
                self.log.append("pitch_released")
        return self.state in wp.FINISHED_STATES



def _run(monkeypatch, tmp_path, script, *, pitch=True):
    created = []

    def factory(arm_ctrl, log, jsonl_path=None, **kw):
        created.append(jsonl_path)
        original = arm_ctrl.deactivate

        def deactivate():
            events.append("arm_deactivate")
            return original()
        arm_ctrl.deactivate = deactivate
        sup = RecordingSupervisor(events)
        sups.append(sup)
        events.append("pitch_create")
        return sup
    events, sups = [], []
    _LAST.update(events=events, sups=sups)
    monkeypatch.setattr(wp, "create_waist_pitch_supervisor", factory)
    monkeypatch.setattr(harness, "BASE_ARGV", PITCH_ARGV if pitch else harness.BASE_ARGV)
    if pitch:
        monkeypatch.setenv("G1D_WAIST_PITCH", "1")
    else:
        monkeypatch.delenv("G1D_WAIST_PITCH", raising=False)
    log, _, state = harness._run(monkeypatch, tmp_path, script, agv=False)
    return log, events, sups, created, state


def _forward(n, tail):
    def script(i):
        if i < n:
            return {"right": (1.0, 0.0)}
        return tail(i)
    return script


def test_without_flag_no_pitch_supervisor_is_constructed(monkeypatch, tmp_path):
    log, events, sups, created, _ = _run(monkeypatch, tmp_path, _forward(5, lambda i: {"q": True}), pitch=False)
    assert created == [] and sups == []


def test_pitch_arms_only_after_tracking_starts_and_receives_right_stick(monkeypatch, tmp_path):
    log, events, sups, created, state = _run(monkeypatch, tmp_path, _forward(8, lambda i: {"q": True}))
    sup = sups[0]
    assert events[0] == "pitch_create" and events[1] == "pitch_arm"
    assert created[0].startswith(str(tmp_path))      # log goes to the pose-log dir
    assert sup.submits, "pitch never received Quest input"
    assert sup.submits[0][0] == (1.0, 0.0)
    assert all(p is False for _, p, _ in sup.submits)


def test_q_homes_pitch_and_waits_for_release_before_arm_shutdown(monkeypatch, tmp_path):
    log, events, sups, _, _ = _run(monkeypatch, tmp_path, _forward(6, lambda i: {"q": True}))
    assert events.index("pitch_home") < events.index("pitch_released") < events.index("arm_deactivate")
    assert sups[0].state == wp.RELEASED
    assert "arm_shutdown" in log


@pytest.mark.parametrize("exc", [RuntimeError("xr died"), KeyboardInterrupt()])
def test_loop_failure_still_homes_pitch_before_release(monkeypatch, tmp_path, exc):
    log, events, sups, _, _ = _run(monkeypatch, tmp_path, _forward(6, lambda i: {"raise": exc}))
    assert events.index("pitch_released") < events.index("arm_deactivate")
    assert sups[0].home_reasons


def test_pitch_flag_without_launcher_env_is_refused(monkeypatch, capsys):
    monkeypatch.delenv("G1D_WAIST_PITCH", raising=False)
    monkeypatch.setattr(sys, "argv", PITCH_ARGV)
    with pytest.raises(SystemExit):
        harness._run_parse_only(monkeypatch)
    assert "launcher-controlled" in capsys.readouterr().err


def test_pitch_flag_rejected_for_non_g1d_arm(monkeypatch, capsys):
    monkeypatch.setenv("G1D_WAIST_PITCH", "1")
    monkeypatch.setattr(sys, "argv", ["teleop_hand_and_arm.py", "--arm", "G1_29", "--g1d-waist-pitch"])
    with pytest.raises(SystemExit):
        harness._run_parse_only(monkeypatch)
    assert "--g1d-waist-pitch is only valid with --arm G1D" in capsys.readouterr().err


def test_pitch_flag_rejected_with_sim(monkeypatch, capsys):
    monkeypatch.setenv("G1D_WAIST_PITCH", "1")
    monkeypatch.setattr(sys, "argv", PITCH_ARGV + ["--sim"])
    with pytest.raises(SystemExit):
        harness._run_parse_only(monkeypatch)
    assert "no simulation backend" in capsys.readouterr().err


def test_shutdown_order_in_source_pitch_before_arm_release():
    src = (ROOT / "teleop" / "teleop_hand_and_arm.py").read_text(encoding="utf-8")
    fin = src.index("    finally:\n        # Explicit StopMove first")
    pitch = src.index("g1d_waist_pitch.finish_waist_pitch(", fin)
    arm = src.index('elif args.arm == "G1D":\n            if outputs_activated:', fin)
    assert fin < pitch < arm


def test_televuer_forwards_left_y_and_right_stick_in_hand_and_controller_modes():
    src = (ROOT / "teleop" / "televuer" / "src" / "televuer" / "tv_wrapper.py").read_text(encoding="utf-8")
    assert src.count("left_ctrl_bButton=self.tvuer.left_ctrl_bButton") >= 2
    assert src.count("right_ctrl_thumbstickValue=self.tvuer.right_ctrl_thumbstickValue") >= 2


def test_pitch_home_requested_before_any_other_shutdown_step_can_fail(monkeypatch, tmp_path):
    """Codex review: a raising cleanup step before the pitch hook must not
    skip the return to q=0.0. The O(1) home request is the first fallible
    shutdown step (the non-daemon pitch thread completes it on its own);
    observed at the first later cleanup call without aborting the harness
    (an aborted in-process finally would leak threads into later tests)."""
    import teleop.utils.robot_state_monitor as rsm
    original = rsm.stop_locomotion_best_effort
    seen = []

    def observing_stop(*a, **k):
        sups = _LAST.get("sups") or []
        seen.append(bool(sups and sups[0].home_reasons))
        return original(*a, **k)
    monkeypatch.setattr(rsm, "stop_locomotion_best_effort", observing_stop)
    _, events, sups, _, _ = _run(monkeypatch, tmp_path, _forward(6, lambda i: {"q": True}))
    assert seen, "shutdown cleanup step not reached"
    assert seen[0] is True, "pitch home was not requested before the first other cleanup step"
    assert sups[0].state == wp.RELEASED


def test_pitch_implies_no_locomotion_wrapper(monkeypatch, capsys):
    """loco_wrapper is provably None whenever the pitch exists (G1D refuses
    --motion), so no StopMove call precedes the pitch home request."""
    monkeypatch.setenv("G1D_WAIST_PITCH", "1")
    monkeypatch.setattr(sys, "argv", PITCH_ARGV + ["--motion"])
    with pytest.raises(SystemExit):
        harness._run_parse_only(monkeypatch)
    assert "--motion is not supported" in capsys.readouterr().err
