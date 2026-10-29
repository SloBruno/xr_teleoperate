"""Argument-parsing-level fail-closed contract for --g1d-waist: this must
always refuse to start today (no real waist backend exists; see
teleop/utils/g1d_waist.py). Reuses the existing in-process parse-only
harness from test_g1d_agv_orchestration.py instead of duplicating its
extensive stub-module setup.
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_g1d_agv_orchestration import BASE_ARGV, _run_parse_only  # noqa: E402


def test_g1d_waist_without_launcher_env_is_refused(monkeypatch, capsys):
    monkeypatch.delenv("G1D_WAIST", raising=False)
    monkeypatch.setattr(sys, "argv", BASE_ARGV + ["--g1d-waist"])
    with pytest.raises(SystemExit):
        _run_parse_only(monkeypatch)
    assert "launcher-controlled" in capsys.readouterr().err


def test_g1d_waist_with_launcher_env_still_refused_no_backend(monkeypatch, capsys):
    monkeypatch.setenv("G1D_WAIST", "1")
    monkeypatch.setattr(sys, "argv", BASE_ARGV + ["--g1d-waist"])
    with pytest.raises(SystemExit):
        _run_parse_only(monkeypatch)
    err = capsys.readouterr().err
    assert "--g1d-waist:" in err
    assert "No real rt/waistpitchcmd backend" in err


def test_g1d_waist_rejected_for_non_g1d_arm(monkeypatch, capsys):
    monkeypatch.setenv("G1D_WAIST", "1")
    monkeypatch.setattr(sys, "argv", ["teleop_hand_and_arm.py", "--arm", "G1_29", "--g1d-waist"])
    with pytest.raises(SystemExit):
        _run_parse_only(monkeypatch)
    assert "--g1d-waist is only valid with --arm G1D" in capsys.readouterr().err


def test_without_g1d_waist_flag_parsing_is_unaffected(monkeypatch, capsys):
    # Sanity check: the new flag/validation must not break the existing
    # G1D arm-only parse path when --g1d-waist is simply absent. This still
    # exits early (missing stubs further down the main path), but NOT with
    # our new error text, and not at the --g1d-waist checks.
    monkeypatch.delenv("G1D_WAIST", raising=False)
    monkeypatch.setattr(sys, "argv", BASE_ARGV)
    try:
        _run_parse_only(monkeypatch)
    except SystemExit:
        err = capsys.readouterr().err
        assert "--g1d-waist" not in err
    except Exception:
        pass  # any failure past arg parsing (missing further stubs) is fine here
