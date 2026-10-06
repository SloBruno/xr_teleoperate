"""XR frame composition always matches the TeleVuer shared-memory shape."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE = Path(__file__).parents[1] / "teleop" / "utils" / "xr_frame.py"


def load():
    spec = importlib.util.spec_from_file_location("xr_frame", MODULE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def img(h, w):
    return SimpleNamespace(bgr=np.full((h, w, 3), 7, dtype=np.uint8))


def test_auto_layout_follows_wrist_stream():
    m = load()
    assert m.resolve_layout("auto", True) == "vertical"
    assert m.resolve_layout("auto", False) == "head"
    with pytest.raises(RuntimeError):
        m.resolve_layout("vertical", False)
    with pytest.raises(ValueError):
        m.resolve_layout("side", True)


def test_vertical_shape_matches_historical_dev_formula():
    m = load()
    assert m.display_shape_for("vertical", [480, 640], [480, 640]) == [495, 320]
    assert m.display_shape_for("head", [480, 640]) == [480, 640]


@pytest.mark.parametrize("layout,shape", [("vertical", [495, 320]), ("head", [480, 640])])
def test_compose_returns_exact_display_shape(layout, shape):
    m = load()
    out = m.compose_xr_frame(layout, img(480, 640), img(480, 640), shape)
    assert out.shape == (shape[0], shape[1], 3) and out.dtype == np.uint8


def test_missing_frames_keep_last_frame_instead_of_wrong_shape():
    m = load()
    assert m.compose_xr_frame("vertical", img(480, 640), None, [495, 320]) is None
    assert m.compose_xr_frame("vertical", img(480, 640), SimpleNamespace(bgr=None), [495, 320]) is None
    assert m.compose_xr_frame("head", None, None, [480, 640]) is None
    assert m.compose_xr_frame("head", SimpleNamespace(bgr=None), None, [480, 640]) is None


def test_head_frame_resized_when_profile_differs_from_buffer():
    m = load()
    out = m.compose_xr_frame("head", img(720, 1280), None, [480, 640])
    assert out.shape == (480, 640, 3)
