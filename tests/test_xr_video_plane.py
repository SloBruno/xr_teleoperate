"""Geometry of the XR video plane: how many degrees of the headset it covers."""
import importlib.util
import math
from pathlib import Path

MODULE = Path(__file__).parents[1] / "teleop" / "utils" / "xr_video_plane.py"


def load():
    spec = importlib.util.spec_from_file_location("xr_video_plane", MODULE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_default_plane_is_the_historical_1m_at_1m():
    m = load()
    assert (m.DEFAULT_HEIGHT_M, m.DEFAULT_DISTANCE_M) == (1.0, 1.0)


def test_angular_size_default_square_plane():
    m = load()
    v, h = m.plane_angular_size_deg(1.0, 1.0, 1.0)
    assert math.isclose(v, 53.13, abs_tol=0.01) and math.isclose(h, 53.13, abs_tol=0.01)


def test_angular_size_depends_on_ratio_only():
    m = load()
    a, b = m.plane_angular_size_deg(2.0, 2.0, 1.5), m.plane_angular_size_deg(1.0, 1.0, 1.5)
    assert all(math.isclose(x, y) for x, y in zip(a, b))


def test_natural_height_shows_sensor_fov_one_to_one():
    m = load()
    # image aspect 2:1 (one 1280x640 cell), sensor 69 deg wide -> plane must span 69 deg
    h = m.natural_height_m(aspect=2.0, distance_m=1.0, sensor_hfov_deg=69.0)
    _, horiz = m.plane_angular_size_deg(h, 1.0, 2.0)
    assert math.isclose(horiz, 69.0, abs_tol=1e-6)


def test_validate_rejects_nonpositive_and_absurd():
    m = load()
    for bad in ((0, 1), (-1, 1), (1, 0), (1, -2), (float("nan"), 1), (1, float("inf")), (50, 1)):
        try:
            m.validate_plane(*bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert m.validate_plane(1.4, 1.0) == (1.4, 1.0)


def test_validate_warns_when_plane_exceeds_headset_fov():
    m = load()
    assert m.plane_exceeds_headset(3.0, 1.0, 1.0)      # ~113 deg tall
    assert not m.plane_exceeds_headset(1.4, 1.0, 1.0)  # ~70 deg


def test_resolve_height_auto_and_numeric():
    m = load()
    h = m.resolve_plane_height("auto", aspect=1.05, distance_m=1.0)
    _, horiz = m.plane_angular_size_deg(h, 1.0, 1.05)
    assert math.isclose(horiz, m.D435I_RGB_HFOV_DEG, abs_tol=1e-6)
    assert m.resolve_plane_height("1.25", aspect=1.05, distance_m=1.0) == 1.25
    assert m.resolve_plane_height(None, aspect=1.05, distance_m=1.0) == 1.0
    try:
        m.resolve_plane_height("abc", aspect=1.0, distance_m=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_describe_mentions_degrees():
    m = load()
    txt = m.describe_plane(1.0, 1.0, 1.05)
    assert "53" in txt and "°" in txt
