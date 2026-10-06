"""Geometry helpers for the XR video plane (Vuer ImageBackground height/distance).

Vuer places a head-locked plane of size (height*aspect) x height at distanceToCamera
metres in front of the headset, so only height/distance matters for degrees covered.
Defaults (1.0 m at 1.0 m) reproduce the historical TeleVuer behaviour.
"""
import math

DEFAULT_HEIGHT_M = 1.0
DEFAULT_DISTANCE_M = 1.0
MAX_PLANE_M = 20.0
# Intel D400 datasheet: D435/D435i RGB sensor FOV 69.4 x 42.5 (deg, H x V).
D435I_RGB_HFOV_DEG = 69.4
# Quest 3 per-eye FOV is roughly 110 deg H x 96 deg V; stay conservative for the warning.
HEADSET_COMFORT_FOV_DEG = 90.0


def plane_angular_size_deg(height_m, distance_m, aspect):
    """Return (vertical_deg, horizontal_deg) subtended by the plane."""
    v = 2 * math.degrees(math.atan((height_m / 2) / distance_m))
    h = 2 * math.degrees(math.atan((height_m * aspect / 2) / distance_m))
    return v, h


def natural_height_m(aspect, distance_m, sensor_hfov_deg):
    """Plane height that makes the plane's horizontal extent equal the sensor's HFOV."""
    width = 2 * distance_m * math.tan(math.radians(sensor_hfov_deg) / 2)
    return width / aspect


def validate_plane(height_m, distance_m):
    for name, val in (("height", height_m), ("distance", distance_m)):
        if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0 or val > MAX_PLANE_M:
            raise ValueError(f"video plane {name} must be in (0, {MAX_PLANE_M}] metres, got {val!r}")
    return float(height_m), float(distance_m)


def plane_exceeds_headset(height_m, distance_m, aspect):
    v, h = plane_angular_size_deg(height_m, distance_m, aspect)
    return v > HEADSET_COMFORT_FOV_DEG or h > HEADSET_COMFORT_FOV_DEG


def resolve_plane_height(value, aspect, distance_m, sensor_hfov_deg=D435I_RGB_HFOV_DEG):
    """None -> default; 'auto' -> 1:1 angular match of the sensor HFOV; else a float in metres."""
    if value is None:
        return DEFAULT_HEIGHT_M
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return natural_height_m(aspect, distance_m, sensor_hfov_deg)
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"video plane height must be a number or 'auto', got {value!r}") from None
    return float(value)


def describe_plane(height_m, distance_m, aspect):
    v, h = plane_angular_size_deg(height_m, distance_m, aspect)
    return f"XR video plane {height_m:.2f} m high at {distance_m:.2f} m -> {v:.0f}° V x {h:.0f}° H in the headset"
