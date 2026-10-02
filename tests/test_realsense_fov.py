"""Read-only RealSense color-profile/FOV inspection helpers."""
import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / "teleop" / "utils" / "realsense_fov.py"


def load():
    spec = importlib.util.spec_from_file_location("realsense_fov", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeIntrinsics:
    width = 1280
    height = 720
    fx = 910.0
    fy = 910.0
    ppx = 640.0
    ppy = 360.0


class FakeProfile:
    def as_video_stream_profile(self):
        return self

    def get_intrinsics(self):
        return FakeIntrinsics()

    def stream_type(self):
        return "color"

    def format(self):
        return "bgr8"

    def fps(self):
        return 15


class FakePipelineProfile:
    def get_stream(self, stream):
        assert stream == "color"
        return FakeProfile()


class FakeConfig:
    def __init__(self):
        self.device = None
        self.stream = None

    def enable_device(self, serial):
        self.device = serial

    def enable_stream(self, stream, width, height, fmt, fps):
        self.stream = (stream, width, height, fmt, fps)

    def resolve(self, wrapper):
        assert wrapper == "wrapper"
        return FakePipelineProfile()


class FakeRs:
    class stream:
        color = "color"

    class format:
        bgr8 = "bgr8"

    def __init__(self):
        self.config_instance = FakeConfig()

    def config(self):
        return self.config_instance

    def pipeline_wrapper(self, pipeline):
        assert pipeline == "pipeline"
        return "wrapper"


def test_fov_from_intrinsics_matches_pinhole_geometry():
    m = load()
    horizontal, vertical = m.fov_from_intrinsics(FakeIntrinsics())
    assert round(horizontal, 1) == 70.2
    assert round(vertical, 1) == 43.2


def test_resolve_color_profile_requests_existing_stream_without_starting_pipeline():
    m = load()
    rs = FakeRs()
    profile, intrinsics = m.resolve_color_profile(
        rs, pipeline="pipeline", serial="243122072230", width=1280, height=720, fps=15
    )
    assert isinstance(profile, FakeProfile)
    assert isinstance(intrinsics, FakeIntrinsics)
    assert rs.config_instance.device == "243122072230"
    assert rs.config_instance.stream == ("color", 1280, 720, "bgr8", 15)


def test_describe_identifies_requested_profile_and_non_optical_limit():
    m = load()
    text = m.describe_profile("243122072230", FakeProfile(), FakeIntrinsics())
    assert "1280x720 @ 15" in text
    assert "70.2° H x 43.2° V" in text
    assert "não altera o FOV óptico" in text
