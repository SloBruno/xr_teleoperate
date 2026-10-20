"""Small, non-blocking policies for XR camera reads and rendering."""

from dataclasses import dataclass
import math
import os
import time


@dataclass(frozen=True)
class CameraReadPlan:
    head: bool
    left_wrist: bool
    right_wrist: bool
    render: bool


@dataclass(frozen=True)
class VideoUpdate:
    head: object = None
    left_wrist: object = None
    right_wrist: object = None
    displayed_frame: object = None
    frame_fresh: bool = False


def validate_head_only_camera(head_camera_config):
    """Fail closed unless the lightweight head stream is binocular ZMQ."""
    if head_camera_config.get("binocular") is not True:
        raise RuntimeError("G1D head-only video requires head_camera.binocular=true")
    if head_camera_config.get("enable_zmq") is not True:
        raise RuntimeError("G1D head-only video requires head_camera.enable_zmq=true")


class VideoFrameGate:
    """Return whether one frame is due without sleeping the control loop."""

    def __init__(self, max_fps=None):
        if max_fps is not None and (not math.isfinite(max_fps) or max_fps <= 0):
            raise ValueError("max_fps must be finite and greater than zero")
        self._period = None if max_fps is None else 1.0 / float(max_fps)
        self._next_due = None

    def due(self, now):
        if self._period is None:
            return True
        if self._next_due is not None and now + 1e-12 < self._next_due:
            return False
        self._next_due = float(now) + self._period
        return True


def resolve_camera_read_plan(camera_layout, *, need_local_image, record, frame_due):
    """Choose camera reads; head layout never consumes either wrist stream."""
    if camera_layout == "head":
        active = bool(frame_due and (need_local_image or record))
        return CameraReadPlan(
            head=active,
            left_wrist=False,
            right_wrist=False,
            render=bool(frame_due and need_local_image),
        )
    if camera_layout != "vertical":
        raise ValueError(f"unsupported camera layout: {camera_layout}")
    return CameraReadPlan(
        head=bool(record or need_local_image),
        left_wrist=bool(record or need_local_image),
        right_wrist=bool(record),
        render=bool(need_local_image),
    )


def service_xr_video(*, client, camera_config, camera_layout, need_local_image,
                     record, frame_due, renderer, stack_vertical=None,
                     head=None, left_wrist=None, right_wrist=None):
    """Perform one policy-controlled camera update and optional XR copy."""
    plan = resolve_camera_read_plan(
        camera_layout,
        need_local_image=need_local_image,
        record=record,
        frame_due=frame_due,
    )
    if plan.head and camera_config["head_camera"]["enable_zmq"]:
        head = client.get_head_frame()
    if plan.left_wrist and camera_config["left_wrist_camera"]["enable_zmq"]:
        left_wrist = client.get_left_wrist_frame()
    if plan.right_wrist and camera_config["right_wrist_camera"]["enable_zmq"]:
        right_wrist = client.get_right_wrist_frame()

    displayed_frame = None
    if plan.render:
        if camera_layout == "head":
            displayed_frame = getattr(head, "bgr", None)
        else:
            if stack_vertical is None:
                raise ValueError("vertical camera layout requires stack_vertical")
            displayed_frame = stack_vertical(head, left_wrist)
        if displayed_frame is not None:
            renderer(displayed_frame)

    return VideoUpdate(
        head=head,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        displayed_frame=displayed_frame,
        frame_fresh=bool(plan.head),
    )


CAMERA_CONFIG_REQUEST_PORT = 60000
_SCENE_CAMERA_KEYS = ("image_shape", "binocular", "enable_zmq", "zmq_port")


def _native_thread_count():
    """Kernel tasks of this process (Linux); None where /proc is unavailable."""
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return None


def fetch_camera_config_once(requester_factory, host, port=CAMERA_CONFIG_REQUEST_PORT, *,
                             native_thread_count=_native_thread_count, sleep=time.sleep,
                             settle_timeout_s=1.0, poll_s=0.01):
    """One-shot camera-config request that leaves no ZMQ context/I/O threads alive.

    TeleVuer forks its Vuer server in __init__, so the XR scene must be built
    while this process is single-threaded; a live zmq.Context owns I/O threads.
    libzmq's I/O threads can still be exiting right after term() returns, so
    wait (bounded) until the native thread count is back to its pre-request
    value, and fail closed otherwise.
    """
    baseline = native_thread_count()
    requester = requester_factory(host, port)
    try:
        config = requester.request()
    finally:
        requester.close()
        context = getattr(requester, "_context", None)
        if context is not None and not getattr(context, "closed", True):
            raise RuntimeError("camera-config requester left its ZMQ context alive")
    if baseline is not None:
        polls = max(1, math.ceil(settle_timeout_s / poll_s))
        for _ in range(polls):
            current = native_thread_count()
            if current is None or current <= baseline:
                break
            sleep(poll_s)
        else:
            raise RuntimeError(
                f"camera-config request left native threads alive ({current} > {baseline})")
    if config is None:
        raise RuntimeError("Failed to get camera configuration.")
    return config


def require_matching_camera_config(expected, actual, camera_names):
    """Fail closed if the subscriber's config differs from the one used for the XR scene."""
    for name in camera_names:
        want = expected.get(name) or {}
        got = (actual or {}).get(name) or {}
        for key in _SCENE_CAMERA_KEYS:
            a, b = want.get(key), got.get(key)
            if isinstance(a, tuple):
                a = list(a)
            if isinstance(b, tuple):
                b = list(b)
            if a != b:
                raise RuntimeError(
                    f"{name}.{key} changed between camera-config requests ({a!r} -> {b!r})")
