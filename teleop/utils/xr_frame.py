"""Compose the image shown in the XR headset with a FIXED shape.

TeleVuer copies every rendered frame into a shared-memory buffer allocated once
with ``img_shape``. A frame of another shape raises inside TeleVuer's writer
thread and the headset video freezes for the rest of the session. This module
always returns a frame of exactly ``display_shape`` (or ``None`` = keep the last
frame on screen), whatever camera is missing.

Layouts:
  * ``vertical``: head camera on top, left-wrist camera below, separated by a
    dark gap, scaled to ``display_shape`` (historical dev layout).
  * ``head``: head camera only (single-camera / head-only Teleimager).
"""
import numpy as np

try:  # cv2 is always present in the teleop env; keep the module importable for tests.
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

GAP_HEIGHT = 30
DISPLAY_SCALE = 0.5
LAYOUTS = ("auto", "head", "vertical")


def resolve_layout(requested, left_wrist_enabled):
    """auto -> vertical when the left-wrist stream is enabled, else head."""
    if requested not in LAYOUTS:
        raise ValueError(f"unknown camera layout {requested!r}; use {'|'.join(LAYOUTS)}")
    if requested == "auto":
        return "vertical" if left_wrist_enabled else "head"
    if requested == "vertical" and not left_wrist_enabled:
        raise RuntimeError("camera layout 'vertical' requires left_wrist_camera.enable_zmq=true; use head/auto")
    return requested


def display_shape_for(layout, head_shape, wrist_shape=None):
    """[height, width] of the XR display buffer for the given layout."""
    hh, hw = int(head_shape[0]), int(head_shape[1])
    if layout == "vertical":
        wh = int(wrist_shape[0])
        return [int((hh + GAP_HEIGHT + wh) * DISPLAY_SCALE), int(hw * DISPLAY_SCALE)]
    return [hh, hw]


def _bgr(image):
    bgr = getattr(image, "bgr", None) if image is not None else None
    if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.size == 0:
        return None
    return bgr


def _fit(frame, display_shape):
    h, w = int(display_shape[0]), int(display_shape[1])
    if frame.shape[:2] == (h, w):
        return np.ascontiguousarray(frame)
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def compose_xr_frame(layout, head_img, left_wrist_img, display_shape):
    """Return a uint8 BGR frame of exactly display_shape, or None to keep the last one."""
    head = _bgr(head_img)
    if head is None:
        return None
    if layout == "vertical":
        wrist = _bgr(left_wrist_img)
        if wrist is None:
            return None  # never show a frame of the wrong shape; keep the last good one
        if wrist.shape[1] != head.shape[1]:
            wrist = cv2.resize(wrist, (head.shape[1], max(1, round(wrist.shape[0] * head.shape[1] / wrist.shape[1]))),
                               interpolation=cv2.INTER_AREA)
        gap = np.zeros((GAP_HEIGHT, head.shape[1], 3), dtype=head.dtype)
        return _fit(np.vstack((head, gap, wrist)), display_shape)
    return _fit(head, display_shape)
