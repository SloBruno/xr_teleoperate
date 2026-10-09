"""Passive, simulation-only startup probe for the G1/Dex3 teleop path.

This module deliberately owns no DDS writer.  The launcher invokes it only for
``--sim --prearm-smoke`` and exits after the proof below succeeds.
"""

from __future__ import annotations

import time


class PrearmOutputBoundary:
    """Fail closed if a pre-arm probe gained any output authority."""

    def __init__(self):
        self._summary = {
            "rt/arm_sdk": False,
            "rt/lowcmd": False,
            "dex3_command": False,
            "process_writer": False,
            "motion_control": False,
        }

    @staticmethod
    def _alive(value):
        return value is not None and bool(getattr(value, "is_alive", lambda: False)())

    def assert_passive(self, arm_controller, dex3_controller):
        arm_publisher = getattr(arm_controller, "lowcmd_publisher", None)
        dex3_publishers = (
            getattr(dex3_controller, "LeftHandCmb_publisher", None),
            getattr(dex3_controller, "RightHandCmb_publisher", None),
        )
        arm_writer = getattr(arm_controller, "publish_thread", None)
        dex3_writer = getattr(dex3_controller, "hand_control_process", None)
        violations = {
            "rt/arm_sdk": arm_publisher is not None,
            "rt/lowcmd": arm_publisher is not None,
            "dex3_command": any(publisher is not None for publisher in dex3_publishers),
            "process_writer": self._alive(arm_writer) or self._alive(dex3_writer),
            "motion_control": bool(getattr(arm_controller, "motion_mode", False)),
        }
        if any(violations.values()) or bool(getattr(arm_controller, "outputs_activated", False)) or bool(
            getattr(dex3_controller, "outputs_activated", False)
        ):
            raise RuntimeError(f"pre-arm output boundary violated: {violations}")
        self._summary = violations

    def summary(self):
        return dict(self._summary)


def _require_frame(client, getter_name, deadline_s=3.0):
    deadline = time.monotonic() + deadline_s
    getter = getattr(client, getter_name)
    while time.monotonic() < deadline:
        frame = getter()
        if frame is not None and getattr(frame, "bgr", None) is not None:
            return frame
        time.sleep(0.02)
    raise RuntimeError(f"fake TeleImager did not provide {getter_name} before timeout")


def run_sim_prearm_smoke(image_client, arm_controller, dex3_controller, camera_layout):
    """Verify the config/JPEG protocol and receive-only DDS startup, then prove no outputs."""
    camera_config = image_client.get_cam_config()
    required = ("head_camera", "left_wrist_camera")
    if any(name not in camera_config for name in required):
        raise RuntimeError("camera config is missing the head or left-wrist stream")
    if not camera_config["head_camera"].get("enable_zmq"):
        raise RuntimeError("pre-arm smoke requires head_camera.enable_zmq=true")
    _require_frame(image_client, "get_head_frame")
    if camera_layout == "vertical":
        if not camera_config["left_wrist_camera"].get("enable_zmq"):
            raise RuntimeError("vertical camera layout requires left_wrist_camera.enable_zmq=true")
        _require_frame(image_client, "get_left_wrist_frame")

    proof = PrearmOutputBoundary()
    proof.assert_passive(arm_controller, dex3_controller)
    return proof.summary()
