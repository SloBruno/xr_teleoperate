import asyncio
import importlib

import numpy as np
import pytest


class StopAfterSceneSetup(Exception):
    pass


class FakeSession:
    def __init__(self):
        self.upserts = []

    def upsert(self, component, **kwargs):
        self.upserts.append((component, kwargs))


@pytest.mark.parametrize(
    "scene_name",
    [
        "main_image_binocular_zmq",
        "main_image_monocular_zmq",
        "main_image_binocular_webrtc",
        "main_image_monocular_webrtc",
        "main_image_binocular_zmq_ego",
        "main_image_monocular_zmq_ego",
        "main_image_binocular_webrtc_ego",
        "main_image_monocular_webrtc_ego",
        "main_pass_through",
    ],
)
def test_hand_tracking_scene_upserts_hands_and_motion_controllers(monkeypatch, scene_name):
    module = importlib.import_module("televuer.televuer")
    monkeypatch.setattr(module, "Hands", lambda **kwargs: ("Hands", kwargs))
    monkeypatch.setattr(module, "MotionControllers", lambda **kwargs: ("MotionControllers", kwargs))

    async def stop_after_scene_setup(_):
        raise StopAfterSceneSetup

    monkeypatch.setattr(module.asyncio, "sleep", stop_after_scene_setup)
    viewer = module.TeleVuer.__new__(module.TeleVuer)
    viewer.use_hand_tracking = True
    viewer.display_fps = 30.0
    viewer.img2display = np.zeros((2, 4, 3), dtype=np.uint8)
    viewer.img_width = 2
    viewer.aspect_ratio = 1.0
    viewer.webrtc_url = "https://example.invalid/offer"
    session = FakeSession()

    with pytest.raises(StopAfterSceneSetup):
        asyncio.run(getattr(viewer, scene_name)(session))

    components = [component for component, _ in session.upserts]
    assert ("Hands", {"stream": True, "key": "hands", "hideLeft": True, "hideRight": True}) in components
    assert ("MotionControllers", {"stream": True, "key": "motionControllers", "left": True, "right": True}) in components
