def test_head_layout_never_requests_wrist_frames_even_while_recording():
    from teleop.utils.xr_video_policy import resolve_camera_read_plan

    plan = resolve_camera_read_plan(
        "head", need_local_image=True, record=True, frame_due=True
    )

    assert plan.head is True
    assert plan.render is True
    assert plan.left_wrist is False
    assert plan.right_wrist is False


def test_head_layout_skips_all_frame_reads_between_ten_hz_deadlines():
    from teleop.utils.xr_video_policy import VideoFrameGate, resolve_camera_read_plan

    gate = VideoFrameGate(max_fps=10.0)
    due = [gate.due(now) for now in (5.0, 5.03, 5.099, 5.1, 5.15, 5.2)]
    plans = [
        resolve_camera_read_plan(
            "head", need_local_image=True, record=False, frame_due=is_due
        )
        for is_due in due
    ]

    assert due == [True, False, False, True, False, True]
    assert [plan.head for plan in plans] == due
    assert [plan.render for plan in plans] == due


def test_frame_gate_does_not_sleep_or_change_control_loop_timing(monkeypatch):
    from teleop.utils.xr_video_policy import VideoFrameGate

    def fail_sleep(_seconds):
        raise AssertionError("video gate must not sleep in the control loop")

    monkeypatch.setattr("time.sleep", fail_sleep)
    gate = VideoFrameGate(max_fps=10.0)

    assert gate.due(1.0)
    assert not gate.due(1.01)


def test_legacy_vertical_plan_preserves_all_existing_reads():
    from teleop.utils.xr_video_policy import resolve_camera_read_plan

    plan = resolve_camera_read_plan(
        "vertical", need_local_image=True, record=True, frame_due=False
    )

    assert (plan.head, plan.left_wrist, plan.right_wrist, plan.render) == (
        True, True, True, True
    )


def test_unlimited_gate_preserves_legacy_every_cycle_behavior():
    from teleop.utils.xr_video_policy import VideoFrameGate

    gate = VideoFrameGate(max_fps=None)
    assert [gate.due(now) for now in (1.0, 1.0, 1.001)] == [True, True, True]


class _FakeClient:
    def __init__(self):
        self.calls = []

    def get_head_frame(self):
        self.calls.append("head")
        return type("Frame", (), {"bgr": "binocular-head"})()

    def get_left_wrist_frame(self):
        self.calls.append("left_wrist")
        return type("Frame", (), {"bgr": "left"})()

    def get_right_wrist_frame(self):
        self.calls.append("right_wrist")
        return type("Frame", (), {"bgr": "right"})()


def test_head_service_reads_and_renders_only_binocular_head_frame():
    from teleop.utils.xr_video_policy import service_xr_video

    client = _FakeClient()
    rendered = []
    update = service_xr_video(
        client=client,
        camera_config={"head_camera": {"enable_zmq": True}},
        camera_layout="head",
        need_local_image=True,
        record=True,
        frame_due=True,
        renderer=rendered.append,
    )

    assert client.calls == ["head"]
    assert rendered == ["binocular-head"]
    assert update.displayed_frame == "binocular-head"
    assert update.left_wrist is None and update.right_wrist is None


def test_head_service_caps_reads_and_render_copies_to_ten_hz_at_thirty_hz_loop():
    from teleop.utils.xr_video_policy import VideoFrameGate, service_xr_video

    client = _FakeClient()
    rendered = []
    gate = VideoFrameGate(max_fps=10.0)
    for cycle in range(30):
        service_xr_video(
            client=client,
            camera_config={"head_camera": {"enable_zmq": True}},
            camera_layout="head",
            need_local_image=True,
            record=False,
            frame_due=gate.due(cycle / 30.0),
            renderer=rendered.append,
        )

    assert client.calls == ["head"] * 10
    assert rendered == ["binocular-head"] * 10


def test_head_service_marks_only_new_reads_as_fresh_for_recording():
    from teleop.utils.xr_video_policy import service_xr_video

    client = _FakeClient()
    first = service_xr_video(
        client=client,
        camera_config={"head_camera": {"enable_zmq": True}},
        camera_layout="head",
        need_local_image=True,
        record=True,
        frame_due=True,
        renderer=lambda _frame: None,
    )
    repeated = service_xr_video(
        client=client,
        camera_config={"head_camera": {"enable_zmq": True}},
        camera_layout="head",
        need_local_image=True,
        record=True,
        frame_due=False,
        renderer=lambda _frame: None,
        head=first.head,
    )

    assert first.frame_fresh is True
    assert repeated.frame_fresh is False


def test_head_only_runtime_requires_binocular_camera_config():
    from teleop.utils.xr_video_policy import validate_head_only_camera

    validate_head_only_camera({"binocular": True, "enable_zmq": True})
    for config in (
        {"binocular": False, "enable_zmq": True},
        {"enable_zmq": True},
    ):
        try:
            validate_head_only_camera(config)
        except RuntimeError as exc:
            assert "binocular=true" in str(exc)
        else:
            raise AssertionError("head-only runtime must fail closed without binocular=true")