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

class _FakeContext:
    def __init__(self):
        self.closed = False

    def term(self):
        self.closed = True


class _FakeRequester:
    instances = []

    def __init__(self, host, port, config=None, term_ok=True):
        self.host, self.port = host, port
        self._config = config
        self._term_ok = term_ok
        self._context = _FakeContext()
        self.closed = False
        _FakeRequester.instances.append(self)

    def request(self):
        if isinstance(self._config, Exception):
            raise self._config
        return self._config

    def close(self):
        self.closed = True
        if self._term_ok:
            self._context.term()


def test_one_shot_camera_config_closes_requester_and_terminates_context():
    from teleop.utils.xr_video_policy import fetch_camera_config_once

    config = {"head_camera": {"binocular": True}}
    seen = []
    factory = lambda host, port: seen.append(_FakeRequester(host, port, config)) or seen[-1]
    assert fetch_camera_config_once(factory, "10.0.0.2") == config
    assert (seen[0].host, seen[0].port) == ("10.0.0.2", 60000)
    assert seen[0].closed and seen[0]._context.closed


def test_one_shot_camera_config_closes_on_failure_and_fails_closed():
    import pytest
    from teleop.utils.xr_video_policy import fetch_camera_config_once

    for bad in (None, RuntimeError("boom")):
        seen = []
        factory = lambda host, port: seen.append(_FakeRequester(host, port, bad)) or seen[-1]
        with pytest.raises(RuntimeError):
            fetch_camera_config_once(factory, "h")
        assert seen[0].closed and seen[0]._context.closed


def test_one_shot_camera_config_refuses_a_context_left_alive():
    import pytest
    from teleop.utils.xr_video_policy import fetch_camera_config_once

    factory = lambda host, port: _FakeRequester(host, port, {"head_camera": {}}, term_ok=False)
    with pytest.raises(RuntimeError, match="ZMQ context"):
        fetch_camera_config_once(factory, "h")


def test_subscriber_config_must_match_the_config_used_to_build_the_xr_scene():
    import pytest
    from teleop.utils.xr_video_policy import require_matching_camera_config

    head = {"image_shape": [480, 1280], "binocular": True, "enable_zmq": True, "zmq_port": 55555}
    wrist = {"image_shape": [480, 640], "binocular": False, "enable_zmq": True, "zmq_port": 55556}
    expected = {"head_camera": head, "left_wrist_camera": wrist}
    require_matching_camera_config(expected, {"head_camera": dict(head), "left_wrist_camera": dict(wrist, zmq_port=1)},
                                   ("head_camera",))
    require_matching_camera_config(expected, {"head_camera": dict(head), "left_wrist_camera": dict(wrist)},
                                   ("head_camera", "left_wrist_camera"))
    with pytest.raises(RuntimeError, match="head_camera"):
        require_matching_camera_config(expected, {"head_camera": dict(head, image_shape=[480, 640])}, ("head_camera",))
    with pytest.raises(RuntimeError, match="left_wrist_camera"):
        require_matching_camera_config(expected, {"head_camera": dict(head), "left_wrist_camera": dict(wrist, binocular=True)},
                                       ("head_camera", "left_wrist_camera"))


def test_one_shot_camera_config_waits_for_exiting_zmq_threads_to_disappear():
    from teleop.utils.xr_video_policy import fetch_camera_config_once

    counts = iter([1, 3, 2, 1])  # before, then libzmq I/O threads still exiting after term()
    sleeps = []
    factory = lambda host, port: _FakeRequester(host, port, {"head_camera": {}})
    fetch_camera_config_once(factory, "h", native_thread_count=lambda: next(counts),
                             sleep=sleeps.append, settle_timeout_s=1.0)
    assert len(sleeps) == 2


def test_one_shot_camera_config_fails_closed_when_native_threads_never_exit():
    import pytest
    from teleop.utils.xr_video_policy import fetch_camera_config_once

    counts = iter([1] + [2] * 1000)
    factory = lambda host, port: _FakeRequester(host, port, {"head_camera": {}})
    with pytest.raises(RuntimeError, match="native threads"):
        fetch_camera_config_once(factory, "h", native_thread_count=lambda: next(counts),
                                 sleep=lambda s: None, settle_timeout_s=0.05, poll_s=0.01)


def test_fork_guard_accepts_only_a_single_threaded_process():
    import threading
    import pytest
    from teleop.utils.xr_video_policy import require_single_threaded_for_fork

    require_single_threaded_for_fork()
    stop = threading.Event()
    worker = threading.Thread(target=stop.wait, name="stray-worker", daemon=True)
    worker.start()
    try:
        with pytest.raises(RuntimeError, match="stray-worker"):
            require_single_threaded_for_fork()
    finally:
        stop.set()
        worker.join()


def test_subscriber_config_match_covers_webrtc_fields_and_missing_cameras():
    import pytest
    from teleop.utils.xr_video_policy import require_matching_camera_config

    head = {"image_shape": [480, 1280], "binocular": True, "enable_zmq": True, "zmq_port": 55555,
            "enable_webrtc": False, "webrtc_port": 60001}
    with pytest.raises(RuntimeError, match="enable_webrtc"):
        require_matching_camera_config({"head_camera": head}, {"head_camera": dict(head, enable_webrtc=True)},
                                       ("head_camera",))
    with pytest.raises(RuntimeError, match="webrtc_port"):
        require_matching_camera_config({"head_camera": head}, {"head_camera": dict(head, webrtc_port=1)},
                                       ("head_camera",))
    with pytest.raises(RuntimeError, match="left_wrist_camera"):
        require_matching_camera_config({"head_camera": head}, {"head_camera": dict(head)},
                                       ("head_camera", "left_wrist_camera"))


def test_one_shot_camera_config_keeps_the_request_error_when_close_also_fails():
    import pytest
    from teleop.utils.xr_video_policy import fetch_camera_config_once

    factory = lambda host, port: _FakeRequester(host, port, RuntimeError("request boom"), term_ok=False)
    with pytest.raises(RuntimeError, match="request boom"):
        fetch_camera_config_once(factory, "h")


def test_fork_guard_allows_the_logging_mp_threads_present_since_import():
    # Real robot: importing teleop starts LogConsole/LogListener and the first log
    # line starts QueueFeederThread. Pass-through has always forked beside them.
    import threading
    import pytest
    from teleop.utils.xr_video_policy import require_single_threaded_for_fork

    stop = threading.Event()
    workers = [threading.Thread(target=stop.wait, name=n, daemon=True)
               for n in ("LogConsole", "LogListener", "QueueFeederThread")]
    for w in workers:
        w.start()
    try:
        require_single_threaded_for_fork()
        stray = threading.Thread(target=stop.wait, name="dds-reader", daemon=True)
        stray.start()
        with pytest.raises(RuntimeError, match="dds-reader"):
            require_single_threaded_for_fork()
    finally:
        stop.set()
        for w in workers:
            w.join()


def test_one_shot_camera_config_primes_logging_before_the_native_baseline():
    # The requester logs; the first log starts QueueFeederThread (one native
    # thread). Priming before the baseline keeps that from looking like a ZMQ leak.
    from teleop.utils.xr_video_policy import fetch_camera_config_once

    events = []
    counts = iter([17, 17])
    factory = lambda host, port: _FakeRequester(host, port, {"head_camera": {}})
    fetch_camera_config_once(
        factory, "h", before_baseline=lambda: events.append("prime"),
        native_thread_count=lambda: events.append("count") or next(counts),
        sleep=lambda s: None)
    assert events[:2] == ["prime", "count"]
