"""Guard: in every mode the Vuer process must be forked before any DDS / keyboard /
ZMQ subscriber thread exists (G1D: child stuck in futex_wait, 8012 not listening).
Runtime proof lives in tests/test_vuer_fork_single_threaded.py."""
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py").read_text()


def _first(pattern, start=0):
    m = re.compile(pattern).search(SRC, start)
    assert m, pattern
    return m.start()


def test_passthrough_tv_wrapper_created_before_threads_and_clients():
    try_pos = _first(r"\n    try:\n        # TeleVuer forks its Vuer server Process")
    early = _first(r"tv_wrapper = TeleVuerWrapper\(", try_pos)
    for later in (r"ChannelFactoryInitialize\(1", r"ChannelFactoryInitialize\(0",
                  r"IPC_Server\(on_press", r"threading\.Thread\(target=listen_keyboard",
                  r"ImageClient\(host="):
        assert early < _first(later, try_pos), later
    guard = SRC[try_pos:early]
    assert "args.display_mode == 'pass-through'" in guard


def test_late_wrapper_only_when_not_already_created():
    assert "if tv_wrapper is None:" in SRC


def test_video_tv_wrapper_also_created_before_threads_and_subscriber():
    try_pos = _first(r"\n    try:\n        # TeleVuer forks its Vuer server Process")
    config = _first(r"fetch_camera_config_once\(ZMQ_Requester", try_pos)
    video_tv = _first(r"tv_wrapper = TeleVuerWrapper\(", _first(r"if tv_wrapper is None:", try_pos))
    assert config < video_tv
    for later in (r"ChannelFactoryInitialize\(0", r"threading\.Thread\(target=listen_keyboard",
                  r"ImageClient\(host="):
        assert video_tv < _first(later, try_pos), later
