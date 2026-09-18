from pathlib import Path


def test_teleop_main_loop_continuously_observes_controller_camera_and_dex3_status():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py").read_text()

    assert "from teleop.utils.teleop_status import TeleopStatusMonitor, camera_frame_is_usable" in source
    assert "head_img = None" in source
    assert "left_wrist_img = None" in source
    assert "camera_frame_is_usable(head_img)" in source
    assert "camera_frame_is_usable(left_wrist_img)" in source
