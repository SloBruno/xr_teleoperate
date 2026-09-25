from pathlib import Path


def test_teleop_main_loop_continuously_observes_controller_camera_and_dex3_status():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py").read_text()

    assert "from teleop.utils.teleop_status import AsyncStatusFileSink, TeleopStatusMonitor, camera_frame_is_usable" in source
    assert "head_img = None" in source
    assert "left_wrist_img = None" in source
    assert "camera_frame_is_usable(head_img)" in source
    assert "AsyncStatusFileSink" in source
    assert "status_sink = AsyncStatusFileSink" in source
    assert "status_monitor = TeleopStatusMonitor(status_sink.emit)" in source
    assert "lambda payload: logger_mp.info(payload)" not in source


def test_teleop_main_loop_records_full_pose_without_serializing_in_control_path():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py").read_text()

    assert "PoseTelemetryJsonlSink" in source
    assert "build_pose_record(" in source
    assert 'left_wrist_pose=getattr(tele_data, "left_wrist_pose", None)' in source
    assert 'right_wrist_pose=getattr(tele_data, "right_wrist_pose", None)' in source
    assert "measured_arm_q=current_lr_arm_q" in source
    assert "commanded_arm_q=None" in source
    assert "commanded_arm_q_reason=commanded_arm_q_reason" in source
    assert "arm_command_request_id=arm_request_id" in source
    assert "selected_arm_q=cycle.selected_q" in source
    assert "dex3_measured_q=dex3_measured_q" in source
    assert "dex3_commanded_q=dex3_commanded_q" in source
    assert "pose_telemetry_sink.emit" in source
    assert "json.dumps" not in source[source.index("while not STOP:", source.index("# main loop")):source.index("except KeyboardInterrupt")]


def test_cleanup_orders_actuator_shutdown_before_optional_telemetry_cleanup():
    source = (Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py").read_text()

    actuator_end = source.index("if hand_outputs_activated:")
    cleanup_telemetry = source.index("# Normal control-path telemetry preserves")
    assert actuator_end < cleanup_telemetry
    assert source.index("_close_telemetry_best_effort(pose_telemetry_sink", cleanup_telemetry) > cleanup_telemetry
    assert source.index("except KeyboardInterrupt:") < source.index("finally:")
