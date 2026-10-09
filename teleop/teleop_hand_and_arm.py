import time
import argparse
import signal
from multiprocessing import Value, Array, Lock
import threading
import cv2
import numpy as np
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)


def _log_best_effort(level, message):
    """Never let diagnostic logging interrupt lifecycle cleanup."""
    try:
        getattr(logger_mp, level)(message)
    except BaseException:
        pass

import os 
from teleop.utils.com_monitor import create_from_env as create_com_monitor_from_env
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController, H2_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper, is_walk_fsm
from teleop.utils.quest_controls import (LocomotionRamp, dispatch_joystick_locomotion, joystick_to_locomotion, stick_snapshot,
                                         resolve_speed_caps, speed_cap_banner, loco_stick_is_fresh)
from teleop.utils.loco_preflight import run_loco_preflight
from teleop.utils.quest_safety import controller_sample_is_fresh, fresh_controller_value
from teleop.utils.controller_wrist_calibration import ControllerWristCalibrator
from teleop.utils.human_arm_calibration import HumanArmSweep, HumanCalibratedWristCalibrator
from teleop.utils.arm_command_gate import publish_if_authorized
from teleop.utils.arm_tracking_orchestration import build_arm_recording_actions, run_arm_tracking_cycle
from teleop.utils.ee_rate_limiter import DualEePoseRateLimiter, G1_29_EE_RATE_LIMITER_CONFIG
from teleop.utils.arm_enable_ramp import ArmEnableRamp, DEFAULT_ENABLE_RAMP_S
from teleop.utils.arm_graceful_shutdown import run_graceful_arm_shutdown
from teleop.utils import dex3_shutdown_hand
from teleop.utils.pose_stream import PoseStreamSender
from teleop.utils import torso_lean
from teleop.utils.xr_video_plane import (
    describe_plane, plane_exceeds_headset, resolve_plane_height, validate_plane,
    DEFAULT_DISTANCE_M,
)
from teleop.utils.robot_state_monitor import RobotStateMonitor, stop_locomotion_best_effort
from teleop.utils import balance_telemetry
from teleop.utils.teleop_status import (
    AsyncStatusFileSink,
    TeleopStatusMonitor,
    camera_frame_is_usable,
    camera_status_for_layout,
    create_status_sink,
)
# from teleop.utils.teleop_status import AsyncStatusFileSink, TeleopStatusMonitor, camera_frame_is_usable
from teleop.utils.full_pose_telemetry import (
    PoseTelemetryJsonlSink,
    ArmPublicationTelemetryBridge,
    build_lifecycle_event,
    build_pose_record,
    create_pose_telemetry_sink,
    emit_lifecycle_event_best_effort,
    emit_pose_record_best_effort,
)
from teleop.utils.loop_diagnostics import LoopDiagnostics, ResourceSampler, GcWatcher
from teleop.utils.dex3_telemetry import Dex3SlowFieldGate, collect_extended_payload
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
ARM_REQUEST_TIMESTAMP = 0.0  # Monotonic time of the most recent terminal r request
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
PREPARATION_COMPLETE = False  # Launcher-time arm preparation has completed.
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
# Serializes r/q lifecycle decisions with the one-way output activation gate.
LIFECYCLE_LOCK = threading.Lock()
arm_calibration = None  # G1_29-only controller-to-wrist calibration state.
LIFECYCLE_EVENTS = []
# Optional pre-arm, read-only operator calibration. It is unavailable once
# tracking has been requested.
HUMAN_SWEEP_REQUESTED = False
HUMAN_SWEEP_ACTIVE = False
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def _request_start_locked():
    """Authorize tracking after preparation, without activating any output."""
    global START, ARM_REQUEST_TIMESTAMP
    LIFECYCLE_EVENTS.append("start_requested")
    if STOP:
        # Stop is terminal: a late r/A during graceful shutdown is ignored.
        return False
    if not PREPARATION_COMPLETE:
        logger_mp.warning("[lifecycle] Ignoring start until arm preparation completes.")
        return False
    if START:
        # Key repeat or a second controller edge must not erase a calibration
        # that the active tracking loop cannot recreate.
        return True
    if HUMAN_SWEEP_ACTIVE or HUMAN_SWEEP_REQUESTED:
        logger_mp.warning("[human-calib] Ignoring start while the arm sweep is running.")
        LIFECYCLE_EVENTS.append("start_refused_human_sweep_active")
        return False
    ARM_REQUEST_TIMESTAMP = time.monotonic()
    if arm_calibration is not None:
        arm_calibration.reset_for_start_request(ARM_REQUEST_TIMESTAMP)
    START = True
    LIFECYCLE_EVENTS.append("start_accepted")
    return True


def _request_stop_locked():
    """Make the unconditional stop request."""
    global STOP, START, HUMAN_SWEEP_REQUESTED
    START = False
    STOP = True
    HUMAN_SWEEP_REQUESTED = False
    LIFECYCLE_EVENTS.append("stop_requested")


def _request_human_sweep_locked():
    """Request a pre-arm sweep; this path never commands an arm."""
    global HUMAN_SWEEP_REQUESTED
    if not isinstance(arm_calibration, HumanCalibratedWristCalibrator):
        return False
    if STOP or START or not PREPARATION_COMPLETE or not READY:
        LIFECYCLE_EVENTS.append("human_sweep_refused")
        return False
    if HUMAN_SWEEP_ACTIVE or HUMAN_SWEEP_REQUESTED:
        return False
    HUMAN_SWEEP_REQUESTED = True
    LIFECYCLE_EVENTS.append("human_sweep_requested")
    _log_best_effort("info", "[human-calib] STARTED: collecting 3 s arm sweep; move both straight arms slowly from down to forward.")
    return True


def poll_human_sweep_button(controller_sample, left_x_was_pressed):
    """Fresh left-controller X rising edge requests a pre-arm sweep."""
    if not controller_sample_is_fresh(controller_sample.controller_sample_timestamp):
        return left_x_was_pressed
    pressed = bool(getattr(controller_sample, "left_ctrl_aButton", False))
    if pressed and not left_x_was_pressed:
        with LIFECYCLE_LOCK:
            _request_human_sweep_locked()
    return pressed


def _emit_human_calibration_record(sink, event, payload_key, payload):
    if sink is None:
        return False
    try:
        record = build_lifecycle_event(event, timestamp=time.time(), timestamp_monotonic=time.monotonic())
        record[payload_key] = payload
        return bool(sink.emit(record))
    except Exception as error:
        _log_best_effort("warning", f"Failed to emit {event} telemetry: {type(error).__name__}")
        return False


def service_human_arm_sweep(sweep, calibrator, tele_data, now, sink):
    """Advance the read-only pre-arm sweep and install only an accepted fit."""
    global HUMAN_SWEEP_REQUESTED, HUMAN_SWEEP_ACTIVE
    with LIFECYCLE_LOCK:
        tracking = START or STOP
        if HUMAN_SWEEP_REQUESTED:
            HUMAN_SWEEP_REQUESTED = False
            if not tracking and not calibrator.calibrated and sweep.request(
                now, preparation_ready=PREPARATION_COMPLETE, tracking_active=tracking
            ):
                HUMAN_SWEEP_ACTIVE = True
        if not HUMAN_SWEEP_ACTIVE:
            return None
    result = sweep.observe(tele_data, now, tracking_active=tracking)
    if result is None:
        return None
    with LIFECYCLE_LOCK:
        HUMAN_SWEEP_ACTIVE = False
        can_install = not (START or STOP) and not calibrator.calibrated
    if result.accepted and can_install:
        calibrator.set_human_calibration(result)
        _log_best_effort("info", "[human-calib] ACCEPTED; assume the robot L pose, then press [r].")
    else:
        if can_install:
            calibrator.set_human_calibration(None)
        _log_best_effort("warning", f"[human-calib] REJECTED ({result.reason}); fixed k remains active.")
    _emit_human_calibration_record(sink, "human_arm_calibration", "human_arm_calibration", result.telemetry())
    return result


def enforce_l_pose_start_gate(calibrator, tele_data, arm_ctrl, arm_ik, sink):
    """With a fit, r requires a mapped controller L-pose within 5 cm of FK."""
    global START
    if calibrator is None or getattr(calibrator, "human_calibration", None) is None:
        return True
    measured = arm_ik.forward_kinematics(arm_ctrl.get_current_dual_arm_q())
    decision = calibrator.check_l_pose((tele_data.left_wrist_pose, tele_data.right_wrist_pose), measured)
    _emit_human_calibration_record(sink, "l_pose_start_gate", "l_pose_start_gate", decision.telemetry())
    if decision.accepted:
        return True
    with LIFECYCLE_LOCK:
        START = False
        LIFECYCLE_EVENTS.append("start_refused_l_pose")
    _log_best_effort("warning", f"[human-calib] START REFUSED: {decision.message()}")
    return False


def graceful_g1_29_shutdown(arm_ctrl, *, arm_ik=None, hand_ctrl=None, sink=None,
                            attempt_return=True, clock=time.monotonic, sleep=time.sleep,
                            hand_mode=None):
    """Terminal q/B, Ctrl+C, SIGTERM or exception: Dex3 close (default) or
    open, return home, release arms.

    ``hand_mode`` (None = env DEX3_SHUTDOWN_HAND, default ``close``): close|hold
    close/hold the Dex3 BEFORE the arm return and stop its writer after the
    weight ramp; ``open`` is the previous sequence unchanged. Only a hand
    controller with ``begin_shutdown_hand`` (Dex3) supports close/hold.
    Tracking must already have stopped. Bounded (worst case ~20 s) and never
    raises; the arm writer always ends deactivated.
    """
    global START, STOP
    try:
        with LIFECYCLE_LOCK:
            START = False
            STOP = True
    except BaseException:
        pass

    def emit(event, detail):
        _cleanup_telemetry_event_best_effort(sink, event, cause=_format_shutdown_detail(detail))
        _log_best_effort("info", f"[shutdown] {event} {detail}")

    gravity = getattr(arm_ik, "gravity_tauff", None) if arm_ik is not None else None
    try:
        _, open_hands, close_hands, release_hands = dex3_shutdown_hand.hand_shutdown_callbacks(
            hand_ctrl, hand_mode, log=lambda message: _log_best_effort("error", message))
    except BaseException as error:
        _log_best_effort("error", f"[Dex3 encerramento] modo indisponível ({type(error).__name__}); abrindo")
        open_hands = None if hand_ctrl is None else (getattr(hand_ctrl, "open_and_deactivate", None) or hand_ctrl.deactivate)
        close_hands = release_hands = None
    try:
        result = run_graceful_arm_shutdown(
            arm_ctrl,
            clock=clock,
            sleep=sleep,
            emit=emit,
            open_hands=open_hands,
            close_hands=close_hands,
            release_hands=release_hands,
            gravity_tauff=gravity,
            attempt_return=attempt_return,
        )
    except BaseException as error:
        _log_best_effort("error", f"Graceful arm shutdown failed: {type(error).__name__}")
        result = None
    if (close_hands is not None and hand_ctrl is not None and result is not None
            and result.deactivated and not (result.hands_released or result.hands_opened)):
        # Close/hold path whose final release failed: never leave the Dex3
        # writer running after the arm writer stopped.
        try:
            hand_ctrl.deactivate()
        except BaseException as error:
            _log_best_effort("error", f"Failed to deactivate Dex3 output: {error}")
    if result is None or not result.deactivated:
        # Last resort: never leave a writer running.
        if hand_ctrl is not None and (result is None or not (result.hands_opened or result.hands_released)):
            try:
                hand_ctrl.deactivate()
            except BaseException as error:
                _log_best_effort("error", f"Failed to deactivate Dex3 output: {error}")
        try:
            arm_ctrl.deactivate()
        except BaseException as error:
            _log_best_effort("error", f"Failed to deactivate arm output: {error}")
    return result


_NAN_POSE = np.full((4, 4), np.nan)


def _pose_stream_hands(cycle):
    """Operator wrist for the 8093 page = the target actually given to IK
    (torso frame when the lean is on), NaN when this cycle had none."""
    target = getattr(cycle, "ik_target", None)
    if target is None:
        return _NAN_POSE, _NAN_POSE
    return target[0], target[1]


def _format_shutdown_detail(detail):
    try:
        return ",".join(f"{key}={detail[key]}" for key in sorted(detail)) or None
    except BaseException:
        return None


def _emit_lifecycle_events(sink):
    """Move callback state markers to the nonblocking telemetry queue."""
    with LIFECYCLE_LOCK:
        events = list(LIFECYCLE_EVENTS)
        del LIFECYCLE_EVENTS[:]
    if sink is None:
        return
    for event in events:
        emit_lifecycle_event_best_effort(sink, event, warn=logger_mp.warning)


def _safe_emit_lifecycle_event(sink, event, *, cause=None):
    emit_lifecycle_event_best_effort(sink, event, cause=cause, warn=logger_mp.warning)


# The best-effort producer owns build_pose_record(...) and the sink handoff.


def _close_telemetry_best_effort(sink, label):
    """Cleanup-only guard: telemetry must not prevent later shutdown steps."""
    if sink is None:
        return
    try:
        sink.close()
    except BaseException as error:
        _log_best_effort("warning", f"Failed to close {label}: {type(error).__name__}")


def _cleanup_telemetry_event_best_effort(sink, event, *, cause=None):
    """Guard optional cleanup telemetry after actuator shutdown has started."""
    try:
        _safe_emit_lifecycle_event(sink, event, cause=cause)
    except BaseException as error:
        _log_best_effort("warning", f"Failed to emit cleanup telemetry {event}: {type(error).__name__}")


def install_sigterm_handler():
    """SIGTERM follows the q / Ctrl+C graceful path (see dex3_shutdown_hand)."""
    try:
        signal.signal(signal.SIGTERM, dex3_shutdown_hand.make_sigterm_handler(
            lambda: STOP, lambda message: _log_best_effort("warning", message)))
        return True
    except (ValueError, OSError):  # not the main thread
        return False


def on_press(key):
    global RECORD_TOGGLE
    with LIFECYCLE_LOCK:
        if key == 'r':
            _request_start_locked()
        elif key == 'q':
            _request_stop_locked()
        elif key == 'c':
            _request_human_sweep_locked()
        elif key == 's' and START == True:
            RECORD_TOGGLE = True
        else:
            logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")


def poll_controller_lifecycle(controller_sample, right_a_was_pressed, right_b_was_pressed):
    """Apply fresh right-controller A/B rising edges through the terminal gates."""
    if not controller_sample_is_fresh(controller_sample.controller_sample_timestamp):
        return right_a_was_pressed, right_b_was_pressed

    right_a_pressed = bool(getattr(controller_sample, "right_ctrl_aButton", False))
    right_b_pressed = bool(getattr(controller_sample, "right_ctrl_bButton", False))
    with LIFECYCLE_LOCK:
        if right_a_pressed and not right_a_was_pressed:
            _request_start_locked()
        if right_b_pressed and not right_b_was_pressed:
            _request_stop_locked()
    return right_a_pressed, right_b_pressed


def _publish_arm_target_if_authorized(arm_ctrl, q_target, tauff_target, target_accepted, sample_fresh):
    """Serialize final lifecycle authority and arm target publication.

    IK runs before this lock. STOP wins the final check: nothing is enqueued
    (never a final IK command); graceful shutdown then owns the writer.
    """
    return publish_if_authorized(
        arm_ctrl,
        q_target,
        tauff_target,
        target_accepted=target_accepted,
        sample_fresh=sample_fresh,
        lifecycle_lock=LIFECYCLE_LOCK,
        is_started=lambda: START,
        is_stopped=lambda: STOP,
    )

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }

def stack_camera_frames_vertical(top_frame, bottom_frame, scale=0.5,
                                 top_crop_bottom=0.89, bottom_crop_top=0.11,
                                 divider_px=4):
    """Crop the overlapping views, resize them equally, and stack with a clean seam."""
    if top_frame is None or bottom_frame is None:
        return None

    top_height, top_width = top_frame.shape[:2]
    bottom_height, bottom_width = bottom_frame.shape[:2]
    top_end = min(top_height, max(1, round(top_height * top_crop_bottom)))
    bottom_start = min(bottom_height - 1, max(0, round(bottom_height * bottom_crop_top)))
    top_frame = top_frame[:top_end]
    bottom_frame = bottom_frame[bottom_start:]

    target_width = max(1, round(top_width * scale))
    target_top_height = max(1, round(top_frame.shape[0] * target_width / top_width))
    target_bottom_height = max(1, round(bottom_frame.shape[0] * target_width / bottom_width))

    top_frame = cv2.resize(top_frame, (target_width, target_top_height), interpolation=cv2.INTER_AREA)
    bottom_frame = cv2.resize(bottom_frame, (target_width, target_bottom_height), interpolation=cv2.INTER_AREA)

    if divider_px > 0:
        divider = np.zeros((divider_px, target_width, 3), dtype=top_frame.dtype)
        return np.vstack((top_frame, divider, bottom_frame))
    return np.vstack((top_frame, bottom_frame))


def stack_camera_images_vertical(top_image, bottom_image, scale=0.5,
                                 top_crop_bottom=0.89, bottom_crop_top=0.11,
                                 divider_px=4):
    """Return a vertical camera stack, skipping unavailable camera images."""
    if top_image is None or bottom_image is None:
        return None
    return stack_camera_frames_vertical(
        top_image.bgr, bottom_image.bgr, scale, top_crop_bottom,
        bottom_crop_top, divider_px)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--camera-layout', type=str, choices=['head', 'vertical'], default='vertical', help='XR camera layout: head camera only, or head above left-wrist camera')
    parser.add_argument('--camera-scale', type=float, default=0.5, help='Scale used for each camera in the vertical XR layout (default: 0.5)')
    parser.add_argument('--head-crop-bottom', type=float, default=0.89, help='Fraction of the head-camera height retained before the seam')
    parser.add_argument('--wrist-crop-top', type=float, default=0.11, help='Fraction removed from the top of the wrist camera before the seam')
    parser.add_argument('--camera-divider-px', type=int, default=4, help='Dark divider thickness between camera views')
    parser.add_argument('--video-plane-height', type=str, default=None, help="XR video plane height in metres, or 'auto' to match the D435i RGB 69.4° HFOV 1:1 (default: 1.0 = historical)")
    parser.add_argument('--video-plane-distance', type=float, default=None, help='XR video plane distance in metres (default: 1.0 = historical; angular size depends on height/distance only)')
    # network parameters
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--walk-speed-cap', type=float, default=None, help='Walk cap m/s (default 0.3 when unset; run_g1_quest_dex3.sh passes 0.6; hard max 0.6; env G1_WALK_SPEED_CAP)')
    parser.add_argument('--loco-request-fsm', type=str, default='none', choices=['none', '500'], help='Opt-in: in the preflight only, request FSM 500 (SetFsmId) when the robot is in 501 and confirm by polling; never 801; default none = send nothing')
    parser.add_argument('--loco-backend', type=str, default='wirelesscontroller', choices=['setvelocity', 'wirelesscontroller'], help='Walking transport: wirelesscontroller (default: continuous 20 Hz rt/wirelesscontroller joystick state; fail-safe, no silent fallback) or setvelocity (legacy RPC 7105, only when forced)')
    parser.add_argument('--turn-rate-cap', type=float, default=None, help='Turn cap rad/s (default 0.3, hard max 1.0; env G1_TURN_RATE_CAP)')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    walk_cap, turn_cap = resolve_speed_caps(args.walk_speed_cap, args.turn_rate_cap, os.environ)
    loco_ramp = LocomotionRamp()  # accel slew + pulse debounce; safety zero bypasses it
    logger_mp.debug(f"args: {args}")
    outputs_activated = False
    hand_outputs_activated = False
    shutdown_return_home = True
    arm_ik = None
    pose_telemetry_sink = None
    dex3_slow_gate = Dex3SlowFieldGate(slow_every=10)
    status_sink = None
    loop_diag = None
    com_monitor = None
    img_client = None
    tv_wrapper = None
    shutdown_cause = None
    loco_wrapper = None
    loco_preflight = None
    robot_monitor = None
    balance_monitor = None
    com_status = None
    # Opt-in UDP side channel for tools/pose_compare_web.py (XR_POSE_STREAM=1);
    # None = disabled, loop unchanged. Never blocks / never raises.
    pose_stream = PoseStreamSender.from_env()
    if pose_stream is not None:
        logger_mp.info(f"[pose_stream] UDP -> {pose_stream.addr[0]}:{pose_stream.addr[1]} @ {1/pose_stream.min_period if pose_stream.min_period else 0:.0f} Hz")
    # Opt-in torso lean from the operator's head displacement (G1_TORSO_LEAN=1,
    # docs/torso_lean.md): waist PITCH/ROLL only, yaw held at the neutral
    # captured at r. None = OFF: the waist (motors 12-14) keeps exactly the
    # previous arm_sdk behaviour. Any invalid env value keeps it OFF.
    lean_cfg = None
    lean_session = None     # TorsoLeanSession after r, torso_lean.REFUSED, or None
    try:
        lean_cfg = torso_lean.config_from_env(os.environ)
    except torso_lean.TorsoLeanConfigError as e:
        logger_mp.error(f"[torso_lean] configuração rejeitada ({e}); inclinação do tronco DESLIGADA")
        lean_cfg = None
    logger_mp.info(lean_cfg.describe() if lean_cfg is not None else "Inclinação do tronco: DESLIGADA")
    install_sigterm_handler()
    if args.ee == "dex3":
        _dex3_shutdown_mode, _dex3_shutdown_warning = dex3_shutdown_hand.resolve_mode(
            os.environ.get(dex3_shutdown_hand.ENV_VAR))
        if _dex3_shutdown_warning:
            logger_mp.error(f"[Dex3 encerramento] {_dex3_shutdown_warning}")
        logger_mp.info(dex3_shutdown_hand.describe_mode(_dex3_shutdown_mode))

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])
        vertical_camera_stack = args.camera_layout == 'vertical'
        if vertical_camera_stack and not camera_config['left_wrist_camera']['enable_zmq']:
            raise RuntimeError("Vertical camera layout requires left_wrist_camera.enable_zmq=true")
        if not 0 < args.camera_scale <= 1:
            raise ValueError("--camera-scale must be greater than 0 and at most 1")
        if not 0 < args.head_crop_bottom <= 1:
            raise ValueError("--head-crop-bottom must be greater than 0 and at most 1")
        if not 0 <= args.wrist_crop_top < 1:
            raise ValueError("--wrist-crop-top must be at least 0 and less than 1")
        if args.camera_divider_px < 0:
            raise ValueError("--camera-divider-px cannot be negative")

        display_img_shape = camera_config['head_camera']['image_shape']
        display_binocular = camera_config['head_camera']['binocular']
        if vertical_camera_stack:
            head_height, head_width = camera_config['head_camera']['image_shape']
            wrist_height, wrist_width = camera_config['left_wrist_camera']['image_shape']
            target_width = max(1, round(head_width * args.camera_scale))
            cropped_head_height = max(1, round(head_height * args.head_crop_bottom))
            cropped_wrist_height = max(1, wrist_height - round(wrist_height * args.wrist_crop_top))
            scaled_head_height = max(1, round(cropped_head_height * target_width / head_width))
            scaled_wrist_height = max(1, round(cropped_wrist_height * target_width / wrist_width))
            display_img_shape = [scaled_head_height + args.camera_divider_px + scaled_wrist_height, target_width]
            display_binocular = False
            logger_mp.info(f"XR vertical camera layout enabled: display shape {display_img_shape}")

        _plane_aspect = display_img_shape[1] / display_img_shape[0]
        _plane_distance = DEFAULT_DISTANCE_M if args.video_plane_distance is None else args.video_plane_distance
        video_plane_height, video_plane_distance = validate_plane(
            resolve_plane_height(args.video_plane_height, _plane_aspect, _plane_distance), _plane_distance)
        logger_mp.info(describe_plane(video_plane_height, video_plane_distance, _plane_aspect))
        if plane_exceeds_headset(video_plane_height, video_plane_distance, _plane_aspect):
            logger_mp.warning("XR video plane exceeds ~90° of the headset FOV; edges may be cut off")

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=display_binocular,
                                     img_shape=display_img_shape,
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     arm_pose_source="controller",
                                     video_plane_height=video_plane_height,
                                     video_plane_distance=video_plane_distance
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            loco_wrapper = LocoClientWrapper(backend=args.loco_backend, walk_cap=walk_cap, turn_cap=turn_cap)
            logger_mp.warning(f"[loco] backend: {args.loco_backend}")
            # BotBrain-style preflight, Regular mode only (FSM 500/501, R1+X). SetFsmId(500) only if --loco-request-fsm 500.
            loco_preflight = run_loco_preflight(loco_wrapper, request_fsm=args.loco_request_fsm)
            logger_mp.info(f"[loco] preflight: {loco_preflight}")
            if loco_preflight.get("fsm_banner"):
                logger_mp.warning(f"[loco] {loco_preflight['fsm_banner']}")
                print(loco_preflight["fsm_banner"], flush=True)
            logger_mp.warning(speed_cap_banner(walk_cap, turn_cap))
            if loco_preflight["loco_enabled"]:
                loco_wrapper.start_move_sender()
            else:
                logger_mp.warning(f"[loco] LOCOMOTION DISABLED: {loco_preflight['message']}")
            # Side channel: rt/sportmodestate callback + 1 Hz GetFsmId thread on
            # its own client. Never touched by the control loop except snapshot().
            try:
                robot_monitor = RobotStateMonitor(fsm_reader=loco_wrapper.make_fsm_reader(), fsm_period_s=1.0)
                robot_monitor.start()
            except BaseException as e:
                robot_monitor = None
                logger_mp.warning(f"[loco] robot state monitor unavailable: {e}")
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        if lean_cfg is not None and (args.arm != "G1_29" or not args.motion):
            logger_mp.error("[torso_lean] requer --arm G1_29 e --motion (rt/arm_sdk); inclinação DESLIGADA")
            lean_cfg = None

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            arm_calibration = HumanCalibratedWristCalibrator(ControllerWristCalibrator())
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        elif args.arm == "H2":
            arm_ik = H2_ArmIK()
            arm_ctrl = H2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

        # Opt-in passive balance/IMU side channel (G1_BALANCE_TELEMETRY=1):
        # DDS readers only, lowstate reused from the arm reader thread.
        balance_monitor = balance_telemetry.create_from_env(os.environ, warn=logger_mp.warning)
        if balance_monitor is not None:
            balance_telemetry.attach_lowstate_tap(balance_monitor, arm_ctrl)
            logger_mp.info(f"[balance_telemetry] enabled at {balance_monitor.rate_hz:.0f} Hz")

        # end-effector
        xr_motion_data_ready = Value('b', False, lock=True)        # [input] whether XR hand/controller motion data has arrived
        if args.ee in ("dex3", "inspire_ftp", "inspire_dfx") and args.input_mode == "controller":
            raise ValueError(f"{args.ee} does not support controller input mode.")
        elif args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            left_ctrl_trigger_in = Value('d', 0.0, lock=True)      # legacy input retained for compatibility
            right_ctrl_trigger_in = Value('d', 0.0, lock=True)     # legacy input retained for compatibility
            left_ctrl_timestamp_in = Value('d', 0.0, lock=True)    # legacy input retained for compatibility
            right_ctrl_timestamp_in = Value('d', 0.0, lock=True)   # legacy input retained for compatibility
            left_ctrl_sample_in = Array('d', 2, lock=True)         # [trigger, monotonic timestamp]
            right_ctrl_sample_in = Array('d', 2, lock=True)        # [trigger, monotonic timestamp]
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready,
                                          left_ctrl_trigger_in=left_ctrl_trigger_in, right_ctrl_trigger_in=right_ctrl_trigger_in,
                                          left_ctrl_timestamp_in=left_ctrl_timestamp_in, right_ctrl_timestamp_in=right_ctrl_timestamp_in,
                                          left_ctrl_sample_in=left_ctrl_sample_in, right_ctrl_sample_in=right_ctrl_sample_in)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "hand":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_hand(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                                dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "controller":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_ctrl
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        status_log_path = os.environ.get(
            "XR_TELEOP_STATUS_LOG",
            "/home/unitree/.local/state/xr_teleoperate/teleop-status.jsonl",
        )
        # status_sink = AsyncStatusFileSink(status_log_path, logger_mp.warning)
        status_sink = create_status_sink(status_log_path, logger_mp.warning)
        status_monitor = TeleopStatusMonitor(status_sink.emit, warn=logger_mp.warning)
        pose_log_dir = os.environ.get(
            "XR_TELEOP_POSE_LOG_DIR",
            "/home/unitree/.local/state/xr_teleoperate",
        )
        pose_telemetry_sink = create_pose_telemetry_sink(pose_log_dir, logger_mp.warning)
        loop_diag = LoopDiagnostics(status_sink.emit, sampler=ResourceSampler(), gc_watcher=GcWatcher())
        loop_diag.start()
        com_monitor = create_com_monitor_from_env(
            os.environ, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "g1", "g1_body29_hand14.urdf"),
            args.arm, logger_mp.warning)
        arm_publication_telemetry = ArmPublicationTelemetryBridge(
            pose_telemetry_sink, profile=args.arm, warn=logger_mp.warning
        )

        # Match the original launcher behavior: connecting the arm motors moves
        # the arms to the all-zero preparation pose immediately, before r.
        # The same lock makes an early terminal q win over output activation.
        # Dex3 remains passive until the later post-r gate.
        with LIFECYCLE_LOCK:
            if STOP:
                logger_mp.info("Launcher cancelled before arm preparation.")
                raise KeyboardInterrupt
            # G1_29 is the only lifecycle-gated arm profile: activating the arm
            # output at launcher time and confirming all-zero arrival before r.
            # The other selectable arm profiles publish continuously from
            # construction, so no activate()/deactivate() call is made for them.
            if args.arm == "G1_29":
                arm_ctrl.activate()
                outputs_activated = True
                preparation_confirmed = arm_ctrl.ctrl_dual_arm_go_home()
                if not preparation_confirmed:
                    # Do not retry the failed motion at shutdown; the finally
                    # block only releases arm_sdk authority and deactivates.
                    shutdown_return_home = False
                    raise RuntimeError("Arm preparation pose was not reached; refusing tracking.")
            PREPARATION_COMPLETE = True
        _safe_emit_lifecycle_event(pose_telemetry_sink, "preparation_ready")

        # Initialize before the pre-arm loop: some display modes intentionally do
        # not fetch local frames, but their status must remain observable.
        head_img = None
        left_wrist_img = None

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        if args.arm == "G1_29":
            logger_mp.info("🟣 Optional: [c] or left-controller X starts a 3 s straight-arm sweep before [r].")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        right_a_was_pressed = False
        right_b_was_pressed = False
        left_x_was_pressed = False
        human_arm_sweep = HumanArmSweep() if args.arm == "G1_29" else None
        first_controller_targets = None
        # Pressing r is only an arm request. Keep the robot pre-armed until a
        # current controller-pose sample exists; zero-initialized pose buffers
        # must never be passed to IK.
        while not STOP:
            time.sleep(0.033)
            _emit_lifecycle_events(pose_telemetry_sink)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                if vertical_camera_stack:
                    left_wrist_img = img_client.get_left_wrist_frame()
                    stacked_img = stack_camera_images_vertical(
                        head_img, left_wrist_img, args.camera_scale,
                        args.head_crop_bottom, args.wrist_crop_top, args.camera_divider_px)
                    if stacked_img is not None:
                        tv_wrapper.render_to_xr(stacked_img)
                elif head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            ready_tele_data = tv_wrapper.get_tele_data()
            right_a_was_pressed, right_b_was_pressed = poll_controller_lifecycle(
                ready_tele_data, right_a_was_pressed, right_b_was_pressed)
            if human_arm_sweep is not None:
                left_x_was_pressed = poll_human_sweep_button(ready_tele_data, left_x_was_pressed)
                service_human_arm_sweep(
                    human_arm_sweep, arm_calibration, ready_tele_data, time.monotonic(), pose_telemetry_sink)
            ready_pressure_timestamps = (0.0, 0.0)
            ready_dex3_measured_q = None
            ready_dex3_commanded_q = None
            ready_dex3_metadata = None
            ready_dex3_extended = None
            if args.ee == "dex3":
                left_pressure_sample, right_pressure_sample = hand_ctrl.get_pressure_samples()
                ready_pressure_timestamps = (left_pressure_sample[1], right_pressure_sample[1])
                ready_dex3_measured_q, ready_dex3_commanded_q, ready_dex3_metadata = hand_ctrl.get_pose_samples()
                ready_dex3_extended = collect_extended_payload(hand_ctrl, dex3_slow_gate, logger_mp.warning)
            status_monitor.observe(
                now=time.monotonic(),
                lifecycle="ready",
                controller_sample_timestamp=ready_tele_data.controller_sample_timestamp,
                head_pose_sample_timestamp=getattr(ready_tele_data, "head_pose_sample_timestamp", 0.0),
                head_pose_is_fallback=getattr(ready_tele_data, "head_pose_is_fallback", True),
                head_pose_source=getattr(ready_tele_data, "head_pose_source", None),
                client_info=getattr(ready_tele_data, "client_info", None),
                cameras=camera_status_for_layout(args.camera_layout, head_img, left_wrist_img),
                dex3_pressure_timestamps=ready_pressure_timestamps,
            )
            get_ready_arm_q = getattr(arm_ctrl, "get_current_dual_arm_q", None)
            ready_arm_joint_split = getattr(arm_ctrl, "arm_joint_split", (7, 7))
            ready_arm_q = (
                get_ready_arm_q()
                if get_ready_arm_q is not None
                else np.zeros(sum(ready_arm_joint_split))
            )
            # Before r there is no IK target (raw controller poses are not in the
            # robot frame until calibration): hands NaN, robot q only. Never raises.
            if pose_stream is not None:
                pose_stream.maybe_send(False, _NAN_POSE, _NAN_POSE, None, ready_arm_q)
            ready_balance = balance_telemetry.balance_snapshot_best_effort(
                balance_monitor,
                loco_backend=(getattr(loco_wrapper, "backend_telemetry", lambda: None)()
                              if loco_wrapper is not None else None),
                warn=logger_mp.warning,
            ) if balance_monitor is not None else None
            ready_wall_clock = time.time()
            emit_pose_record_best_effort(
                pose_telemetry_sink.emit,
                warn=logger_mp.warning,
                timestamp=ready_wall_clock,
                timestamp_monotonic=time.monotonic(),
                lifecycle="ready",
                controller_sample_timestamp=ready_tele_data.controller_sample_timestamp,
                head_pose=getattr(ready_tele_data, "head_pose", None),
                left_wrist_pose=getattr(ready_tele_data, "left_wrist_pose", None),
                right_wrist_pose=getattr(ready_tele_data, "right_wrist_pose", None),
                measured_arm_q=ready_arm_q,
                commanded_arm_q=ready_arm_q,
                arm_joint_split=ready_arm_joint_split,
                dex3_configured=args.ee == "dex3",
                dex3_measured_q=ready_dex3_measured_q,
                dex3_commanded_q=ready_dex3_commanded_q,
                dex3_sample_metadata=ready_dex3_metadata,
                dex3_extended=ready_dex3_extended,
                drop_count=pose_telemetry_sink.drop_count,
                now=time.monotonic(),
                balance=ready_balance,
            )
            # Dex3 has controller/trigger authority only. Start its command
            # process after a post-r controller sample, independently of hand
            # skeleton availability needed by arm IK.
            if (
                args.ee == "dex3"
                and not hand_outputs_activated
                and START
                and ready_tele_data.controller_sample_timestamp > ARM_REQUEST_TIMESTAMP
                and controller_sample_is_fresh(ready_tele_data.controller_sample_timestamp)
            ):
                with LIFECYCLE_LOCK:
                    if not STOP and not hand_outputs_activated:
                        hand_ctrl.activate()
                        hand_outputs_activated = True

            if (
                START
                and ready_tele_data.controller_sample_timestamp > ARM_REQUEST_TIMESTAMP
                and controller_sample_is_fresh(ready_tele_data.controller_sample_timestamp)
            ):
                if (
                    args.arm == "G1_29"
                    and not arm_calibration.calibrated
                    and not enforce_l_pose_start_gate(
                        arm_calibration, ready_tele_data, arm_ctrl, arm_ik, pose_telemetry_sink)
                ):
                    continue
                if args.arm == "G1_29" and not arm_calibration.calibrated:
                    measured_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
                    measured_wrist_poses = arm_ik.forward_kinematics(measured_lr_arm_q)
                    calibrated = arm_calibration.calibrate(
                        (ready_tele_data.left_wrist_pose, ready_tele_data.right_wrist_pose),
                        measured_wrist_poses,
                        ready_tele_data.controller_sample_timestamp,
                        ARM_REQUEST_TIMESTAMP,
                    )
                    if calibrated:
                        first_controller_targets = arm_calibration.consume_first_target()
                        if first_controller_targets is not None:
                            break
                    continue
                break

        # The arm writer was activated at launcher-time preparation.  Here r
        # only authorizes tracking after a fresh controller-pose pair; Dex3 may
        # already be active from its independent post-r controller gate.
        with LIFECYCLE_LOCK:
            prearm_cancelled = STOP
        if prearm_cancelled:
            logger_mp.info("Preparation complete; tracking was cancelled before r.")
            raise KeyboardInterrupt

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        _safe_emit_lifecycle_event(pose_telemetry_sink, "tracking_started")
        arm_ctrl.speed_gradual_max()

        head_img = None
        left_wrist_img = None
        right_wrist_img = None

        # G1_29 only: one Cartesian three-band limiter and one enable ramp per
        # tracking session.  Both start fresh here (after the post-r
        # calibration); a repeated r/A never recreates them.  Merge point with
        # feat/graceful-shutdown-release: the ramp blends only the joint
        # target -- the arm_sdk authority weight and disarm return-to-pose
        # stay owned by that branch.
        arm_rate_limiter = None
        arm_enable_ramp = None
        if args.arm == "G1_29":
            arm_ik.reset_warm_start()
            arm_rate_limiter = DualEePoseRateLimiter(G1_29_EE_RATE_LIMITER_CONFIG)
            arm_enable_ramp = ArmEnableRamp(duration_s=DEFAULT_ENABLE_RAMP_S)

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            _diag = loop_diag
            if _diag is not None: _diag.begin()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record or (xr_need_local_img and vertical_camera_stack):
                    left_wrist_img = img_client.get_left_wrist_frame()
            if _diag is not None: _diag.mark("camera")
            if xr_need_local_img and head_img is not None:
                if vertical_camera_stack:
                    stacked_img = stack_camera_images_vertical(
                        head_img, left_wrist_img, args.camera_scale,
                        args.head_crop_bottom, args.wrist_crop_top, args.camera_divider_px)
                    if stacked_img is not None:
                        tv_wrapper.render_to_xr(stacked_img)
                        if _diag is not None: _diag.record_video(stacked_img)
                elif head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
                    if _diag is not None: _diag.record_video(head_img.bgr)
            if _diag is not None: _diag.mark("render")
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            right_a_was_pressed, right_b_was_pressed = poll_controller_lifecycle(
                tele_data, right_a_was_pressed, right_b_was_pressed)
            if _diag is not None: _diag.mark("controller")

            if args.ee in ("inspire_ftp", "inspire_dfx", "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            if args.ee == "dex3":
                # Dex3 has no hand-skeleton input path: only timestamped Quest
                # controller triggers are published to its process.
                with left_ctrl_sample_in.get_lock():
                    left_ctrl_sample_in[:] = [
                        tele_data.left_ctrl_triggerValue,
                        tele_data.controller_sample_timestamp,
                    ]
                with right_ctrl_sample_in.get_lock():
                    right_ctrl_sample_in[:] = [
                        tele_data.right_ctrl_triggerValue,
                        tele_data.controller_sample_timestamp,
                    ]
                tv_wrapper.set_pressure_samples(*hand_ctrl.get_pressure_samples())
            elif args.ee == "brainco" and args.input_mode == "controller":
                with left_gripper_trigger_in.get_lock():
                    left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                with left_gripper_squeeze_in.get_lock():
                    left_gripper_squeeze_in.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_trigger_in.get_lock():
                    right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                with right_gripper_squeeze_in.get_lock():
                    right_gripper_squeeze_in.value = tele_data.right_ctrl_squeezeValue
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready
            
            if _diag is not None: _diag.mark("hand")
            # Controller samples own arm IK, locomotion, and Dex3 freshness.
            controller_is_fresh = controller_sample_is_fresh(tele_data.controller_sample_timestamp)
            loco_on = bool(args.motion and loco_preflight and loco_preflight["loco_enabled"])
            locomotion = dispatch_joystick_locomotion(
                loco_wrapper if loco_on else None,
                motion_enabled=loco_on,
                controller_is_fresh=loco_stick_is_fresh(tele_data.controller_sample_timestamp),  # 0.2 s dead-man
                left_xy=tele_data.left_ctrl_thumbstickValue,
                right_xy=tele_data.right_ctrl_thumbstickValue,
                walk_cap=walk_cap,
                turn_cap=turn_cap,
                ramp=loco_ramp,
            )

            # In-memory only (no I/O); raw sticks are logged before any shaping.
            stick_log = stick_snapshot(
                tele_data.left_ctrl_thumbstickValue,
                tele_data.right_ctrl_thumbstickValue,
                loco_wrapper if args.motion else None,
            )
            stick_log["dispatched_command"] = [float(v) for v in locomotion]
            stick_log["raw_command"] = loco_ramp.last["raw_command"]      # post-curve, pre-ramp
            stick_log["ramp_command"] = loco_ramp.last["ramp_command"]    # post-ramp/debounce
            stick_log["ramp_reason"] = loco_ramp.last["reason"]
            stick_log["controller_fresh"] = bool(controller_is_fresh)
            if robot_monitor is not None:
                stick_log["robot_state"] = robot_monitor.snapshot()
            if loco_wrapper is not None:
                stick_log["stop_count"] = loco_wrapper.stop_count
                stick_log["last_stop_code"] = loco_wrapper.last_stop_code
                stick_log["last_stop_reason"] = loco_wrapper.last_stop_reason
                stick_log["loco_preflight"] = loco_preflight
                stick_log["walk_cap"] = walk_cap
                stick_log["turn_cap"] = turn_cap
                stick_log["loco_enabled"] = bool(loco_preflight and loco_preflight.get("loco_enabled"))
                stick_log["loco_disabled_reason"] = None if stick_log["loco_enabled"] else (loco_preflight or {}).get("refusal_reason")
                for _k in ("fsm_before", "fsm_requested", "fsm_after", "set_fsm_rc", "fsm_confirm_s"):
                    stick_log[_k] = (loco_preflight or {}).get(_k)
                stick_log["loco_backend"] = getattr(loco_wrapper, "backend_telemetry", lambda: {"backend": args.loco_backend})()
                stick_log["watchdog_stop_code"] = getattr(loco_wrapper, "watchdog_stop_code", None)
                if getattr(loco_wrapper, "sender", None) is not None:
                    stick_log["watchdog_trips"] = loco_wrapper.sender.watchdog.trips

            if _diag is not None: _diag.mark("locomotion")
            tracking_pressure_timestamps = (0.0, 0.0)
            if args.ee == "dex3":
                left_pressure_sample, right_pressure_sample = hand_ctrl.get_pressure_samples()
                tracking_pressure_timestamps = (left_pressure_sample[1], right_pressure_sample[1])
            status_monitor.observe(
                now=time.monotonic(),
                lifecycle="tracking",
                controller_sample_timestamp=tele_data.controller_sample_timestamp,
                head_pose_sample_timestamp=getattr(tele_data, "head_pose_sample_timestamp", 0.0),
                head_pose_is_fallback=getattr(tele_data, "head_pose_is_fallback", True),
                head_pose_source=getattr(tele_data, "head_pose_source", None),
                client_info=getattr(tele_data, "client_info", None),
                motion_enabled=args.motion,
                locomotion=locomotion,
                stick=stick_log,
                cameras=camera_status_for_layout(args.camera_layout, head_img, left_wrist_img),
                dex3_pressure_timestamps=tracking_pressure_timestamps,
                torso_lean=torso_lean.status_telemetry(lean_session),
            )

            if _diag is not None: _diag.mark("telemetry")
            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            # Recheck immediately before IK; state reads may consume the final
            # part of the controller-pose freshness window.
            controller_pose_is_fresh = controller_sample_is_fresh(tele_data.controller_sample_timestamp)

            if com_monitor is not None:
                com_status = com_monitor.update(current_lr_arm_q)  # warn-only; never publishes
            if _diag is not None: _diag.mark("state_read")
            candidate_targets = None
            calibration = arm_calibration if args.arm == "G1_29" else None
            first_target = first_controller_targets if args.arm == "G1_29" else None
            if args.arm != "G1_29" and controller_pose_is_fresh:
                candidate_targets = (tele_data.left_wrist_pose, tele_data.right_wrist_pose)
            # Torso lean (opt-in, G1_29 + --motion only): head displacement ->
            # waist pitch/roll command; world-fixed controller targets are
            # re-expressed in the leaning torso frame (IK model frame) and the
            # arm gravity feed-forward uses R^T g. Off: nothing below runs.
            lean_transform = None
            waist_cmd = None
            if lean_cfg is not None and lean_session is None:
                lean_session = torso_lean.TorsoLeanSession.try_engage(
                    lean_cfg, arm_ctrl, getattr(tele_data, "head_pose", None), time.monotonic(), log=logger_mp)
            if isinstance(lean_session, torso_lean.TorsoLeanSession):
                lean_now = time.monotonic()
                try:
                    measured_waist, measured_waist_age = arm_ctrl.get_waist_q_snapshot()
                except Exception:
                    measured_waist, measured_waist_age = None, float("inf")
                # Compensation is derived exclusively from measured lowstate
                # waist feedback. The watchdog may disable new lean before this
                # cycle's command is generated; telemetry failure never raises.
                R_torso = lean_session.observe_measured_waist(
                    measured_waist, measured_waist_age, lean_now)
                waist_cmd = lean_session.step(
                    getattr(tele_data, "head_pose", None), lean_now)
                if R_torso is None:
                    R_torso = np.eye(3)
                arm_ik.set_torso_rotation(R_torso)
                lean_transform = lambda pose, _R=R_torso: torso_lean.retarget_world_fixed_to_torso(pose, _R)
            time_ik_start = time.time()
            cycle = run_arm_tracking_cycle(
                arm_ctrl=arm_ctrl,
                arm_ik=arm_ik,
                calibrator=calibration,
                controller_poses=(tele_data.left_wrist_pose, tele_data.right_wrist_pose),
                sample_timestamp=tele_data.controller_sample_timestamp,
                current_q=current_lr_arm_q,
                current_dq=current_lr_arm_dq,
                first_target=first_target,
                candidate_targets=candidate_targets,
                lifecycle_lock=LIFECYCLE_LOCK,
                is_started=lambda: START,
                is_stopped=lambda: STOP,
                rate_limiter=arm_rate_limiter,
                enable_ramp=arm_enable_ramp,
                target_transform=lean_transform,
            )
            if waist_cmd is not None:
                with LIFECYCLE_LOCK:
                    if not STOP:
                        arm_ctrl.set_waist_target(waist_cmd)
            if pose_stream is not None:
                pose_stream.maybe_send(True, *_pose_stream_hands(cycle), cycle.selected_q, current_lr_arm_q,
                                       lean=torso_lean.stream_telemetry(lean_session, arm_ctrl))
            if first_target is not None:
                first_controller_targets = None
            if cycle.target_accepted:
                logger_mp.debug(f"ik:\t{round(time.time() - time_ik_start, 6)}")

            if _diag is not None: _diag.mark("arm_cycle")
            dex3_measured_q = None
            dex3_commanded_q = None
            dex3_metadata = None
            dex3_extended = None
            if args.ee == "dex3":
                dex3_measured_q, dex3_commanded_q, dex3_metadata = hand_ctrl.get_pose_samples()
                dex3_extended = collect_extended_payload(hand_ctrl, dex3_slow_gate, logger_mp.warning)
            arm_request_id = (
                int(cycle.publication)
                if isinstance(cycle.publication, (int, np.integer))
                else None
            )
            commanded_arm_q_reason = (
                "arm_command_publication_pending"
                if arm_request_id is not None
                else "arm_command_publication_unavailable"
            )
            if _diag is not None: _diag.mark("hand")
            _loop_timing, _loop_diag = (_diag.pose_fields(arm_rate_limiter, arm_calibration) if _diag is not None else (None, None))
            tracking_balance = balance_telemetry.balance_snapshot_best_effort(
                balance_monitor,
                loco_backend=stick_log.get("loco_backend"),
                dispatched=locomotion,
                loco_raw={"left": tele_data.left_ctrl_thumbstickValue,
                          "right": tele_data.right_ctrl_thumbstickValue},
                com_status=com_status,
                arm_commanded_q=cycle.selected_q,
                warn=logger_mp.warning,
            ) if balance_monitor is not None else None
            tracking_wall_clock = time.time()
            emit_pose_record_best_effort(
                lambda record: arm_publication_telemetry.emit_cycle(record, arm_ctrl),
                warn=logger_mp.warning,
                timestamp=tracking_wall_clock,
                timestamp_monotonic=time.monotonic(),
                lifecycle="tracking",
                controller_sample_timestamp=tele_data.controller_sample_timestamp,
                head_pose=getattr(tele_data, "head_pose", None),
                left_wrist_pose=getattr(tele_data, "left_wrist_pose", None),
                right_wrist_pose=getattr(tele_data, "right_wrist_pose", None),
                measured_arm_q=current_lr_arm_q,
                commanded_arm_q=None,
                commanded_arm_q_reason=commanded_arm_q_reason,
                arm_command_request_id=arm_request_id,
                requested_arm_q=cycle.requested_q,
                selected_arm_q=cycle.selected_q,
                requested_arm_tauff=cycle.requested_tauff,
                selected_arm_tauff=cycle.selected_tauff,
                calibrated_cartesian_target=cycle.target,
                ik_target_accepted=cycle.target_accepted,
                ik_sample_fresh=cycle.sample_fresh,
                ik_published=cycle.published,
                ik_hold=cycle.hold,
                ik_reason=cycle.decision_reason,
                arm_publication_drop_count=getattr(arm_ctrl, "publication_receipt_drop_count", 0),
                arm_joint_split=arm_ctrl.arm_joint_split,
                dex3_configured=args.ee == "dex3",
                dex3_measured_q=dex3_measured_q,
                dex3_commanded_q=dex3_commanded_q,
                dex3_sample_metadata=dex3_metadata,
                dex3_extended=dex3_extended,
                drop_count=pose_telemetry_sink.drop_count,
                now=time.monotonic(),
                locomotion=stick_log,
                loop_timing=_loop_timing,
                loop_diag=_loop_diag,
                balance=tracking_balance,
            )
            if _diag is not None: _diag.mark("telemetry")

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = list(joystick_to_locomotion(
                            tele_data.left_ctrl_thumbstickValue,
                            tele_data.right_ctrl_thumbstickValue,
                            walk_cap, turn_cap,
                        ))
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                elif (args.ee == "brainco" and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = list(joystick_to_locomotion(
                            tele_data.left_ctrl_thumbstickValue,
                            tele_data.right_ctrl_thumbstickValue,
                            walk_cap, turn_cap,
                        ))
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_joint_count, right_joint_count = arm_ctrl.arm_joint_split
                left_arm_state  = current_lr_arm_q[:left_joint_count]
                right_arm_state = current_lr_arm_q[left_joint_count:left_joint_count + right_joint_count]
                recorded_arm_actions = build_arm_recording_actions(cycle)
                left_arm_action = recorded_arm_actions["left_arm"]["qpos"]
                right_arm_action = recorded_arm_actions["right_arm"]["qpos"]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None and head_img.bgr is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None and left_wrist_img.bgr is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None and right_wrist_img.bgr is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None and head_img.bgr is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None and left_wrist_img.bgr is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None and right_wrist_img.bgr is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action,
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action,
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            if _diag is not None: _diag.mark("record")
            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            if _diag is not None:
                _diag.mark("sleep")
                _diag.cycle_end(tele_data.controller_sample_timestamp, arm_rate_limiter, arm_calibration)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        shutdown_cause = "shutdown_interrupted"
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        shutdown_cause = "shutdown_exception"
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        # Explicit StopMove first (bounded, never raises): the robot keeps the
        # last SetVelocity for its duration (1 s), so do not wait for the arm
        # shutdown to cancel it. Retried again after the arm shutdown below.
        if getattr(loco_wrapper, "stop_move_sender", None) is not None:
            loco_wrapper.stop_move_sender()
        stop_locomotion_best_effort(loco_wrapper, shutdown_cause or "shutdown")
        # G1_29 graceful shutdown (q/B, Ctrl+C, exception): tracking has
        # already stopped; return to the all-zero preparation pose with a
        # velocity-limited trajectory, open Dex3, ramp the arm_sdk authority
        # weight 1 -> 0 so the Unitree controller takes the arms back, then
        # deactivate the writer. Invalid/stale state or a dead writer skips
        # the motion (release/deactivate only). Every phase is time-bounded.
        try:
            if isinstance(lean_session, torso_lean.TorsoLeanSession) and arm_ik is not None:
                # the waist returns to its neutral during the shutdown: the arm
                # feed-forward uses the neutral (upright) gravity again.
                arm_ik.set_torso_rotation(np.eye(3))
        except BaseException as e:
            _log_best_effort("error", f"[torso_lean] reset da gravidade do IK falhou: {e!r}")
        if args.arm == "G1_29":
            if outputs_activated:
                graceful_g1_29_shutdown(
                    arm_ctrl,
                    arm_ik=arm_ik,
                    hand_ctrl=hand_ctrl if hand_outputs_activated else None,
                    sink=pose_telemetry_sink,
                    attempt_return=shutdown_return_home,
                )
                _log_best_effort("info", "Arm output released; exiting.")
            elif hand_outputs_activated:
                try:
                    hand_ctrl.deactivate()
                except Exception as e:
                    _log_best_effort("error", f"Failed to deactivate Dex3 output: {e}")
        else:
            if hand_outputs_activated:
                try:
                    hand_ctrl.deactivate()
                except Exception as e:
                    _log_best_effort("error", f"Failed to deactivate Dex3 output: {e}")
            # Legacy arm profiles publish continuously from construction;
            # preserve the previous shutdown behavior of returning home.
            try:
                arm_ctrl.ctrl_dual_arm_go_home()
            except Exception as e:
                _log_best_effort("error", f"Failed to ctrl_dual_arm_go_home: {e}")

        stop_locomotion_best_effort(loco_wrapper, "shutdown_post_arm")
        try:
            if robot_monitor is not None:
                robot_monitor.close()
        except BaseException as e:
            _log_best_effort("error", f"Failed to close robot state monitor: {e}")
        try:
            if balance_monitor is not None:
                balance_monitor.close()
        except BaseException as e:
            _log_best_effort("error", f"Failed to close balance telemetry: {e}")

        # Normal control-path telemetry preserves KeyboardInterrupt/SystemExit
        # for the outer shutdown handler. Cleanup telemetry is different: it is
        # optional and fully guarded so one failing emit cannot skip later
        # actuator, listener, client, or sink cleanup.
        try:
            _emit_lifecycle_events(pose_telemetry_sink)
        except BaseException as error:
            _log_best_effort("warning", f"Failed to emit queued cleanup telemetry: {type(error).__name__}")
        if pose_telemetry_sink is not None:
            if shutdown_cause is not None:
                _cleanup_telemetry_event_best_effort(
                    pose_telemetry_sink, shutdown_cause, cause=shutdown_cause)
            _cleanup_telemetry_event_best_effort(pose_telemetry_sink, "shutdown_finalization")
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            _log_best_effort("error", f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            _log_best_effort("error", f"Failed to close image client: {e}")

        try:
            if tv_wrapper is not None:
                tv_wrapper.close()
        except Exception as e:
            _log_best_effort("error", f"Failed to close televuer wrapper: {e}")

        if loop_diag is not None:
            try:
                loop_diag.close()
            except Exception:
                pass
        _close_telemetry_best_effort(pose_telemetry_sink, "pose telemetry sink")
        _close_telemetry_best_effort(status_sink, "teleop status sink")
        if pose_stream is not None:
            pose_stream.close()

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            _log_best_effort("error", f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            _log_best_effort("error", f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            _log_best_effort("error", f"Failed to close recorder: {e}")
        _log_best_effort("info", "✅ Finally, exiting program.")
