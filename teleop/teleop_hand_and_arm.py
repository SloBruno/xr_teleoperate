import time
import argparse
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
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from teleop.utils.quest_controls import joystick_to_locomotion
from teleop.utils.quest_safety import controller_sample_is_fresh, fresh_controller_value
from teleop.utils.teleop_status import (
    AsyncStatusFileSink,
    TeleopStatusMonitor,
    camera_frame_is_usable,
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
    publish_arm_command_for_telemetry,
)
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
LIFECYCLE_EVENTS = []
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
    if not PREPARATION_COMPLETE:
        logger_mp.warning("[lifecycle] Ignoring start until arm preparation completes.")
        return False
    ARM_REQUEST_TIMESTAMP = time.monotonic()
    START = True
    LIFECYCLE_EVENTS.append("start_accepted")
    return True


def _request_stop_locked():
    """Make the unconditional stop request."""
    global STOP, START
    START = False
    STOP = True
    LIFECYCLE_EVENTS.append("stop_requested")


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


def on_press(key):
    global RECORD_TOGGLE
    with LIFECYCLE_LOCK:
        if key == 'r':
            _request_start_locked()
        elif key == 'q':
            _request_stop_locked()
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
    # network parameters
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
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
    logger_mp.debug(f"args: {args}")
    outputs_activated = False
    hand_outputs_activated = False
    pose_telemetry_sink = None
    status_sink = None
    img_client = None
    tv_wrapper = None
    shutdown_cause = None

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
                                     arm_pose_source="controller"
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
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
        status_monitor = TeleopStatusMonitor(status_sink.emit)
        pose_log_dir = os.environ.get(
            "XR_TELEOP_POSE_LOG_DIR",
            "/home/unitree/.local/state/xr_teleoperate",
        )
        pose_telemetry_sink = create_pose_telemetry_sink(pose_log_dir, logger_mp.warning)
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
                    arm_ctrl.deactivate()
                    outputs_activated = False
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
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        right_a_was_pressed = False
        right_b_was_pressed = False
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
            ready_pressure_timestamps = (0.0, 0.0)
            ready_dex3_measured_q = None
            ready_dex3_commanded_q = None
            ready_dex3_metadata = None
            if args.ee == "dex3":
                left_pressure_sample, right_pressure_sample = hand_ctrl.get_pressure_samples()
                ready_pressure_timestamps = (left_pressure_sample[1], right_pressure_sample[1])
                ready_dex3_measured_q, ready_dex3_commanded_q, ready_dex3_metadata = hand_ctrl.get_pose_samples()
            status_monitor.observe(
                now=time.monotonic(),
                lifecycle="ready",
                controller_sample_timestamp=ready_tele_data.controller_sample_timestamp,
                cameras={"head": camera_frame_is_usable(head_img), "left_wrist": camera_frame_is_usable(left_wrist_img)},
                dex3_pressure_timestamps=ready_pressure_timestamps,
            )
            get_ready_arm_q = getattr(arm_ctrl, "get_current_dual_arm_q", None)
            ready_arm_q = get_ready_arm_q() if get_ready_arm_q is not None else np.zeros(14)
            ready_wall_clock = time.time()
            emit_pose_record_best_effort(
                pose_telemetry_sink.emit,
                warn=logger_mp.warning,
                timestamp=ready_wall_clock,
                timestamp_monotonic=time.monotonic(),
                lifecycle="ready",
                controller_sample_timestamp=ready_tele_data.controller_sample_timestamp,
                left_wrist_pose=getattr(ready_tele_data, "left_wrist_pose", None),
                right_wrist_pose=getattr(ready_tele_data, "right_wrist_pose", None),
                measured_arm_q=ready_arm_q,
                commanded_arm_q=ready_arm_q,
                dex3_configured=args.ee == "dex3",
                dex3_measured_q=ready_dex3_measured_q,
                dex3_commanded_q=ready_dex3_commanded_q,
                dex3_sample_metadata=ready_dex3_metadata,
                drop_count=pose_telemetry_sink.drop_count,
                now=time.monotonic(),
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

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record or (xr_need_local_img and vertical_camera_stack):
                    left_wrist_img = img_client.get_left_wrist_frame()
            if xr_need_local_img and head_img is not None:
                if vertical_camera_stack:
                    stacked_img = stack_camera_images_vertical(
                        head_img, left_wrist_img, args.camera_scale,
                        args.head_crop_bottom, args.wrist_crop_top, args.camera_divider_px)
                    if stacked_img is not None:
                        tv_wrapper.render_to_xr(stacked_img)
                elif head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
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
            
            # Controller samples own arm IK, locomotion, and Dex3 freshness.
            controller_is_fresh = controller_sample_is_fresh(tele_data.controller_sample_timestamp)
            locomotion = (0.0, 0.0, 0.0)
            if args.motion:
                if controller_is_fresh:
                    locomotion = joystick_to_locomotion(
                        tele_data.left_ctrl_thumbstickValue,
                        tele_data.right_ctrl_thumbstickValue,
                    )
                loco_wrapper.Move(*locomotion)

            tracking_pressure_timestamps = (0.0, 0.0)
            if args.ee == "dex3":
                left_pressure_sample, right_pressure_sample = hand_ctrl.get_pressure_samples()
                tracking_pressure_timestamps = (left_pressure_sample[1], right_pressure_sample[1])
            status_monitor.observe(
                now=time.monotonic(),
                lifecycle="tracking",
                controller_sample_timestamp=tele_data.controller_sample_timestamp,
                motion_enabled=args.motion,
                locomotion=locomotion,
                cameras={"head": camera_frame_is_usable(head_img), "left_wrist": camera_frame_is_usable(left_wrist_img)},
                dex3_pressure_timestamps=tracking_pressure_timestamps,
            )

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            # Recheck immediately before IK; state reads may consume the final
            # part of the controller-pose freshness window.
            controller_pose_is_fresh = controller_sample_is_fresh(tele_data.controller_sample_timestamp)

            # Only solve new arm targets while the controller poses are fresh.
            # On loss of controller authority, hold the measured joint position
            # with zero feed-forward torque instead of advancing stale IK.
            if controller_pose_is_fresh:
                time_ik_start = time.time()
                sol_q, sol_tauff = arm_ik.solve_ik(
                    tele_data.left_wrist_pose,
                    tele_data.right_wrist_pose,
                    current_lr_arm_q,
                    current_lr_arm_dq,
                )
                time_ik_end = time.time()
                logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            else:
                sol_q = current_lr_arm_q.copy()
                sol_tauff = np.zeros_like(current_lr_arm_q)
            commanded_arm_q = None
            commanded_arm_q_reason = "arm_command_publication_unavailable"
            if (
                controller_pose_is_fresh
                and controller_sample_is_fresh(tele_data.controller_sample_timestamp)
            ):
                arm_publication = publish_arm_command_for_telemetry(
                    arm_ctrl, sol_q, sol_tauff
                )
            else:
                # The sample expired during IK; discard its target and hold the
                # most recently measured arm position instead.
                hold_q = arm_ctrl.get_current_dual_arm_q().copy()
                arm_publication = publish_arm_command_for_telemetry(
                    arm_ctrl, hold_q, np.zeros_like(current_lr_arm_q)
                )
            commanded_arm_q = arm_publication.published_q
            commanded_arm_q_reason = arm_publication.reason

            dex3_measured_q = None
            dex3_commanded_q = None
            dex3_metadata = None
            if args.ee == "dex3":
                dex3_measured_q, dex3_commanded_q, dex3_metadata = hand_ctrl.get_pose_samples()
            tracking_wall_clock = time.time()
            emit_pose_record_best_effort(
                lambda record: arm_publication_telemetry.emit_cycle(record, arm_ctrl),
                warn=logger_mp.warning,
                timestamp=tracking_wall_clock,
                timestamp_monotonic=time.monotonic(),
                lifecycle="tracking",
                controller_sample_timestamp=tele_data.controller_sample_timestamp,
                left_wrist_pose=getattr(tele_data, "left_wrist_pose", None),
                right_wrist_pose=getattr(tele_data, "right_wrist_pose", None),
                measured_arm_q=current_lr_arm_q,
                commanded_arm_q=commanded_arm_q,
                commanded_arm_q_reason=commanded_arm_q_reason,
                arm_command_request_id=arm_publication.request_id,
                requested_arm_q=arm_publication.requested_q,
                selected_arm_q=arm_publication.selected_q,
                arm_publication_drop_count=getattr(arm_ctrl, "publication_receipt_drop_count", 0),
                arm_joint_split=arm_ctrl.arm_joint_split,
                dex3_configured=args.ee == "dex3",
                dex3_measured_q=dex3_measured_q,
                dex3_commanded_q=dex3_commanded_q,
                dex3_sample_metadata=dex3_metadata,
                drop_count=pose_telemetry_sink.drop_count,
                now=time.monotonic(),
            )

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
                        ))
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
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
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
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

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        shutdown_cause = "shutdown_interrupted"
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        shutdown_cause = "shutdown_exception"
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        # Dex3 is an independent post-r output. Stop its child process before
        # the potentially slower arm return-to-preparation motion.
        if hand_outputs_activated:
            try:
                hand_ctrl.deactivate()
            except Exception as e:
                _log_best_effort("error", f"Failed to deactivate Dex3 output: {e}")
        if args.arm == "G1_29":
            if outputs_activated:
                try:
                    # Return to the original all-zero preparation pose before
                    # releasing arm DDS output.
                    arm_ctrl.ctrl_dual_arm_go_home(release_motion_authority=True)
                    arm_ctrl.deactivate()
                except Exception as e:
                    _log_best_effort("error", f"Failed to deactivate arm output: {e}")
                _log_best_effort("info", "Arm preparation output ended; exiting.")
        else:
            # Legacy arm profiles publish continuously from construction;
            # preserve the previous shutdown behavior of returning home.
            try:
                arm_ctrl.ctrl_dual_arm_go_home()
            except Exception as e:
                _log_best_effort("error", f"Failed to ctrl_dual_arm_go_home: {e}")

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

        _close_telemetry_best_effort(pose_telemetry_sink, "pose telemetry sink")
        _close_telemetry_best_effort(status_sink, "teleop status sink")

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
