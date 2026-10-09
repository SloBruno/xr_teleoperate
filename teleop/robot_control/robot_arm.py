import numpy as np
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import ( LowCmd_  as hg_LowCmd, LowState_ as hg_LowState) # idl for g1, h1_2
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

from unitree_sdk2py.idl.unitree_go.msg.dds_ import ( LowCmd_  as go_LowCmd, LowState_ as go_LowState)  # idl for h1
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

kTopicLowCommand_Debug  = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"

G1_29_Num_Motors = 35
G1_23_Num_Motors = 35
H1_2_Num_Motors = 35
H1_Num_Motors = 20
H2_Num_Motors = 35
 

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None
        self.motorstate = None   # driver status word (0 = ok); filled for G1_29 only

class G1_29_LowState:
    def __init__(self):
        self.mode_machine = None
        self.imu_quaternion = None   # pelvis IMU (w, x, y, z) from rt/lowstate
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]

class G1_23_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_23_Num_Motors)]

class H1_2_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_2_Num_Motors)]

class H1_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_Num_Motors)]

class H2_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H2_Num_Motors)]


class DataBuffer:
    def __init__(self):
        self.data = None
        self.timestamp = 0.0
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def GetSnapshot(self):
        with self.lock:
            return self.data, self.timestamp

    def SetData(self, data):
        with self.lock:
            self.data = data
            self.timestamp = time.monotonic()


@dataclass(frozen=True)
class ArmPublicationReceipt:
    request_id: int
    published_q: tuple[float, ...] | None
    published_tauff: tuple[float, ...] | None
    reason: str
    timestamp_monotonic: float
    arm_joint_split: tuple[int, int]


class _ArmPublicationMixin:
    arm_joint_split: tuple[int, int]

    def _init_arm_publication_state(self):
        self._command_request_id = 0
        self._publication_receipts = deque(maxlen=64)
        self._publication_receipt_drop_count = 0
        self._publication_receipt_lock = threading.Lock()

    def _capture_arm_command(self):
        with self.ctrl_lock:
            return (
                np.asarray(self.q_target, dtype=float).copy(),
                np.asarray(self.tauff_target, dtype=float).copy(),
                self._command_request_id,
            )

    def _set_arm_command(self, q_target, tauff_target):
        with self.ctrl_lock:
            self.q_target = np.asarray(q_target, dtype=float).copy()
            self.tauff_target = np.asarray(tauff_target, dtype=float).copy()
            self._command_request_id += 1
            return self._command_request_id

    def _record_arm_publication(self, request_id, published_q, reason, published_tauff=None):
        if published_q is None:
            frozen_q = None
        else:
            frozen_q = tuple(float(value) for value in np.asarray(published_q, dtype=float).reshape(-1))
        if published_tauff is None:
            frozen_tauff = None
        else:
            frozen_tauff = tuple(float(value) for value in np.asarray(published_tauff, dtype=float).reshape(-1))
        receipt = ArmPublicationReceipt(
            request_id=int(request_id),
            published_q=frozen_q,
            published_tauff=frozen_tauff,
            reason=str(reason),
            timestamp_monotonic=time.monotonic(),
            arm_joint_split=self.arm_joint_split,
        )
        with self._publication_receipt_lock:
            if len(self._publication_receipts) == self._publication_receipts.maxlen:
                self._publication_receipt_drop_count += 1
            self._publication_receipts.append(receipt)

    def drain_arm_publication_receipts(self):
        with self._publication_receipt_lock:
            receipts = tuple(self._publication_receipts)
            self._publication_receipts.clear()
            return receipts

    @property
    def publication_receipt_drop_count(self):
        with self._publication_receipt_lock:
            return self._publication_receipt_drop_count

    def _record_failed_arm_publication(self, request_id, error):
        self._record_arm_publication(
            request_id, None, f"arm_command_publication_failed:{type(error).__name__}"
        )

    def _arm_command_is_finite(self, q_target, tauff_target):
        try:
            q = np.asarray(q_target, dtype=float)
            tau = np.asarray(tauff_target, dtype=float)
            expected_size = sum(int(count) for count in self.arm_joint_split)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            q.ndim == 1
            and expected_size > 0
            and q.size == expected_size
            and tau.shape == q.shape
            and np.all(np.isfinite(q))
            and np.all(np.isfinite(tau))
        )


class G1_29_ArmController(_ArmPublicationMixin):
    arm_joint_split = (7, 7)

    def __init__(self, motion_mode = False, simulation_mode = False):
        logger_mp.info("Initialize G1_29_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5
        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        # Pre-arm is receive-only: create the state subscriber and wait for a
        # measured robot state, but do not create a command publisher/message.
        self.lowcmd_publisher = None
        self.crc = None
        self.msg = None
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[G1_29_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[G1_29_ArmController] Subscribe dds ok.")

        # Command message and all target positions are initialized from a fresh
        # low-state sample in activate(), immediately before first output.

        # Construct the publisher now, but never start its write loop until the
        # terminal pre-arm gate has accepted a post-r controller sample.
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self._init_arm_publication_state()
        self.publish_thread.daemon = True
        self.output_enabled = threading.Event()
        self.outputs_activated = False
        # rt/arm_sdk authority weight (kNotUsedJoint0.q): 1.0 = teleop owns
        # the arms, 0.0 = the Unitree motion controller owns them again.
        self._motion_authority_weight = 1.0
        self._last_publish_monotonic = None
        self._last_published_weight = None
        # Optional torso lean (teleop/utils/torso_lean.py, G1_TORSO_LEAN=1).
        # Until configure_waist_command() is called (after activate(), at r)
        # the writer NEVER touches motor_cmd[12..14]: they keep the activate()
        # values (q = measured at activation, kp/kd = kp_high/kd_high, dq = tau = 0).
        self._waist_enabled = False
        self._waist_target = None
        self._waist_written = None
        self._waist_lower = None
        self._waist_upper = None
        self._waist_max_step = 0.0
        self._waist_max_rate = None
        self._waist_neutral = None
        # Optional waist gravity feed-forward (teleop/utils/waist_gravity_ff.py,
        # G1_TORSO_LEAN_WAIST_FF): None = tau of 12..14 stays exactly 0.
        self._waist_ff = None

        logger_mp.info("Initialize G1_29_ArmController OK (passive pre-arm).")

    def activate(self):
        """Start arm DDS publication after the terminal pre-arm gate only."""
        if self.outputs_activated:
            return
        # Build the complete command from one atomic fresh state snapshot.
        lowstate, lowstate_timestamp = self.lowstate_buffer.GetSnapshot()
        if lowstate is None or time.monotonic() - lowstate_timestamp > 0.1:
            raise RuntimeError("[G1_29_ArmController] LowState is stale; refusing activation.")
        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = lowstate.mode_machine
        self.all_motor_q = np.array([lowstate.motor_state[id].q for id in G1_29_JointIndex])
        arm_indices = set(member.value for member in G1_29_JointArmIndex)
        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            elif self._Is_weak_motor(id):
                self.msg.motor_cmd[id].kp = self.kp_low
                self.msg.motor_cmd[id].kd = self.kd_low
            else:
                self.msg.motor_cmd[id].kp = self.kp_high
                self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q = self.all_motor_q[id]
        # First arm command is the same fresh measured configuration.
        with self.ctrl_lock:
            self.q_target = np.array([
                lowstate.motor_state[id].q for id in G1_29_JointArmIndex]).copy()
            self.tauff_target = np.zeros_like(self.q_target)
        self.output_enabled.set()
        self.outputs_activated = True
        self.publish_thread.start()
        logger_mp.info("[G1_29_ArmController] Arm DDS output activated.")

    def deactivate(self):
        """Stop the arm write loop immediately when terminal q is processed."""
        self.output_enabled.clear()
        logger_mp.info("[G1_29_ArmController] Arm DDS output deactivated.")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()
                lowstate.mode_machine = msg.mode_machine
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                try:
                    lowstate.imu_quaternion = tuple(float(v) for v in msg.imu_state.quaternion)
                    for id in G1_29_WAIST_INDICES:
                        lowstate.motor_state[id].motorstate = int(msg.motor_state[id].motorstate)
                except Exception:
                    pass   # missing fields: the waist feed-forward fails closed (tau 0)
                self.lowstate_buffer.SetData(lowstate)
                # Optional passive tap (balance telemetry): store-only, never raises.
                observer = getattr(self, "lowstate_observer", None)
                if observer is not None:
                    try:
                        observer(msg)
                    except Exception:
                        pass
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        while self.output_enabled.is_set():
            start_time = time.time()

            arm_q_target, arm_tauff_target, request_id = self._capture_arm_command()
            with self.ctrl_lock:
                motion_weight = self._motion_authority_weight
            if self.motion_mode:
                self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = motion_weight
            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            if not self._arm_command_is_finite(cliped_arm_q_target, arm_tauff_target):
                self._record_failed_arm_publication(request_id, ValueError("non-finite arm command"))
                time.sleep(self.control_dt)
                continue

            for idx, id in enumerate(G1_29_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]   

            waist_q = self._next_waist_frame()
            if waist_q is not None:
                waist_tau = self._next_waist_ff_frame()
                for idx, id in enumerate(G1_29_WAIST_INDICES):
                    self.msg.motor_cmd[id].q = float(waist_q[idx])
                    self.msg.motor_cmd[id].dq = 0
                    self.msg.motor_cmd[id].tau = 0 if waist_tau is None else float(waist_tau[idx])

            self.msg.crc = self.crc.Crc(self.msg)
            try:
                self.lowcmd_publisher.Write(self.msg)
            except Exception as error:
                self._record_failed_arm_publication(request_id, error)
            else:
                self._record_arm_publication(request_id, cliped_arm_q_target, "published", arm_tauff_target)
                with self.ctrl_lock:
                    self._last_publish_monotonic = time.monotonic()
                    self._last_published_weight = motion_weight if self.motion_mode else None
                    if waist_q is not None:
                        self._waist_written = waist_q

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        request_id = self._set_arm_command(q_target, tauff_target)
        return request_id

    def set_motion_authority_weight(self, weight):
        '''Set the rt/arm_sdk authority weight written by the arm writer (0..1).'''
        weight = float(weight)
        if not np.isfinite(weight):
            raise ValueError("motion authority weight must be finite")
        with self.ctrl_lock:
            self._motion_authority_weight = min(1.0, max(0.0, weight))

    # ---- optional waist (torso lean) command, motors 12 yaw, 13 roll, 14 pitch
    def _next_waist_frame(self):
        '''Writer side: next waist q to write, or None (= do not touch 12..14).

        Final authority before the DDS write: the target is clamped to the
        configured box (neutral +- lean limit, inside the URDF limits) and the
        step from the last WRITTEN value is bounded by max_rate * control_dt.
        '''
        with self.ctrl_lock:
            if not getattr(self, "_waist_enabled", False):
                return None
            target = self._waist_target
            last = self._waist_written
            lower, upper, max_step = self._waist_lower, self._waist_upper, self._waist_max_step
        step = np.clip(np.clip(target, lower, upper) - last, -max_step, max_step)
        q = np.clip(last + step, lower, upper)
        if q.shape != (3,) or not np.all(np.isfinite(q)):
            return last.copy()
        return q

    # ---- optional waist gravity feed-forward (roll 13 / pitch 14 tau only)
    def configure_waist_gravity_ff(self, model, cap_nm=None):
        '''Enable the model-based waist gravity tau (after configure_waist_command).

        ``model.waist_tau(waist_q, arm_q, pelvis_quat_wxyz) -> [roll, pitch]``
        (teleop/utils/waist_gravity_ff.WaistGravityModel). The writer computes
        it every frame from ONE lowstate snapshot; cap/ramp/fail-to-zero by
        WaistFFShaper. kp/kd are never changed. Raises ValueError.
        '''
        from teleop.utils import waist_gravity_ff as wff
        if not callable(getattr(model, "waist_tau", None)):
            raise ValueError("waist gravity model must provide waist_tau()")
        shaper = wff.WaistFFShaper() if cap_nm is None else wff.WaistFFShaper(cap_nm=cap_nm)
        with self.ctrl_lock:
            if not getattr(self, "_waist_enabled", False):
                raise ValueError("waist command not configured; gravity feed-forward unavailable")
            self._waist_ff = {"model": model, "shaper": shaper, "wff": wff, "last_t": None,
                              "tau": np.zeros(3), "raw": np.zeros(3)}

    def _next_waist_ff_frame(self):
        '''Writer side: waist tau [yaw=0, roll, pitch] for this frame, None = not configured.'''
        ff = getattr(self, "_waist_ff", None)
        if ff is None:
            return None
        wff = ff["wff"]
        now = time.monotonic()
        dt = self.control_dt if ff["last_t"] is None else now - ff["last_t"]
        ff["last_t"] = now
        raw, reason = None, None
        lowstate, stamp = self.lowstate_buffer.GetSnapshot()
        age = now - stamp
        if lowstate is None or not (0.0 <= age <= wff.STATE_MAX_AGE_S or -0.01 <= age < 0.0):
            reason = "state_stale"
        else:
            words = [lowstate.motor_state[i].motorstate for i in G1_29_WAIST_INDICES]
            if any(w is None or w != 0 for w in words):
                reason = "motor_fault"
            else:
                try:
                    waist = [lowstate.motor_state[i].q for i in G1_29_WAIST_INDICES]
                    arms = [lowstate.motor_state[i].q for i in G1_29_JointArmIndex]
                    raw = ff["model"].waist_tau(waist, arms, lowstate.imu_quaternion)
                except Exception:
                    raw, reason = None, "model_rejected"
        with self.ctrl_lock:
            out = ff["shaper"].step(raw, dt, reason=reason)
            tau = np.array([0.0, float(out[0]), float(out[1])])
            if not np.all(np.isfinite(tau)):
                tau = np.zeros(3)
            ff["tau"] = tau
            ff["raw"] = np.array([0.0, *ff["shaper"].raw])
        return tau

    def get_waist_gravity_ff(self):
        '''Telemetry snapshot of the waist feed-forward (copies; never the model).'''
        with self.ctrl_lock:
            ff = getattr(self, "_waist_ff", None)
            if ff is None:
                return {"configured": False}
            shaper = ff["shaper"]
            return {"configured": True, "gain": float(shaper.gain), "reason": shaper.reason,
                    "tau_nm": ff["tau"].tolist(), "raw_nm": ff["raw"].tolist(),
                    "finished": bool(shaper.finished), "cap_nm": float(shaper.cap)}

    def waist_gravity_ff_ramp_out(self):
        '''Ramp the waist tau to 0 over RAMP_S (graceful shutdown). False if not configured.'''
        with self.ctrl_lock:
            ff = getattr(self, "_waist_ff", None)
            if ff is None:
                return False
            ff["shaper"].ramp_out()
            return True

    def configure_waist_command(self, lower, upper, max_rate, initial_target):
        '''Take the waist (12..14) from the q currently in the message.

        kp/kd/mode of 12..14 are NOT changed (they keep the values written since
        construction, see docs/torso_lean.md). Raises ValueError on bad input.
        '''
        lower = np.asarray(lower, dtype=float).reshape(-1).copy()
        upper = np.asarray(upper, dtype=float).reshape(-1).copy()
        target = np.asarray(initial_target, dtype=float).reshape(-1).copy()
        max_rate = float(max_rate)
        if (lower.shape != (3,) or upper.shape != (3,) or target.shape != (3,)
                or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
                or not np.all(np.isfinite(target)) or np.any(lower > upper)
                or not np.isfinite(max_rate) or max_rate <= 0.0
                or max_rate > G1_29_WAIST_HARD_MAX_RATE + 1e-9):
            raise ValueError("invalid waist command configuration")
        if self.msg is None or not self.outputs_activated:
            # Dex3 line: the command message only exists after activate().
            raise ValueError("arm writer not activated; waist command unavailable")
        with self.ctrl_lock:
            start = np.array([float(self.msg.motor_cmd[i].q) for i in G1_29_WAIST_INDICES])
            self._waist_written = np.clip(start, lower, upper)
            self._waist_lower, self._waist_upper = lower, upper
            self._waist_max_step = max_rate * self.control_dt
            self._waist_max_rate = max_rate
            self._waist_neutral = np.clip(target, lower, upper)
            self._waist_target = self._waist_neutral.copy()
            self._waist_enabled = True

    def set_waist_target(self, q):
        '''Set the absolute waist target [yaw, roll, pitch] (rad); ignored unless configured.'''
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (3,) or not np.all(np.isfinite(q)):
            raise ValueError("waist target must be 3 finite values")
        with self.ctrl_lock:
            if self._waist_enabled:
                self._waist_target = np.clip(q, self._waist_lower, self._waist_upper)

    def waist_return_to_neutral(self):
        '''Target = neutral captured at r (writer slews at the configured rate). False if not configured.'''
        with self.ctrl_lock:
            if not self._waist_enabled:
                return False
            self._waist_target = self._waist_neutral.copy()
            return True

    def get_waist_command(self):
        '''Snapshot dict {enabled, target, written, neutral, max_rate}; arrays are copies.'''
        with self.ctrl_lock:
            if not self._waist_enabled:
                return {"enabled": False, "target": None, "written": None, "neutral": None, "max_rate": None}
            return {"enabled": True, "target": self._waist_target.copy(), "written": self._waist_written.copy(),
                    "neutral": self._waist_neutral.copy(), "max_rate": self._waist_max_rate}

    def get_waist_command_written(self):
        '''Waist q currently in the outgoing message (what the servos are holding).'''
        with self.ctrl_lock:
            if self.msg is None:
                return None
            return np.array([float(self.msg.motor_cmd[i].q) for i in G1_29_WAIST_INDICES])

    def get_waist_q_snapshot(self):
        '''(measured waist q [yaw, roll, pitch], age s) from one atomic read.'''
        lowstate, timestamp = self.lowstate_buffer.GetSnapshot()
        if lowstate is None:
            return None, float("inf")
        q = np.array([lowstate.motor_state[i].q for i in G1_29_WAIST_INDICES], dtype=float)
        return q, time.monotonic() - timestamp

    def get_arm_command(self):
        '''Return copies of the current (pre-limit) arm q/tau command.'''
        q, tau, _ = self._capture_arm_command()
        return q, tau

    def get_dual_arm_q_snapshot(self):
        '''Return (measured arm q, sample age in seconds) from one atomic read.'''
        lowstate, timestamp = self.lowstate_buffer.GetSnapshot()
        if lowstate is None:
            return None, float("inf")
        q = np.array([lowstate.motor_state[id].q for id in G1_29_JointArmIndex], dtype=float)
        return q, time.monotonic() - timestamp

    def get_publication_status(self):
        '''Nonblocking writer liveness snapshot used by graceful shutdown.'''
        with self.ctrl_lock:
            return {
                "active": self.output_enabled.is_set() and self.publish_thread.is_alive(),
                "last_publish_monotonic": self._last_publish_monotonic,
                "last_published_weight": self._last_published_weight,
                "commanded_weight": self._motion_authority_weight,
            }

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_29_JointArmIndex])

    def ctrl_dual_arm_go_home(self, release_motion_authority=False):
        '''Move both arms home; release rt/arm_sdk authority only during shutdown.'''
        logger_mp.info("[G1_29_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) <= tolerance):
                if self.motion_mode and release_motion_authority:
                    # The writer publishes this weight every frame.
                    for weight in np.linspace(1, 0, num=101):
                        self.set_motion_authority_weight(weight)
                        time.sleep(0.02)
                logger_mp.info("[G1_29_ArmController] both arms have reached the home position.")
                return True
            current_attempts += 1
            time.sleep(0.05)
        logger_mp.error("[G1_29_ArmController] timed out waiting for arms to reach home position.")
        return False

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34

# Waist motors (yaw, roll, pitch) = G1_29_JointIndex 12, 13, 14; same order as
# the Unitree g1_arm7_sdk_dds_example (kWaistYaw, kWaistRoll, kWaistPitch).
G1_29_WAIST_INDICES = (G1_29_JointIndex.kWaistYaw, G1_29_JointIndex.kWaistRoll, G1_29_JointIndex.kWaistPitch)
# Hard ceiling of the waist slew rate at the final writer (= torso_lean
# HARD_MAX_RATE_DPS, 90 deg/s); the writer is the single waist rate limiter.
G1_29_WAIST_HARD_MAX_RATE = np.deg2rad(90.0)


class G1_23_ArmController(_ArmPublicationMixin):
    arm_joint_split = (5, 5)

    def __init__(self, motion_mode = False, simulation_mode = False):
        self.simulation_mode = simulation_mode
        self.motion_mode = motion_mode

        logger_mp.info("Initialize G1_23_ArmController...")
        self.q_target = np.zeros(10)
        self.tauff_target = np.zeros(10)

        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        
        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[G1_23_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[G1_23_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in G1_23_JointArmIndex)
        for id in G1_23_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self._init_arm_publication_state()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize G1_23_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_23_LowState()
                for id in range(G1_23_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[G1_23_JointIndex.kNotUsedJoint0].q = 1.0;

        while True:
            start_time = time.time()

            arm_q_target, arm_tauff_target, request_id = self._capture_arm_command()

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            if not self._arm_command_is_finite(cliped_arm_q_target, arm_tauff_target):
                self._record_failed_arm_publication(request_id, ValueError("non-finite arm command"))
                time.sleep(self.control_dt)
                continue

            for idx, id in enumerate(G1_23_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            try:
                self.lowcmd_publisher.Write(self.msg)
            except Exception as error:
                self._record_failed_arm_publication(request_id, error)
            else:
                self._record_arm_publication(request_id, cliped_arm_q_target, "published", arm_tauff_target)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        request_id = self._set_arm_command(q_target, tauff_target)
        return request_id

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_23_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_23_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_23_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[G1_23_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(10)
            # self.tauff_target = np.zeros(10)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[G1_23_JointIndex.kNotUsedJoint0].q = weight;
                        time.sleep(0.02)
                logger_mp.info("[G1_23_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_23_JointIndex.kLeftAnklePitch.value,
            G1_23_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_23_JointIndex.kLeftShoulderPitch.value,
            G1_23_JointIndex.kLeftShoulderRoll.value,
            G1_23_JointIndex.kLeftShoulderYaw.value,
            G1_23_JointIndex.kLeftElbow.value,
            # Right arm
            G1_23_JointIndex.kRightShoulderPitch.value,
            G1_23_JointIndex.kRightShoulderRoll.value,
            G1_23_JointIndex.kRightShoulderYaw.value,
            G1_23_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_23_JointIndex.kLeftWristRoll.value,
            G1_23_JointIndex.kRightWristRoll.value,
        ]
        return motor_index.value in wrist_motors

class G1_23_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26

class G1_23_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRollNotUsed = 13
    kWaistPitchNotUsed = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitchNotUsed = 20
    kLeftWristyawNotUsed = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitchNotUsed = 27
    kRightWristYawNotUsed = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34

class H1_2_ArmController(_ArmPublicationMixin):
    arm_joint_split = (7, 7)

    def __init__(self, motion_mode = False, simulation_mode = False):
        self.simulation_mode = simulation_mode
        self.motion_mode = motion_mode
        
        logger_mp.info("Initialize H1_2_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)

        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0
        self.kp_wrist = 50.0
        self.kd_wrist = 2.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None


        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[H1_2_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[H1_2_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in H1_2_JointArmIndex)
        for id in H1_2_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self._init_arm_publication_state()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize H1_2_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H1_2_LowState()
                for id in range(H1_2_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[H1_2_JointIndex.kNotUsedJoint0].q = 1.0;

        while True:
            start_time = time.time()

            arm_q_target, arm_tauff_target, request_id = self._capture_arm_command()

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            if not self._arm_command_is_finite(cliped_arm_q_target, arm_tauff_target):
                self._record_failed_arm_publication(request_id, ValueError("non-finite arm command"))
                time.sleep(self.control_dt)
                continue

            for idx, id in enumerate(H1_2_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            try:
                self.lowcmd_publisher.Write(self.msg)
            except Exception as error:
                self._record_failed_arm_publication(request_id, error)
            else:
                self._record_arm_publication(request_id, cliped_arm_q_target, "published", arm_tauff_target)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        request_id = self._set_arm_command(q_target, tauff_target)
        return request_id

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_2_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_2_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H1_2_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[H1_2_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[H1_2_JointIndex.kNotUsedJoint0].q = weight;
                        time.sleep(0.02)
                logger_mp.info("[H1_2_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H1_2_JointIndex.kLeftAnkle.value,
            H1_2_JointIndex.kRightAnkle.value,
            # Left arm
            H1_2_JointIndex.kLeftShoulderPitch.value,
            H1_2_JointIndex.kLeftShoulderRoll.value,
            H1_2_JointIndex.kLeftShoulderYaw.value,
            H1_2_JointIndex.kLeftElbowPitch.value,
            # Right arm
            H1_2_JointIndex.kRightShoulderPitch.value,
            H1_2_JointIndex.kRightShoulderRoll.value,
            H1_2_JointIndex.kRightShoulderYaw.value,
            H1_2_JointIndex.kRightElbowPitch.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            H1_2_JointIndex.kLeftElbowRoll.value,
            H1_2_JointIndex.kLeftWristPitch.value,
            H1_2_JointIndex.kLeftWristyaw.value,
            H1_2_JointIndex.kRightElbowRoll.value,
            H1_2_JointIndex.kRightWristPitch.value,
            H1_2_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors
    
class H1_2_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

class H1_2_JointIndex(IntEnum):
    # Left leg
    kLeftHipYaw = 0
    kLeftHipRoll = 1
    kLeftHipPitch = 2
    kLeftKnee = 3
    kLeftAnkle = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipYaw = 6
    kRightHipRoll = 7
    kRightHipPitch = 8
    kRightKnee = 9
    kRightAnkle = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12

    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

    kNotUsedJoint0 = 27
    kNotUsedJoint1 = 28
    kNotUsedJoint2 = 29
    kNotUsedJoint3 = 30
    kNotUsedJoint4 = 31
    kNotUsedJoint5 = 32
    kNotUsedJoint6 = 33
    kNotUsedJoint7 = 34

class H1_ArmController(_ArmPublicationMixin):
    arm_joint_split = (4, 4)

    def __init__(self, simulation_mode = False):
        self.simulation_mode = simulation_mode
        
        logger_mp.info("Initialize H1_ArmController...")
        self.q_target = np.zeros(8)
        self.tauff_target = np.zeros(8)

        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, go_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, go_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[H1_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[H1_ArmController] Subscribe dds ok.")

        # initialize h1's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_go_msg_dds__LowCmd_()
        self.msg.head[0] = 0xFE
        self.msg.head[1] = 0xEF
        self.msg.level_flag = 0xFF
        self.msg.gpio = 0

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        for id in H1_JointIndex:
            if self._Is_weak_motor(id):
                self.msg.motor_cmd[id].kp = self.kp_low
                self.msg.motor_cmd[id].kd = self.kd_low
                self.msg.motor_cmd[id].mode = 0x01
            else:
                self.msg.motor_cmd[id].kp = self.kp_high
                self.msg.motor_cmd[id].kd = self.kd_high
                self.msg.motor_cmd[id].mode = 0x0A
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self._init_arm_publication_state()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize H1_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H1_LowState()
                for id in range(H1_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        while True:
            start_time = time.time()

            arm_q_target, arm_tauff_target, request_id = self._capture_arm_command()

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            if not self._arm_command_is_finite(cliped_arm_q_target, arm_tauff_target):
                self._record_failed_arm_publication(request_id, ValueError("non-finite arm command"))
                time.sleep(self.control_dt)
                continue

            for idx, id in enumerate(H1_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            try:
                self.lowcmd_publisher.Write(self.msg)
            except Exception as error:
                self._record_failed_arm_publication(request_id, error)
            else:
                self._record_arm_publication(request_id, cliped_arm_q_target, "published", arm_tauff_target)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        request_id = self._set_arm_command(q_target, tauff_target)
        return request_id
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H1_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[H1_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(8)
            # self.tauff_target = np.zeros(8)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                logger_mp.info("[H1_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H1_JointIndex.kLeftAnkle.value,
            H1_JointIndex.kRightAnkle.value,
            # Left arm
            H1_JointIndex.kLeftShoulderPitch.value,
            H1_JointIndex.kLeftShoulderRoll.value,
            H1_JointIndex.kLeftShoulderYaw.value,
            H1_JointIndex.kLeftElbow.value,
            # Right arm
            H1_JointIndex.kRightShoulderPitch.value,
            H1_JointIndex.kRightShoulderRoll.value,
            H1_JointIndex.kRightShoulderYaw.value,
            H1_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
class H1_JointArmIndex(IntEnum):
    # Unlike G1 and H1_2, the arm order in DDS messages for H1 is right then left. 
    # Therefore, the purpose of switching the order here is to maintain consistency with G1 and H1_2.
    # Left arm
    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19
    # Right arm
    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15

class H1_JointIndex(IntEnum):
    kRightHipRoll = 0
    kRightHipPitch = 1
    kRightKnee = 2
    kLeftHipRoll = 3
    kLeftHipPitch = 4
    kLeftKnee = 5
    kWaistYaw = 6
    kLeftHipYaw = 7
    kRightHipYaw = 8
    kNotUsedJoint = 9
    kLeftAnkle = 10
    kRightAnkle = 11
    # Right arm
    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15
    # Left arm
    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19

class H2_ArmController(_ArmPublicationMixin):
    arm_joint_split = (7, 7)

    def __init__(self, motion_mode=False, simulation_mode=False):
        logger_mp.info("Initialize H2_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0
        self.kp_wrist = 50.0
        self.kd_wrist = 2.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None
        
        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[H2_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[H2_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.debug(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.debug(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in H2_JointArmIndex)
        for id in H2_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            logger_mp.info(
                f"Motor {id.value} ({id.name}): kp={self.msg.motor_cmd[id].kp}, kd={self.msg.motor_cmd[id].kd}"
            )
            self.msg.motor_cmd[id].q = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self._init_arm_publication_state()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize H2_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H2_LowState()
                for id in range(35):
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[H2_JointIndex.kNotUsedJoint0].q = 1.0

        while True:
            start_time = time.time()

            arm_q_target, arm_tauff_target, request_id = self._capture_arm_command()

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit=self.arm_velocity_limit)

            if not self._arm_command_is_finite(cliped_arm_q_target, arm_tauff_target):
                self._record_failed_arm_publication(request_id, ValueError("non-finite arm command"))
                time.sleep(self.control_dt)
                continue

            for idx, id in enumerate(H2_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]

            self.msg.crc = self.crc.Crc(self.msg)
            try:
                self.lowcmd_publisher.Write(self.msg)
            except Exception as error:
                self._record_failed_arm_publication(request_id, error)
            else:
                self._record_arm_publication(request_id, cliped_arm_q_target, "published", arm_tauff_target)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)

    def ctrl_dual_arm(self, q_target, tauff_target):
        """Set control target values q & tau of the left and right arm motors."""
        request_id = self._set_arm_command(q_target, tauff_target)
        return request_id

    def get_mode_machine(self):
        """Return current dds mode machine."""
        return self.lowstate_subscriber.Read().mode_machine

    def get_current_motor_q(self):
        """Return current state q of all body motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H2_JointIndex])

    def get_current_dual_arm_q(self):
        """Return current state q of the left and right arm motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H2_JointArmIndex])

    def get_current_dual_arm_dq(self):
        """Return current state dq of the left and right arm motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H2_JointArmIndex])

    def ctrl_dual_arm_go_home(self):
        """Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero."""
        logger_mp.info("[H2_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
        tolerance = 0.05
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[H2_JointIndex.kNotUsedJoint0].q = weight
                        time.sleep(0.02)
                logger_mp.info("[H2_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t=5.0):
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H2_JointIndex.kLeftAnklePitch.value,
            H2_JointIndex.kRightAnklePitch.value,
            # Left arm
            H2_JointIndex.kLeftShoulderPitch.value,
            H2_JointIndex.kLeftShoulderRoll.value,
            H2_JointIndex.kLeftShoulderYaw.value,
            H2_JointIndex.kLeftElbow.value,
            # Right arm
            H2_JointIndex.kRightShoulderPitch.value,
            H2_JointIndex.kRightShoulderRoll.value,
            H2_JointIndex.kRightShoulderYaw.value,
            H2_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors

    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            H2_JointIndex.kLeftWristRoll.value,
            H2_JointIndex.kLeftWristPitch.value,
            H2_JointIndex.kLeftWristyaw.value,
            H2_JointIndex.kRightWristRoll.value,
            H2_JointIndex.kRightWristPitch.value,
            H2_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

class H2_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


class H2_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

    # Head
    kHeadPitch = 29
    kHeadYaw = 30

    # not used
    kNotUsedJoint0 = 31
    kNotUsedJoint1 = 32
    kNotUsedJoint2 = 33
    kNotUsedJoint3 = 34

if __name__ == "__main__":
    from robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
    import pinocchio as pin

    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation

    # arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = False)
    # arm = G1_29_ArmController(simulation_mode=True)
    # arm_ik = G1_23_ArmIK(Unit_Test = True, Visualization = False)
    # arm = G1_23_ArmController()
    # arm_ik = H1_2_ArmIK(Unit_Test = True, Visualization = False)
    # arm = H1_2_ArmController()
    # arm_ik = H1_ArmIK(Unit_Test = True, Visualization = True)
    # arm = H1_ArmController()
    arm_ik = H2_ArmIK(Unit_Test = True, Visualization = False)
    arm = H2_ArmController()


    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program): \n")
    if user_input.lower() == 's':
        step = 0
        arm.speed_gradual_max()
        while True:
            if step <= 120:
                angle = rotation_speed * step
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation += np.array([0.001,  0.001, 0.001])
                R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            else:
                angle = rotation_speed * (240 - step)
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation -= np.array([0.001,  0.001, 0.001])
                R_tf_target.translation -= np.array([0.001, -0.001, 0.001])

            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()

            current_lr_arm_q  = arm.get_current_dual_arm_q()
            current_lr_arm_dq = arm.get_current_dual_arm_dq()

            sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, current_lr_arm_q, current_lr_arm_dq)

            arm.ctrl_dual_arm(sol_q, sol_tauff)

            step += 1
            if step > 240:
                step = 0
            time.sleep(0.01)
