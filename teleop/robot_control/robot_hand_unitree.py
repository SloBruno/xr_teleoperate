# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_                               # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
# for gripper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
from multiprocessing import Process, Array, Value, Lock

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from teleop.utils.weighted_moving_filter import WeightedMovingFilter
from teleop.utils.dex3_controls import (
    trigger_to_dex3_targets, GripLatch, open_reasons,
)
from teleop.utils.quest_safety import controller_sample_is_fresh
from teleop.utils.haptics import extract_dex3_pressure
from teleop.utils.dex3_telemetry import (
    RateEstimator, extract_hand_snapshot, extract_published_command,
)
from teleop.utils.dex3_state_grace import Dex3StateGrace, state_is_fresh
from teleop.utils.dex3_protection import (
    Dex3HandProtector, ProtectionWarner, extract_protection_state,
)
from teleop.utils.dex3_shutdown_hand import (
    SideShutdownPlan, close_blocker, REASON_TEXT,
)

import logging_mp
logger_mp = logging_mp.getLogger(__name__)


Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"

# PD gains (unchanged); protection never raises them.
Dex3_Kp = 1.5
Dex3_Kd = 0.2
Dex3_Open_Pose = np.zeros(Dex3_Num_Motors)
# Index/middle closed pose = ~87% of the URDF range (joint0 +-1.571, joint1
# +-1.745) with ~0.2 rad margin to the stops (>=0.15 required). Was 73%
# (1.15/1.30) and 50% before; the box was only wrapped partway. More closure =
# more heating. GRIP_HOLD_CMD_LIMIT_RAD (1.47/1.65) stays above these targets.
# Calibrar no teste fisico. Thumb0
# stays neutral; Thumb1/Thumb2 use the Unitree full-grasp targets so the thumb
# reaches full closure at trigger=1.0.
Dex3_Left_Closed_Pose = np.array([
    0.0, 1.05, 1.75,
    -1.37, -1.53, -1.37, -1.53,
])
Dex3_Right_Closed_Pose = np.array([
    0.0, -1.05, -1.75,
    1.37, 1.53, 1.37, 1.53,
])
# Final shutdown frame (DEX3_SHUTDOWN_HAND=close|hold): a joint farther than
# this from its goal (blocked by an object) is left at q_cmd = measured q so
# the frame the firmware keeps after the publisher stops has no squeeze.
SHUTDOWN_FINAL_SQUEEZE_TOL_RAD = 0.15


class Dex3_1_Controller:
    def __init__(self, left_hand_array_in, right_hand_array_in, dual_hand_data_lock = None, dual_hand_state_array_out = None,
                       dual_hand_action_array_out = None, fps = 100.0, Unit_Test = False, simulation_mode = False, xr_motion_data_ready_in = None,
                       left_ctrl_trigger_in = None, right_ctrl_trigger_in = None,
                       left_ctrl_timestamp_in = None, right_ctrl_timestamp_in = None,
                       left_ctrl_sample_in = None, right_ctrl_sample_in = None):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_hand_array_in: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process

        right_hand_array_in: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process

        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array

        dual_hand_state_array_out: [output] Return left(7), right(7) hand motor state

        dual_hand_action_array_out: [output] Return left(7), right(7) hand motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """
        logger_mp.info("Initialize Dex3_1_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        # Dex3 joint targets are controller-trigger owned; no hand-retargeting
        # object is constructed for this controller.

        # Pre-arm is receive-only; command publishers are constructed in the
        # child process only after activate() starts it.
        self.LeftHandCmb_publisher = None
        self.RightHandCmb_publisher = None
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Dex3_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)
        # Verified HandState_ pressure source: press_sensor_state[*].pressure[12].
        # Timestamped, side-local pressure handoff for the parent/XR process.
        self.left_pressure = Value('d', 0.0, lock=True)
        self.right_pressure = Value('d', 0.0, lock=True)
        self.left_pressure_timestamp = Value('d', 0.0, lock=True)
        self.right_pressure_timestamp = Value('d', 0.0, lock=True)
        self._telemetry_lock = threading.Lock()
        self._left_state_valid = False
        self._right_state_valid = False
        self._left_state_timestamp = 0.0
        self._right_state_timestamp = 0.0
        self._left_action_valid = False
        self._right_action_valid = False
        self._left_action_timestamp = 0.0
        self._right_action_timestamp = 0.0
        self._left_action = np.zeros(Dex3_Num_Motors)
        self._right_action = np.zeros(Dex3_Num_Motors)
        # Torque/thermal protection (pure state machines, no I/O); gains untouched.
        self._protectors = {
            "left": Dex3HandProtector(Dex3_Open_Pose),
            "right": Dex3HandProtector(Dex3_Open_Pose),
        }
        self._protection_warner = ProtectionWarner()
        self._left_state_sampled = threading.Event()
        self._right_state_sampled = threading.Event()

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if self._left_state_sampled.is_set() and self._right_state_sampled.is_set():
                break
            time.sleep(0.01)
            logger_mp.warning("[Dex3_1_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex3_1_Controller] Subscribe dds ok.")

        # Save the command-loop arguments, but keep the process absent until
        # the explicit terminal r gate activates actuator authority.
        self._control_process_args = (
            left_hand_array_in, right_hand_array_in, self.left_hand_state_array,
            self.right_hand_state_array, dual_hand_data_lock,
            dual_hand_state_array_out, dual_hand_action_array_out,
            xr_motion_data_ready_in, left_ctrl_trigger_in, right_ctrl_trigger_in,
            left_ctrl_timestamp_in, right_ctrl_timestamp_in,
            left_ctrl_sample_in, right_ctrl_sample_in,
        )
        self.hand_control_process = None
        self.outputs_activated = False

        logger_mp.info("Initialize Dex3_1_Controller OK (passive pre-arm).")

    def activate(self):
        """Start Dex3 command publication after terminal r has been accepted."""
        if self.outputs_activated:
            return
        # Cyclone DDS was initialized by the parent process before this point.
        # Forking here leaves the child with an unusable inherited participant,
        # so keep command publication in a thread of the initialized process.
        self.running = True
        self.hand_control_process = threading.Thread(
            target=self.control_process, args=self._control_process_args, daemon=True)
        self.hand_control_process.start()
        self.outputs_activated = True
        logger_mp.info("[Dex3_1_Controller] Dex3 DDS output activated.")

    def open_and_deactivate(self, open_hold_s=0.5, poll_s=0.01):
        """Shutdown: command the open rest pose for a bounded time, then stop."""
        self._force_open = True
        thread = getattr(self, "hand_control_process", None)
        if getattr(self, "outputs_activated", False) and thread is not None and thread.is_alive():
            deadline = time.monotonic() + max(0.0, float(open_hold_s))
            while time.monotonic() < deadline and thread.is_alive():
                time.sleep(poll_s)
        self.deactivate()

    # ---- shutdown hand mode (DEX3_SHUTDOWN_HAND=close|hold; see
    # teleop/utils/dex3_shutdown_hand.py). ``open`` keeps open_and_deactivate.
    def begin_shutdown_hand(self, mode="close", timeout_s=2.5, poll_s=0.01):
        """Revoke trigger authority and ramp to closed (or hold) under protection.

        Blocks (bounded) until both sides finished the ramp or were blocked by
        a safety rule. The command thread keeps holding the result until
        ``finish_shutdown_hand``. Returns a per-side summary; never raises on
        a missing/inactive thread (nothing is commanded then).
        """
        if mode not in ("close", "hold"):
            raise ValueError(f"unsupported shutdown hand mode {mode!r}")
        if getattr(self, "_shutdown_hand_mode", None) is None:
            self.__dict__.setdefault("_shutdown_plans", {})
            self._shutdown_hand_mode = mode
            try:
                logger_mp.info(f"[Dex3 encerramento] gatilhos ignorados; modo '{mode}': "
                               + ("fechando a mão em rampa" if mode == "close" else "mantendo o último alvo"))
            except Exception:
                pass
        thread = getattr(self, "hand_control_process", None)
        if getattr(self, "outputs_activated", False) and thread is not None and thread.is_alive():
            deadline = time.monotonic() + max(0.0, float(timeout_s))
            while time.monotonic() < deadline and thread.is_alive():
                plans = self.__dict__.get("_shutdown_plans", {})
                if len(plans) == 2 and all(p.done for p in plans.values()):
                    break
                time.sleep(poll_s)
        return self.shutdown_hand_summary()

    def shutdown_hand_summary(self):
        plans = dict(self.__dict__.get("_shutdown_plans", {}))
        return {side: {"mode": p.mode, "done": p.done, "blocked_reason": p.blocked_reason,
                       "duration_s": round(p.duration, 3)} for side, p in plans.items()}

    def finish_shutdown_hand(self, timeout_s=0.3, poll_s=0.01):
        """Publish the final frame (no unmonitored squeeze) and stop the thread."""
        self._shutdown_finalize = True
        thread = getattr(self, "hand_control_process", None)
        if getattr(self, "outputs_activated", False) and thread is not None and thread.is_alive():
            deadline = time.monotonic() + max(0.0, float(timeout_s))
            while (time.monotonic() < deadline and thread.is_alive()
                   and not self.__dict__.get("_shutdown_final_published", False)):
                time.sleep(poll_s)
        self.deactivate()
        return bool(self.__dict__.get("_shutdown_final_published", False))

    def _shutdown_step(self, left_info, right_info):
        """One control cycle while DEX3_SHUTDOWN_HAND=close|hold is active."""
        now = time.monotonic()
        mode = self._shutdown_hand_mode
        plans = self.__dict__.setdefault("_shutdown_plans", {})
        protectors = self.__dict__.get("_protectors") or {}
        last_targets = self.__dict__.get("_last_trigger_target", {})
        finalize = bool(self.__dict__.get("_shutdown_finalize", False))
        closed = {"left": Dex3_Left_Closed_Pose, "right": Dex3_Right_Closed_Pose}
        out = {}
        for side, info in (("left", left_info), ("right", right_info)):
            plan = plans.get(side)
            if plan is None:
                # Ramp from the last published (post-protection) command the
                # servo is tracking; hold mode keeps the last trigger target.
                with self._telemetry_lock:
                    published = (getattr(self, f"_{side}_action", None)
                                 if getattr(self, f"_{side}_action_valid", False) else None)
                start = published if published is not None else last_targets.get(side)
                plan = plans[side] = SideShutdownPlan(
                    side, mode, start, closed[side], Dex3_Open_Pose, now,
                    hold_target=last_targets.get(side))
            protector = protectors.get(side)
            with self._telemetry_lock:
                state = self.__dict__.get("_protection_state", {}).get(side)
            if protector is None:
                reason = "protection_unavailable"
            else:
                reason = close_blocker(state, now, fault_latched=protector.fault_latched,
                                       hot_latched=protector.hot_latched)
            if reason is not None and plan.block(reason):
                try:
                    logger_mp.warning(
                        f"[Dex3 encerramento {side}] NÃO fecha: {REASON_TEXT.get(reason, reason)}; "
                        "mantendo a regra de proteção (abrir/relaxar)")
                except Exception:
                    pass
            target = plan.target(now)
            if protector is None:
                q_cmd, enable, flags = Dex3_Open_Pose.copy(), None, None
            else:
                q_cmd, enable, flags = self._apply_protection_detail(side, now, target)
            if finalize and plan.blocked_reason is None and isinstance(state, dict):
                # The last frame persists in the Dex3 firmware after the
                # publisher stops (mode timeout bit = 0) with no software
                # thermal protection left: joints that did not reach the goal
                # (blocked by an object) get q_cmd = measured q (zero implicit
                # squeeze, hand stays closed around it); reached joints keep
                # the protected closed command (~zero error).
                q_meas = state.get("q") or []
                q_cmd = np.asarray(q_cmd, dtype=float).copy()
                for i in range(Dex3_Num_Motors):
                    qi = q_meas[i] if i < len(q_meas) else None
                    if qi is not None and np.isfinite(qi) and abs(plan.goal[i] - qi) > SHUTDOWN_FINAL_SQUEEZE_TOL_RAD:
                        q_cmd[i] = float(qi)
            info = dict(info)
            info["exact_open_reason"] = plan.blocked_reason
            info["shutdown_hand_mode"] = mode
            self._record_trigger_path(side, info, target, q_cmd)
            out[side] = (np.asarray(q_cmd, dtype=float), enable)
        self.ctrl_dual_hand(out["left"][0], out["right"][0], 0.0, 0.0, out["left"][1], out["right"][1])
        if finalize:
            self._shutdown_final_published = True
        return out["left"][0], out["right"][0]

    def deactivate(self):
        """Stop the Dex3 command thread when terminal q is processed."""
        self.running = False
        if self.hand_control_process is not None and self.hand_control_process.is_alive():
            self.hand_control_process.join(timeout=1.0)
        self.outputs_activated = False
        logger_mp.info("[Dex3_1_Controller] Dex3 DDS output deactivated.")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None:
                # Update left hand state
                with self.left_hand_state_array.get_lock():
                    for idx, id in enumerate(Dex3_1_Left_JointIndex):
                        self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                with self._telemetry_lock:
                    self._left_state_valid = True
                    self._left_state_timestamp = time.monotonic()
                self._record_extended_state("left", left_hand_msg, Dex3_1_Left_JointIndex, time.monotonic())
                self._record_protection_state("left", left_hand_msg, Dex3_1_Left_JointIndex, time.monotonic())
                self._left_state_sampled.set()
                with self.left_pressure.get_lock():
                    self.left_pressure.value = extract_dex3_pressure(left_hand_msg)
                with self.left_pressure_timestamp.get_lock():
                    self.left_pressure_timestamp.value = time.monotonic()
            if right_hand_msg is not None:
                # Update right hand state
                with self.right_hand_state_array.get_lock():
                    for idx, id in enumerate(Dex3_1_Right_JointIndex):
                        self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
                with self._telemetry_lock:
                    self._right_state_valid = True
                    self._right_state_timestamp = time.monotonic()
                self._record_extended_state("right", right_hand_msg, Dex3_1_Right_JointIndex, time.monotonic())
                self._record_protection_state("right", right_hand_msg, Dex3_1_Right_JointIndex, time.monotonic())
                self._right_state_sampled.set()
                with self.right_pressure.get_lock():
                    self.right_pressure.value = extract_dex3_pressure(right_hand_msg)
                with self.right_pressure_timestamp.get_lock():
                    self.right_pressure_timestamp.value = time.monotonic()
            time.sleep(0.002)

    extended_telemetry_failure_count = 0

    def _note_extended_failure(self):
        """Count a telemetry failure; must itself never raise."""
        try:
            lock = self.__dict__.get("_telemetry_lock")
            if lock is None:
                self.extended_telemetry_failure_count += 1
            else:
                with lock:
                    self.extended_telemetry_failure_count += 1
        except Exception:
            pass

    def _record_extended_state(self, side, hand_msg, joint_ids, timestamp):
        """Side-channel: keep the latest extended HandState_ snapshot in memory.

        Pure in-memory bookkeeping (no I/O); a failure only bumps a counter.
        """
        try:
            snapshot = extract_hand_snapshot(hand_msg, joint_ids)
            with self._telemetry_lock:
                store = self.__dict__.setdefault("_extended_state", {})
                rates = self.__dict__.setdefault("_extended_rate", {})
                estimator = rates.setdefault(side, RateEstimator())
                estimator.add(timestamp)
                store[side] = {
                    "state": snapshot,
                    "state_timestamp": timestamp,
                    "state_count": estimator.count,
                    "rate_hz": estimator.rate_hz(),
                }
        except Exception:
            self._note_extended_failure()

    def _record_protection_state(self, side, hand_msg, joint_ids, timestamp):
        """Keep the latest protection inputs in memory (no I/O; never raises)."""
        try:
            snapshot = extract_protection_state(hand_msg, joint_ids, timestamp)
            with self._telemetry_lock:
                self.__dict__.setdefault("_protection_state", {})[side] = snapshot
        except Exception:
            self._note_extended_failure()

    def _apply_protection_detail(self, side, now, target, *, warn_state_stale=True):
        """Return (q_cmd, enable, flags); a stale result is never fresh feedback."""
        protectors = self.__dict__.get("_protectors")
        if not protectors:
            return target, None, None
        try:
            with self._telemetry_lock:
                state = self.__dict__.get("_protection_state", {}).get(side)
            result = protectors[side].update(now, target, state)
            warner = self.__dict__.get("_protection_warner")
            if warner is not None and result.active:
                active = result.active if warn_state_stale else {
                    key: text for key, text in result.active.items() if key[0] != "state_stale"
                }
                for message in warner.messages(now, side, active):
                    try:
                        logger_mp.warning(message)
                    except Exception:
                        pass
            flags = result.flags()
            self._record_protection_flags(side, flags)
            return result.q_cmd, result.enable, flags
        except Exception:
            # Fail safe: unexpected protection failure -> open rest pose.
            self._note_extended_failure()
            return Dex3_Open_Pose.copy(), None, None

    def _apply_protection(self, side, now, target):
        """Compatibility wrapper: return the public two-value protection result."""
        q_cmd, enable, _ = self._apply_protection_detail(side, now, target)
        return q_cmd, enable

    def _protection_state_freshness(self, side, now):
        """Snapshot state+age under one lock; all timestamps are monotonic."""
        with self._telemetry_lock:
            state = self.__dict__.get("_protection_state", {}).get(side)
        ts = state.get("timestamp") if isinstance(state, dict) else None
        try:
            age_s = now - ts if np.isfinite(ts) and np.isfinite(now) else None
        except TypeError:
            age_s = None
        return state_is_fresh(now, state), age_s

    @staticmethod
    def _safe_to_retain(flags, enable):
        """Legacy whole-hand predicate retained for callers/tests outside grace."""
        return all(Dex3_1_Controller._safe_joints_to_retain(flags, enable))

    @staticmethod
    def _safe_joints_to_retain(flags, enable):
        """Return per-joint cache eligibility from fresh protection output only."""
        try:
            fault = list(flags["fault"])
            derate = [float(v) for v in flags["derate"]]
            enabled = [True] * 7 if enable is None else [bool(v) for v in enable]
        except (KeyError, TypeError, ValueError):
            return [False] * 7
        if len(fault) != 7 or len(derate) != 7 or len(enabled) != 7:
            return [False] * 7
        return [not bool(fault[i]) and enabled[i] and np.isfinite(derate[i]) and derate[i] > 0.0
                for i in range(7)]

    def _warn_state_grace(self, side, event, age_s):
        if event is None:
            return
        detail = "?" if age_s is None else f"{age_s * 1000.0:.0f} ms"
        messages = {
            "state_gap_started": f"[Dex3 protecao {side}] estado DDS antigo/ausente ({detail}): retendo comando protegido sem abrir automaticamente",
            "state_gap_expired": f"[Dex3 protecao {side}] estado DDS ainda ausente ({detail}): graca expirou, comando aberto fail-safe",
            "state_gap_recovered": f"[Dex3 protecao {side}] estado DDS recuperado ({detail}): protecao recalculada",
        }
        try:
            logger_mp.warning(messages[event])
        except Exception:
            pass

    def _record_trigger_path(self, side, info, q_target, q_cmd):
        """Side-channel (in-memory, no I/O): trigger path of the latest cycle."""
        try:
            flags = self.__dict__.get("_protection_flags", {}).get(side)
            item = dict(info)
            item["open_reasons"] = open_reasons(
                info["trigger_state"], info["trigger_effective"], flags,
                force_open=(info.get("exact_open_reason") == "stop"),
            )
            if item.get("exact_open_reason") and item["exact_open_reason"] not in item["open_reasons"]:
                item["open_reasons"].insert(0, item["exact_open_reason"])
            item["q_target"] = [round(float(v), 4) for v in q_target]
            item["q_cmd"] = [round(float(v), 4) for v in q_cmd]
            state = self.__dict__.get("_extended_state", {}).get(side, {}).get("state", {})
            measured = [j.get("q") for j in (state.get("joints") or [])[3:7]]
            previous = self.__dict__.setdefault("_trigger_measured", {}).get(side)
            item["measured_q"] = measured if len(measured) == 4 else None
            item["measured_q_drop_event"] = None
            if (previous and item["measured_q"] and info.get("trigger_effective", 0.0) > 0.05
                    and all(v is not None for v in item["measured_q"])):
                drops = [abs(a) - abs(b) for a, b in zip(previous, item["measured_q"])]
                if all(d > 0.2 for d in drops):
                    item["measured_q_drop_event"] = {"drops": [round(d, 3) for d in drops],
                                                       "q_cmd": item["q_cmd"],
                                                       "trigger_effective": info["trigger_effective"]}
            if item["measured_q"] and all(v is not None for v in item["measured_q"]):
                self.__dict__.setdefault("_trigger_measured", {})[side] = item["measured_q"]
            item["pressure_peak"] = (self.left_pressure if side == "left" else self.right_pressure).value \
                if hasattr(self, "left_pressure") else None
            with self._telemetry_lock:
                self.__dict__.setdefault("_trigger_path", {})[side] = item
        except Exception:
            self._note_extended_failure()

    def _record_protection_flags(self, side, flags):
        try:
            with self._telemetry_lock:
                self.__dict__.setdefault("_protection_flags", {})[side] = flags
        except Exception:
            pass

    def get_extended_samples(self):
        """Latest extended state + published command per side (None if absent)."""
        out = {"left": None, "right": None}
        try:
            with self._telemetry_lock:
                states = dict(self.__dict__.get("_extended_state", {}))
                commands = dict(self.__dict__.get("_published_command", {}))
                failures = self.extended_telemetry_failure_count
            for side in out:
                state = states.get(side)
                command = commands.get(side)
                if state is None and command is None:
                    continue
                item = dict(state) if state else {"state": None}
                if command:
                    item.update(command)
                item["failure_count"] = failures
                flags = self.__dict__.get("_protection_flags", {}).get(side)
                if flags is not None:
                    item["protection"] = flags
                tp = self.__dict__.get("_trigger_path", {}).get(side)
                if tp is not None:
                    item["trigger_path"] = tp
                out[side] = item
        except Exception:
            self._note_extended_failure()
        return out

    def _record_published_command(self, side, msg, joint_ids):
        try:
            snapshot = extract_published_command(msg.motor_cmd, joint_ids)
            with self._telemetry_lock:
                store = self.__dict__.setdefault("_published_command", {})
                previous = store.get(side) or {}
                store[side] = {
                    "published_command": snapshot,
                    "command_timestamp": time.monotonic(),
                    "command_count": previous.get("command_count", 0) + 1,
                }
        except Exception:
            self._note_extended_failure()

    def get_pressure_samples(self):
        """Read the latest timestamped pressure sample for each Dex3 side."""
        with self.left_pressure.get_lock(), self.left_pressure_timestamp.get_lock():
            left = (self.left_pressure.value, self.left_pressure_timestamp.value)
        with self.right_pressure.get_lock(), self.right_pressure_timestamp.get_lock():
            right = (self.right_pressure.value, self.right_pressure_timestamp.value)
        return left, right

    def get_pose_samples(self):
        """Return only DDS/control samples that have actually been observed."""
        with self.left_hand_state_array.get_lock():
            left_state = np.asarray(self.left_hand_state_array[:], dtype=float).copy()
        with self.right_hand_state_array.get_lock():
            right_state = np.asarray(self.right_hand_state_array[:], dtype=float).copy()
        with self._telemetry_lock:
            left_action = self._left_action.copy()
            right_action = self._right_action.copy()
            metadata = {
                "left": {
                    "state_valid": self._left_state_valid,
                    "state_timestamp": self._left_state_timestamp,
                    "action_valid": self._left_action_valid,
                    "action_timestamp": self._left_action_timestamp,
                },
                "right": {
                    "state_valid": self._right_state_valid,
                    "state_timestamp": self._right_state_timestamp,
                    "action_valid": self._right_action_valid,
                    "action_timestamp": self._right_action_timestamp,
                },
            }
        return np.concatenate((left_state, right_state)), np.concatenate((left_action, right_action)), metadata
    
    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    @staticmethod
    def _set_gains(motor_cmd, enabled):
        """Faulted/disabled motor: stop commanding torque (kp=kd=0, tau=0)."""
        motor_cmd.kp = Dex3_Kp if enabled else 0.0
        motor_cmd.kd = Dex3_Kd if enabled else 0.0
        motor_cmd.tau = 0.0

    def ctrl_dual_hand(self, left_q_target, right_q_target,
                       left_sample_timestamp=0.0, right_sample_timestamp=0.0,
                       left_enable=None, right_enable=None):
        """Publish already-authorized targets; authority is decided in control_step."""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
            if left_enable is not None:
                self._set_gains(self.left_msg.motor_cmd[id], left_enable[idx])
        self.LeftHandCmb_publisher.Write(self.left_msg)
        self._record_published_command("left", self.left_msg, Dex3_1_Left_JointIndex)

        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]
            if right_enable is not None:
                self._set_gains(self.right_msg.motor_cmd[id], right_enable[idx])
        self.RightHandCmb_publisher.Write(self.right_msg)
        self._record_published_command("right", self.right_msg, Dex3_1_Right_JointIndex)

    def control_step(self, left_hand_array_in, right_hand_array_in,
                     left_ctrl_trigger_in=None, right_ctrl_trigger_in=None,
                     xr_motion_data_ready=True, previous_targets=None,
                     left_ctrl_timestamp_in=None, right_ctrl_timestamp_in=None,
                     left_ctrl_sample_in=None, right_ctrl_sample_in=None):
        """Publish Dex3 targets from controller triggers only.

        Hand-array and XR-readiness parameters are retained only to avoid
        breaking the existing process call signature; they intentionally have
        no authority over finger targets.
        """

        # Read each controller value and its monotonic timestamp atomically.
        # A missing, invalid, or stale sample fails open at the publisher.
        if left_ctrl_sample_in is not None:
            with left_ctrl_sample_in.get_lock():
                left_trigger, left_sample_timestamp = left_ctrl_sample_in[:]
        else:
            left_trigger, left_sample_timestamp = 0.0, 0.0
        if right_ctrl_sample_in is not None:
            with right_ctrl_sample_in.get_lock():
                right_trigger, right_sample_timestamp = right_ctrl_sample_in[:]
        else:
            right_trigger, right_sample_timestamp = 0.0, 0.0
        # Per-side latch owns grip authority. It bridges observed controller gaps
        # up to 2 s and requires 0.6 s of fresh low samples to deliberately open.
        now_t = time.monotonic()
        latches = self.__dict__.setdefault("_grip_latches", {"left": GripLatch(), "right": GripLatch()})
        stopping = bool(getattr(self, "_force_open", False))
        # Shutdown close/hold also revokes trigger authority (latches stopped).
        revoke = stopping or getattr(self, "_shutdown_hand_mode", None) is not None
        left_info = latches["left"].update_sample(left_trigger, left_sample_timestamp, now_t, stop=revoke)
        right_info = latches["right"].update_sample(right_trigger, right_sample_timestamp, now_t, stop=revoke)
        left_trigger = left_info["trigger_effective"]
        right_trigger = right_info["trigger_effective"]
        if stopping:
            # Explicit lifecycle stop wins over latches and publishes rest once.
            left_q_target = Dex3_Open_Pose.copy()
            right_q_target = Dex3_Open_Pose.copy()
            self._record_trigger_path("left", left_info, left_q_target, left_q_target)
            self._record_trigger_path("right", right_info, right_q_target, right_q_target)
            self.ctrl_dual_hand(left_q_target, right_q_target)
            return left_q_target, right_q_target
        if getattr(self, "_shutdown_hand_mode", None) is not None:
            # DEX3_SHUTDOWN_HAND=close|hold: triggers have no authority (the
            # latches above were revoked with stop=True); ramp/hold under the
            # normal protection.
            return self._shutdown_step(left_info, right_info)

        # Dex3 finger targets are controller-only: released is the explicit
        # open pose and trigger travel interpolates to the explicit close pose.
        # Hand tracking still supplies arm/wrist tracking elsewhere, but never
        # contributes to Dex3 joint targets.
        left_q_target = trigger_to_dex3_targets(
            left_trigger, Dex3_Open_Pose, Dex3_Left_Closed_Pose)
        right_q_target = trigger_to_dex3_targets(
            right_trigger, Dex3_Open_Pose, Dex3_Right_Closed_Pose)
        self.__dict__["_last_trigger_target"] = {
            "left": left_q_target.copy(), "right": right_q_target.copy()}

        # A short DDS state gap cannot be passed to protection as if it were a
        # new feedback sample.  Instead retain only the last output protection
        # already approved from fresh, non-faulted, sub-cutoff feedback.
        now = time.monotonic()
        left_pre, right_pre = left_q_target.copy(), right_q_target.copy()
        graces = self.__dict__.setdefault("_state_graces", {
            "left": Dex3StateGrace(), "right": Dex3StateGrace(),
        })

        def protected_or_grace(side, target, info):
            # Test/pre-arm compatibility: without configured protectors there is
            # no state feedback authority and this class historically bypassed.
            if not self.__dict__.get("_protectors"):
                info.update({"state_age_ms": None, "state_valid": False,
                             "state_grace_state": "expired", "held_command": None,
                             "state_hold_duration_s": 0.0, "state_grace_reason": "protection_unavailable",
                             "state_gap_count": 0, "state_gap_max_s": 0.0})
                return target, None
            fresh, age_s = self._protection_state_freshness(side, now)
            grip_active = info["trigger_effective"] > 0.05 and info["trigger_state"] in ("active", "held_stale")
            if fresh:
                q_cmd, enable, flags = self._apply_protection_detail(side, now, target)
                decision = graces[side].update(
                    now, fresh=True, grip_active=grip_active,
                    safe=self._safe_joints_to_retain(flags, enable), q_cmd=q_cmd, enable=enable,
                    open_q=Dex3_Open_Pose,
                )
            else:
                decision = graces[side].update(
                    now, fresh=False, grip_active=grip_active, safe=False,
                    gap_eligible=(age_s is None or age_s >= 0.0), open_q=Dex3_Open_Pose,
                    fallback_q=target,
                )
                if decision["q_cmd"] is not None:
                    q_cmd, enable = decision["q_cmd"], decision["enable"]
                else:
                    # Deliberate trigger release still owns the open command.
                    q_cmd, enable = Dex3_Open_Pose.copy(), None
            info.update({
                "state_age_ms": None if age_s is None else round(age_s * 1000.0, 1),
                "state_valid": bool(fresh),
                "state_grace_state": decision["state"],
                "held_command": decision["held_command"],
                "state_hold_duration_s": decision["hold_duration_s"],
                "state_grace_reason": decision["reason"],
                "state_gap_count": decision["gap_count"],
                "state_gap_max_s": decision["gap_max_s"],
                "state_grace_joint_hold": decision["state_grace_joint_hold"],
                "state_grace_blocked_joints": decision["state_grace_blocked_joints"],
            })
            self._warn_state_grace(side, decision["warning"], age_s)
            return q_cmd, enable

        left_q_target, left_enable = protected_or_grace("left", left_q_target, left_info)
        right_q_target, right_enable = protected_or_grace("right", right_q_target, right_info)
        self._record_trigger_path("left", left_info, left_pre, left_q_target)
        self._record_trigger_path("right", right_info, right_pre, right_q_target)

        self.ctrl_dual_hand(
            left_q_target, right_q_target, left_sample_timestamp, right_sample_timestamp,
            left_enable, right_enable)
        return left_q_target, right_q_target
    
    def control_process(self, left_hand_array_in, right_hand_array_in, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array_out = None, dual_hand_action_array_out = None, xr_motion_data_ready_in = None,
                              left_ctrl_trigger_in = None, right_ctrl_trigger_in = None,
                              left_ctrl_timestamp_in = None, right_ctrl_timestamp_in = None,
                              left_ctrl_sample_in = None, right_ctrl_sample_in = None):

        # DDS is already initialized in this process; construct publishers only
        # after the explicit terminal-r activation gate.
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        left_q_target  = np.full(Dex3_Num_Motors, 0)
        right_q_target = np.full(Dex3_Num_Motors, 0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2
        assert kp == Dex3_Kp and kd == Dex3_Kd

        # initialize dex3-1's left hand cmd msg
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q    = q
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = kp
            self.left_msg.motor_cmd[id].kd   = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode  
            self.right_msg.motor_cmd[id].q    = q
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd  

        try:
            while self.running:
                start_time = time.time()
                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                left_q_target, right_q_target = self.control_step(
                    left_hand_array_in, right_hand_array_in,
                    left_ctrl_trigger_in, right_ctrl_trigger_in,
                    left_ctrl_timestamp_in=left_ctrl_timestamp_in,
                    right_ctrl_timestamp_in=right_ctrl_timestamp_in,
                    left_ctrl_sample_in=left_ctrl_sample_in,
                    right_ctrl_sample_in=right_ctrl_sample_in,
                )

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array_out and dual_hand_action_array_out:
                    with dual_hand_data_lock:
                        dual_hand_state_array_out[:] = state_data
                        dual_hand_action_array_out[:] = action_data
                with self._telemetry_lock:
                    self._left_action = left_q_target.copy()
                    self._right_action = right_q_target.copy()
                    action_timestamp = time.monotonic()
                    self._left_action_timestamp = action_timestamp
                    self._right_action_timestamp = action_timestamp
                    self._left_action_valid = True
                    self._right_action_valid = True

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex3_1_Controller has been closed.")

class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"

class Dex1_1_Gripper_Controller:
    def __init__(self, left_gripper_value_in, right_gripper_value_in, dual_gripper_data_lock = None, dual_gripper_state_out = None, dual_gripper_action_out = None, 
                       filter = True, fps = 200.0, Unit_Test = False, simulation_mode = False, xr_motion_data_ready_in = None):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_gripper_value_in: [input] Left ctrl data (required from XR device) to control_thread

        right_gripper_value_in: [input] Right ctrl data (required from XR device) to control_thread

        dual_gripper_data_lock: Data synchronization lock for dual_gripper_state_array and dual_gripper_action_array

        dual_gripper_state_out: [output] Return left(1), right(1) gripper motor state

        dual_gripper_action_out: [output] Return left(1), right(1) gripper motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """

        logger_mp.info("Initialize Dex1_1_Gripper_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.gripper_sub_ready = False
        self.simulation_mode = simulation_mode
        
        if filter and not self.simulation_mode:
            self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), 2)
        else:
            self.smooth_filter = None
 
        # initialize handcmd publisher and handstate subscriber
        self.LeftGripperCmb_publisher = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self.LeftGripperCmb_publisher.Init()
        self.RightGripperCmb_publisher = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self.RightGripperCmb_publisher.Init()

        self.LeftGripperState_subscriber = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self.LeftGripperState_subscriber.Init()
        self.RightGripperState_subscriber = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self.RightGripperState_subscriber.Init()

        # Shared Arrays for gripper states
        self.left_gripper_state_value = Value('d', 0.0, lock=True)
        self.right_gripper_state_value = Value('d', 0.0, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while not self.gripper_sub_ready:
            time.sleep(0.01)
            logger_mp.warning("[Dex1_1_Gripper_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex1_1_Gripper_Controller] Subscribe dds ok.")

        self.gripper_control_thread = threading.Thread(target=self.control_thread, args=(left_gripper_value_in, right_gripper_value_in, self.left_gripper_state_value, self.right_gripper_state_value,
                                                                                         dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out, xr_motion_data_ready_in))
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        logger_mp.info("Initialize Dex1_1_Gripper_Controller OK!")

    def _subscribe_gripper_state(self):
        while True:
            left_gripper_msg  = self.LeftGripperState_subscriber.Read()
            right_gripper_msg  = self.RightGripperState_subscriber.Read()
            if left_gripper_msg is not None and right_gripper_msg is not None:
                self.left_gripper_state_value.value = left_gripper_msg.states[0].q
                self.right_gripper_state_value.value = right_gripper_msg.states[0].q
                self.gripper_sub_ready = True
            time.sleep(0.002)
    
    def ctrl_dual_gripper(self, dual_gripper_action):
        """set current left, right gripper motor cmd target q"""
        self.left_gripper_msg.cmds[0].q  = dual_gripper_action[0]
        self.right_gripper_msg.cmds[0].q = dual_gripper_action[1]

        self.LeftGripperCmb_publisher.Write(self.left_gripper_msg)
        self.RightGripperCmb_publisher.Write(self.right_gripper_msg)
        # logger_mp.debug("gripper ctrl publish ok.")
    
    def control_thread(self, left_gripper_value_in, right_gripper_value_in, left_gripper_state_value, right_gripper_state_value, dual_hand_data_lock = None, 
                             dual_gripper_state_out = None, dual_gripper_action_out = None, xr_motion_data_ready_in = None):
        self.running = True
        DELTA_GRIPPER_CMD = 0.18     # The motor rotates 5.4 radians, the clamping jaw slide open 9 cm, so 0.6 rad <==> 1 cm, 0.18 rad <==> 3 mm
        THUMB_INDEX_DISTANCE_MIN = 5.0
        THUMB_INDEX_DISTANCE_MAX = 7.0
        LEFT_MAPPED_MIN  = 0.0           # The minimum initial motor position when the gripper closes at startup.
        RIGHT_MAPPED_MIN = 0.0           # The minimum initial motor position when the gripper closes at startup.
        # The maximum initial motor position when the gripper closes before calibration (with the rail stroke calculated as 0.6 cm/rad * 9 rad = 5.4 cm).
        LEFT_MAPPED_MAX = LEFT_MAPPED_MIN + 5.40 
        RIGHT_MAPPED_MAX = RIGHT_MAPPED_MIN + 5.40
        left_target_action  = (LEFT_MAPPED_MAX - LEFT_MAPPED_MIN) / 2.0
        right_target_action = (RIGHT_MAPPED_MAX - RIGHT_MAPPED_MIN) / 2.0

        dq = 0.0
        tau = 0.0
        kp = 5.00
        kd = 0.05
        # initialize gripper cmd msg
        self.left_gripper_msg  = MotorCmds_()
        self.left_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self.right_gripper_msg = MotorCmds_()
        self.right_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]

        self.left_gripper_msg.cmds[0].dq  = dq
        self.left_gripper_msg.cmds[0].tau = tau
        self.left_gripper_msg.cmds[0].kp  = kp
        self.left_gripper_msg.cmds[0].kd  = kd

        self.right_gripper_msg.cmds[0].dq  = dq
        self.right_gripper_msg.cmds[0].tau = tau
        self.right_gripper_msg.cmds[0].kp  = kp
        self.right_gripper_msg.cmds[0].kd  = kd
        try:
            while self.running:
                start_time = time.time()
                # get dual hand skeletal point state from XR device
                with left_gripper_value_in.get_lock():
                    left_gripper_value  = left_gripper_value_in.value
                with right_gripper_value_in.get_lock():
                    right_gripper_value = right_gripper_value_in.value
                if xr_motion_data_ready_in is not None:
                    with xr_motion_data_ready_in.get_lock():
                        xr_motion_data_ready = xr_motion_data_ready_in.value
                else:
                    xr_motion_data_ready = True
                # get current dual gripper motor state
                dual_gripper_state = np.array([left_gripper_state_value.value, right_gripper_state_value.value])

                if xr_motion_data_ready:
                    # Linear mapping from [0, THUMB_INDEX_DISTANCE_MAX] to gripper action range
                    left_target_action  = np.interp(left_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [LEFT_MAPPED_MIN, LEFT_MAPPED_MAX])
                    right_target_action = np.interp(right_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [RIGHT_MAPPED_MIN, RIGHT_MAPPED_MAX])
                else:
                    left_target_action = dual_gripper_state[0]
                    right_target_action = dual_gripper_state[1]
                # clip dual gripper action to avoid overflow
                if not self.simulation_mode:
                    left_actual_action  = np.clip(left_target_action,  dual_gripper_state[0] - DELTA_GRIPPER_CMD, dual_gripper_state[0] + DELTA_GRIPPER_CMD) 
                    right_actual_action = np.clip(right_target_action, dual_gripper_state[1] - DELTA_GRIPPER_CMD, dual_gripper_state[1] + DELTA_GRIPPER_CMD)
                else:
                    left_actual_action  = left_target_action
                    right_actual_action = right_target_action
                dual_gripper_action = np.array([left_actual_action, right_actual_action])

                if self.smooth_filter:
                    self.smooth_filter.add_data(dual_gripper_action)
                    dual_gripper_action = self.smooth_filter.filtered_data

                if dual_gripper_state_out and dual_gripper_action_out:
                    with dual_hand_data_lock:
                        dual_gripper_state_out[:] = dual_gripper_state - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])
                        dual_gripper_action_out[:] = dual_gripper_action - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])

                self.ctrl_dual_gripper(dual_gripper_action)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex1_1_Gripper_Controller has been closed.")

class Gripper_JointIndex(IntEnum):
    kGripper = 0


if __name__ == "__main__":
    import argparse
    from televuer import TeleVuerWrapper
    from teleimager import ImageClient

    parser = argparse.ArgumentParser()
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'], help='Select end effector controller')
    args = parser.parse_args()
    logger_mp.info(f"args:{args}\n")

    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation
    
    # image client
    img_client = ImageClient(host='127.0.0.1') #host='192.168.123.164'
    if not img_client.has_head_cam():
        logger_mp.error("Head camera is required. Please enable head camera on the image server side.")
    head_img_shape = img_client.get_head_shape()
    tv_binocular = img_client.head_is_binocular()

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVuerWrapper(binocular=tv_binocular, use_hand_tracking=args.xr_mode == "hand", img_shape=head_img_shape, return_hand_rot_data = False)

# end-effector
    if args.ee == "dex3":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
    elif args.ee == "dex1":
        left_gripper_value = Value('d', 0.0, lock=True)        # [input]
        right_gripper_value = Value('d', 0.0, lock=True)       # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == 's':
        while True:
            head_img, head_img_fps = img_client.get_head_frame()
            tv_wrapper.set_display_image(head_img)
            tele_data = tv_wrapper.get_tele_data()
            if args.ee == "dex3" and args.xr_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex1" and args.xr_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.xr_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass

            # with dual_hand_data_lock:
            #     logger_mp.info(f"state : {list(dual_hand_state_array)} \naction: {list(dual_hand_action_array)} \n")
            time.sleep(0.01)
