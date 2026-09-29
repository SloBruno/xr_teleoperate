#!/usr/bin/env python3
"""INERT per-stage timing of the G1 teleop main loop on the robot host.

Never creates DDS participants, publishers, subscribers, Vuer servers, or
motor commands. It only:
  * subscribes (ZMQ SUB, read-only) to the already running Teleimager streams;
  * runs the pure-Python/numpy pieces of one tracking cycle (stack, render
    copy, get_tele_data math, calibrator, IK+FK gate, telemetry build/enqueue)
    against logged real controller samples;
  * optionally starts background threads that emulate the in-process CPU/GIL
    load of the real teleop (arm writer message fill + CRC at 250 Hz, low-state
    copy at 500 Hz, Dex3 state/command fill, XR render copy), without any
    transport write.

Usage (robot, from the deployed teleop dir, read-only):
  PYTHONDONTWRITEBYTECODE=1 python bench_loop_stages.py --pose <pose.jsonl> \
      --out <result.json> [--load] [--cycles 400]
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np

sys.dont_write_bytecode = True


def pct(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "max": round(float(a.max()), 3),
        "mean": round(float(a.mean()), 3),
    }


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "full_pose_telemetry" and row.get("lifecycle") == "tracking":
                rows.append(row)
    return rows


class _Stop:
    flag = False


class _LoadRates:
    dds = False
    lowstate_hz = 500.0
    hand_state_hz = 500.0
    loops = ("lowstate", "handstate", "armwriter", "handcmd")


def start_background_load(render_fn_holder):
    """Emulate in-process threads of teleop_hand_and_arm (no transport)."""
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.utils.crc import CRC
    from teleop.robot_control.robot_arm import (
        G1_29_LowState, DataBuffer, G1_29_JointArmIndex, G1_29_Num_Motors, _ArmPublicationMixin)

    crc = CRC()
    lowcmd = unitree_hg_msg_dds__LowCmd_()
    fake_state_msg = unitree_hg_msg_dds__LowState_()
    buf = DataBuffer()
    threads = []

    def lowstate_loop():  # robot_arm._subscribe_motor_state body without Read()
        while not _Stop.flag:
            s = G1_29_LowState()
            s.mode_machine = fake_state_msg.mode_machine
            for i in range(G1_29_Num_Motors):
                s.motor_state[i].q = fake_state_msg.motor_state[i].q
                s.motor_state[i].dq = fake_state_msg.motor_state[i].dq
            buf.SetData(s)
            time.sleep(0.002)

    class Pub(_ArmPublicationMixin):
        arm_joint_split = (7, 7)

    pub = Pub()
    pub._init_arm_publication_state()

    def arm_writer_loop():  # robot_arm._ctrl_motor_state body without Write()
        rid = 0
        while not _Stop.flag:
            t0 = time.time()
            q = np.array([buf.GetData().motor_state[j].q for j in G1_29_JointArmIndex]) if buf.GetData() else np.zeros(14)
            target = q + 0.001
            for idx, j in enumerate(G1_29_JointArmIndex):
                lowcmd.motor_cmd[j].q = target[idx]
                lowcmd.motor_cmd[j].dq = 0
                lowcmd.motor_cmd[j].tau = 0.0
            lowcmd.crc = crc.Crc(lowcmd)
            rid += 1
            pub._record_arm_publication(rid, target, "published", np.zeros(14))
            time.sleep(max(0.0, 0.004 - (time.time() - t0)))

    def dex3_state_loop():  # two HandState copies at 500 Hz
        from multiprocessing import Array
        arr_l = Array('d', 7, lock=True)
        arr_r = Array('d', 7, lock=True)
        while not _Stop.flag:
            for arr in (arr_l, arr_r):
                with arr.get_lock():
                    for i in range(7):
                        arr[i] = 0.1 * i
            time.sleep(0.002)

    # --- DDS-realistic variant: the SDK deserializes every received sample and
    # serializes every written one in Python (holding the GIL). Emulate that on
    # local bytes only; nothing is ever sent or received on a network.
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_, unitree_hg_msg_dds__HandCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, HandState_
    low_bytes = fake_state_msg.serialize()
    hand_bytes = unitree_hg_msg_dds__HandState_().serialize()
    hand_cmd = unitree_hg_msg_dds__HandCmd_()
    rates = _LoadRates

    def dds_lowstate_loop():
        period = 1.0 / rates.lowstate_hz
        while not _Stop.flag:
            t0 = time.perf_counter()
            msg = LowState_.deserialize(low_bytes)
            s = G1_29_LowState()
            s.mode_machine = msg.mode_machine
            for i in range(G1_29_Num_Motors):
                s.motor_state[i].q = msg.motor_state[i].q
                s.motor_state[i].dq = msg.motor_state[i].dq
            buf.SetData(s)
            time.sleep(max(0.002, period - (time.perf_counter() - t0)))

    def dds_hand_state_loop():
        period = 1.0 / rates.hand_state_hz
        while not _Stop.flag:
            t0 = time.perf_counter()
            for _ in range(2):
                HandState_.deserialize(hand_bytes)
            time.sleep(max(0.002, period - (time.perf_counter() - t0)))

    def dds_arm_writer_loop():
        while not _Stop.flag:
            t0 = time.perf_counter()
            data = buf.GetData()
            q = np.array([data.motor_state[j].q or 0.0 for j in G1_29_JointArmIndex]) if data else np.zeros(14)
            target = q + 0.001
            for idx, j in enumerate(G1_29_JointArmIndex):
                lowcmd.motor_cmd[j].q = target[idx]
                lowcmd.motor_cmd[j].dq = 0
                lowcmd.motor_cmd[j].tau = 0.0
            lowcmd.crc = crc.Crc(lowcmd)
            lowcmd.serialize()  # what ChannelPublisher.Write does in Python
            pub._record_arm_publication(0, target, "published", np.zeros(14))
            time.sleep(max(0.0, 0.004 - (time.perf_counter() - t0)))

    def dds_hand_cmd_loop():
        while not _Stop.flag:
            t0 = time.perf_counter()
            for _ in range(2):
                hand_cmd.serialize()
            time.sleep(max(0.0, 0.01 - (time.perf_counter() - t0)))

    if rates.dds:
        named = {"lowstate": dds_lowstate_loop, "handstate": dds_hand_state_loop,
                 "armwriter": dds_arm_writer_loop, "handcmd": dds_hand_cmd_loop}
        loops = tuple(named[n] for n in rates.loops)
    else:
        loops = (lowstate_loop, arm_writer_loop, dex3_state_loop)
    for fn in loops:
        t = threading.Thread(target=fn, daemon=True)
        t.start()
        threads.append(t)
    return pub, buf, threads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycles", type=int, default=400)
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--img-host", default="192.168.123.164")
    ap.add_argument("--frequency", type=float, default=30.0)
    ap.add_argument("--dds-load", action="store_true", help="emulate SDK (de)serialization per DDS sample")
    ap.add_argument("--lowstate-hz", type=float, default=500.0)
    ap.add_argument("--hand-state-hz", type=float, default=500.0)
    ap.add_argument("--video-thread", action="store_true", help="candidate fix: stack/render in own thread")
    ap.add_argument("--dds-loops", default=None, help="comma list: lowstate,handstate,armwriter,handcmd")
    ap.add_argument("--switch-interval", type=float, default=None)
    args = ap.parse_args()
    _LoadRates.dds = args.dds_load
    if args.dds_loops:
        _LoadRates.loops = tuple(args.dds_loops.split(","))
    _LoadRates.lowstate_hz = args.lowstate_hz
    _LoadRates.hand_state_hz = args.hand_state_hz
    if args.switch_interval:
        sys.setswitchinterval(args.switch_interval)

    import cv2
    # Same code as teleop_hand_and_arm.stack_camera_frames_vertical (74ded80);
    # inlined so this probe never imports the teleop entry module.
    def stack_camera_images_vertical(top_image, bottom_image, scale=0.5, top_crop_bottom=0.89,
                                     bottom_crop_top=0.11, divider_px=4):
        if top_image is None or bottom_image is None:
            return None
        top_frame, bottom_frame = top_image.bgr, bottom_image.bgr
        if top_frame is None or bottom_frame is None:
            return None
        th, tw = top_frame.shape[:2]
        bh, bw = bottom_frame.shape[:2]
        top_end = min(th, max(1, round(th * top_crop_bottom)))
        bottom_start = min(bh - 1, max(0, round(bh * bottom_crop_top)))
        top_frame = top_frame[:top_end]
        bottom_frame = bottom_frame[bottom_start:]
        width = max(1, round(tw * scale))
        top_h = max(1, round(top_frame.shape[0] * width / tw))
        bot_h = max(1, round(bottom_frame.shape[0] * width / bw))
        top_frame = cv2.resize(top_frame, (width, top_h), interpolation=cv2.INTER_AREA)
        bottom_frame = cv2.resize(bottom_frame, (width, bot_h), interpolation=cv2.INTER_AREA)
        divider = np.zeros((divider_px, width, 3), dtype=top_frame.dtype)
        return np.vstack((top_frame, divider, bottom_frame))
    from teleimager.image_client import ZMQ_SubscriberManager
    from teleop.utils.controller_wrist_calibration import ControllerWristCalibrator
    from teleop.utils.arm_tracking_orchestration import _fk_matches_target
    from teleop.utils.full_pose_telemetry import (
        PoseTelemetryJsonlSink, ArmPublicationTelemetryBridge, emit_pose_record_best_effort)
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

    rows = load_rows(args.pose)
    ctrl = [(np.array(r["controller"]["wrist_pose"]["left"]), np.array(r["controller"]["wrist_pose"]["right"])) for r in rows]
    meas = [np.array(r["arm"]["left"]["measured_q"] + r["arm"]["right"]["measured_q"]) for r in rows]
    first = rows[0]["arm"]["calibrated_cartesian_target"]
    W0 = (np.array(first["left"]), np.array(first["right"]))

    mgr = ZMQ_SubscriberManager.get_instance()
    head_port, wrist_port = 55555, 55556
    mgr.subscribe(args.img_host, head_port, request_bgr=True)
    mgr.subscribe(args.img_host, wrist_port, request_bgr=True)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        h = mgr.subscribe(args.img_host, head_port, request_bgr=True)
        w = mgr.subscribe(args.img_host, wrist_port, request_bgr=True)
        if h is not None and h.bgr is not None and w is not None and w.bgr is not None:
            break
        time.sleep(0.05)
    print("frames", None if h.bgr is None else h.bgr.shape, None if w.bgr is None else w.bgr.shape, "fps", h.fps, w.fps)

    # XR render thread emulation: identical to TeleVuer._xr_render_loop
    stacked_probe = stack_camera_images_vertical(h, w, 0.5, 0.89, 0.11, 4)
    img2display = np.zeros(stacked_probe.shape, dtype=np.uint8)
    latest = {"frame": None}
    ev = threading.Event()
    render_times = []

    def render_loop():
        while not _Stop.flag:
            if not ev.wait(timeout=0.1):
                continue
            ev.clear()
            f = latest["frame"]
            if f is None:
                continue
            t0 = time.perf_counter()
            img2display[:] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            render_times.append((time.perf_counter() - t0) * 1000)

    threading.Thread(target=render_loop, daemon=True).start()

    pub = None
    if args.load or args.dds_load:
        pub, _, _ = start_background_load(None)

    video_times = []
    if args.video_thread:
        def video_loop():
            while not _Stop.flag:
                t0 = time.perf_counter()
                hv = mgr.subscribe(args.img_host, head_port, request_bgr=True)
                wv = mgr.subscribe(args.img_host, wrist_port, request_bgr=True)
                sv = stack_camera_images_vertical(hv, wv, 0.5, 0.89, 0.11, 4)
                if sv is not None:
                    latest["frame"] = sv
                    ev.set()
                video_times.append((time.perf_counter() - t0) * 1000)
                time.sleep(max(0.0, 1.0 / 30.0 - (time.perf_counter() - t0)))
        threading.Thread(target=video_loop, daemon=True).start()

    arm_ik = G1_29_ArmIK()
    cal = ControllerWristCalibrator()
    now0 = time.monotonic()
    cal.reset_for_start_request(now0 - 0.01)
    assert cal.calibrate(ctrl[0], W0, now0, now0 - 0.01, now=now0)

    os.makedirs("/tmp/loop_perf_telemetry", exist_ok=True)
    sink = PoseTelemetryJsonlSink("/tmp/loop_perf_telemetry")

    class FakeCtrl:
        publication_receipt_drop_count = 0
        arm_joint_split = (7, 7)

        def drain_arm_publication_receipts(self):
            if pub is None:
                return ()
            got = pub.drain_arm_publication_receipts()
            writer_counts.append(len(got))
            return got

    writer_counts = []
    bridge = ArmPublicationTelemetryBridge(sink, profile="G1_29")
    fake_ctrl = FakeCtrl()

    stages = {k: [] for k in ("img_get", "stack", "render_handoff", "calib_targets", "ik_solve", "fk_gate",
                              "telemetry", "cycle_work", "period", "jpeg_q80")}
    last_start = None
    n = min(args.cycles, len(ctrl) - 1)
    for i in range(1, n + 1):
        start = time.perf_counter()
        if last_start is not None:
            stages["period"].append((start - last_start) * 1000)
        last_start = start
        if not args.video_thread:
            t = time.perf_counter()
            head = mgr.subscribe(args.img_host, head_port, request_bgr=True)
            wrist = mgr.subscribe(args.img_host, wrist_port, request_bgr=True)
            stages["img_get"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            stacked = stack_camera_images_vertical(head, wrist, 0.5, 0.89, 0.11, 4)
            stages["stack"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            latest["frame"] = stacked
            ev.set()
            stages["render_handoff"].append((time.perf_counter() - t) * 1000)

        now = time.monotonic()
        t = time.perf_counter()
        target = cal.targets(ctrl[i], now, now=now)
        stages["calib_targets"].append((time.perf_counter() - t) * 1000)
        if target is not None:
            t = time.perf_counter()
            q, tau = arm_ik.solve_ik(target[0], target[1], meas[i], np.zeros(14))
            stages["ik_solve"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            _fk_matches_target(arm_ik, q, target)
            stages["fk_gate"].append((time.perf_counter() - t) * 1000)
        else:
            q, tau = meas[i], np.zeros(14)
        t = time.perf_counter()
        emit_pose_record_best_effort(
            lambda record: bridge.emit_cycle(record, fake_ctrl), warn=None,
            timestamp=time.time(), timestamp_monotonic=time.monotonic(), lifecycle="tracking",
            controller_sample_timestamp=now, head_pose=np.eye(4),
            left_wrist_pose=ctrl[i][0], right_wrist_pose=ctrl[i][1],
            measured_arm_q=meas[i], commanded_arm_q=None, commanded_arm_q_reason="arm_command_publication_pending",
            arm_command_request_id=i, requested_arm_q=q, selected_arm_q=q, requested_arm_tauff=tau,
            selected_arm_tauff=tau, calibrated_cartesian_target=target, ik_target_accepted=target is not None,
            ik_sample_fresh=True, ik_published=True, ik_hold=False, ik_reason="ik_command_selected",
            arm_publication_drop_count=0, arm_joint_split=(7, 7), dex3_configured=True,
            dex3_measured_q=np.zeros(14), dex3_commanded_q=np.zeros(14),
            dex3_sample_metadata={"left": {"state_valid": True, "state_timestamp": now, "action_valid": True, "action_timestamp": now},
                                  "right": {"state_valid": True, "state_timestamp": now, "action_valid": True, "action_timestamp": now}},
            drop_count=sink.drop_count, now=time.monotonic())
        stages["telemetry"].append((time.perf_counter() - t) * 1000)
        work = time.perf_counter() - start
        stages["cycle_work"].append(work * 1000)
        time.sleep(max(0.0, 1.0 / args.frequency - work))

    # Vuer-side cost (separate process in production): JPEG encode of the display buffer
    for _ in range(60):
        t = time.perf_counter()
        cv2.imencode(".jpg", img2display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        stages["jpeg_q80"].append((time.perf_counter() - t) * 1000)

    _Stop.flag = True
    sink.close()
    result = {k: pct(v) for k, v in stages.items()}
    result["video_thread_iter"] = pct(video_times)
    if writer_counts:
        total_s = sum(stages["period"]) / 1000.0
        result["arm_writer_hz_est"] = round(sum(writer_counts) / max(total_s, 1e-6), 1)
        result["hz_main"] = round(1000.0 / float(np.mean(stages["period"])), 2)
    if pub is not None:
        receipts = len(pub.drain_arm_publication_receipts()) + 0
        result["arm_writer_receipt_drops"] = pub.publication_receipt_drop_count
    result["args"] = vars(args)
    result["render_thread_cvtcolor_copy"] = pct(render_times)
    result["stacked_shape"] = list(stacked_probe.shape)
    result["load"] = args.load
    result["loadavg"] = os.getloadavg()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
