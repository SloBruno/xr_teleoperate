"""Opt-in, non-blocking UDP side channel: teleop -> tools/pose_compare_web.py.

Enabled only when ``XR_POSE_STREAM=1`` (the Inspire launcher exports it with
``XR_POSE_WEB=1``). Per control cycle the teleop calls ``maybe_send`` with the
IK wrist targets (operator hand already in the robot waist frame), the
commanded arm q (sol_q) and the measured arm q (same lowstate buffer the loop
already read). The sender:

* decimates to ``XR_POSE_STREAM_HZ`` (default 50 Hz);
* packs one fixed-size struct (no JSON, no FK, no file I/O);
* ``sendto`` with MSG_DONTWAIT on a non-blocking socket to 127.0.0.1;
* never raises: any failure only increments ``errors``.

Robot FK is computed by the web server, not here.

Torso lean telemetry (docs/torso_lean.md): when ``maybe_send`` gets a
``lean`` dict the packet is the extended ``XPS2`` layout (XPS1 + lean target
pitch/roll, commanded lean pitch/roll, waist q commanded and measured
[yaw, roll, pitch], rad). Without it the packet is byte-identical to XPS1, and
``unpack_sample`` accepts both (old senders keep working; missing fields are
``None``).
"""
from __future__ import annotations

import math
import os
import socket
import struct
import time

MAGIC = b"XPS1"
DEFAULT_PORT = 47555
DEFAULT_HZ = 50.0
N_ARM = 14  # G1_29 arm joints: left 7 then right 7

# magic, seq, t_mono, t_unix, flags, hand_l xyz, hand_r xyz, q_cmd[14], q_meas[14]
_FMT = "<4sIddB3f3f14f14f"
PACKET_SIZE = struct.calcsize(_FMT)
MAGIC2 = b"XPS2"
# XPS1 body + lean_target(pitch, roll), lean_cmd(pitch, roll), waist_cmd[3], waist_meas[3]
_FMT2 = _FMT + "2f2f3f3f"
PACKET_SIZE2 = struct.calcsize(_FMT2)
FLAG_TRACKING = 0x01
FLAG_FRESH = 0x02
FLAG_LEAN_ACTIVE = 0x04

_NAN14 = (float("nan"),) * N_ARM


def _vec(v, n):
    if v is None:
        return (float("nan"),) * n
    out = tuple(float(x) for x in v)
    if len(out) != n:
        raise ValueError(f"expected {n} values, got {len(out)}")
    return out


def pack_sample(seq, t_mono, t_unix, tracking, fresh, hand_l, hand_r, q_cmd, q_meas) -> bytes:
    flags = (FLAG_TRACKING if tracking else 0) | (FLAG_FRESH if fresh else 0)
    return struct.pack(_FMT, MAGIC, int(seq) & 0xFFFFFFFF, float(t_mono), float(t_unix), flags,
                       *_vec(hand_l, 3), *_vec(hand_r, 3), *_vec(q_cmd, N_ARM), *_vec(q_meas, N_ARM))


def pack_sample_lean(seq, t_mono, t_unix, tracking, fresh, hand_l, hand_r, q_cmd, q_meas,
                     lean_active, lean_target, lean_cmd, waist_cmd, waist_meas) -> bytes:
    flags = ((FLAG_TRACKING if tracking else 0) | (FLAG_FRESH if fresh else 0)
             | (FLAG_LEAN_ACTIVE if lean_active else 0))
    return struct.pack(_FMT2, MAGIC2, int(seq) & 0xFFFFFFFF, float(t_mono), float(t_unix), flags,
                       *_vec(hand_l, 3), *_vec(hand_r, 3), *_vec(q_cmd, N_ARM), *_vec(q_meas, N_ARM),
                       *_vec(lean_target, 2), *_vec(lean_cmd, 2), *_vec(waist_cmd, 3), *_vec(waist_meas, 3))


def unpack_sample(data: bytes):
    """Return a dict, or None for anything that is not a valid packet (XPS1 or XPS2)."""
    if len(data) == PACKET_SIZE and data[:4] == MAGIC:
        fmt = _FMT
    elif len(data) == PACKET_SIZE2 and data[:4] == MAGIC2:
        fmt = _FMT2
    else:
        return None
    try:
        v = struct.unpack(fmt, data)
    except struct.error:
        return None
    flags = v[4]
    out = {
        "seq": v[1], "t_mono": v[2], "t_unix": v[3],
        "tracking": bool(flags & FLAG_TRACKING), "fresh": bool(flags & FLAG_FRESH),
        "hand_l": list(v[5:8]), "hand_r": list(v[8:11]),
        "q_cmd": list(v[11:11 + N_ARM]), "q_meas": list(v[11 + N_ARM:11 + 2 * N_ARM]),
        "lean_active": bool(flags & FLAG_LEAN_ACTIVE),
        "lean_target": None, "lean_cmd": None, "waist_cmd": None, "waist_meas": None,
    }
    if fmt is _FMT2:
        b = 11 + 2 * N_ARM
        out.update(lean_target=list(v[b:b + 2]), lean_cmd=list(v[b + 2:b + 4]),
                   waist_cmd=list(v[b + 4:b + 7]), waist_meas=list(v[b + 7:b + 10]))
    return out


def _xyz(pose):
    """Translation of a 4x4 pose (numpy or nested list)."""
    return (float(pose[0][3]), float(pose[1][3]), float(pose[2][3]))


class PoseStreamSender:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, rate_hz: float = DEFAULT_HZ):
        self.addr = (host, int(port))
        self.min_period = 1.0 / float(rate_hz) if rate_hz and rate_hz > 0 else 0.0
        self.seq = 0
        self.sent = 0
        self.errors = 0
        self._next_t = None
        self._last_targets = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._flags = getattr(socket, "MSG_DONTWAIT", 0)

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        if env.get("XR_POSE_STREAM", "0").strip().lower() not in ("1", "true", "yes", "on"):
            return None
        try:
            return cls(host=env.get("XR_POSE_STREAM_HOST", "127.0.0.1"),
                       port=int(env.get("XR_POSE_STREAM_PORT", DEFAULT_PORT)),
                       rate_hz=float(env.get("XR_POSE_STREAM_HZ", DEFAULT_HZ)))
        except Exception:
            return None

    def due(self, now=None) -> bool:
        now = time.monotonic() if now is None else now
        return self._next_t is None or now >= self._next_t - 1e-9

    def maybe_send(self, tracking, left_wrist_pose, right_wrist_pose, q_cmd, q_meas, now=None, lean=None) -> bool:
        """Send one decimated sample. Never raises; returns True when a packet left.

        ``lean`` (optional dict: active, target (pitch, roll), cmd (pitch, roll),
        waist_cmd[3], waist_meas[3]; values may be None) selects the XPS2 layout.
        """
        try:
            now = time.monotonic() if now is None else now
            if not self.due(now):
                return False
            # Fixed-grid decimation; after a long gap restart from now.
            if self._next_t is None or now - self._next_t > self.min_period:
                self._next_t = now + self.min_period
            else:
                self._next_t += self.min_period
            hl = _xyz(left_wrist_pose)
            hr = _xyz(right_wrist_pose)
            targets = hl + hr
            fresh = targets != self._last_targets
            self._last_targets = targets
            if lean is None:
                pkt = pack_sample(self.seq, now, time.time(), tracking, fresh, hl, hr, q_cmd, q_meas)
            else:
                pkt = pack_sample_lean(self.seq, now, time.time(), tracking, fresh, hl, hr, q_cmd, q_meas,
                                       lean.get("active"), lean.get("target"), lean.get("cmd"),
                                       lean.get("waist_cmd"), lean.get("waist_meas"))
            self.seq += 1
            self._sock.sendto(pkt, self._flags, self.addr)
            self.sent += 1
            return True
        except Exception:
            self.errors += 1
            return False

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass
