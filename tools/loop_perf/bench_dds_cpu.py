#!/usr/bin/env python3
"""INERT: cost of the pure-Python work the SDK does per DDS sample.

No DomainParticipant, reader, writer or publisher is ever created.  Only IDL
serialize/deserialize round trips on local bytes and the SDK CRC packing.
"""
import json
import sys
import time

import numpy as np

sys.dont_write_bytecode = True

from unitree_sdk2py.idl.default import (  # noqa: E402
    unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_,
    unitree_hg_msg_dds__HandState_, unitree_hg_msg_dds__HandCmd_)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, HandState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402


def timeit(fn, n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    a = np.array(out)
    return {"p50": round(float(np.percentile(a, 50)), 4), "p95": round(float(np.percentile(a, 95)), 4)}


def main():
    low = unitree_hg_msg_dds__LowState_()
    low_bytes = low.serialize()
    hand = unitree_hg_msg_dds__HandState_()
    hand_bytes = hand.serialize()
    cmd = unitree_hg_msg_dds__LowCmd_()
    hcmd = unitree_hg_msg_dds__HandCmd_()
    crc = CRC()
    res = {
        "lowstate_deserialize_ms": timeit(lambda: LowState_.deserialize(low_bytes), 500),
        "handstate_deserialize_ms": timeit(lambda: HandState_.deserialize(hand_bytes), 500),
        "lowcmd_crc_ms": timeit(lambda: crc.Crc(cmd), 500),
        "lowcmd_serialize_ms": timeit(lambda: cmd.serialize(), 500),
        "handcmd_serialize_ms": timeit(lambda: hcmd.serialize(), 500),
        "lowstate_bytes": len(low_bytes),
    }
    rates = {"lowstate_deserialize_ms": 500, "handstate_deserialize_ms": 2 * 500,
             "lowcmd_crc_ms": 250, "lowcmd_serialize_ms": 250, "handcmd_serialize_ms": 2 * 100}
    res["gil_ms_per_s_estimate"] = round(sum(res[k]["p50"] * r for k, r in rates.items()), 1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
