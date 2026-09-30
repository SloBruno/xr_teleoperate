# for motion switcher
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
# for loco client
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
import json
import time

# MotionSwitcher used to switch mode between debug mode and ai mode
class MotionSwitcher:
    def __init__(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(1.0)
        self.msc.Init()

    def Enter_Debug_Mode(self):
        try:
            status, result = self.msc.CheckMode()
            while result['name']:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)
            return status, result
        except Exception as e:
            return None, None
    
    def Exit_Debug_Mode(self):
        try:
            status, result = self.msc.SelectMode(nameOrAlias='ai')
            return status, result
        except Exception as e:
            return None, None

# FSM ids that accept SetVelocity as a walking command (Unitree G1 "Expert
# interface": 500 Walk, 501 Walk with 3-DoF waist).  801/802 are Run.
WALK_FSM_IDS = frozenset({500, 501})


def is_walk_fsm(fsm_id):
    return fsm_id in WALK_FSM_IDS


class LocoClientWrapper:
    def __init__(self):
        self.client = LocoClient()
        self.client.SetTimeout(0.0001)
        self.client.Init()
        self.last_move_code = None
        self.nonzero_move_codes = 0

    def Enter_Damp_Mode(self):
        self.client.Damp()

    def read_fsm_id(self, timeout=1.0):
        """Read-only GetFsmId (7001); returns None when unreadable."""
        try:
            self.client.SetTimeout(timeout)
            code, data = self.client._Call(7001, "{}")
            if code != 0 or not data:
                return None
            return int(json.loads(data)["data"])
        except Exception:
            return None
        finally:
            self.client.SetTimeout(0.0001)

    def Move(self, vx, vy, vyaw):
        # Same wire call as LocoClient.Move(continous_move=False) (duration=1 s)
        # but the RPC status is kept instead of discarded.  NOTE: the client
        # timeout is 0.1 ms (non-blocking), so a reply is rarely in time and
        # code 3104 (timeout) is expected; use read_fsm_id / rt/sportmodestate
        # as the authoritative state, not this code.
        code = self.client.SetVelocity(vx, vy, vyaw, 1.0)
        self.last_move_code = code
        if code != 0:
            self.nonzero_move_codes += 1
        return code


if __name__ == '__main__':
    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation
    ms = MotionSwitcher()
    status, result = ms.Enter_Debug_Mode()
    print("Enter debug mode:", status, result)
    time.sleep(5)
    status, result = ms.Exit_Debug_Mode()
    print("Exit debug mode:", status, result)
    time.sleep(2)
