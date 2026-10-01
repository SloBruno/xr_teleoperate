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
        self.stop_count = 0
        self.stop_failures = 0
        self.last_stop_code = None
        self.last_stop_reason = None
        self.sender = None
        self.watchdog_stop_code = None
        self._cfg_client = None

    # --- BotBrain-style Regular-mode walking (FSM 500/501) -------------------
    # Config RPCs use a separate client with a short BLOCKING timeout so the
    # return code is real; the control loop only ever hands Move to a
    # latest-only sender thread (own client, 0.2 s blocking), never blocks.
    def _cfg(self, timeout=0.3):
        if self._cfg_client is None:
            self._cfg_client = LocoClient()
            self._cfg_client.Init()
        self._cfg_client.SetTimeout(timeout)
        return self._cfg_client

    def set_speed_mode(self, mode=0):
        import json as _json
        code, _ = self._cfg()._Call(7107, _json.dumps({"data": mode}))
        return code

    def set_balance_mode(self, mode=0):
        # ContinuousGait(false) == SetBalanceMode(0); never enable it.
        return self._cfg().SetBalanceMode(mode)

    def checked_zero(self):
        return self._cfg().SetVelocity(0.0, 0.0, 0.0, 1.0)

    def start_move_sender(self, move_timeout=0.2):
        from teleop.utils.loco_preflight import LatestMoveSender
        mc = LocoClient()
        mc.SetTimeout(move_timeout)
        mc.Init()

        def send(vx, vy, w):
            return mc.SetVelocity(vx, vy, w, 1.0)

        def stop(reason):
            self.watchdog_stop_code = mc.SetVelocity(0.0, 0.0, 0.0, 1.0)
            return self.watchdog_stop_code
        self.sender = LatestMoveSender(send, stop_fn=stop)
        self.sender.start()

    def stop_move_sender(self):
        if self.sender is not None:
            self.sender.stop()

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

    def make_fsm_reader(self, timeout=1.0):
        """Callable doing GetFsmId on a SEPARATE client (own timeout), so a
        background poller can never change the control client's 0.1 ms timeout."""
        reader_client = LocoClient()
        reader_client.SetTimeout(timeout)
        reader_client.Init()

        def read():
            code, data = reader_client._Call(7001, "{}")
            if code != 0 or not data:
                return None
            return int(json.loads(data)["data"])
        return read

    def StopMove(self, reason="", timeout=None, attempts=1):
        """Explicit stop = SetVelocity(0,0,0,1.0), the SDK's StopMove wire call.

        timeout=None: non-blocking (control-loop safe).  timeout=x: bounded
        blocking retries (cleanup only).  Never raises; returns last RPC code
        or None when nothing could be sent.
        """
        self.stop_count += 1
        self.last_stop_reason = reason
        code = None
        blocking = timeout is not None
        try:
            if blocking:
                self.client.SetTimeout(timeout)
            for _ in range(max(1, int(attempts))):
                code = self.client.SetVelocity(0.0, 0.0, 0.0, 1.0)
                if code == 0:
                    break
        except BaseException:
            self.stop_failures += 1
            code = None
        finally:
            if blocking:
                try:
                    self.client.SetTimeout(0.0001)
                except BaseException:
                    pass
        self.last_stop_code = code
        return code

    def Move(self, vx, vy, vyaw):
        # Same wire call as LocoClient.Move(continous_move=False) (duration=1 s)
        # but the RPC status is kept instead of discarded.  NOTE: the client
        # timeout is 0.1 ms (non-blocking), so a reply is rarely in time and
        # code 3104 (timeout) is expected; use read_fsm_id / rt/sportmodestate
        # as the authoritative state, not this code.
        if self.sender is not None:
            self.sender.submit((vx, vy, vyaw))
            self.last_move_code = self.sender.last_rc  # rc of an earlier send
            self.nonzero_move_codes = self.sender.nonzero_rc
            return self.last_move_code
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
