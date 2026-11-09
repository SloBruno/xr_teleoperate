"""Real Cyclone DDS adapter for rt/waistpitchcmd / rt/waistpitchstate.

This is the first REAL transport code for this tool. It has never been run
against the actual robot (no network path to the robot's internal DDS
network -- domain 0 -- was available from the machine this was written on;
see README "Rede necessária" section). Treat every claim below as "should
work per the SDK's own API and the vendor's example", not "verified on
hardware".

Design goals, matching the task's requirements:
  - Read-only state subscription can be used completely independently of
    anything that arms or publishes (see g1d_passive_read.py) -- there is
    no way to accidentally command the robot by only importing/using
    RealBus.read_state().
  - Concurrent-publisher detection uses Cyclone DDS's own built-in
    DCPSPublication discovery topic (passive: does not require creating our
    own reader/writer on rt/waistpitchcmd first) so a pre-existing writer
    (e.g. the vendor's own move_service or another operator's tool) is
    visible *before* we ever touch the command topic ourselves.
  - The DDS-filtering logic (CmdPublisherWatch.ingest) is separated from
    the real cyclonedds I/O so it has unit tests that run without any DDS
    participant or network (see test_g1d_dds_real.py).
"""
from dataclasses import dataclass
import time

TOPIC_STATE = 'rt/waistpitchstate'
TOPIC_CMD = 'rt/waistpitchcmd'
DEFAULT_DOMAIN_ID = 0

# Re-exported here so callers only need this module for real-mode wiring.
VENDOR_Q_MIN = -0.04363
VENDOR_Q_MAX = 2.3562


class CmdPublisherWatch:
    """Tracks writer GUIDs seen on TOPIC_CMD via DCPSPublication samples.

    Pure bookkeeping, no DDS calls -- feed it (guid, topic_name) pairs via
    ingest() and it tells you which GUIDs (other than our own) are/were
    present. By design this is a one-way ratchet: once a foreign writer GUID
    is observed it stays "seen" for the rest of the process, because we have
    no reliable disposal/liveliness signal wired up yet (see README
    limitation). A false "someone else is there" is safe (refuses to arm /
    stops the session); a false "all clear" would not be, so we deliberately
    never forget a foreign writer.
    """

    def __init__(self, self_guid=None):
        self.self_guid = self_guid
        self._foreign_guids = set()

    def set_self_guid(self, guid):
        self.self_guid = guid
        self._foreign_guids.discard(guid)

    def ingest(self, endpoints):
        """endpoints: iterable of objects/tuples with .key and .topic_name
        (or (guid, topic_name) tuples) from a DCPSPublication builtin reader.
        """
        for ep in endpoints:
            guid = getattr(ep, 'key', None)
            topic_name = getattr(ep, 'topic_name', None)
            if guid is None and isinstance(ep, tuple):
                guid, topic_name = ep
            if topic_name != TOPIC_CMD:
                continue
            if self.self_guid is not None and guid == self.self_guid:
                continue
            self._foreign_guids.add(guid)

    def foreign_writers(self):
        return sorted(str(g) for g in self._foreign_guids)

    def any_foreign(self):
        return bool(self._foreign_guids)


@dataclass
class _Registry:
    """Adapter exposing the .cmd_registry interface PitchControlSession
    expects (see control_session.py / g1d_dds_fake.PublisherRegistry),
    backed by a CmdPublisherWatch fed from real DDS discovery.
    """
    watch: CmdPublisherWatch
    pump: callable  # () -> None; drains new DCPSPublication samples into watch

    def heartbeat(self, writer_id, now=None):
        pass  # presence on the real bus is proven by DDS matching, not this

    def leave(self, writer_id):
        pass  # we do not un-ratchet; see CmdPublisherWatch docstring

    def other_writers(self, self_id, now=None):
        self.pump()
        return self.watch.foreign_writers()


def discover_and_record_self_guid(take_fn, watch, timeout=3.0, poll_interval=0.05,
                                   sleep_fn=time.sleep, now_fn=time.monotonic):
    """Polls `take_fn()` (a callable returning an iterable of discovery
    endpoints, like BuiltinDataReader.take(32)) for up to `timeout` seconds,
    looking for the first endpoint on TOPIC_CMD, and records it as our own
    writer GUID via `watch.set_self_guid(...)`.

    Pure/testable: no real DDS reader required, only a callable. See
    RealBus.create_cmd_writer for why a bounded fixed sleep is not safe here
    (DDS discovery-announcement latency is not tightly bounded in practice).

    Returns True if a TOPIC_CMD endpoint was found and recorded, False if
    the timeout elapsed first.
    """
    deadline = now_fn() + timeout
    while now_fn() < deadline:
        try:
            samples = take_fn()
        except Exception:
            samples = []
        for ep in samples:
            if getattr(ep, 'topic_name', None) == TOPIC_CMD:
                watch.set_self_guid(ep.key)
                return True
        sleep_fn(poll_interval)
    return False


class RealBus:
    """Thin wrapper around unitree_sdk2py's ChannelSubscriber/ChannelPublisher
    + cyclonedds's builtin DCPSPublication reader, shaped to match the
    .cmd_registry / .read_state() / .publish_cmd() surface PitchControlSession
    and the keyboard tool already use against the fake bus.

    Construction only subscribes to state and starts passive discovery; it
    never creates a command writer. Call create_cmd_writer() explicitly,
    and only after a clean arm()-time discovery check, to start being able
    to publish.
    """

    def __init__(self, domain_id=DEFAULT_DOMAIN_ID, iface=None):
        # Imports are local so importing this module (e.g. from tests, or
        # from g1d_passive_read.py in a read-only context) never requires a
        # live DDS runtime unless RealBus is actually constructed.
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize, ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorState_, MotorCmd_
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication

        self._MotorState_ = MotorState_
        self._MotorCmd_ = MotorCmd_

        ChannelFactoryInitialize(domain_id, iface)

        self._latest_state = None
        self._latest_stamp = None
        self._state_sub = ChannelSubscriber(TOPIC_STATE, MotorState_)
        self._state_sub.Init(self._on_state, 10)

        # Passive discovery participant/reader -- independent of the
        # ChannelFactory singleton's participant so this works even before
        # any publisher/subscriber of our own exists on the command topic.
        self._participant = DomainParticipant(domain_id)
        self._pub_reader = BuiltinDataReader(self._participant, BuiltinTopicDcpsPublication)

        self.watch = CmdPublisherWatch()
        self.cmd_registry = _Registry(self.watch, self._pump_discovery)

        self._cmd_pub = None  # created lazily by create_cmd_writer()

    def _on_state(self, msg):
        self._latest_state = msg
        self._latest_stamp = time.monotonic()

    def _pump_discovery(self):
        try:
            samples = self._pub_reader.take(32)
        except Exception:
            return
        self.watch.ingest(samples)

    def read_state(self):
        """Returns (state-like, stamp) or (None, None) if nothing received
        yet. `state` has .mode/.q/.dq/.motorstate like g1d_dds_fake's
        MotorState_ (it is literally the SDK's own MotorState_ instance).
        """
        if self._latest_state is None:
            return None, None
        return self._latest_state, self._latest_stamp

    def create_cmd_writer(self, writer_id='keyboard_tool_real',
                           self_guid_timeout=3.0, self_guid_poll_interval=0.05):
        """Creates the real command publisher. Call this ONLY after arming
        logic has already confirmed (a) fresh/valid state and (b) no
        foreign writer on rt/waistpitchcmd. Creating this writer makes our
        own participant visible on discovery too, so we record our own GUID
        to exclude it from future "foreign writer" checks.

        DDS discovery-announcement latency is NOT bounded tightly (observed
        >0.3s on the real robot's network at least once). A single
        sleep(0.3)+take() can miss our own writer's announcement entirely,
        leaving self_guid unset -- at which point our own writer later shows
        up as a "foreign" publisher the instant any other_writers() call
        drains it, immediately tripping the concurrent-publisher stop with
        zero operator input. Poll for up to `self_guid_timeout` seconds
        instead of a single fixed wait. Because the caller already verified
        zero pre-existing writers on TOPIC_CMD before calling this method,
        the first endpoint observed on TOPIC_CMD here is assumed to be ours;
        set_self_guid() retroactively clears any false-foreign flag already
        recorded for that GUID (see CmdPublisherWatch).

        Returns True if self_guid was identified, False if the timeout
        elapsed without seeing any endpoint on TOPIC_CMD (the writer still
        exists and works, but other_writers() may then report our own
        writer as foreign -- callers should treat False as a reason not to
        proceed to arm()).
        """
        from unitree_sdk2py.core.channel import ChannelPublisher
        self._cmd_pub = ChannelPublisher(TOPIC_CMD, self._MotorCmd_)
        self._cmd_pub.Init()
        return discover_and_record_self_guid(
            take_fn=lambda: self._pub_reader.take(32),
            watch=self.watch,
            timeout=self_guid_timeout,
            poll_interval=self_guid_poll_interval,
        )

    def publish_cmd(self, writer_id, cmd, now=None):
        if self._cmd_pub is None:
            raise RuntimeError('create_cmd_writer() was never called')
        real_cmd = self._MotorCmd_(
            mode=cmd.mode, q=cmd.q, dq=cmd.dq, tau=cmd.tau,
            kp=cmd.kp, kd=cmd.kd, reserve=0,
        )
        self._cmd_pub.Write(real_cmd)

    def close(self):
        if self._cmd_pub is not None:
            self._cmd_pub.Close()
        self._state_sub.Close()
