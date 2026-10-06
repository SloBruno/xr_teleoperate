#!/usr/bin/env python3
"""Inspire DFQ (RS-485 / Modbus RTU) -> DDS driver, versioned in this fork.

Same behavior as the robot SDK's example/Headless_driver_485_double.py
(inspire_sdkpy.inspire_sdk.ModbusDataHandler per hand, use_serial=True,
states_structure=[('angle_act', 1546, 6, 'short')], one shared DDS factory),
but without editing the SDK:

  * serial ports / baudrate / Modbus device id by argument or environment
    (INSPIRE_LEFT_PORT, INSPIRE_RIGHT_PORT, INSPIRE_BAUDRATE, INSPIRE_LEFT_ID,
    INSPIRE_RIGHT_ID); /dev/serial/by-id/... and by-path/... are accepted;
  * explicit DDS interface: ChannelFactoryInitialize(0, --iface) (INSPIRE_DDS_IFACE,
    default enP8p1s0);
  * clear error when a port is missing or not readable/writable (no sudo);
  * clean shutdown on SIGINT/SIGTERM (a second signal exits immediately);
  * decimated frequency log.

Topics (unchanged from the SDK): publishes rt/inspire_hand/state/{l,r};
subscribes rt/inspire_hand/ctrl/{l,r} and writes the hand registers when a
ctrl message arrives. NOTE: the SDK constructor itself writes register 1004=1
("clear error") once per hand; it commands no motion.

Read-only modes (never write a register, never publish a ctrl):
  --probe         open each port, read angle_act (1546, 6 regs) once; no DDS.
  --health-check  DDS subscriber only; wait for state samples on l and r.

Requires PYTHONPATH with unitree_sdk2_python and inspire_hand_sdk (for the
driver/health check); --probe only needs pymodbus.
"""
from __future__ import annotations

import argparse
import glob
import os
import signal
import stat
import sys
import threading
import time

ANGLE_ACT_ADDR = 1546
ANGLE_ACT_COUNT = 6
STATES_STRUCTURE = [("angle_act", ANGLE_ACT_ADDR, ANGLE_ACT_COUNT, "short")]
DEFAULT_LEFT_PORT = "/dev/ttyUSB1"   # same as the SDK example (fragile: set by-id)
DEFAULT_RIGHT_PORT = "/dev/ttyUSB2"
DEFAULT_IFACE = "enP8p1s0"
STATE_TOPIC = "rt/inspire_hand/state/"

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2
EXIT_PORT = 3
EXIT_HEALTH = 4


class PortError(Exception):
    pass


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name}={raw!r} não é inteiro")


def build_parser():
    p = argparse.ArgumentParser(
        description="Inspire DFQ RS-485 -> DDS driver (rt/inspire_hand/{state,ctrl}/{l,r}).")
    p.add_argument("--left-port", default=os.environ.get("INSPIRE_LEFT_PORT") or DEFAULT_LEFT_PORT,
                   help="porta serial da mão esquerda (env INSPIRE_LEFT_PORT; default %(default)s)")
    p.add_argument("--right-port", default=os.environ.get("INSPIRE_RIGHT_PORT") or DEFAULT_RIGHT_PORT,
                   help="porta serial da mão direita (env INSPIRE_RIGHT_PORT; default %(default)s)")
    p.add_argument("--baudrate", type=int, default=_env_int("INSPIRE_BAUDRATE", 115200),
                   help="env INSPIRE_BAUDRATE (default %(default)s)")
    p.add_argument("--left-id", type=int, default=_env_int("INSPIRE_LEFT_ID", 1),
                   help="Modbus device id esquerda (env INSPIRE_LEFT_ID; default %(default)s)")
    p.add_argument("--right-id", type=int, default=_env_int("INSPIRE_RIGHT_ID", 1),
                   help="Modbus device id direita (env INSPIRE_RIGHT_ID; default %(default)s)")
    p.add_argument("--iface", default=os.environ.get("INSPIRE_DDS_IFACE", DEFAULT_IFACE),
                   help="interface DDS (env INSPIRE_DDS_IFACE; default %(default)s; '' = automática)")
    p.add_argument("--log-every-s", type=float, default=5.0, help="intervalo do log de frequência")
    p.add_argument("--max-consecutive-errors", type=int, default=50,
                   help="sai com erro após N leituras seguidas falhando numa mão")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--probe", action="store_true",
                      help="somente leitura: abre cada porta e lê angle_act uma vez (sem DDS, sem escrita)")
    mode.add_argument("--health-check", action="store_true",
                      help="somente DDS subscriber: espera amostras em state/l e state/r")
    p.add_argument("--timeout", type=float, default=15.0, help="timeout do --health-check (s)")
    p.add_argument("ports", nargs="*", help="(--probe) portas extras/alternativas a testar")
    return p


# ---------------------------------------------------------------- ports
def check_port(path, label="porta"):
    """Return the resolved device path, or raise PortError with a clear message."""
    if not path:
        raise PortError(f"{label}: caminho vazio")
    if not os.path.lexists(path):
        raise PortError(f"{label} {path} não existe (adaptador RS-485 desconectado?)")
    real = os.path.realpath(path)
    try:
        st = os.stat(real)
    except OSError as exc:
        raise PortError(f"{label} {path} -> {real}: {exc}")
    if not stat.S_ISCHR(st.st_mode):
        raise PortError(f"{label} {path} -> {real} não é um dispositivo de caractere")
    if not os.access(real, os.R_OK | os.W_OK):
        raise PortError(f"{label} {path} -> {real}: sem permissão de leitura/escrita "
                        f"(usuário no grupo dialout? relogar após adicionar)")
    return real


def port_aliases(real):
    """Stable /dev/serial/by-id and by-path names that point to `real` (read-only)."""
    out = []
    for pattern in ("/dev/serial/by-id/*", "/dev/serial/by-path/*", "/dev/inspire_*"):
        for link in sorted(glob.glob(pattern)):
            if os.path.realpath(link) == real:
                out.append(link)
    return out


def usb_serial_attr(real):
    """USB serial number of the adapter behind /dev/ttyUSBx (sysfs, read-only)."""
    name = os.path.basename(real)
    dev = os.path.realpath(f"/sys/class/tty/{name}/device")
    for _ in range(6):
        cand = os.path.join(dev, "serial")
        if os.path.isfile(cand):
            try:
                with open(cand, encoding="utf-8") as fh:
                    return fh.read().strip()
            except OSError:
                return None
        dev = os.path.dirname(dev)
    return None


# ---------------------------------------------------------------- probe
def _make_serial_client(port, baudrate, timeout=1.0):
    from pymodbus.client import ModbusSerialClient
    return ModbusSerialClient(port=port, baudrate=baudrate, timeout=timeout)


def probe_port(port, baudrate, device_ids, client_factory=_make_serial_client):
    """Read angle_act once per device id. Reads only: never writes a register."""
    result = {"port": port, "real": None, "aliases": [], "usb_serial": None,
              "responses": {}, "error": None}
    try:
        real = check_port(port)
    except PortError as exc:
        result["error"] = str(exc)
        return result
    result.update(real=real, aliases=port_aliases(real), usb_serial=usb_serial_attr(real))
    client = client_factory(real, baudrate)
    try:
        if not client.connect():
            result["error"] = "falha ao abrir a porta"
            return result
        for dev_id in device_ids:
            try:
                rsp = client.read_holding_registers(ANGLE_ACT_ADDR, ANGLE_ACT_COUNT, dev_id)
            except Exception as exc:  # noqa: BLE001 - report, never escalate to a write
                result["responses"][dev_id] = f"erro: {exc}"
                continue
            if rsp is None or rsp.isError():
                result["responses"][dev_id] = "sem resposta"
            else:
                regs = [r - 0x10000 if r >= 0x8000 else r for r in rsp.registers]
                result["responses"][dev_id] = regs
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    return result


def run_probe(args, client_factory=_make_serial_client, out=print):
    ports = []
    for p in [args.left_port, args.right_port, *args.ports]:
        if p not in ports:
            ports.append(p)
    ids = sorted({args.left_id, args.right_id})
    out(f"PROBE somente leitura (angle_act {ANGLE_ACT_ADDR}x{ANGLE_ACT_COUNT}, ids {ids}, "
        f"{args.baudrate} baud); nenhuma escrita, sem DDS.")
    any_ok = False
    for port in ports:
        res = probe_port(port, args.baudrate, ids, client_factory)
        role = "ESQUERDA(config)" if port == args.left_port else (
            "DIREITA(config)" if port == args.right_port else "extra")
        out(f"- {port} [{role}] -> {res['real'] or '?'}")
        if res["aliases"]:
            out(f"    aliases estáveis: {', '.join(res['aliases'])}")
        if res["usb_serial"]:
            out(f"    serial USB do adaptador: {res['usb_serial']}")
        if res["error"]:
            out(f"    ERRO: {res['error']}")
            continue
        for dev_id, val in res["responses"].items():
            out(f"    id {dev_id}: {val}")
            if isinstance(val, list):
                any_ok = True
    out("Obs.: as duas mãos usam id 1, então o registro não diz esquerda/direita. "
        "Para identificar: conecte um adaptador por vez e rode --probe, depois fixe "
        "INSPIRE_LEFT_PORT/INSPIRE_RIGHT_PORT com os caminhos /dev/serial/by-id/...")
    return EXIT_OK if any_ok else EXIT_PORT


# ---------------------------------------------------------------- health check
def run_health_check(args, out=print):
    """Passive: subscribers only, never a publisher."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from inspire_sdkpy.inspire_dds import inspire_hand_state

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)
    got = {"l": threading.Event(), "r": threading.Event()}
    subs = []
    for side in ("l", "r"):
        sub = ChannelSubscriber(STATE_TOPIC + side, inspire_hand_state)
        sub.Init(lambda _msg, s=side: got[s].set(), 10)
        subs.append(sub)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and not all(e.is_set() for e in got.values()):
        time.sleep(0.05)
    missing = [s for s, e in got.items() if not e.is_set()]
    if missing:
        out(f"HEALTH: sem amostras em {', '.join(STATE_TOPIC + s for s in missing)} "
            f"em {args.timeout:.0f}s (iface {args.iface or 'auto'})")
        return EXIT_HEALTH
    out("HEALTH: amostras recebidas em rt/inspire_hand/state/l e /r")
    return EXIT_OK


# ---------------------------------------------------------------- driver
class StopFlag:
    def __init__(self):
        self.running = True
        self.signals = 0

    def handler(self, signum, _frame):
        self.signals += 1
        if self.signals > 1:
            print(f"\nsinal {signum} repetido: saída imediata", flush=True)
            os._exit(EXIT_RUNTIME)
        self.running = False
        print(f"\nsinal {signum}: encerrando...", flush=True)


def _read_ok(data):
    try:
        return data["states"]["ANGLE_ACT"] is not None
    except (TypeError, KeyError):
        return False


def run_driver(args, stop, sdk=None, channel_init=None, out=print, clock=time.monotonic):
    # Ports first: a missing adapter gives a clear error before any import/DDS.
    hands = [("l", "ESQUERDA", args.left_port, args.left_id),
             ("r", "DIREITA", args.right_port, args.right_id)]
    resolved = []
    for lr, label, port, dev_id in hands:
        resolved.append((lr, label, port, check_port(port, f"porta {label}"), dev_id))
    if resolved[0][3] == resolved[1][3]:
        raise PortError(f"esquerda e direita apontam para o mesmo dispositivo {resolved[0][3]}")

    if sdk is None:
        from inspire_sdkpy import inspire_sdk as sdk  # noqa: N813
    if channel_init is None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize as channel_init

    if args.iface:
        channel_init(0, args.iface)
    else:
        channel_init(0)
    out(f"DDS inicializado (domain 0, iface {args.iface or 'auto'})")

    handlers = []
    try:
        for lr, label, port, real, dev_id in resolved:
            out(f"Inicializando mão {label} em {port} -> {real} (id {dev_id}, {args.baudrate} baud)...")
            try:
                h = sdk.ModbusDataHandler(LR=lr, device_id=dev_id, use_serial=True, serial_port=real,
                                          baudrate=args.baudrate, states_structure=STATES_STRUCTURE,
                                          initDDS=False)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"falha ao inicializar mão {label} ({port}): {exc}") from exc
            handlers.append((label, h))
            if not stop.running:
                return EXIT_OK
        out("Ambas as mãos inicializadas; publicando rt/inspire_hand/state/{l,r}.")

        counts = [0, 0]
        errors = [0, 0]
        consecutive = [0, 0]
        t0 = last_log = clock()
        while stop.running:
            for i, (label, h) in enumerate(handlers):
                try:
                    ok = _read_ok(h.read())
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    if consecutive[i] == 0:
                        out(f"[{label}] erro de leitura: {exc}")
                if ok:
                    counts[i] += 1
                    consecutive[i] = 0
                else:
                    errors[i] += 1
                    consecutive[i] += 1
                    if consecutive[i] >= args.max_consecutive_errors:
                        out(f"[{label}] {consecutive[i]} leituras seguidas falharam; saindo.")
                        return EXIT_RUNTIME
                if not stop.running:
                    break
            now = clock()
            if now - last_log >= args.log_every_s:
                dt = max(now - t0, 1e-9)
                out(" | ".join(f"[{handlers[i][0]}] {counts[i] / dt:.1f} Hz ok={counts[i]} err={errors[i]}"
                               for i in range(len(handlers))))
                last_log = now
        return EXIT_OK
    finally:
        for _label, h in handlers:
            try:
                h.client.close()
            except Exception:  # noqa: BLE001
                pass
        out("Driver encerrado.")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.ports and not args.probe:
        print("portas posicionais só valem com --probe", file=sys.stderr)
        return EXIT_USAGE
    if args.probe:
        return run_probe(args)
    if args.health_check:
        return run_health_check(args)
    stop = StopFlag()
    signal.signal(signal.SIGINT, stop.handler)
    signal.signal(signal.SIGTERM, stop.handler)
    try:
        return run_driver(args, stop)
    except PortError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return EXIT_PORT
    except RuntimeError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.exit(main())
