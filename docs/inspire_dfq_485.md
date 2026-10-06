# Inspire DFQ (RS-485) — driver e portas

Driver versionado: `teleop/robot_control/inspire_dfq_485_driver.py` (equivalente ao
`example/Headless_driver_485_double.py` do SDK, sem editar o SDK). O launcher
`teleop/run_g1_quest_inspire.sh` o gerencia com `INSPIRE_DRIVER=auto` (padrão).

## Variáveis

| Variável | Padrão | Uso |
|---|---|---|
| `INSPIRE_DRIVER` | `auto` (`skip` se `G1_EE=inspire_ftp`) | `auto` / `external` (só health check) / `skip` |
| `INSPIRE_LEFT_PORT` | `/dev/ttyUSB1` | porta da mão esquerda (aceita by-id/by-path/symlink) |
| `INSPIRE_RIGHT_PORT` | `/dev/ttyUSB2` | porta da mão direita |
| `INSPIRE_BAUDRATE` | `115200` | |
| `INSPIRE_LEFT_ID` / `INSPIRE_RIGHT_ID` | `1` / `1` | Modbus device id |
| `INSPIRE_DDS_IFACE` | `enP8p1s0` | `ChannelFactoryInitialize(0, iface)`; vazio = automático |
| `INSPIRE_DRIVER_TIMEOUT_S` | `15` | health check passivo |
| `INSPIRE_DRIVER_STOP_TIMEOUT_S` | `8` | espera após SIGINT antes do SIGTERM |
| `INSPIRE_STATE_DIR` | `~/.local/state/xr_teleoperate_inspire` | `inspire_driver.{log,pid,lock}` |
| `INSPIRE_SDK_DIR`, `INSPIRE_UNITREE_SDK`, `INSPIRE_PYTHON` | caminhos do robô | ambiente do driver |

`/dev/ttyUSB1`/`ttyUSB2` são o padrão do SDK, mas a numeração muda com a ordem de
conexão; o launcher avisa enquanto as portas não forem by-id/`/dev/inspire_*`.

## Checagem sem iniciar nada

    bash teleop/run_g1_quest_inspire.sh --inspire-preflight

## Descobrir as portas (somente leitura)

`--probe` abre cada porta, lê `angle_act` (reg. 1546, 6) uma vez, sem DDS e sem
escrever nenhum registrador; mostra alias by-id/by-path e serial USB do adaptador.

    cd <checkout-dev-inspire-implantado>
    PYTHONNOUSERSITE=1 ~/miniconda3/envs/tv/bin/python \
      teleop/robot_control/inspire_dfq_485_driver.py --probe /dev/ttyUSB0 /dev/ttyUSB3

As duas mãos usam id 1, então o registrador não diz qual é qual. Para identificar:
conecte só o adaptador da mão esquerda, rode `--probe` e anote o caminho
`/dev/serial/by-id/...`; repita com a direita. Depois fixe:

    export INSPIRE_LEFT_PORT=/dev/serial/by-id/usb-...-if00-port0
    export INSPIRE_RIGHT_PORT=/dev/serial/by-id/usb-...-if00-port0

Se os dois adaptadores forem iguais e sem número de série, `by-id` colide: use
`/dev/serial/by-path/...` (fixo por porta USB física) ou a regra udev abaixo.

## Opcional (requer sudo; NÃO aplicado): udev com nomes estáveis

`/etc/udev/rules.d/99-inspire-hands.rules` (troque os seriais pelos de `--probe`
ou `udevadm info -a -n /dev/ttyUSBx | grep -m1 serial`):

    SUBSYSTEM=="tty", ATTRS{serial}=="SERIAL_ESQ", SYMLINK+="inspire_left",  GROUP="dialout", MODE="0660"
    SUBSYSTEM=="tty", ATTRS{serial}=="SERIAL_DIR", SYMLINK+="inspire_right", GROUP="dialout", MODE="0660"

    sudo udevadm control --reload && sudo udevadm trigger
    export INSPIRE_LEFT_PORT=/dev/inspire_left INSPIRE_RIGHT_PORT=/dev/inspire_right

## Opcional (NÃO instalado): unit systemd de usuário para boot

`~/.config/systemd/user/inspire-dfq-485.service` — com ela rodando, o launcher
detecta o driver pelo argv e o reaproveita sem pará-lo.

    [Unit]
    Description=Inspire DFQ RS-485 -> DDS driver
    [Service]
    WorkingDirectory=/home/unitree/inspire_hand_ws/inspire_hand_sdk/example
    Environment=PYTHONNOUSERSITE=1
    Environment=PYTHONPATH=/home/unitree/unitree_sdk2_python:/home/unitree/inspire_hand_ws/inspire_hand_sdk
    Environment=LD_LIBRARY_PATH=/home/unitree/miniconda3/envs/tv/lib:/home/unitree/cyclonedds/build/lib
    ExecStart=/home/unitree/miniconda3/envs/tv/bin/python -u -s <checkout-dev-inspire>/teleop/robot_control/inspire_dfq_485_driver.py --left-port /dev/inspire_left --right-port /dev/inspire_right --iface enP8p1s0
    KillSignal=SIGINT
    TimeoutStopSec=8
    Restart=on-failure
    RestartSec=3
    [Install]
    WantedBy=default.target

    systemctl --user daemon-reload && systemctl --user enable --now inspire-dfq-485
    sudo loginctl enable-linger unitree   # para iniciar no boot sem login

## Escritas na mão

O driver não faz movimento de inicialização. Porém o construtor do SDK
(`inspire_sdk.ModbusDataHandler.__init__`) escreve **uma vez** o registrador
1004 = 1 ("reset error") em cada mão. Fora isso, só escreve registradores
(1486 angle, 1474 pos, 1498 force, 1522 speed) quando chega mensagem em
`rt/inspire_hand/ctrl/{l,r}`. `--probe` e `--health-check` nunca escrevem.
