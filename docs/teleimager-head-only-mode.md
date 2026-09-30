# Teleimager: modo somente cabeça

## Comando único (detecção automática)

```bash
bash teleop/run_g1_quest_dex3.sh
```

Sem variáveis. O padrão é `TELEIMAGER_CAMERA_MODE=auto`: o launcher detecta por pyrealsense2 (leitura, timeout `TELEIMAGER_DETECT_TIMEOUT_S`=15s, sessão própria, sem herdar o lock FD 9) as câmeras {cabeça 243122072230, pulso esquerdo 233622070789}:

| Conectadas | Modo | Teleimager / probe | Teleop |
|---|---|---|---|
| 2 | `both` | servidor padrão; probe 60000/55555/55556 + 2 frames | `--camera-layout vertical` |
| 1 (qualquer) | `any` | `head_camera` = a presente; probe 60000/55555 | `--camera-layout head` (aviso se for o pulso) |
| 0, ou erro/timeout de detecção | — | falha clara (exit 3 / exit 4), nada é iniciado | — |

Reinício seguro: se o Teleimager já no ar está em modo/câmera diferente do detectado agora (`teleimager.mode`, `teleimager.source`), o launcher o reinicia (SIGTERM no PID de `teleimager.pid` após checar que é o Teleimager, espera limitada, limpa mode/source/pid, sobe no modo certo) **somente se não houver `teleop_hand_and_arm.py` rodando**; com teleop ativo recusa com mensagem clara e não mexe em nada. Servidor saudável no mesmo modo/câmera é reutilizado.

`TELEIMAGER_CAMERA_MODE=both|head|any` explícito continua valendo como override (sem auto-detecção no `both`/`head`).

Descrição abaixo: modos explícitos (histórico). O "padrão" anterior (`both` rígido) agora é `auto`.

Modo explícito, com uma só câmera (cabeça, RealSense 243122072230):

```bash
TELEIMAGER_CAMERA_MODE=head bash teleop/run_g1_quest_dex3.sh
# só garantir o Teleimager, sem teleop:
TELEIMAGER_CAMERA_MODE=head G1_LAUNCHER_SKIP_TELEOP=1 bash teleop/run_g1_quest_dex3.sh
```

- Imprime `TELEIMAGER: modo SOMENTE CABEÇA (pulso esquerdo desativado)`.
- Servidor: `teleop.utils.teleimager_head_only_server` gera `~/.local/state/xr_teleoperate/cam_config_server.head_only.yaml` (cópia com câmeras de pulso `enable_zmq/enable_webrtc=false`) e aponta `image_server.CONFIG_PATH` para ela. O submodule e o `cam_config_server.yaml` não são alterados. Recusa iniciar se o serial da cabeça não estiver presente.
- Probe de saúde: portas 60000/55555 e frame BGR 720x1280x3 da cabeça; o pulso não é consultado.
- Teleop inicia com `--camera-layout head`; a telemetria `cameras` omite `left_wrist` nesse layout.
- Servidor saudável já rodando é reutilizado. Se houver servidor vivo em outro modo e não saudável para o modo pedido, o launcher recusa duplicar (pare-o antes: `kill $(cat ~/.local/state/xr_teleoperate/teleimager.pid)`). Modo registrado em `~/.local/state/xr_teleoperate/teleimager.mode`.
- Voltar ao padrão: religar a segunda câmera, parar o servidor head e rodar sem a variável.


## Modo `any` (câmera única, qualquer uma conectada)

```bash
TELEIMAGER_CAMERA_MODE=any bash teleop/run_g1_quest_dex3.sh      # alias: =single
# só garantir o Teleimager, sem teleop:
G1_LAUNCHER_SKIP_TELEOP=1 TELEIMAGER_CAMERA_MODE=any bash teleop/run_g1_quest_dex3.sh
```

- Detecta por pyrealsense2 quais seriais {cabeça 243122072230, pulso esquerdo 233622070789} estão conectados agora. Nenhuma → falha clara (exit 3) sem iniciar. Uma → usa essa. Duas → usa só a cabeça e avisa.
- A câmera presente é publicada como `head_camera` (zmq 55555, 720x1280 @15, monocular); `left_wrist_camera` fica com zmq/webrtc desativados. Config derivada em `~/.local/state/xr_teleoperate/cam_config_server.head_only.yaml`; o cliente a recebe do servidor na 60000. Submodule/YAML local intactos. Teleop roda com `--camera-layout head`.
- Banner: `TELEIMAGER: modo CÂMERA ÚNICA (<cabeça|pulso esquerdo> serial X publicada como imagem principal)`. Com o pulso, aviso extra (ponto de vista da mão, pode confundir).
- Telemetria `teleop_status` ganha `camera_source` (`head` | `left_wrist`).
- Probe de saúde: portas 60000/55555 e frame BGR 720x1280x3. Reutiliza servidor saudável no mesmo modo; recusa duplicar em modo diferente. Se trocar a câmera, pare o servidor (`kill $(cat ~/.local/state/xr_teleoperate/teleimager.pid)`) e rode de novo.
