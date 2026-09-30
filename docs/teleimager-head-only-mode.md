# Teleimager: modo somente cabeça

Padrão (inalterado): `bash teleop/run_g1_quest_dex3.sh` exige cabeça + pulso esquerdo e usa `--camera-layout vertical`.

Modo explícito, com uma só câmera (cabeça, RealSense 243122072230):

```bash
TELEIMAGER_CAMERA_MODE=head bash teleop/run_g1_quest_dex3.sh
# só garantir o Teleimager, sem teleop:
TELEIMAGER_CAMERA_MODE=head G1_LAUNCHER_SKIP_TELEOP=1 bash teleop/run_g1_quest_dex3.sh
```

- Imprime `TELEIMAGER: modo SOMENTE CABEÇA (pulso esquerdo desativado)`.
- Servidor: `teleop.utils.teleimager_head_only_server` gera `teleop/teleimager/.cam_config_server.head_only.yaml` (cópia com câmeras de pulso `enable_zmq/enable_webrtc=false`) e aponta `image_server.CONFIG_PATH` para ela. O submodule e o `cam_config_server.yaml` não são alterados. Recusa iniciar se o serial da cabeça não estiver presente.
- Probe de saúde: portas 60000/55555 e frame BGR 720x1280x3 da cabeça; o pulso não é consultado.
- Teleop inicia com `--camera-layout head`; a telemetria `cameras` omite `left_wrist` nesse layout.
- Servidor saudável já rodando é reutilizado. Se houver servidor vivo em outro modo e não saudável para o modo pedido, o launcher recusa duplicar (pare-o antes: `kill $(cat ~/.local/state/xr_teleoperate/teleimager.pid)`). Modo registrado em `~/.local/state/xr_teleoperate/teleimager.mode`.
- Voltar ao padrão: religar a segunda câmera, parar o servidor head e rodar sem a variável.
