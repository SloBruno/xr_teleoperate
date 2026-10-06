# Teleimager: perfil RealSense USB 2.0

Este robô não tem USB 3.0, então o launcher normal usa `XR_REALSENSE_PROFILE=usb2` **por padrão**:

```bash
bash teleop/run_g1_quest_inspire.sh                               # usb2 (padrão)
XR_REALSENSE_PROFILE=normal bash teleop/run_g1_quest_inspire.sh   # volta a 1280x720@15
# alias aceito: XR_REALSENSE_PROFILE=low-bandwidth
```

O mesmo launcher também usa por padrão `XR_VIDEO_PLANE_HEIGHT=auto` (plano XR dimensionado 1:1 ao HFOV de 69,4° da D435i, calculado pelo aspecto do layout). Para o plano antigo (1,0 m a 1,0 m): `XR_VIDEO_PLANE_HEIGHT=1.0`. Sem crop nem upscale.

Rodando `teleimager.image_server` direto (sem launcher), sem a variável, o servidor continua em `normal`.

`usb2` aplica **a ambas as câmeras configuradas como `realsense`** o mesmo stream colorido BGR em **640×480 a 6 fps**. A taxa bruta do color BGR é aproximadamente 44,2 Mbit/s por câmera (88,5 Mbit/s para duas), antes de overhead; é substancialmente menor que 1280×720@15 (aprox. 331,8 Mbit/s por câmera). Não há crop nem tentativa de simular FOV maior: reduzir resolução/FPS não altera a óptica.

O perfil não pede MJPEG à RealSense: o caminho atual de `RealSenseCamera` pede `rs.format.bgr8`; manter esse formato evita assumir que um perfil MJPEG é suportado pela câmera/SDK e que o SDK o decodificará como BGR. O servidor continua JPEG-encodando para ZMQ.

O launcher guarda o perfil usado pelo processo. Uma alteração de perfil requer reiniciar o Teleimager; se `teleop_hand_and_arm.py` estiver ativo, o launcher recusa a alteração e não mata/reconfigura nenhum processo. Pare a teleop primeiro.

Quando uma captura falha, os buffers JPEG/BGR válidos não são apagados e o servidor não derruba as outras câmeras. A telemetria em memória por câmera (`ImageServer.telemetry_snapshot()`) contém `capture_seq`, `publish_seq`, `capture_age_s`, `publish_age_s`, `capture_errors` e `publish_errors`. A gravação ignora BGR `None` com segurança. Cabos, hub, alimentação e erros UVC/USB ainda podem desconectar fisicamente o dispositivo.
