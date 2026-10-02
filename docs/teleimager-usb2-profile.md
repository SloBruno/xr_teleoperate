# Teleimager: perfil RealSense USB 2.0

O modo padrão não muda: `XR_REALSENSE_PROFILE=normal` mantém a configuração normal das duas câmeras.

Para operação em USB 2.0, selecione o perfil explicitamente antes de iniciar o launcher:

```bash
XR_REALSENSE_PROFILE=usb2 bash teleop/run_g1_quest_dex3.sh
# alias aceito: XR_REALSENSE_PROFILE=low-bandwidth
```

`usb2` aplica **a ambas as câmeras configuradas como `realsense`** o mesmo stream colorido BGR em **640×480 a 6 fps**. A taxa bruta do color BGR é aproximadamente 44,2 Mbit/s por câmera (88,5 Mbit/s para duas), antes de overhead; é substancialmente menor que 1280×720@15 (aprox. 331,8 Mbit/s por câmera). Não há crop nem tentativa de simular FOV maior: reduzir resolução/FPS não altera a óptica.

O perfil não pede MJPEG à RealSense: o caminho atual de `RealSenseCamera` pede `rs.format.bgr8`; manter esse formato evita assumir que um perfil MJPEG é suportado pela câmera/SDK e que o SDK o decodificará como BGR. O servidor continua JPEG-encodando para ZMQ.

O launcher guarda o perfil usado pelo processo. Uma alteração de perfil requer reiniciar o Teleimager; se `teleop_hand_and_arm.py` estiver ativo, o launcher recusa a alteração e não mata/reconfigura nenhum processo. Pare a teleop primeiro.

Quando uma captura falha, os buffers JPEG/BGR válidos não são apagados e o servidor não derruba as outras câmeras. A telemetria em memória por câmera (`ImageServer.telemetry_snapshot()`) contém `capture_seq`, `publish_seq`, `capture_age_s`, `publish_age_s`, `capture_errors` e `publish_errors`. A gravação ignora BGR `None` com segurança. Cabos, hub, alimentação e erros UVC/USB ainda podem desconectar fisicamente o dispositivo.
