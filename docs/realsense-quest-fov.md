# FOV da RealSense no Quest

## Limite físico e perfil solicitado

A D435i não ganha campo de visão óptico ao alterar resolução, FPS, escala ou o
plano no Quest. O launcher solicita ao Teleimager RGB **1280×720 BGR8 a 15
FPS**. No D435i, o valor de referência do RGB é aproximadamente **69,4° H ×
42,5° V**; o valor efetivo deve ser lido das intrínsecas do perfil resolvido,
não inferido da resolução.

A disposição `vertical` de duas câmeras não é uma visão óptica única mais larga:
`stack_camera_frames_vertical` remove deliberadamente as faixas sobrepostas
(11% inferior da imagem de cima e 11% superior da de baixo) antes de compor
cabeça + pulso. Não habilite esse modo esperando ampliar a RealSense da cabeça.
O modo automático de uma só câmera usa a câmera inteira como imagem principal,
sem crop ou upscale adicional no launcher.

## Confirmar intrínsecas sem iniciar stream

Com **Teleimager e teleop encerrados**, no ambiente Python que contém
`pyrealsense2`, execute a consulta somente leitura:

```bash
cd ~/xr_teleoperate
PYTHONPATH="$PWD:$PWD/teleop/televuer/src:$PWD/teleop/teleimager/src" \
  /home/unitree/miniconda3/envs/tv/bin/python -m teleop.utils.realsense_fov \
  --serial 243122072230 --width 1280 --height 720 --fps 15
```

O comando usa `config.resolve(pipeline_wrapper)`, **não** chama
`pipeline.start()` e não altera opções da câmera. Ele imprime o perfil RGB
resolvido, `fx`/`fy` e o FOV calculado pelas intrínsecas. Não execute a consulta
em paralelo com uma teleoperação ou Teleimager ativo: embora não abra stream,
o dispositivo já está em uso e o diagnóstico não é necessário durante operação.

## Ampliar somente o FOV percebido

Para fazer a imagem RGB completa ocupar no Quest a mesma largura angular do
perfil D435i de referência, use o plano XR `auto` ao iniciar uma nova sessão:

```bash
XR_VIDEO_PLANE_HEIGHT=auto XR_VIDEO_PLANE_DISTANCE=1 \
  bash teleop/run_g1_quest_dex3.sh
```

`auto` preserva o comportamento normal de câmera e calcula a altura do plano
para cerca de **69,4° horizontais** no layout monocular ZMQ; não muda pixels,
resolução, FPS, crop, intrínsecas nem FOV óptico. A distância pode ser alterada
junto com `auto` (por exemplo, `XR_VIDEO_PLANE_DISTANCE=1.2`) para conforto: a
altura é recalculada para manter a mesma cobertura angular. Sem essas variáveis,
o padrão histórico continua **1,0 m de altura a 1,0 m de distância**.

Não use altura manual maior para alegar FOV de câmera maior: isso apenas aumenta
uma mesma imagem no display e pode exceder a zona confortável do headset. O
teleop registra a cobertura angular real do plano no log.

A implementação de plano configurável só é aplicada ao fluxo monocular ZMQ
usado pelo launcher no modo de uma câmera. Os planos WebRTC/estéreo mantêm seu
comportamento próprio; esta mudança não tenta ampliar ou recortar esses fluxos.
