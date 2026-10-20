# G1D + Meta Quest + Dex3 (perfil somente braços)

Este perfil escreve somente os 14 slots dos braços (`15..28`). Os slots
`WaistYaw=12` e `WaistPitch=14` permanecem no `LowCmd`, mas são passivos:
posição medida, `kp=0`, `kd=0` e `tau=0`. `WaistRoll=13` é inválido no G1D.
Não há comando de cintura, caminhada ou autoridade do corpo neste perfil.

O SDK oficial fixa os índices dos braços do G1D em `15..28`, mas este
repositório não contém URDF nem especificação geométrica oficial do G1D. Por
isso o perfil reutiliza temporariamente `G1_29_ArmIK` somente após a confirmação
explícita abaixo. Isso não é validação em hardware: valide a geometria, limites,
orientações de punho e a transformação controlador→punho em bancada antes de
armar perto do robô.

## Câmeras e preflight obrigatório

No Teleimager, os caminhos de imagem são representados no
`cam_config_server.yaml` por `head_camera`, `left_wrist_camera` e
`right_wrist_camera`. O campo `binocular` descreve se aquele caminho entrega um
par estereoscópico; `enable_zmq` e a porta ZMQ definem se o fluxo é publicado.

O perfil G1D consome o Teleimager remoto em `192.168.123.164` por padrão
(configurável por `G1D_TELEIMAGER_HOST`). O vídeo continua desligado por
padrão. `G1D_VIDEO=1` preserva sem alteração o fluxo vertical legado: exige
`head_camera` em TCP/ZMQ `55555` e `left_wrist_camera` em `55556`, confere as
portas `60000`, `55555` e `55556` e decodifica um frame BGR real de cada fluxo.
Nesse modo não existe fallback para uma câmera, para a câmera de pulso publicada
como cabeça, nem para `head`/`any`: sem fallback.

O modo explícito `G1D_VIDEO_MODE=head` é o caminho leve recomendado para o G1D.
O preflight consulta somente a configuração (`60000`) e `head_camera` ZMQ
(`55555`), exige `enable_zmq=true`, porta correta, `binocular=true` e um frame
BGR com a forma declarada. Ele força ZMQ mesmo quando o servidor anuncia WebRTC,
preserva o frame binocular lado a lado e não consome nenhuma câmera de pulso.
Leitura de frames e cópia para XR ficam limitadas a 10 Hz sem inserir espera no
loop de controle de 30 Hz. A codificação XR usa JPEG quality 60 neste modo; os
padrões legados continuam 30 Hz/quality 80. Os limites podem ser reduzidos com
`G1D_HEAD_VIDEO_FPS` (1..10) e a qualidade pode ser configurada com
`G1D_HEAD_JPEG_QUALITY` (1..100).

O launcher não inicia, para ou gerencia Teleimager/Realsense localmente. Se o
fluxo exigido pelo modo selecionado não estiver saudável, o perfil falha fechado
e não inicia a teleoperação.

## Runtime isolado G1D

O ambiente Python do G1D fixa `numpy==1.26.4`, necessário para o conjunto
Pinocchio/IK, e `pyarrow==20.0.0` para `rerun-sdk==0.20.1`. Não atualize NumPy
para a série 2: versões recentes de PyArrow podem exigir NumPy 2, enquanto
PyArrow 20 possui wheel CPython 3.12 ARM e permanece compatível com este runtime.
Reinstale estes pins a partir de `requirements.txt` ao preparar um release.

## Comando de operador

No Jetson/PC2, com o Teleimager e o ambiente Quest já preparados, execute a
partir de `teleop/`:

```bash
bash run_g1d_quest_dex3.sh
```

Para o modo leve de vídeo binocular ZMQ (somente cabeça), use exatamente:

```bash
G1D_VIDEO_MODE=head bash run_g1d_quest_dex3.sh
```

O launcher é separado de `run_g1_quest_dex3.sh`; não modifique nem reutilize o
perfil G1. Ele fixa DDS em `enP2p1s0` por padrão. Para outra interface, use
`G1D_DDS_INTERFACE=<interface> bash run_g1d_quest_dex3.sh`.

Argumentos extras são repassados à teleop, após as flags fixas (para testes A/B
sem redeploy):

```bash
bash run_g1d_quest_dex3.sh --display-mode pass-through   # sem vídeo no headset
bash run_g1d_quest_dex3.sh --camera-scale 0.3            # vídeo menor
```

O launcher recusa (exit 2) `--motion`, `--arm`, `--ee`, `--input-mode`,
`--network-interface`, `--camera-layout`, `--force-zmq-video`, `--video-fps` e
`--jpeg-quality` (também na forma `--flag=valor`). Use as variáveis
`G1D_VIDEO_MODE` e `G1D_HEAD_*` para selecionar/configurar o modo seguro.

Todo processo invocado recebe:

```text
--arm G1D --g1d-ik-geometry-assumption --ee dex3 --input-mode hand
```

O modo legado acrescenta `--camera-layout vertical`; o modo leve acrescenta
`--camera-layout head --force-zmq-video --video-fps 10 --jpeg-quality 60`.

Ele não habilita controle de movimento: G1D é somente braços.

`--input-mode hand` mantém o caminho híbrido existente do Quest: poses dos
controladores alimentam os braços via calibração controlador→punho, e cada
trigger controla exclusivamente os dedos Dex3 (repouso aberto e fechamento
proporcional). Não habilite rastreamento de mãos para os dedos.

O processo só arma rastreamento com `r` no terminal e encerra com `q`. Antes de
`r`, o G1D mantém a pose medida; após `q`, ele libera a saída sem retorno
automático à pose zero e sem abrir/fechar automaticamente as mãos.
