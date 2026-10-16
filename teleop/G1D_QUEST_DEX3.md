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

O perfil G1D exige especificamente dois caminhos físicos/configurados:
`head_camera` e `left_wrist_camera`. Antes de iniciar, o launcher verifica os
dois dispositivos RealSense esperados, as duas seções configuradas com
`enable_zmq=true`, seus seriais e os dois frames ZMQ. Não existe fallback para
uma câmera, para a câmera de pulso publicada como cabeça, nem para `head`/`any`:
sem fallback.
Se qualquer fluxo não estiver saudável, o perfil falha fechado e não inicia a
teleoperação.

## Comando de operador

No Jetson/PC2, com o Teleimager e o ambiente Quest já preparados, execute a
partir de `teleop/`:

```bash
bash run_g1d_quest_dex3.sh
```

O launcher é separado de `run_g1_quest_dex3.sh`; não modifique nem reutilize o
perfil G1. Ele fixa DDS em `enP2p1s0` por padrão. Para outra interface, use
`G1D_DDS_INTERFACE=<interface> bash run_g1d_quest_dex3.sh`.

O processo invocado pelo launcher recebe:

```text
--arm G1D --g1d-ik-geometry-assumption --ee dex3 --input-mode hand
--camera-layout vertical
```

Ele não habilita controle de movimento: G1D é somente braços.

`--input-mode hand` mantém o caminho híbrido existente do Quest: poses dos
controladores alimentam os braços via calibração controlador→punho, e cada
trigger controla exclusivamente os dedos Dex3 (repouso aberto e fechamento
proporcional). Não habilite rastreamento de mãos para os dedos.

O processo só arma rastreamento com `r` no terminal e encerra com `q`. Antes de
`r`, o G1D mantém a pose medida; após `q`, ele libera a saída sem retorno
automático à pose zero e sem abrir/fechar automaticamente as mãos.
