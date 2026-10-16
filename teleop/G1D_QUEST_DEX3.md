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

## Comando de operador

Depois de iniciar o Teleimager compatível e configurar o ambiente DDS/Quest,
execute a partir de `teleop/`:

```bash
python teleop_hand_and_arm.py \
  --arm G1D \
  --g1d-ik-geometry-assumption \
  --ee dex3 \
  --input-mode hand \
  --camera-layout vertical
```

`--input-mode hand` mantém o caminho híbrido existente do Quest: poses dos
controladores alimentam os braços via calibração controlador→punho, e cada
trigger controla exclusivamente os dedos Dex3 (repouso aberto e fechamento
proporcional). Não habilite rastreamento de mãos para os dedos.

O processo só arma rastreamento com `r` no terminal e encerra com `q`. Antes de
`r`, o G1D mantém a pose medida; após `q`, ele libera a saída sem retorno
automático à pose zero e sem abrir/fechar automaticamente as mãos.
