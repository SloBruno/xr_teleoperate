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

## Base AGV e coluna de elevação (opt-in, `G1D_AGV=1`)

Desligado por padrão. Sem `G1D_AGV=1` nada relacionado ao AGV é construído
(nenhum cliente RPC, thread ou subscriber). A flag `--g1d-agv` só é adicionada
pelo launcher; passá-la (ou abreviações `--g1d-a…`) como argumento extra é
recusado. `--motion` continua recusado. Funciona nos três modos de vídeo
(`off`, `head`, `vertical`).

Comando exato (modo de vídeo leve recomendado), a partir de `teleop/`:

```bash
G1D_AGV=1 G1D_VIDEO_MODE=head bash run_g1d_quest_dex3.sh
```

Limite de velocidade linear (padrão 0.6 m/s, máximo 1.5 m/s; validado no
launcher e no Python, valor inválido aborta antes de qualquer preflight):

```bash
G1D_AGV=1 G1D_AGV_MAX_VX=0.3 G1D_VIDEO_MODE=head bash run_g1d_quest_dex3.sh
```

Mapeamento (mesma convenção de sinais já validada na caminhada do G1: o Quest
reporta y<0 com o stick para frente):

| Entrada | Comando | Faixa |
|---|---|---|
| stick esquerdo frente/trás | `Move.vx` (+frente) | ±`G1D_AGV_MAX_VX` m/s |
| stick esquerdo esquerda/direita | `Move.vyaw` (esquerda = anti-horário, +) | ±0.6 rad/s |
| stick direito frente/trás | `HeightAdjust` (frente = sobe) | ±1 (×76.5 mm/s) |

`vy` é sempre 0 (não suportado). Zona morta 0.12 + curva cúbica (as mesmas da
caminhada do G1). Sem limites de altura em software: a coluna tem fim de curso
mecânico.

Segurança: só há movimento após `r` (rastreamento armado) com amostra do
controle fresca (≤0.25 s). O cliente RPC `agv` e a thread de envio (20 Hz) só são
criados no início do rastreamento; o loop de 30 Hz apenas publica o alvo em
memória. Com controle velho, não armado, `q`/B, exceção ou encerramento: envia
`Move(0,0,0)` e `HeightAdjust(0)` no próximo ciclo do sender (≤50 ms + latência
RPC); no encerramento, rajada de zeros antes da limpeza lenta dos braços (join
limitado; pior caso ~2 s com robô sem resposta). Se uma RPC travada retornar
depois do encerramento, o próprio sender envia um par de zeros final. A coluna também para se
`rt/hispeed_state` (altura) ficar sem amostra por >0.5 s. Códigos RPC diferentes
de 0 (9101 não iniciado, 9102 falha de movimento, 9103 falha de altura) são
contados e os primeiros são logados.

Status (`teleop-status.jsonl`, campo `locomotion.agv`): `raw_left_xy`,
`raw_right_xy` (sticks crus), `target` [vx, vyaw, vz], `sent`
[vx, vy, vyaw, vz] efetivamente enviado, `reason`/`column_reason`,
`move_code`/`height_code`, contadores de erro e `height` (altura em metros,
idade e frescor).

Validação física (primeira vez, área livre, emergência à mão, robô suspenso ou
em espaço amplo): `G1D_AGV_MAX_VX=0.2`; antes de `r` confirmar que nada se
move; depois de `r`, um eixo por vez com deflexão pequena — stick esquerdo
frente (status: `raw_left_xy[1]<0`, `sent[0]>0`, base anda para frente), esquerda
(`raw_left_xy[0]<0`, `sent[2]>0`, gira anti-horário), stick direito frente
(`raw_right_xy[1]<0`, `sent[3]>0`, coluna sobe, `height.height_m` aumenta);
soltar → zeros; tirar o controle do rastreamento → zeros em ≤0.25 s; `q` → para.
Se o sinal físico divergir, compare o stick cru com o comando antes de mudar
qualquer sinal.

Limitação conhecida (verificar antes do uso em campo): todos os zeros dependem
do processo Python vivo. SIGKILL/OOM/segfault, queda do link Ethernet ou SIGTERM
(a teleop não instala handler de SIGTERM; o `finally` não roda) deixam o último
`Move`/`HeightAdjust` a cargo do firmware do AGV. Em bancada (rodas fora do chão,
coluna livre), envie um único comando pequeno e pare de enviar: meça se e quando
base e coluna param sozinhas e registre aqui. Se o comando ficar travado, a
emergência física é a única proteção contra queda do processo. Encerre sempre
com `q`/B/Ctrl+C, nunca `kill`.
