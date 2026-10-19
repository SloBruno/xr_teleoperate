# G1D teach-and-replay (braços + Dex3)

Modo separado da teleop (`run_g1d_quest_dex3.sh` não muda). Sem Quest, sem IK,
sem vídeo. Usa a mesma interface DDS (`enP2p1s0`, `G1D_DDS_INTERFACE`) e o
mesmo ambiente `xr-g1d`.

```bash
cd teleop
bash run_g1d_teach_replay.sh                 # exige as duas mãos Dex3
bash run_g1d_teach_replay.sh --hands left    # opt-in: só mão esquerda
bash run_g1d_teach_replay.sh --hands none    # opt-in: só braços
bash run_g1d_teach_replay.sh --replay 20261019T120000_000000Z.jsonl
```

Teclas (terminal interativo):

| tecla | ação |
|---|---|
| `r` | TEACH: braços com kp=0, kd pequeno + feed-forward de gravidade (modelo) no q medido; Dex3 kp=0, kd=0.05. Grava q/dq a 50 Hz. |
| `s` | para de ensinar, salva `~/.local/state/xr_teleoperate/teach/<UTC>.jsonl` e segura a pose (ganhos normais + gravidade). |
| `p` | REPLAY da última gravação salva (ou `--replay`): rampa ≤0.5 rad/s (braço) / ≤2 rad/s (dedos) até a 1ª amostra, depois reprodução interpolada com limite de passo e de junta; segura a pose final (não volta para casa). |
| `q` | para tudo: segura a pose medida com gravidade por 2 s e libera as saídas. |

Segurança:

* Antes do primeiro `r`/`p` só existem subscribers; nenhum publisher.
* `r`/`p` são recusados com estado de braço ou Dex3 ausente/velho, modelo de
  gravidade não finito, ou rampa > 15 s. Recusa também se outro
  `teleop_hand_and_arm.py`/`teach_replay_g1d.py` estiver rodando.
* Estado velho durante teach/replay → HOLD no último comando validado (não
  continua a trajetória). Gravação interrompida fica como `*.aborted.jsonl` e
  nunca é escolhida como "última".
* Erro de rastreamento > 0.35 rad no replay → HOLD na pose medida.
* Proteção Dex3 existente (falha/térmica/torque) aplicada no replay.
* Um único escritor para `rt/lowcmd` e cada `rt/dex3/<lado>/cmd`.

Todos os números (kd de teach, fração de gravidade 50% do esforço URDF, limites
de velocidade, 2 s de hold no `q`) são palpites iniciais — calibrar no teste
físico, robô suspenso, R3/e-stop na mão.
