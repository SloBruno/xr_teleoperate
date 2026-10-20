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
| `p` | REPLAY da última gravação salva (ou `--replay`): uma execução; rampa segura ≤0.5 rad/s (braço) / ≤2 rad/s (dedos) até a 1ª amostra, depois reprodução interpolada com limite de passo e de junta; segura a pose final (não volta para casa). |
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

## Painel web opcional (`--web`)

```bash
bash run_g1d_teach_replay.sh --web          # porta 8095
bash run_g1d_teach_replay.sh --web 9000
```

O terminal imprime `http://<ip>:PORT/?t=TOKEN` para cada interface (use a do
Wi-Fi). Abra no celular/Quest/PC — página única, sem internet/CDN. As teclas
`r/s/p/q` continuam funcionando e são o fallback.

* Botões Ensinar (r), Salvar (s), Reproduzir (p) e **PARAR (q)** fixo embaixo;
  status (modo, mensagem, idade do estado braço/mãos, gravação + tempo,
  progresso do replay) atualizado a ~5 Hz.
* Lista de gravações (mais recente primeiro, só salvas; "mostrar abortadas"
  opcional) com duração/data: Selecionar para reproduzir, Renomear (mantém o
  prefixo UTC: `<UTC>_<rótulo>.jsonl`), Apagar (move para `teach/trash/`).
  Renomear/apagar recusados enquanto o arquivo está gravando/reproduzindo.
* Token aleatório por execução, exigido em toda `/api` (comparação em tempo
  constante); sem token a página só mostra "token necessário". Qualquer um com a
  URL completa controla o robô: não compartilhe.
* Um navegador controla por vez; outros ficam somente leitura até o primeiro
  ficar >3 s sem consultar. PARAR é aceito de qualquer navegador com token e
  descarta ações web pendentes.
* A perda da página/heartbeat **não** interrompe replay finito ou contínuo.
  Frescura do estado DDS, proteção das mãos, erro de rastreamento, limites,
  ciclo de vida e **PARAR** continuam sendo gates independentes.
* O painel permite `Repetições` (N) ou `Repetir continuamente`; cada ciclo
  volta à primeira amostra pela mesma rampa segura a partir da pose medida.
  Upload aceita somente JSONL G1D validado, com limite conservador de **16 MiB**;
  o arquivo é escrito em chunks, validado e promovido atomicamente sem substituir
  outro arquivo. Upload nunca inicia replay.
* O estado da sessão fica em `~/.local/state/xr_teleoperate/teach_replay.session.json`
  com modo 0600. Invocar o launcher novamente para uma sessão viva imprime
  somente a URL existente (ou informa que não há painel), sem inicializar DDS.
* O servidor HTTP só enfileira pedidos para os mesmos `request_*` das teclas;
  nunca faz IO/DDS e nunca espera o ciclo de controle. Se a porta estiver
  ocupada, o painel não sobe e o modo terminal segue normal.

Todos os números (kd de teach, fração de gravidade 50% do esforço URDF, limites
de velocidade, 2 s de hold no `q`) são palpites iniciais — calibrar no teste
físico, robô suspenso, R3/e-stop na mão.
