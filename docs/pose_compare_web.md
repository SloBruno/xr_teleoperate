# Página de comparação de punho: robô (FK) × mão (alvo do IK)

Ferramenta só-leitura para a teleop Inspire (`dev-inspire`): dois gráficos X/Y/Z
vs tempo — posição do punho do robô e posição de referência do punho do
operador — mais um painel de erro, e o botão **Salvar tarefa**.

> **Linha Dex3:** `XR_POSE_WEB=1 bash teleop/run_g1_quest_dex3.sh` (env `tv`,
> estado em `~/.local/state/xr_teleoperate/`, tarefas em `.../tasks/`). Na Dex3
> a "mão" é o **alvo do IK vindo dos controladores** já calibrado (e, com
> `G1_TORSO_LEAN=1`, expresso no referencial do tronco), não o hand tracking.
> Antes do `r` não há alvo (controles ainda não calibrados): só o robô é
> enviado, mão = NaN. Sem `XR_POSE_WEB` nada é enviado (loop inalterado).

## O que é plotado (interpretação)

| Série | Origem | Frame |
|---|---|---|
| **Mão** (gráfico 2) | translação de `tele_data.left/right_wrist_pose` = o **alvo do IK** que o teleop passa a `arm_ik.solve_ik` (punho do Quest hand tracking já convertido para o robô) | cintura do robô (modelo reduzido do IK), m, x frente, y esquerda, z cima |
| **Robô medido** (gráfico 1, sólido) | FK de `q` medido (`arm_ctrl.get_current_dual_arm_q()`, o mesmo lowstate que o loop já leu) | mesmo |
| **Robô comandado** (gráfico 1, tracejado) | FK de `sol_q` (saída do IK filtrada, enviada ao braço) | mesmo |
| **Erro** (gráfico 3) | `|mão − robô medido|` por eixo e norma, mm | — |

O ponto do robô é o frame `L_ee`/`R_ee` do IK (+0,05 m em x de
`{left,right}_wrist_yaw_joint`), com o mesmo URDF e as mesmas juntas travadas
(`teleop/utils/arm_fk.py`; o teste `tests/test_arm_fk.py` compara o modelo com
`G1_29_ArmIK` e verifica FK(sol_q) ≈ alvo < 1 cm quando o IK converge).

Gráficos 1 e 2 usam a mesma janela de tempo e a mesma escala Y. Antes do `r`
os pacotes vão com `tracking=false` (mão + robô medido; comandado vazio).

## Painel 3D

Seletor **2D / 3D / 2D + 3D** na barra de controles. O painel 3D é um `<canvas>`
com projeção perspectiva em JS puro (sem bibliotecas, sem CDN, funciona offline),
no **mesmo referencial do FK/IK** (cintura do modelo reduzido, m):

- eixos na origem: **X** vermelho (frente), **Y** verde (esquerda), **Z** azul
  (cima); grade leve de 10 cm abaixo dos dados;
- **rastro** dos últimos 10/30/60 s (mesma janela dos gráficos 2D), mais apagado
  quanto mais antigo: mão/alvo do IK (amarelo), robô medido (roxo, sólido),
  robô comandado (roxo, fino tracejado), conforme o seletor Medido/Comandado;
- **ponto atual** de cada um e uma linha fina tracejada mão ↔ robô medido com o
  erro atual em mm;
- **esqueleto do braço medido**: origens das juntas ombro pitch → roll → yaw →
  cotovelo → punho roll → pitch → yaw → ponto `L_ee`/`R_ee`, mais pelve ↔ tronco
  ↔ ombros. Calculado no servidor (`G1_29_WristFK.skeleton`, mesmo modelo
  pinocchio de `arm_fk.py`) a partir do q **medido** já recebido, decimado a
  ~20 Hz (`--skeleton-hz`) e enviado em `/api/samples` no campo `sk`
  (`{"l": 8 pts, "r": 8 pts, "b": [pelvis, torso]}`). O braço não selecionado
  aparece apagado. Com `--no-fk` (ou sem pinocchio) o esqueleto some e o painel
  avisa.
- Seletores Braço (E/D/Ambos), Janela e **Pausar** valem também para o 3D.

Interação: **arrastar** (mouse ou um dedo) gira em órbita (yaw/pitch);
**roda** ou **pinça** = zoom; botões **Frente / Lado / Topo / Isométrica**;
**Recentrar** enquadra os dados da janela. Caixas para ocultar esqueleto/rastro.
Redesenho só quando chegam dados novos ou há interação, limitado a ~30 fps.
O CSV salvo não mudou.

## Arquitetura

1. Teleop: `teleop/utils/pose_stream.py` — com `XR_POSE_STREAM=1`, envia por
   UDP não bloqueante (`MSG_DONTWAIT`, 127.0.0.1:47555) um struct fixo de
   ~150 B, decimado a 50 Hz (`XR_POSE_STREAM_HZ`). Sem FK, sem arquivo, sem
   JSON no loop; erro de envio só incrementa contador. Sem a variável, o
   objeto é `None` e o loop não muda.
2. `tools/pose_compare_web.py` (stdlib `ThreadingHTTPServer`, polling 10 Hz):
   recebe UDP, calcula FK com pinocchio (punho a cada amostra; esqueleto do
   braço medido a ~20 Hz), buffer de 120 s, gravação de tarefas. Página `tools/pose_compare_web.html`: canvas puro, sem CDN,
   funciona offline. Não publica nada em DDS.
3. Launcher: `XR_POSE_WEB=1` exporta `XR_POSE_STREAM=1` e inicia o servidor
   como filho (`setsid`, log `~/.local/state/xr_teleoperate_inspire/pose_web.log`),
   parado no EXIT trap. Padrão desligado.

## Como usar (robô)

```bash
cd ~/xr_teleoperate_inspire
XR_POSE_WEB=1 TELEIMAGER_PYTHON=/home/unitree/miniconda3/envs/tv_inspire/bin/python \
TELEIMAGER_STATE_DIR=/home/unitree/.local/state/xr_teleoperate_inspire \
  bash teleop/run_g1_quest_inspire.sh
# página: http://<ip-wifi-ou-tailscale-do-robô>:8093/
```

Manual (teleop já rodando com `XR_POSE_STREAM=1` exportado):

```bash
cd ~/xr_teleoperate_inspire
PYTHONPATH=$PWD ~/miniconda3/envs/tv_inspire/bin/python tools/pose_compare_web.py   # --port 8093
# teste sem robô: ... tools/pose_compare_web.py --fake-source
```

Variáveis: `POSE_WEB_PORT=8093`, `POSE_WEB_TOKEN` (exige `?token=...`),
`XR_POSE_STREAM_PORT=47555`, `XR_POSE_STREAM_HZ=50`, `POSE_WEB_TASK_DIR`.

## Salvar tarefa

Nome + modo: **Duração fixa** (s, contagem regressiva opcional de 3 s; clicar
de novo para antes) ou **Iniciar/Parar**. Arquivos em
`~/.local/state/xr_teleoperate_inspire/tasks/<UTC>_<nome>.{csv,json}` e
listados na página para download.

CSV (uma linha por amostra, ~50 Hz): `t_mono_s, t_rel_s, t_utc, seq, tracking,
fresh, hand_{L,R}_{x,y,z}, robot_cmd_{L,R}_{x,y,z}, robot_meas_{L,R}_{x,y,z},
q_cmd_0..13, q_meas_0..13` (m, rad; vazio = indisponível). `t_mono_s` é o
relógio monotônico do teleop. JSON: nome, modo, início/fim UTC, nº de
amostras, taxa, fração em tracking, frame, unidades, ordem de q, versão git.

## Limitações

- O alvo da mão é o que o IK recebe; não inclui o filtro/limites do IK (isso
  aparece na diferença mão × comandado).
- q medido vem do lowstate lido no mesmo ciclo (idade ≤ 1 ciclo DDS); o FK
  ignora cintura/pernas (travadas em 0 como no IK), então é frame da cintura
  do modelo, não do mundo.
- Latência da página: ~100–200 ms (polling); os timestamps gravados são os do
  teleop.
- `fresh` = alvo mudou desde a amostra anterior (proxy de mão nova do Quest;
  o teleop não expõe idade da amostra nesta branch).
