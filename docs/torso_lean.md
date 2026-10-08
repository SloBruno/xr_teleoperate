# Inclinação do tronco pelo Quest (G1_TORSO_LEAN)

Linha `dev-inspire` (G1_29 + mãos Inspire, hand tracking). O operador inclina o
próprio corpo (desloca a cabeça) e o robô inclina o tronco pela cintura
(pitch = frente/trás, roll = lados), no máximo **±10°**. **Desligado por padrão.**

Código: `teleop/utils/torso_lean.py` (lógica pura), writer em
`teleop/robot_control/robot_arm.py` (`G1_29_ArmController`), integração em
`teleop/teleop_hand_and_arm.py`, shutdown em `teleop/utils/arm_graceful_shutdown.py`.
Testes: `tests/test_torso_lean.py`.

## Ligar

```bash
G1_TORSO_LEAN=1 G1_TORSO_LEAN_MAX_DEG=3 XR_POSE_WEB=1 bash teleop/run_g1_quest_inspire.sh
```

| variável | padrão | limite | significado |
|---|---|---|---|
| `G1_TORSO_LEAN` | `0` | 0/1 | liga a função (exige `G1_MOTION=1`) |
| `G1_TORSO_LEAN_MAX_DEG` | `10` | (0, 10] — maior é **rejeitado** (launcher sai com código 2; o Python mantém DESLIGADO) | saturação de pitch e roll |
| `G1_TORSO_LEAN_GAIN_DEG_PER_M` | `66.7` | (0, 200] | 15 cm além da zona morta = 10° |
| `G1_TORSO_LEAN_DEADBAND_M` | `0.03` | [0, 0.20] | zona morta do deslocamento |
| `G1_TORSO_LEAN_RATE_DPS` | `15` | (0, 30] | limite de velocidade da cintura |
| `G1_TORSO_LEAN_ACCEL_DPS2` | `0` (desligado) | [0, 200] | limite de aceleração opcional |

O launcher imprime `Inclinação do tronco: LIGADA, máx N°` ou `DESLIGADA`.
Sem embreagem (o operador usa hand tracking; não há botão de controle).

## Entrada: deslocamento, não orientação

* Fonte: `tele_data.head_pose` do TeleVuer (já recebido pelo teleop; base do robô:
  x frente, y esquerda, z cima; origem = mundo do Quest).
* Usa-se o **pivô do pescoço** = posição do headset − `R_cabeça · (0.0805, 0, 0.075)`
  (modelo de pescoço padrão da Meta). Girar a cabeça (olhar para baixo/lados)
  não move o pivô → **não inclina** o robô (teste `test_looking_down_or_around_does_not_lean`).
* No `r`: guarda o pivô neutro e o yaw do operador. O deslocamento horizontal é
  expresso no referencial do yaw neutro (frente/lado do operador); a altura é
  ignorada. Operador virado 90° no mundo do Quest: frente continua = pitch.
* Frente → `+pitch`; esquerda → `−roll` (ver convenção abaixo). Sem yaw de cintura.

Mapa: zona morta 3 cm → ganho linear → **saturação dura** ±máx por eixo →
passa-baixa (τ = 0,3 s) → limite de velocidade (15°/s) [+ aceleração] →
clamp no envelope `neutro ± máx` ∩ limites do URDF → writer (clamp final de
novo, ver abaixo).

Robustez da entrada:
* pose inválida (NaN, não rígida, pose de fallback `CONST_HEAD_POSE` do TeleVuer)
  ou amostra repetida há mais de 0,2 s: mantém o último alvo;
  perda por mais de 1 s: alvo → 0 (cintura volta ao neutro suavemente);
* salto do pivô > 0,25 m (ou do yaw > 45°) entre amostras = recentralização do
  Quest/guardian: a amostra é ignorada, o neutro é reancorado para que o
  deslocamento continue do último valor aceito, e o teleop loga um aviso.

## Convenção de sinais (confirmada)

Fonte: `assets/g1/g1_body29_hand14.urdf` e enums do código.

| motor | junta | eixo URDF | q > 0 |
|---|---|---|---|
| 12 | `waist_yaw_joint` | z | gira para a esquerda (não usado) |
| 13 | `waist_roll_joint` | **+x** | tronco tomba para a **direita** (−y) |
| 14 | `waist_pitch_joint` | **+y** | tronco inclina para a **frente** (+x) |

Limites URDF: yaw ±2,618 rad, roll/pitch ±0,52 rad (±29,8°). Os índices 12/13/14
batem com `G1_29_JointIndex` e com o exemplo oficial Unitree
`unitree_sdk2/example/g1/high_level/g1_arm7_sdk_dds_example.cpp`
(`kWaistYaw, kWaistRoll, kWaistPitch`). O teste
`test_urdf_sign_convention_pitch_forward_roll_positive_is_right` fixa isso por
FK (pinocchio) de um ponto 40 cm acima do `torso_link`.

## O que vai para a cintura em `rt/arm_sdk`

**Hoje (e com a função DESLIGADA, idêntico):** no construtor o
`G1_29_ArmController` escreve em TODOS os 35 motores `mode=1` e
`q = posição medida na construção`; para 12–14 (não-braço, não “fraco”)
`kp = 300`, `kd = 3`, `dq = tau = 0`. O writer de 250 Hz só reescreve os 14
braços e o peso (`motor 29 = kNotUsedJoint0.q`), então 12–14 seguem com esse
`q` congelado da construção, com peso 1,0 desde a construção (comportamento
atual da linha Inspire). Teste de regressão
`test_feature_off_writer_frames_identical_to_base_commit`: roda o writer real do
commit base `5902937` e o novo com o mesmo publisher falso e compara todos os
campos de todos os 35 motores em 20 frames — idênticos.

**Exemplo oficial Unitree** (`g1_arm7_sdk_dds_example`): coloca a cintura
(12–14) junto com os 14 braços na lista de juntas controladas por `rt/arm_sdk`,
`kp = 60`, `kd = 1,5`, `dq = tau = 0`, velocidade máx 0,5 rad/s, rampa de peso
0,2/s no fim.

**Com a função LIGADA:** antes do `r`, nada muda. No `r` (primeiro ciclo com
cabeça válida e cintura medida fresca): o neutro é o `q` que os servos **já
estão segurando** (o da construção). Só ativa se a cintura medida estiver a
≤ 0,05 rad desse comando; senão loga erro e a sessão segue sem inclinação.
(Re-amostrar o medido como novo alvo faria o “ratchet” da queda gravitacional.)
Depois, a cada frame de 250 Hz o writer escreve em 12–14
`q = clamp(neutro + [0, roll, pitch])`, `dq = tau = 0`, yaw fixo no neutro.
**kp/kd/mode não mudam (300 / 3 / 1)** — são os que a cintura já recebe hoje
nesta linha; trocar o ganho no meio da sessão (para 60/1,5 do exemplo) seria
uma mudança de rigidez não testada no hardware e foi deixada de fora. A
autoridade do vendor sobre a cintura continua governada pelo mesmo peso.

Clamp final no writer (imediatamente antes do `Write`): alvo preso ao
envelope `neutro ± máx` (∩ URDF), e passo por frame ≤ `taxa · 4 ms` a partir
do último valor **escrito**; valor não finito → repete o último escrito.

## Consistência da IK (decisão)

O modelo reduzido da IK trava a cintura em 0 → sua base é, rigidamente, o
**referencial do tronco** (teste (1) de `test_ik_frame_is_torso_frame_and_retarget_keeps_operator_vector`:
FK do modelo completo com cintura = `pelvis_T_torso(q)·pelvis_T_torso(0)⁻¹ ·` FK reduzida).

O TeleVuer entrega o alvo do punho como `ponto_da_cabeça + v` (`ponto_da_cabeça`
= (0,15; 0; 0,45) m, `v` = mão − cabeça do operador em eixos alinhados à
gravidade/yaw). No robô, esse “ponto da cabeça” é fixo ao **tronco**. Logo, com
o tronco inclinado por `R` (rotação de cintura comandada, relativa ao neutro), o
alvo geometricamente correto em coordenadas do tronco é
`ponto + Rᵀ v` com orientação `Rᵀ R_alvo` (`retarget_to_torso`). Consequências,
provadas por FK no teste:

* se o operador inclina o corpo inteiro rigidamente, o alvo no tronco não muda
  → os braços mantêm a postura relativa ao tronco (as mãos “acompanham” a
  inclinação, como no corpo do operador);
* no referencial da pelve o robô reproduz exatamente o vetor cabeça→mão do
  operador (`R v`) e a orientação do punho (`R R_alvo`), a partir da sua
  própria cabeça;
* sem essa correção o alvo erraria vários centímetros (≈ 4–9 cm a 10° para alvos típicos).

Usa-se o ângulo **comandado** (o mesmo que vai para o writer, já suavizado),
não o medido: é determinístico, sem ruído de encoder, e o atraso de
rastreamento da cintura a 15°/s é pequeno. Além disso o feed-forward de
gravidade dos braços (rnea do modelo reduzido) usa a gravidade no referencial
do tronco, `Rᵀ g` (`G1_29_ArmIK.set_torso_rotation`; teste compara com rnea do
modelo completo com a cintura inclinada — igual a 1e-9). No shutdown a gravidade
volta ao neutro. O problema casadi não muda.

**Sites 8093/8095:** continuam no referencial da IK (= tronco). Em 8093 o
punho do robô (FK de q) e o alvo (agora o alvo retargetado, o que a IK
realmente recebeu) estão no mesmo referencial → comparação coerente. A
inclinação vai à parte na telemetria (abaixo). 8095 (passivo, `rt/lowstate`)
segue no referencial cintura-travada = tronco; documentado em `arm_fk.py`.

## Shutdown

`q` / Ctrl+C / SIGTERM / erro, ordem (testes `test_shutdown_returns_waist_to_neutral_before_weight_ramp`
e `test_teleop_feature_on_takes_waist_after_r_and_returns_it_before_weight_ramp`):

1. para a mão (como antes);
2. alvo da cintura = neutro do `r` (o writer desliza a ≤ taxa configurada), em
   paralelo ao retorno dos braços a 0 (≤ 0,5 rad/s);
3. espera limitada (`distância/taxa + 1,5 s`) até o comando **escrito** da
   cintura = neutro; loga se o medido está a ≤ 0,05 rad;
4. só então a rampa do peso do `arm_sdk` 1 → 0 em 2 s; desativa o writer.

Com a função desligada nenhuma etapa nova roda (mesma sequência de antes).

## Telemetria (pose_stream / 8093)

Com a função ativa, o pacote UDP passa a `XPS2` = `XPS1` + alvo de inclinação
(pitch, roll), inclinação comandada (pitch, roll), cintura comandada
[yaw, roll, pitch] e medida (motores 12–14), em rad. Sem a função os bytes são
exatamente `XPS1`. O servidor `pose_compare_web` aceita os dois; `/api/status`
ganha `lean` (graus, `null` para pacotes antigos), o CSV ganha colunas no fim
(vazias para `XPS1`) e a página 8093 mostra a pílula
“tronco pedida P/R · cmd P/R · medida P/R” (amarela se medida ≠ cmd > 3°).

## Custo no loop

O(1): algumas multiplicações 3×3, sem I/O, sem alocação relevante; o writer
faz 3 clamps a mais por frame quando ativo e nada quando desligado.

## Protocolo do primeiro teste físico (NÃO executado)

Pré-requisitos: robô **suspenso no pórtico ou apoiado** com folga para o tronco
inclinar, área livre, **R3 (e-stop) na mão** de um segundo operador, deploy
conferido (`git -C ~/xr_teleoperate_inspire log -1`), Quest com hand tracking.

1. `G1_TORSO_LEAN=1 G1_TORSO_LEAN_MAX_DEG=3 XR_POSE_WEB=1 bash teleop/run_g1_quest_inspire.sh`;
   conferir no terminal “Inclinação do tronco: LIGADA, máx 3°”; abrir
   `http://<ip>:8093`.
2. Operador parado, ereto, olhando para frente → `r`. Conferir no log
   `[torso_lean] ativada: cintura neutra …` (e não a mensagem de recusa).
   Robô não deve mover a cintura.
3. Só girar a cabeça (olhar para baixo, lados): pílula “pedida” deve ficar 0.
4. Deslocar a cabeça ~5 cm para a frente (dentro da zona morta + 2 cm): pedida
   ≈ +1,3°; depois ~20 cm: pedida satura em +3°. Observar no robô inclinação
   **para a frente**, lenta (≤ 15°/s), e “medida” acompanhando “cmd”.
5. Repetir para trás, esquerda (roll **negativo**, robô tomba para a ESQUERDA)
   e direita. Qualquer sentido trocado → `q` imediatamente.
6. Observar os braços: com o corpo inteiro inclinado, as mãos devem acompanhar
   o tronco sem saltos; no 8093 o erro mão×robô não deve crescer com a
   inclinação.
7. Tirar o headset/cobrir sensores > 1 s: cintura volta ao neutro sozinha.
8. `q`: cintura volta ao neutro antes da rampa de peso; braços voltam como hoje.
9. Abortar (R3) em: movimento sem deslocamento da cabeça, sentido errado,
   oscilação, ruído/vibração na cintura, “medida” longe de “cmd” (> 3°) por
   mais de 1 s, perda de equilíbrio.
10. Só depois de 3° limpo em todas as direções, repetir com 5° e 10°; só então
    testar com o robô em pé no chão (balanço: o controlador da Unitree vai
    reagir ao deslocamento do CoM).
