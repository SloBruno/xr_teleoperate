# Dex3 ao encerrar a teleoperação (`DEX3_SHUTDOWN_HAND`)

Pedido do operador: "quando apertar encerrar, a mão fechar".

## Modo

| `DEX3_SHUTDOWN_HAND` | Comportamento |
|---|---|
| `close` (padrão) | fecha em rampa até a pose do gatilho = 1,0 |
| `open` | comportamento anterior (0c5ba31): abre após o retorno dos braços |
| `hold` | mantém o último alvo do gatilho (rampa a partir do último comando publicado) |

`bash teleop/run_g1_quest_dex3.sh` imprime o modo e recusa (exit 2) valores
inválidos. No Python, valor inválido → `open` + log de erro.
Exemplo: `DEX3_SHUTDOWN_HAND=open bash teleop/run_g1_quest_dex3.sh`.

## "Fechada"

É a mesma pose que o gatilho em 1,0 produz (`Dex3_{Left,Right}_Closed_Pose`:
Thumb0 neutro, polegar reto em pegada total, indicador/médio na pegada
configurada) — não um limite mecânico. Rampa smoothstep, pico ≤ 3 rad/s por
junta, mínimo 0,9 s (curso total aberto→fechado = 0,9 s). Cada ciclo passa pelo
mesmo `Dex3HandProtector` da teleoperação: kp = 1,5, kd = 0,2, tau = 0, mesmos
tetos de torque de fechamento, derate térmico 65–80 °C sobre o q medido e o
mesmo grip-hold de stall. Logo, fechar no encerramento nunca aperta mais do que
puxar o gatilho até o fim (teste comparativo com objeto bloqueando os dedos).

## Sequência no q / B / Ctrl+C / SIGTERM / erro no loop

1. O loop para de gerar IK; `STOP = True` (SIGTERM vira `KeyboardInterrupt`,
   mesmo caminho do q; um segundo SIGTERM durante o encerramento é ignorado).
2. StopMove da locomoção (como antes).
3. **Dex3**: gatilhos perdem autoridade (latches revogados); a mão fecha em
   rampa (~0,9 s; espera limitada a 2,5 s).
4. Braços voltam à pose segura (≤ 0,5 rad/s) — a mão continua fechada,
   protegida, publicada a 100 Hz. Cintura volta ao neutro se o torso lean estava ligado.
5. Rampa de peso do `arm_sdk` 1 → 0 em 2 s.
6. Último quadro da Dex3: juntas que chegaram ao alvo ficam no comando fechado;
   juntas bloqueadas por um objeto (> 0,15 rad do alvo) recebem `q_cmd = q medido`
   (zero torque implícito, sem aperto residual). Depois a thread da Dex3 para.
7. Writer dos braços desativado.

Se o fechamento falhar/for interrompido (exceção, segundo Ctrl+C), a
sequência anterior roda: abre a mão (`open_and_deactivate`) e libera os braços.

## Segurança (mantida)

Por lado, a cada ciclo: estado DDS ausente ou antigo (> 0,5 s), fault
(motorstate ≠ 0 ou fault já travado) ou temperatura ≥ 80 °C (ou ainda no
latch térmico) → **não fecha**: comando aberto/relaxado da regra existente
(motor em fault continua com kp = kd = 0) e log PT-BR
`[Dex3 encerramento <lado>] NÃO fecha: <motivo>`. O bloqueio trava até o fim.
Ex.: a Dex3 esquerda sem `rt/dex3/left/state` fica aberta e só a direita fecha.
`_force_open` (`open_and_deactivate`) continua vencendo qualquer modo. Não há
caminho de emergência separado no código; o R3/E-stop físico não é afetado.

## O que o firmware faz quando o publisher para

Os comandos usam `mode` com o bit de timeout = 0 (`_RIS_Mode(timeout=0)`,
inalterado). Pela documentação da Unitree (G1 dexterous hand, `RIS_Mode_t`):
"Master->Motor: 0 = Disable timeout protection, 1 = Enable (default 1 s)".
Portanto, **sem timeout, o motor continua seguindo o último comando** (PD
kp = 1,5 em torno do último q): a mão deve permanecer fechada, sem relaxar
sozinha, e **sem a proteção térmica de software** (que roda só no processo do
teleop). Por isso o último quadro zera o aperto das juntas bloqueadas.
NÃO VERIFICADO no hardware: confirmar no teste físico que a mão fica fechada
após o processo sair e que a temperatura não sobe (consultar `rt/dex3/*/state`).
Para abrir depois: rodar de novo com `DEX3_SHUTDOWN_HAND=open` e encerrar, ou
desligar/religar a mão.
