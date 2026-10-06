# Calibração de caminhada (trim) e offset da IMU — G1

Ferramentas offline/passivas. O terminal de marcadores **não** cria DDS, não publica
nada e não toca na teleop: só lê o teclado e grava JSONL. O analisador só lê arquivos.

## Pré-requisitos
- Teleop rodando com `G1_BALANCE_TELEMETRY=1` (bloco `balance` na
  `pose-telemetry-*.jsonl`, em `/home/unitree/.local/state/xr_teleoperate/`).
- Relógio do robô é a referência: marcadores e telemetria rodam na mesma máquina
  (alinhamento por `time.time()`/UTC).

## Abrir o terminal de marcadores (SSH, em paralelo à teleop)
Em outra janela/terminal (pode ser outra pessoa operando):

```bash
ssh -t unitree@100.126.188.19
cd /home/unitree/xr_teleoperate_slo
/home/unitree/miniconda3/envs/tv/bin/python tools/mark_calibration_segments.py
# opcional: --out /caminho/calib-markers-teste.jsonl
```
Grava `calib-markers-<UTC>.jsonl` (append-only, flush+fsync a cada tecla).

## Teclas (letra + Enter)
| tecla | efeito |
|---|---|
| `s` / `c` | condição atual: sem caixa / com caixa |
| `t N` | tentativa N |
| `o R P Y` | offset IMU da pelve em uso (graus roll pitch yaw, valor do app Unitree Explorer); persiste até mudar |
| `r` | início da RETA |
| `e` | início giro 360° à ESQUERDA |
| `d` | início giro 360° à DIREITA |
| `f` | fim do trecho aberto |
| `x texto` | nota livre |
| `u` | desfaz o último marcador |
| `?` | ajuda |
| `q` | sair |

Abrir um trecho fecha o anterior. **Não existe tecla de parada**: o operador está com
as mãos no controle. O analisador detecta a soltura do joystick automaticamente
(`loco_command` efetivo ~0 por ≥0,2 s após a reta) e mede a parada até o robô
ficar parado por 0,5 s, até o próximo marcador ou no máx. 5 s.

## Protocolo por tentativa
1. (marcador) `c` ou `s`, `t N`, e `o R P Y` se o offset mudou.
2. Robô na fita, início do espaço. (marcador) `r` quando ele começar a andar.
3. Operador anda até o fim **corrigindo com o joystick** para seguir a fita.
4. Operador solta o joystick e espera ~3 s (parada automática). Opcional: `f`.
5. (marcador) `e` ao iniciar o giro de 360° à esquerda.
6. (marcador) `d` ao iniciar o giro de 360° à direita; `f` ao terminar.
7. 3 tentativas por condição (sem caixa / com caixa, mesma pose da caixa).

Exemplo: `c`, `t 1`, `r`, …, `e`, `d`, `f`, `t 2`, `r`, …

## Varredura de offset da IMU (com caixa)
O offset é alterado **pelo app Unitree Explorer** (não por estas ferramentas).
1. Baseline: anote o valor atual no marcador (`o R P Y`), 3 tentativas com caixa.
2. Roll: baseline −0,5° e +0,5° (e opcional ±1,0°), 3 tentativas cada; registrar `o` a cada mudança.
3. Pitch: idem, com roll de volta ao baseline.
4. Yaw: não varrer (só é reportado).
5. Rodar com `--fit-offset`. Precisa de ≥3 níveis distintos por eixo; o
   "offset que zera" fora da faixa medida é extrapolação (aviso).

Se a telemetria trouxer `config_changes` com `imu_offset_json`, o analisador usa
esse valor quando não há marcador e avisa em caso de conflito (marcador vence).

## Analisar
```bash
cd /home/unitree/xr_teleoperate_slo
D=/home/unitree/.local/state/xr_teleoperate
/home/unitree/miniconda3/envs/tv/bin/python tools/analyze_walk_calibration.py \
    $D/pose-telemetry-*.jsonl --markers $D/calib-markers-<UTC>.jsonl \
    [--fit-offset] [--json] [--csv /tmp/calib.csv] [--trim-start 1.0 --trim-end 0.5]
```
Saída: por trecho (reta: trim sugerido = média de vy/omega comandados, distância,
deriva lateral/heading; parada: deslocamento, deriva frente/lateral, tempo até
parar; giros: ângulo desembrulhado, omega, translação), agregados por condição e
por condição+offset (média±desvio entre tentativas), diferença com−sem caixa, ajuste
de offset e avisos de campos ausentes.

## Lacunas conhecidas
- `loco_command` está em unidades **normalizadas** do joystick
  (`rt/wirelesscontroller`); o trim sugerido está nessa escala, não em m/s ou rad/s.
  O limite de yaw do robô (`ROBOT_MAX_YAW_RADPS=1.0`) é uma suposição.
- A odometria (`odommodestate`/`odom.*`) não foi validada contra medição externa:
  distâncias e derivas são estimativas; a fita no chão continua sendo a referência.
- `pitch_vs_erro_vx` mistura m/s reais com vx normalizado: use só a variação entre offsets.
