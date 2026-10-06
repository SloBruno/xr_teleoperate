# Calibração de caminhada (trim) e offset da IMU — G1

Duas formas de marcar os trechos:
- **UI web (`tools/calib_web.py`, recomendada)**: botões grandes no celular/tablet,
  lê o offset da IMU do DDS e permite alterá-lo com travas de segurança. Ver abaixo.
- Terminal antigo (`tools/mark_calibration_segments.py`): sem DDS, só teclado. Continua funcionando.

O analisador só lê arquivos e aceita os dois formatos.

## UI web de calibração

### Iniciar (no robô, em paralelo à teleop; processo independente)
```bash
ssh unitree@100.126.188.19
cd /home/unitree/xr_teleoperate_slo
export LD_LIBRARY_PATH=/home/unitree/cyclonedds/build/lib
export PYTHONPATH=/home/unitree/unitree_sdk2_python:/home/unitree/xr_teleoperate_slo
# opcional: export CALIB_WEB_TOKEN=algum-segredo
/home/unitree/miniconda3/envs/tv/bin/python tools/calib_web.py --iface enP8p1s0
#   --read-only  só leitura passiva (sem GET, sem SET)
#   --dry-run    SET simulado (nada é enviado), sem GET
#   --port 8090 --window 3 --max-step 0.5 --min-interval 1 --allow-yaw
```
O programa imprime as URLs: `http://10.22.16.175:8090/` (Wi-Fi) e
`http://100.126.188.19:8090/` (Tailscale). Com token, use `...:8090/?token=SEGREDO`.
Abra no celular/tablet. Para sair: Ctrl+C. Se o offset final diferir da base, o
log avisa; **nada é restaurado automaticamente**.

Arquivos (em `/home/unitree/.local/state/xr_teleoperate/`):
`calib-markers-<UTC>.jsonl` (mesmo formato do terminal; cada marcador leva
`imu_offset` + `imu_offset_source`) e `offset-changes-<UTC>.jsonl` (cada escrita:
antes, depois, code, verificação, UTC/monotônico). Cada escrita também entra no
calib-markers como `type: "offset_set"`.

### Painéis
- **Sessão/marcadores**: Sem caixa / Com caixa, tentativa (auto-incrementa ao iniciar
  uma nova reta; ± para corrigir), Reta, Giro Esq., Giro Dir., Fim, Desfazer, Nota;
  trecho aberto com cronômetro; checklist t1..t3 por condição; últimos marcadores.
- **Status ao vivo (passivo)**: `rt/wirelesscontroller` (lx, ly, rx e se ~0),
  velocidade e yaw rate (`rt/odommodestate`), roll/pitch da IMU da pelve
  (`rt/lf/lowstate`) e do torso (`rt/secondary_imu`), idade de cada fonte,
  indicador "teleop ativa".
- **Offset da IMU**: `imu_offset_json` (pelve) e `secondaryimu_offset_json` (torso)
  com origem (`dds_get`, `dds_passive_app`, `dds_passive_status`, `manual`,
  `ui_set`) e idade; base da sessão; ±0,1° / ±0,5° em roll e pitch da pelve; valor
  exato; "Restaurar base" (passo a passo, cada passo confirmado); "Reler (GET)";
  torso só em "Avançado"; histórico.

### Como o offset é lido
1. Ao iniciar (modo normal), um GET (api 1002, timeout 1 s) por chave. **Única ação
   automática.** Nunca em `--read-only`/`--dry-run`.
2. Se o GET falhar: valor visto passivamente (SET do app Unitree Explorer em
   `rt/api/config/request` com resposta code=0, ou `rt/config_change_status`).
3. Se nada for conhecido: "desconhecido" e escrita **bloqueada** até você confirmar o
   valor do app em "Confirmar valor manual / base". A UI nunca chuta a base.

### Regras de segurança implementadas
- Base = primeiro valor confiável da sessão; janela padrão ±3° em torno dela;
  |x| ≤ 10° rígido; passo máx. 0,5° por comando; ≥1,0 s entre escritas
  (os tetos 10°/0,5°/1 s não podem ser afrouxados por CLI); NaN/inf recusados;
  yaw só leitura (salvo `--allow-yaw`); torso só no modo avançado.
- Modal "Aplicar X→Y?" em toda escrita; se o joystick não estiver ~0, a velocidade
  > 0,05 m/s, o yaw rate alto ou as fontes estiverem sem dado, aparece aviso e é
  exigida **segunda confirmação**. O servidor revalida tudo e recusa se o valor
  atual mudou desde o modal.
- SET no formato exato do app (`{"name":"imu_offset_json","content":"{\"imu\":[r,p,y]}"}`,
  api 1001), espera resposta ≤2 s e mostra o code. Sem resposta/erro → offset
  marcado **incerto** e novas escritas bloqueadas até "Reler" ou confirmação manual.
- Verificação: GET (se disponível) e/ou variação do rpy da IMU ≈ delta em ~1 s →
  "verificado"/"não verificado".
- Nada é enviado no encerramento. O DDS writer só é criado na primeira escrita/GET.

### Uso seguro (operador)
- Alterar offset **só com o robô parado, apoiado/na fita, joystick solto**; R3/e-stop à mão.
- Passos pequenos (0,1–0,5°). Observe o robô após cada passo.
- Ao final, **Restaurar base** (ou deixe conscientemente o novo valor e anote).
- Não se sabe se o SET persiste após reboot; confira no app após reiniciar.

### Protocolo das 6 sessões (3 sem caixa + 3 com caixa)
Por tentativa: (UI) condição → **Reta** quando começar a andar; operador corrige
com o joystick do Quest até o fim da fita; solta e espera ~3 s (parada automática,
não precisa marcar); **Giro Esq.** (360°); **Giro Dir.** (360°); **Fim**.
A próxima **Reta** abre a tentativa seguinte automaticamente. Confira o checklist.

### Varredura de offset (com caixa)
1. Baseline (base lida): 3 tentativas com caixa.
2. Roll: base −0,5° e +0,5° (opcional ±1,0°), 3 tentativas cada. Mude pelo painel
   (robô parado), um eixo por vez, volte à base entre eixos.
3. Pitch: idem, com roll na base.
4. Yaw: não varrer.
5. Analisar com `--fit-offset` (o offset de cada trecho já está nos marcadores).
6. **Restaurar base** no fim, salvo decisão explícita de manter o novo valor.

## Terminal de marcadores (antigo)

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

## Varredura de offset da IMU com o terminal antigo (com caixa)
Com o terminal, o offset é alterado **pelo app Unitree Explorer**.
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

O analisador lê, nesta ordem: offset gravado automaticamente no marcador pela UI
(`imu_offset_fonte` = `marcador:<origem>`), eventos `offset_set` bem-sucedidos
(não dry-run), marcador `o R P Y` do terminal, e `config_changes` da telemetria
(`{"imu":[r,p,y]}`; `secondaryimu` é ignorado).

## Lacunas conhecidas
- GET api 1002 do serviço `config` **não confirmado** no G1 (vem do `config_api.hpp`
  do Go2/B2). Se falhar, a UI usa leitura passiva ou valor manual.
- Formato real de `rt/config_change_status` não capturado ainda (parser tolerante).
- Não se sabe se o SET persiste em flash/após reboot.
- Efeito do `secondaryimu_offset_json` no `rt/secondary_imu` não medido.
- `loco_command` está em unidades **normalizadas** do joystick
  (`rt/wirelesscontroller`); o trim sugerido está nessa escala, não em m/s ou rad/s.
  O limite de yaw do robô (`ROBOT_MAX_YAW_RADPS=1.0`) é uma suposição.
- A odometria (`odommodestate`/`odom.*`) não foi validada contra medição externa:
  distâncias e derivas são estimativas; a fita no chão continua sendo a referência.
- `pitch_vs_erro_vx` mistura m/s reais com vx normalizado: use só a variação entre offsets.
