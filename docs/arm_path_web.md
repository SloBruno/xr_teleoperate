# Caminho do punho / centro da mão (G1) — `tools/arm_path_web.py`

Programa web **independente e somente leitura** que grava o caminho 3D do
punho (ou do centro da mão) de **cada braço** do G1 durante uma tarefa que você
inicia/para na página, desenha o caminho e mede o **comprimento percorrido**
por braço. Os dados são salvos **separados por braço**.

> **Linha Dex3:** rodar no env `tv` (`/home/unitree/miniconda3/envs/tv/bin/python`),
> tarefas em `~/.local/state/xr_teleoperate/arm_paths/`. O "centro da mão"
> agora tem perfil selecionável `--hand dex3|inspire` (padrão **dex3**,
> env `ARM_PATH_HAND`): Dex3-1 = **(0,080; ±0,004; 0) m** no `wrist_yaw_link`,
> **ESTIMADO** do URDF `assets/g1/g1_body29_hand14.urdf` (palma +0,0415 m x;
> base dos dedos indicador/médio +0,119 m x; centro = ponto médio; não medido).
> `--hand inspire` mantém o valor (0,110; 0; 0) descrito abaixo.

## Iniciar

No robô (env `tv_inspire`, com ou sem teleop rodando — Inspire ou Dex3):

```bash
export LD_LIBRARY_PATH=/home/unitree/cyclonedds/build/lib
export PYTHONPATH=<checkout>:/home/unitree/unitree_sdk2_python
cd <checkout>
/home/unitree/miniconda3/envs/tv_inspire/bin/python tools/arm_path_web.py            # :8095, iface enP8p1s0
#   --point hand                      começa no centro da mão (troca também na página)
#   --hand-center-offset 0.11,0,0     offset (m) no frame wrist_yaw_link (ambas); ou -left/-right
#   --rate 100                        taxa de amostragem/FK (Hz)
#   --filter-window-s 0.05 --min-step-m 0.001   filtro do comprimento "filtrado"
#   --fake-source                     q sintético (PC, sem robô)
#   ARM_PATH_TOKEN=segredo            exige ?token=segredo
```

URLs (Tailscale/Wi-Fi/local) são impressas no início; porta padrão **8095**
(8090 = calib_web, 8093 = pose_compare). Ctrl+C encerra limpo (se estiver
gravando, salva antes).

## Fonte de dados (só leitura)

- `ChannelFactoryInitialize(0, iface)` + `ChannelSubscriber("rt/lowstate",
  unitree_hg LowState_)`. **Nenhum publisher** é criado (há teste que garante
  que o arquivo nem referencia `ChannelPublisher`/`Write`).
- q medido dos 14 motores de braço **15..28** (mesma ordem de
  `G1_29_JointArmIndex` do teleop: esquerdo 15..21, direito 22..28).
- O callback DDS só copia os 14 q sob um lock curto; uma thread separada
  amostra a **100 Hz** (configurável) e calcula FK. Amostra descartada se o
  lowstate tiver mais de 100 ms (contador `n_stale`).

## Ponto medido

FK com o **mesmo modelo/frames do IK** (`teleop/utils/arm_fk.py`,
`G1_29_WristFK.points_xyz`): referencial = cintura do modelo reduzido (pernas e
cintura travadas em 0), m, X frente, Y esquerda, Z cima.

| Ponto | Definição |
|---|---|
| **Punho** (padrão) | `L_ee`/`R_ee` do IK = +0,05 m em x de `{left,right}_wrist_yaw_joint` |
| **Centro da mão** | ponto fixo no frame `{left,right}_wrist_yaw_link` (gira com o punho), offset padrão **(0,110; 0; 0) m** |

Offset do centro da mão (Inspire RH56DFQ) — **derivado do URDF, não medido**:
no URDF oficial Unitree `g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf`
(unitree_ros) a base da mão é montada a +0,0415 m em x do `wrist_yaw_link`
(`L/R_base_link_joint`) e as articulações MCP dos 4 dedos ficam a +0,178 m em x
(y≈0, z≈+3 mm). O centro da palma ≈ ponto médio = **+0,110 m em x**; o CoM da
base da mão fica a +0,108 m. O URDF do repositório (`g1_body29_hand14.urdf`,
Dex3) não tem a Inspire montada, por isso o valor vem do URDF oficial. A página
mostra o offset e a origem; o `summary.json` grava ponto, offset e origem.
Para um valor medido, use `--hand-center-offset x,y,z`.

## Página

- Barra: conexão, taxa do lowstate, **idade do lowstate**, taxa de amostras
  (real/alvo, erros de FK), indicador **GRAVANDO** + cronômetro.
- Lateral: ponto medido (Punho/Centro da mão), nome da tarefa, botão grande
  **Iniciar / Parar**, opcional duração fixa e contagem de 3 s; **comprimento
  acumulado ao vivo** por braço (filtrado, bruto, líquido); resultado final;
  lista de tarefas salvas com métricas, botão **Abrir** (redesenha a partir
  dos arquivos) e links para baixar `left.csv`, `right.csv`, `summary.json`.
- Gráficos (canvas JS puro, sem CDN; código em `tools/path_plot.js`), seletor
  **Esquerdo / Direito / Ambos lado a lado**, por braço:
  - **3D** interativo (arrastar gira, roda/pinça zoom, vistas Frente/Lado/
    Topo/Iso compartilhadas entre os braços, Enquadrar), sombra no plano da
    grade, **cor = tempo** (azul → vermelho), início ● azul, fim ■ vermelho;
  - projeções **Topo XY, Lateral XZ, Frontal YZ** com **mesma escala** nos
    dois eixos (cm);
  - **X/Y/Z × tempo** (tarefa salva: claro = bruto, forte = filtrado);
  - comprimento acumulado × tempo (ambos os braços; filtrado sólido, bruto
    tracejado).
- Fora da gravação mostra os últimos 10/30/60 s ao vivo; gravando mostra a
  tarefa inteira; ao parar abre automaticamente a tarefa salva.

## Arquivos

`~/.local/state/xr_teleoperate_inspire/arm_paths/<AAAAMMDDTHHMMSSZ>_<nome>/`
(`--out-dir` / `ARM_PATH_DIR`):

- `left.csv`, `right.csv` (um por braço, mesmas linhas/tempos):
  `t_rel_s, t_utc, x_raw, y_raw, z_raw, x_filt, y_filt, z_filt,
  dist_cum_raw_m, dist_cum_filt_m, lowstate_age_s, q_shoulder_pitch,
  q_shoulder_roll, q_shoulder_yaw, q_elbow, q_wrist_roll, q_wrist_pitch,
  q_wrist_yaw` (m, rad, s).
- `summary.json`: nome, início/fim UTC, duração, nº de amostras, taxa medida,
  idade máx. do lowstate, referencial, **ponto/offset/origem**, fonte DDS,
  **filtro (método + parâmetros)**, definições das métricas, versão git e
  `arms.left` / `arms.right` com: `length_raw_m`, `length_filtered_m`,
  `net_displacement_m` (+ vetor), `length_axis_raw_m` / `length_axis_filtered_m`
  (Σ|dx|, Σ|dy|, Σ|dz|), `duration_s`, `mean_speed_m_s`, `max_speed_m_s`,
  `raw_over_filtered`, início/fim, caixa envolvente.

## Como medir o comprimento (bruto × filtrado)

Comprimento = Σ‖p[i+1] − p[i]‖ (3D) entre amostras consecutivas.

- **Bruto**: sobre a posição FK direta. Ruído/tremor soma sempre (com ruído
  branco σ por eixo, cada passo ganha ~σ·√2·√3 mesmo com o braço parado), então
  o bruto **superestima** e cresce com a taxa de amostragem. Ex.: círculo de
  r = 10 cm a 100 Hz com 1 mm de ruído → bruto 3,7× o real.
- **Filtrado (use este)**: média móvel **centrada** de **50 ms** (fase zero:
  sem atraso nem encurtamento de movimento lento) + **histerese de 1 mm**
  (um ponto só conta quando se afasta ≥ 1 mm do último ponto contado). Mesmo
  exemplo → +10 % (1 mm de ruído) e < 2 % com 0,3 mm; braço parado → 0.
  Ajuste: `--filter-window-s`, `--min-step-m` (no `summary.json`).
- **Líquido**: ‖p_fim − p_início‖ (filtrado) — não é o caminho.
- **Por eixo**: Σ|dx|, Σ|dy|, Σ|dz|.
- **Velocidade** média = filtrado / duração; máxima = máx ‖dp/dt‖ filtrado.

Ao vivo a página mostra bruto (incremental) e filtrado (recalculado a cada
~0,5 s); os números finais são os do `summary.json`.

## Relatório offline (PC)

```bash
python tools/arm_path_report.py <pasta_da_tarefa> [--window-s 0.05] [--min-step-m 0.001] [--no-png]
```

Recalcula as métricas a partir das colunas brutas com o filtro escolhido,
imprime tabela, grava `report.json` e (com matplotlib) `left.png`/`right.png`:
3D colorido pelo tempo + projeções XY/XZ/YZ com eixos iguais + X/Y/Z × tempo +
comprimento acumulado.

## API

`GET /api/status`, `GET /api/samples?since=N`, `GET /api/record/path`,
`GET /api/tasks`, `GET /api/task?id=<id>`, `GET /tasks/<id>/<left.csv|right.csv|summary.json>`,
`POST /api/record/start {name, duration?, countdown?}`, `POST /api/record/stop`,
`POST /api/config {point: wrist|hand}` (só parado).

## Limitações

- Referencial da cintura do modelo com pernas/cintura travadas: movimento do
  tronco/cintura real **não** entra no caminho (igual ao IK).
- Centro da mão é um offset fixo derivado do URDF; não medido no robô.
- FK é do q medido (encoders); erros de calibração do modelo não são vistos.
- Gravações muito longas ficam em memória até Parar (100 Hz × 1 h ≈ 360 k
  linhas, ok).
