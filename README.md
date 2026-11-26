# AlignFit — dados e modelos

Repositório de **dados, treinamento e experimentos** do AlignFit, sistema que
recebe o vídeo de um exercício físico e informa **qual exercício** foi feito e
se a **execução está correta**.

Exercícios suportados: agachamento, rosca bíceps, desenvolvimento de ombro e
elevação lateral.

> Repositórios relacionados:
> [`iac-infra`](https://github.com/AlignFit/iac-infra) (infraestrutura AWS que usa os modelos daqui) ·
> [`align-fit-web`](https://github.com/AlignFit/align-fit-web) (site onde o usuário envia o vídeo).

---

## Como o sistema funciona

```
 vídeo ──► 1. extrair keypoints ──► 2. qual exercício? ──► 3. execução correta?
           (YOLOv8-Pose,            (classificador de     (um classificador
            17 pontos do corpo)      exercício)             por exercício)
```

1. **Extração de pose** — o YOLOv8-Pose encontra 17 pontos do corpo (ombros,
   cotovelos, joelhos…) em 1 a cada 5 frames do vídeo.
2. **Classificação do exercício** — as coordenadas de todos os frames são
   resumidas em 136 atributos por vídeo (média, desvio, mínimo e máximo de cada
   coordenada) e um XGBoost identifica o exercício.
3. **Classificação da execução** — o modelo específico daquele exercício diz se
   a execução está correta ou errada.

---

## Estrutura

```
exercises-dataset/
├── pipeline/              treino, rodar na ordem 1 → 2 → 3
│   ├── 1_extrair_keypoints.py            vídeos → dados/base_<data>.csv
│   ├── 2_treinar_classificador_exercicio.py   → modelos/classificador-exercicio.pkl
│   └── 3_treinar_classificador_execucao.py    → modelos/execucao-<exercicio>.pkl
│
├── inferencia/            testar um vídeo localmente (mesmo fluxo da produção)
│   ├── extrair_keypoints_video.py
│   ├── prever_exercicio.py
│   └── prever_execucao.py
│
├── alignfit/              código compartilhado
│   ├── caminhos.py        todos os caminhos do projeto num só lugar
│   ├── keypoints.py       nomes dos 17 pontos e lista de exercícios
│   ├── atributos.py       keypoints por frame → 136 atributos por vídeo
│   └── sessoes.py         identifica a sessão de gravação (usado nos experimentos da base v1)
│
├── dados/                 bases de keypoints (CSV)        → ver dados/README.md
├── modelos/               modelos treinados (.pkl)        → ver modelos/README.md
├── experimentos/          notebooks e scripts das Seções 4–6 do TCC → ver experimentos/README.md
├── yolo/                  pesos do YOLOv8n-Pose
└── videos/                vídeos brutos (fora do git, 2,4 GB)
```

---

## Instalação

```bash
py -m pip install -r requirements.txt              # Windows
python -m pip install -r requirements.txt          # Linux / macOS

# só se for rodar os notebooks de experimentos/notebooks/
py -m pip install -r requirements-notebooks.txt
```

> No Windows, use `py -m pip` em vez de `pip`: o comando `pip` sozinho
> normalmente não está no PATH.

Todos os scripts podem ser executados **a partir de qualquer pasta**; os
caminhos são resolvidos por `alignfit/caminhos.py`.

## Treinar os modelos

```bash
# 1. (opcional) gerar a base a partir dos vídeos — precisa da pasta videos/
python pipeline/1_extrair_keypoints.py --externos

# 2. classificador de exercício
python pipeline/2_treinar_classificador_exercicio.py

# 3. classificadores de execução (os 4, ou um só com --exercicio)
python pipeline/3_treinar_classificador_execucao.py
python pipeline/3_treinar_classificador_execucao.py --exercicio biceps
```

O passo 3 imprime a **acurácia balanceada por pessoa** (Leave-One-Person-Out):
treina com todas as pessoas menos uma e testa nela. É a avaliação usada no TCC,
porque o split aleatório superestima o desempenho (Seção 6.3).

### Convenção de nome dos vídeos (passo 1)

```
{pessoa}_{sessao}_{exercicio}_{execucao}_{seq}.mp4
ex.: bruno_S01_agachamento_correta_01.mp4
```

## Testar um vídeo

```bash
python inferencia/extrair_keypoints_video.py --video meu_video.mp4
python inferencia/prever_execucao.py          # identifica o exercício e avalia a execução
```

## Publicar os modelos na AWS

Os arquivos de `modelos/` (exceto `arquivo/`) são enviados ao bucket de modelos
pelo `iac-infra`:

```bash
cd ../iac-infra && ./scripts/update.sh models
```

> **Atenção:** a produção (`iac-infra/lambda/user_dataset_generator.py`) usa
> **só a pessoa principal** de cada frame e descarta keypoints com confiança
> < 0,5, enquanto o treino usa todas as pessoas detectadas. Os atributos são
> calculados da mesma forma (`alignfit/atributos.py` ↔ `lambda/inference.py`).

---

## Resultados atuais (base v2)

| Modelo | Avaliação | Resultado |
|---|---|---|
| Exercício | split aleatório 80/20 (48 vídeos) | 79,2% |
| Exercício | por pessoa da equipe | 93,7% |
| Execução — agachamento | por pessoa | 75,0% |
| Execução — bíceps | por pessoa | 70,8% |
| Execução — desenvolvimento | por pessoa | 62,5% |
| Execução — elevação lateral | por pessoa | 65,3% |

Acurácia balanceada; 50% equivale a um palpite aleatório.

## Histórico

| Tag / versão | O que é |
|---|---|
| `v1-antes-da-reorganizacao` | estado do repositório antes desta reorganização (nomes antigos: `adm/`, `user/`, `models/`…) |
