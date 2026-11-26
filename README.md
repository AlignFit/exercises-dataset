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
2. **Classificação do exercício** — só a pessoa principal do vídeo é
   considerada. Dos seus pontos saem medidas biomecânicas (ângulos de joelho,
   quadril, cotovelo e ombro, inclinação do tronco, alturas e larguras
   normalizadas pelo tamanho do tronco), resumidas por vídeo junto com as
   coordenadas (409 atributos). Um XGBoost identifica o exercício.
3. **Classificação da execução** — um XGBoost por exercício diz se a execução
   está correta ou errada, usando os mesmos atributos. Alternativa: o modelo de
   **conformidade** (Seção 8 do TCC), que aprende só o padrão correto
   (`pipeline/3_treinar_classificador_execucao.py --modelo conformidade`).

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
│   ├── atributos.py       pessoa principal + keypoints por frame → atributos por vídeo
│   ├── modelos.py         ClassificadorExercicio e ModeloConformidade
│   └── sessoes.py         identifica a sessão de gravação (usado nos experimentos da base v1)
│
├── dados/                 bases de keypoints (CSV)        → ver dados/README.md
├── modelos/               modelos treinados (.pkl)        → ver modelos/README.md
├── experimentos/          otimização, notebooks e versões antigas → ver experimentos/README.md
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

> ⚠ **Os modelos atuais ainda não são compatíveis com a produção.** O Lambda
> `iac-infra/lambda/inference.py` calcula só os 136 atributos de coordenadas
> e espera um XGBoost. Antes de publicar estes modelos, o Lambda precisa usar
> `alignfit.atributos.atributos_completos` e `alignfit.modelos` (ver
> `modelos/README.md`). Até lá, a produção segue com o modelo de coordenadas
> (`experimentos/legado/modelo_coordenadas/`).

---

## Resultados atuais (base v2)

Acurácia balanceada **por pessoa** (o modelo nunca viu a pessoa testada);
50% equivale a um palpite aleatório.

**Exercício (XGBoost):** 94,0% (modelo anterior, só coordenadas: 93,7%).

**Execução** — modelo em uso: XGBoost biomecânico. Para comparação, a
conformidade e o modelo anterior (XGBoost só com coordenadas):

| Exercício | XGBoost biomecânico (atual) | Erradas detectadas | Corretas aceitas | Conformidade | Anterior |
|---|---|---|---|---|---|
| Agachamento | 58,3% | 58% | 67% | 68,1% | 75,0% |
| Bíceps | 70,8% | 91% | 42% | 79,2% | 70,8% |
| Desenvolvimento | 79,2% | 92% | 83% | 77,8% | 62,5% |
| Elevação lateral | 59,7% | 67% | 50% | 72,2% | 65,3% |
| **Média** | **67,0%** | | | **74,3%** | **68,4%** |

**Teste externo** (11 vídeos de elevação lateral correta, gravados fora da
base): exercício identificado em 11/11 (anterior: 10/11); execução aceita
como correta em 8/11 pelo XGBoost biomecânico e pela conformidade (anterior:
3/11).

> O XGBoost de execução usa a mesma taxa de aprendizado (0,03) nos 4
> exercícios; ela não foi ajustada por exercício com os atributos novos. Com
> a base atual, a conformidade tem média maior; reavaliar quando houver mais
> vídeos (principalmente execuções erradas).

O retorno ao usuário deve ser tratado como alerta ("possível erro, revise a
técnica"), não como veredito.

## Histórico

| Versão | O que é |
|---|---|
| tag `v1-antes-da-reorganizacao` | estado antes da reorganização (nomes antigos: `adm/`, `user/`, `models/`…) |
| commit `1a8be2c` | última versão com o modelo de coordenadas (XGBoost) |
| commit `3cb0705` | atributos biomecânicos + XGBoost (exercício) + conformidade (execução) |
| atual | atributos biomecânicos + XGBoost (exercício e execução) |
