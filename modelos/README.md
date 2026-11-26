# modelos/

Modelos treinados pelo `pipeline/` e usados na inferência (`inferencia/` e,
em produção, `iac-infra/lambda/inference.py`).

| Arquivo | Gerado por | Classe | O que faz |
|---|---|---|---|
| `classificador-exercicio.pkl` | `pipeline/2_treinar_classificador_exercicio.py` | `alignfit.modelos.ClassificadorExercicio` | identifica o exercício (`.predict` devolve o nome) |
| `execucao-agachamento.pkl` | `pipeline/3_treinar_classificador_execucao.py` | `alignfit.modelos.ClassificadorExecucaoXGB` (ou `ModeloConformidade` com `--modelo conformidade`) | 0 = correta, 1 = errada |
| `execucao-biceps.pkl` | idem | idem | idem |
| `execucao-desenvolvimento.pkl` | idem | idem | idem |
| `execucao-elevacao-lateral.pkl` | idem | idem | idem |

## Como usar

Os modelos são objetos das classes de `alignfit/modelos.py`, então o pacote
`alignfit` precisa estar importável ao carregá-los. Todos recebem a tabela de
`alignfit.atributos.atributos_completos` (uma linha por vídeo) e escolhem
sozinhos as colunas com que foram treinados:

```python
import joblib, pandas as pd
from alignfit.atributos import atributos_completos

X = atributos_completos(pd.read_csv("keypoints.csv"))
exercicio = joblib.load("modelos/classificador-exercicio.pkl").predict(X)[0]
execucao  = joblib.load(f"modelos/execucao-{exercicio.replace('_', '-')}.pkl").predict(X)[0]
```

O XGBoost de execução expõe `probabilidade_errada(X)`; a conformidade expõe `afastamento(X)` (quanto a execução se
afasta do padrão correto) e `limiar` (acima dele, a execução é considerada
errada).

## arquivo/

| Pasta | Conteúdo |
|---|---|
| `antes-dos-externos-20261108/` | modelos de coordenadas treinados antes de incluir os vídeos externos (nomes antigos) |

A versão anterior a esta (XGBoost só com coordenadas, em produção até
26/11/2026) pode ser recuperada pelo git
(`git show 1a8be2c:modelos/execucao-biceps.pkl > ...`) ou regenerada por
`experimentos/legado/modelo_coordenadas/`.
