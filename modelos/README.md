# modelos/

Modelos treinados pelo `pipeline/` e usados em produção
(`iac-infra/lambda/inference.py`).

| Arquivo | Gerado por | O que faz |
|---|---|---|
| `classificador-exercicio.pkl` | `pipeline/2_treinar_classificador_exercicio.py` | XGBoost que identifica o exercício |
| `rotulos-exercicio.pkl` | idem | `LabelEncoder`: número da classe → nome do exercício |
| `execucao-agachamento.pkl` | `pipeline/3_treinar_classificador_execucao.py` | 0 = correta, 1 = errada |
| `execucao-biceps.pkl` | idem | idem |
| `execucao-desenvolvimento.pkl` | idem | idem |
| `execucao-elevacao-lateral.pkl` | idem | idem |

Todos recebem os 136 atributos de `alignfit/atributos.py` (um vetor por vídeo).

## arquivo/

| Pasta | Conteúdo |
|---|---|
| `antes-dos-externos-20261108/` | modelos treinados antes de incluir os vídeos externos (nomes antigos mantidos) |

Versões anteriores também podem ser recuperadas pelo git
(`git checkout v1-antes-da-reorganizacao -- models/`).
