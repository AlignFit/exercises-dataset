# experimentos/

Código que gerou os resultados do TCC, mas que **não faz parte do pipeline de
produção**.

## notebooks/ — comparação de modelos (Seção 5, base v1)

| Notebook | O que faz | Dados |
|---|---|---|
| `Comparativo_Modelos_AlignFit_V2.ipynb` | compara XGBoost, MLP, SVM, Árvore de Decisão e KNN (Tabela 3, Figuras 7–9) | `dados/base-v1.csv` |
| `Comparativo_Modelos_AlignFit.ipynb` | primeira versão do comparativo | `dados/base-v1.csv` |
| `MLP_Softmax_AlignFit_Paulo.ipynb` | MLP com Softmax (Tabela 5) | `dados/base-v1.csv` |
| `MLP_Softmax_AlignFit_Bruno.ipynb` | MLP com Softmax, versão para Colab | `dataset_exercicios.csv` (upload no Colab) |
| `SVM_AlignFit.ipynb` | SVM com kernel RBF | `dataset_exercicios.csv` (upload no Colab) |
| `KNN_AlignFit.ipynb` | KNN (k = 5) | `dataset_exercicios.csv` (upload no Colab) |

Os notebooks de Colab esperam a base v1 enviada com o nome
`dataset_exercicios.csv`.

> O notebook `Comparativo_Modelos_AlignFit_V2.ipynb` reproduz MLP = 94,44%,
> enquanto a Tabela 3 do TCC publica 95,37%. A diferença precisa ser
> conciliada antes da entrega.

## legado/ — versões antigas dos scripts

| Pasta | O que é | TCC |
|---|---|---|
| `extracao/extrair_keypoints_v1.py` | extrator usado para gerar a base v1 (rótulo vinha da pasta) | Seção 4 |
| `execucao_split_por_sessao/treinar_execucao_<exercicio>.py` | classificadores de execução avaliados com validação cruzada agrupada por sessão na base v1 | Seções 6.3–6.5, Tabelas 8–10 |

Os scripts legados gravam saídas em `legado/**/saida/` e **nunca** sobrescrevem
`modelos/`. A versão original dos scripts com split aleatório por vídeo
(Tabela 7) está no histórico do git (tag `v1-antes-da-reorganizacao`, pasta
`buildClassifierExerciseModel/XGB/`).
