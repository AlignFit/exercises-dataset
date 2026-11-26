# experimentos/

Código que gerou os resultados do TCC, mas que **não faz parte do pipeline de
produção**.

## otimizacao/ — escolha do modelo atual (Seção 9)

Scripts exploratórios que compararam atributos, algoritmos e abordagens até
chegar aos modelos atuais. Todos avaliam com Leave-One-Person-Out nas 3
pessoas da equipe. Rodar de dentro da pasta, nesta ordem (o 1 gera
`feats.pkl`, usado pelo 2 e pelo 4):

| Script | O que compara | Principal conclusão |
|---|---|---|
| `avalia_modelo_coordenadas.py <base.csv> <saida.json>` | modelo antigo (coordenadas) por split, por pessoa e por sessão | execução 62,5–75,0% por pessoa |
| `externos_no_binario.py <base.csv>` | modelo binário antigo com e sem vídeos externos no treino | externos **pioram** o binário |
| `1_atributos_e_modelos.py <base.csv>` | coordenadas × pessoa principal × biomecânicos; XGB, LogReg, SVM, RF, ExtraTrees | atributos biomecânicos: +7 a 8 pontos |
| `2_combinacoes_e_conformidade.py` | combinações de modelos, atributos por exercício e **conformidade** | conformidade + externos detecta 75–100% dos erros |
| `3_frame_a_frame.py <base.csv>` | classificar cada frame e votar por vídeo | não supera os atributos por vídeo |
| `4_classificador_exercicio.py` | classificador de exercício com os novos atributos | biomecânico + coordenadas: RF 95,8%, XGBoost 94,0% (mantido o XGBoost) |
| `teste_externo.py <keypoints.csv>` | modelos treinados na base inteira aplicados a vídeos novos | conformidade aceitou 8/11 corretas externas |

`feats_exploracao.py` é a primeira versão dos atributos biomecânicos; a versão
de produção é `alignfit/atributos.py` (mesmo cálculo, reproduz os mesmos
números).

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
| `modelo_coordenadas/` | treino do modelo anterior (XGBoost só com coordenadas, em produção até 26/11/2026) | comparação na Seção 9 |
| `extracao/extrair_keypoints_v1.py` | extrator usado para gerar a base v1 (rótulo vinha da pasta) | Seção 4 |
| `execucao_split_por_sessao/treinar_execucao_<exercicio>.py` | classificadores de execução avaliados com validação agrupada por sessão na base v1 | Seções 6.3–6.5, Tabelas 8–10 |

Os scripts legados gravam saídas em `legado/**/saida/` e **nunca** sobrescrevem
`modelos/`. A versão com split aleatório por vídeo (Tabela 7) está no
histórico do git (tag `v1-antes-da-reorganizacao`, pasta
`buildClassifierExerciseModel/XGB/`).
