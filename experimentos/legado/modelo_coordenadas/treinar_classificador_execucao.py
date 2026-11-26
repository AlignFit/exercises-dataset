# LEGADO: modelo baseado só em coordenadas (versão em produção até 26/11/2026).
# Salva em experimentos/legado/modelo_coordenadas/saida/, nunca em modelos/.
# =========================================================
# 3_treinar_classificador_execucao.py
#
# Estágio 3 do pipeline: treina, para cada exercício, o modelo
# que diz se a execução está CORRETA ou ERRADA.
#
# Substitui os 4 scripts antigos (BuildSquat/Biceps/ShoulderPress/
# LateralRaiseExecutionClassifierModel.py), que eram idênticos a
# não ser pelo exercício e pelos hiperparâmetros abaixo.
#
# Entrada : dados/base-v2.csv
# Saída   : modelos/execucao-<exercicio>.pkl
#
# Uso (de qualquer pasta):
#   python pipeline/3_treinar_classificador_execucao.py               # os 4 exercícios
#   python pipeline/3_treinar_classificador_execucao.py --exercicio biceps
# =========================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
SAIDA = Path(__file__).resolve().parent / "saida"
SAIDA.mkdir(exist_ok=True)

import joblib
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

from alignfit import caminhos
from alignfit.atributos import atributos_por_video
from alignfit.keypoints import EXERCICIOS

# =========================================================
# HIPERPARÂMETROS POR EXERCÍCIO
# =========================================================
# Escolhidos por busca em grade com validação Leave-One-Person-Out
# (grupos = pessoa), métrica balanced_accuracy. Valores LOPO obtidos:
#   agachamento       0.750 ± 0.000  (antes 0.722 com max_depth=8, 300 árvores)
#   biceps            0.708 ± 0.156  (antes 0.625)
#   desenvolvimento   0.625 ± 0.177  (antes 0.625)
#   elevacao_lateral  0.653 ± 0.071  (igual, modelo mais simples)

TAXA_APRENDIZADO = {
    "agachamento": 0.1,
    "biceps": 0.03,
    "desenvolvimento": 0.1,
    "elevacao_lateral": 0.03,
}


def criar_modelo(exercicio):
    return XGBClassifier(
        n_estimators=100,        # nº de árvores do boosting
        max_depth=2,             # árvores rasas: poucas amostras -> menor variância
        learning_rate=TAXA_APRENDIZADO[exercicio],
        min_child_weight=1,
        subsample=0.8,           # 80% dos vídeos por árvore
        colsample_bytree=0.8,    # 80% dos atributos por árvore
        random_state=42,
    )


def avaliar_por_pessoa(X, y, pessoas, exercicio):
    """Leave-One-Person-Out: treina com todas as pessoas menos uma e testa
    nela. É a avaliação honesta usada no TCC (sem vazamento de sessão)."""
    resultados = {}
    for pessoa in sorted(set(pessoas)):
        teste = pessoas == pessoa
        modelo = criar_modelo(exercicio).fit(X[~teste], y[~teste])
        resultados[pessoa] = balanced_accuracy_score(y[teste], modelo.predict(X[teste]))
    return resultados


def treinar(df, exercicio):
    print("\n========================================")
    print(f"Exercício: {exercicio}")
    print("========================================")

    dados = df[df["tipo_exercicio"] == exercicio]

    # Vídeos externos só têm execução "correta": usá-los aqui criaria um
    # atalho (origem do vídeo -> rótulo). Eles entram apenas no
    # classificador de exercício.
    dados = dados[dados["pessoa"] != "externo"]

    X = atributos_por_video(dados, chave="video")
    meta = dados.groupby("video")[["tipo_execucao", "pessoa"]].first().loc[X.index]
    y = meta["tipo_execucao"].map({"correta": 0, "errada": 1}).values
    X = X.values

    print(f"Vídeos: {len(y)}  (corretas={int((y == 0).sum())}, erradas={int((y == 1).sum())})")

    # Avaliação 1 — por pessoa (número reportado no TCC)
    lopo = avaliar_por_pessoa(X, y, meta["pessoa"].values, exercicio)
    media = sum(lopo.values()) / len(lopo)
    print("\nAcurácia balanceada por pessoa (Leave-One-Person-Out):")
    for pessoa, acc in lopo.items():
        print(f"  {pessoa:8s} {acc:.3f}")
    print(f"  média    {media:.3f}")

    # Avaliação 2 — split aleatório 80/20 (referência)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    modelo = criar_modelo(exercicio).fit(X_treino, y_treino)
    print("\nSplit aleatório 80/20:")
    print(classification_report(y_teste, modelo.predict(X_teste),
                                target_names=["correta", "errada"], zero_division=0))

    # Modelo salvo = o treinado no split 80/20 (mesmo comportamento da versão anterior)
    saida = SAIDA / caminhos.modelo_execucao(exercicio).name
    joblib.dump(modelo, saida)
    print(f"Modelo salvo: {saida}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercicio", choices=EXERCICIOS,
                        help="treina só este exercício (padrão: todos)")
    args = parser.parse_args()

    df = pd.read_csv(caminhos.BASE_TREINO)
    for exercicio in [args.exercicio] if args.exercicio else EXERCICIOS:
        treinar(df, exercicio)


if __name__ == "__main__":
    main()
