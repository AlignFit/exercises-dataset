# =========================================================
# 2_treinar_classificador_exercicio.py
#
# Estágio 2 do pipeline: treina o modelo que identifica QUAL
# exercício aparece no vídeo (agachamento, bíceps,
# desenvolvimento ou elevação lateral).
#
# Modelo   : Random Forest (alignfit.modelos.ClassificadorExercicio)
# Atributos: biomecânicos + coordenadas da pessoa principal
#            (alignfit.atributos.atributos_completos)
#
# Entrada : dados/base-v2.csv
# Saída   : modelos/classificador-exercicio.pkl
#
# Uso (de qualquer pasta):
#   python pipeline/2_treinar_classificador_exercicio.py
# =========================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from alignfit import caminhos
from alignfit.atributos import atributos_completos
from alignfit.modelos import ClassificadorExercicio


def main():
    print("Carregando base e calculando atributos...")
    df = pd.read_csv(caminhos.BASE_TREINO)
    X = atributos_completos(df)
    meta = df.groupby("video")[["tipo_exercicio", "pessoa"]].first().loc[X.index]
    y, pessoas = meta["tipo_exercicio"].values, meta["pessoa"].values
    print(f"Vídeos: {len(X)}  |  atributos: {X.shape[1]}")

    # Avaliação 1 — por pessoa: treina sem a pessoa e testa nela
    print("\nAcurácia balanceada deixando cada grupo de fora:")
    equipe = []
    for grupo in ["bruno", "caio", "paulo", "externo"]:
        teste = pessoas == grupo
        modelo = ClassificadorExercicio().fit(X[~teste], y[~teste])
        acc = balanced_accuracy_score(y[teste], modelo.predict(X[teste]))
        if grupo != "externo":
            equipe.append(acc)
        print(f"  {grupo:8s} {acc:.3f}")
    print(f"  média da equipe {np.mean(equipe):.3f}")

    # Avaliação 2 — split aleatório 80/20 (referência)
    treino, teste = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    modelo = ClassificadorExercicio().fit(X.iloc[treino], y[treino])
    previsto = modelo.predict(X.iloc[teste])
    print(f"\nSplit aleatório 80/20: acurácia {accuracy_score(y[teste], previsto):.3f}")
    print(classification_report(y[teste], previsto, zero_division=0))

    # Modelo final: treinado com TODOS os vídeos
    final = ClassificadorExercicio().fit(X, y)
    joblib.dump(final, caminhos.MODELO_EXERCICIO)
    print(f"Modelo salvo: {caminhos.MODELO_EXERCICIO}")


if __name__ == "__main__":
    main()
