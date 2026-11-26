# =========================================================
# 3_treinar_classificador_execucao.py
#
# Estágio 3 do pipeline: treina, para cada exercício, o modelo
# que diz se a execução está CORRETA ou ERRADA.
#
# Modelo   : conformidade (alignfit.modelos.ModeloConformidade,
#            Seção 8 do TCC). Aprende só o padrão da execução
#            correta e marca como errada o que se afasta dele.
# Dados    : execuções corretas da equipe + vídeos externos
#            (todos corretos). As execuções erradas da equipe
#            servem só para AVALIAR o modelo.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from alignfit import caminhos
from alignfit.atributos import atributos_completos
from alignfit.keypoints import EXERCICIOS
from alignfit.modelos import ModeloConformidade

PESSOAS_EQUIPE = ["bruno", "caio", "paulo"]


def treinar_modelo(exercicio, X, y, externo):
    """Padrão = corretas (equipe + externos); limiar = corretas da equipe."""
    corretas = y == 0
    return ModeloConformidade(exercicio).fit(X[corretas], X[corretas & ~externo])


def avaliar_por_pessoa(exercicio, X, y, pessoas, externo):
    """Leave-One-Person-Out: o modelo nunca viu a pessoa testada."""
    previsto = np.zeros(len(y), dtype=int)
    por_pessoa = {}
    for pessoa in PESSOAS_EQUIPE:
        teste = pessoas == pessoa
        modelo = treinar_modelo(exercicio, X[~teste], y[~teste], externo[~teste])
        previsto[teste] = modelo.predict(X[teste])
        por_pessoa[pessoa] = balanced_accuracy_score(y[teste], previsto[teste])
    return por_pessoa, previsto


def treinar(exercicio, X_todos, meta_todos):
    print("\n========================================")
    print(f"Exercício: {exercicio}")
    print("========================================")

    sel = (meta_todos["tipo_exercicio"] == exercicio).values
    X, meta = X_todos[sel], meta_todos[sel]
    y = (meta["tipo_execucao"] == "errada").astype(int).values
    pessoas = meta["pessoa"].values
    externo = pessoas == "externo"
    print(f"Equipe: {int((~externo & (y == 0)).sum())} corretas, {int((y == 1).sum())} erradas"
          f"  |  externos: {int(externo.sum())} corretas")

    por_pessoa, previsto = avaliar_por_pessoa(exercicio, X, y, pessoas, externo)
    print("\nAcurácia balanceada por pessoa (Leave-One-Person-Out):")
    for pessoa, acc in por_pessoa.items():
        print(f"  {pessoa:8s} {acc:.3f}")
    print(f"  média    {np.mean(list(por_pessoa.values())):.3f}")

    eq = ~externo
    (vn, fp), (fn, vp) = confusion_matrix(y[eq], previsto[eq], labels=[0, 1])
    print(f"\nErradas detectadas : {vp}/{vp + fn} ({vp / (vp + fn):.0%})")
    print(f"Corretas aceitas   : {vn}/{vn + fp} ({vn / (vn + fp):.0%})")

    final = treinar_modelo(exercicio, X, y, externo)
    saida = caminhos.modelo_execucao(exercicio)
    joblib.dump(final, saida)
    print(f"\nModelo salvo: {saida}  (limiar = {final.limiar:.3f}, {len(final.colunas)} atributos)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercicio", choices=EXERCICIOS, help="treina só este exercício (padrão: todos)")
    args = parser.parse_args()

    print("Carregando base e calculando atributos...")
    df = pd.read_csv(caminhos.BASE_TREINO)
    X = atributos_completos(df)
    meta = df.groupby("video")[["tipo_exercicio", "tipo_execucao", "pessoa"]].first().loc[X.index]

    for exercicio in [args.exercicio] if args.exercicio else EXERCICIOS:
        treinar(exercicio, X, meta)


if __name__ == "__main__":
    main()
