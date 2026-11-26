# =========================================================
# 3_treinar_classificador_execucao.py
#
# Estágio 3 do pipeline: treina, para cada exercício, o modelo
# que diz se a execução está CORRETA ou ERRADA.
#
# Dois modelos disponíveis (--modelo):
#
#   xgboost (padrão)  XGBoost binário correta x errada, treinado
#                     só com os vídeos da equipe
#                     (alignfit.modelos.ClassificadorExecucaoXGB)
#
#   conformidade      aprende só o padrão da execução correta
#                     (equipe + externos) e marca como errada o
#                     que se afasta dele (Seção 8 do TCC;
#                     alignfit.modelos.ModeloConformidade)
#
# Atributos: biomecânicos + coordenadas da pessoa principal
#            (alignfit.atributos.atributos_completos)
#
# Entrada : dados/base-v2.csv
# Saída   : modelos/execucao-<exercicio>.pkl
#
# Uso (de qualquer pasta):
#   python pipeline/3_treinar_classificador_execucao.py
#   python pipeline/3_treinar_classificador_execucao.py --exercicio biceps
#   python pipeline/3_treinar_classificador_execucao.py --modelo conformidade
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
from alignfit.modelos import ClassificadorExecucaoXGB, ModeloConformidade

PESSOAS_EQUIPE = ["bruno", "caio", "paulo"]


def treinar_modelo(tipo, exercicio, X, y, externo):
    if tipo == "xgboost":
        # só a equipe: externos (todos corretos) criariam um atalho
        return ClassificadorExecucaoXGB(exercicio).fit(X[~externo], y[~externo])
    # conformidade: padrão = corretas (equipe + externos); limiar = corretas da equipe
    corretas = y == 0
    return ModeloConformidade(exercicio).fit(X[corretas], X[corretas & ~externo])


def avaliar_por_pessoa(tipo, exercicio, X, y, pessoas, externo):
    """Leave-One-Person-Out: o modelo nunca viu a pessoa testada."""
    previsto = np.zeros(len(y), dtype=int)
    por_pessoa = {}
    for pessoa in PESSOAS_EQUIPE:
        teste = pessoas == pessoa
        modelo = treinar_modelo(tipo, exercicio, X[~teste], y[~teste], externo[~teste])
        previsto[teste] = modelo.predict(X[teste])
        por_pessoa[pessoa] = balanced_accuracy_score(y[teste], previsto[teste])
    return por_pessoa, previsto


def treinar(tipo, exercicio, X_todos, meta_todos):
    print("\n========================================")
    print(f"Exercício: {exercicio}  |  modelo: {tipo}")
    print("========================================")

    sel = (meta_todos["tipo_exercicio"] == exercicio).values
    X, meta = X_todos[sel], meta_todos[sel]
    y = (meta["tipo_execucao"] == "errada").astype(int).values
    pessoas = meta["pessoa"].values
    externo = pessoas == "externo"
    print(f"Equipe: {int((~externo & (y == 0)).sum())} corretas, {int((y == 1).sum())} erradas"
          f"  |  externos: {int(externo.sum())} corretas"
          f"{' (não usados no XGBoost)' if tipo == 'xgboost' else ''}")

    por_pessoa, previsto = avaliar_por_pessoa(tipo, exercicio, X, y, pessoas, externo)
    print("\nAcurácia balanceada por pessoa (Leave-One-Person-Out):")
    for pessoa, acc in por_pessoa.items():
        print(f"  {pessoa:8s} {acc:.3f}")
    print(f"  média    {np.mean(list(por_pessoa.values())):.3f}")

    eq = ~externo
    (vn, fp), (fn, vp) = confusion_matrix(y[eq], previsto[eq], labels=[0, 1])
    print(f"\nErradas detectadas : {vp}/{vp + fn} ({vp / (vp + fn):.0%})")
    print(f"Corretas aceitas   : {vn}/{vn + fp} ({vn / (vn + fp):.0%})")

    final = treinar_modelo(tipo, exercicio, X, y, externo)
    saida = caminhos.modelo_execucao(exercicio)
    joblib.dump(final, saida)
    print(f"\nModelo salvo: {saida}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercicio", choices=EXERCICIOS, help="treina só este exercício (padrão: todos)")
    parser.add_argument("--modelo", choices=["xgboost", "conformidade"], default="xgboost")
    args = parser.parse_args()

    print("Carregando base e calculando atributos...")
    df = pd.read_csv(caminhos.BASE_TREINO)
    X = atributos_completos(df)
    meta = df.groupby("video")[["tipo_exercicio", "tipo_execucao", "pessoa"]].first().loc[X.index]

    for exercicio in [args.exercicio] if args.exercicio else EXERCICIOS:
        treinar(args.modelo, exercicio, X, meta)


if __name__ == "__main__":
    main()
