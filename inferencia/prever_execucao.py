# =========================================================
# prever_execucao.py
#
# Inferência local, passo 3: diz se a execução está correta ou
# errada. Se o exercício não for informado, ele é identificado
# antes pelo classificador de exercício (fluxo completo).
#
# Uso:
#   python inferencia/prever_execucao.py
#   python inferencia/prever_execucao.py --csv dados/exemplo-video-usuario.csv
#   python inferencia/prever_execucao.py --exercicio desenvolvimento
# =========================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from alignfit import caminhos
from alignfit.atributos import atributos_completos
from alignfit.keypoints import EXERCICIOS
from prever_exercicio import prever_exercicio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=caminhos.DADOS / "video-usuario.csv")
    parser.add_argument("--exercicio", choices=EXERCICIOS, help="pula a identificação do exercício")
    args = parser.parse_args()

    X = atributos_completos(pd.read_csv(args.csv))
    if args.exercicio:
        exercicios = pd.DataFrame({"exercicio": args.exercicio, "confianca": float("nan")}, index=X.index)
    else:
        exercicios = prever_exercicio(X)

    for video, linha in exercicios.iterrows():
        modelo = joblib.load(caminhos.modelo_execucao(linha.exercicio))
        x = X.loc[[video]]
        errada = modelo.predict(x)[0] == 1
        conf = "" if pd.isna(linha.confianca) else f" ({linha.confianca:.0%})"
        if hasattr(modelo, "probabilidade_errada"):      # XGBoost
            detalhe = f"chance de erro {modelo.probabilidade_errada(x)[0]:.0%}"
        else:                                            # conformidade
            detalhe = f"afastamento {modelo.afastamento(x)[0]:.2f} / limiar {modelo.limiar:.2f}"
        print(f"{video}: exercício={linha.exercicio}{conf}  execução={'ERRADA' if errada else 'CORRETA'}"
              f"  [{detalhe}]")


if __name__ == "__main__":
    main()
