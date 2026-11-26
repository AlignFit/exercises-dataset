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
from alignfit.atributos import atributos_por_video
from alignfit.keypoints import EXERCICIOS
from prever_exercicio import prever_exercicio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=caminhos.DADOS / "video-usuario.csv")
    parser.add_argument("--exercicio", choices=EXERCICIOS,
                        help="pula a identificação do exercício")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    X = atributos_por_video(df)
    exercicios = (pd.Series(args.exercicio, index=X.index) if args.exercicio
                  else prever_exercicio(df))

    for video, exercicio in exercicios.items():
        modelo = joblib.load(caminhos.modelo_execucao(exercicio))
        errada = modelo.predict(X.loc[[video]].values)[0] == 1
        print(f"{video}: exercício={exercicio}  execução={'ERRADA' if errada else 'CORRETA'}")


if __name__ == "__main__":
    main()
