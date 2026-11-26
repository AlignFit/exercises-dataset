# =========================================================
# prever_exercicio.py
#
# Inferência local, passo 2: identifica o exercício a partir do
# CSV gerado por extrair_keypoints_video.py.
#
# Uso:
#   python inferencia/prever_exercicio.py
#   python inferencia/prever_exercicio.py --csv dados/exemplo-video-usuario.csv
# =========================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from alignfit import caminhos
from alignfit.atributos import atributos_completos


def prever_exercicio(X: pd.DataFrame) -> pd.DataFrame:
    """Recebe a tabela de atributos_completos e devolve, por vídeo, o
    exercício previsto e a confiança do modelo."""
    modelo = joblib.load(caminhos.MODELO_EXERCICIO)
    proba = modelo.predict_proba(X)
    return pd.DataFrame(
        {"exercicio": [modelo.classes[i] for i in proba.argmax(1)], "confianca": proba.max(1)},
        index=X.index,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=caminhos.DADOS / "video-usuario.csv")
    args = parser.parse_args()

    X = atributos_completos(pd.read_csv(args.csv))
    for video, linha in prever_exercicio(X).iterrows():
        print(f"{video}: {linha.exercicio} (confiança {linha.confianca:.0%})")


if __name__ == "__main__":
    main()
