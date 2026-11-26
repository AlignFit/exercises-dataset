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
from alignfit.atributos import atributos_por_video


def prever_exercicio(df: pd.DataFrame) -> pd.Series:
    """Retorna o exercício previsto para cada vídeo do DataFrame."""
    modelo = joblib.load(caminhos.MODELO_EXERCICIO)
    rotulos = joblib.load(caminhos.ROTULOS_EXERCICIO)
    X = atributos_por_video(df)
    return pd.Series(rotulos.inverse_transform(modelo.predict(X.values)), index=X.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=caminhos.DADOS / "video-usuario.csv")
    args = parser.parse_args()

    for video, exercicio in prever_exercicio(pd.read_csv(args.csv)).items():
        print(f"{video}: {exercicio}")


if __name__ == "__main__":
    main()
