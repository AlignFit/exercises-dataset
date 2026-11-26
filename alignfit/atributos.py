"""Transforma os keypoints de cada frame em um vetor de atributos por vídeo.

Cada coordenada (34 colunas) é resumida por média, desvio padrão, mínimo
e máximo ao longo dos frames, gerando 136 atributos por vídeo
(Seção 4.4 do TCC). É o mesmo cálculo feito em produção
(iac-infra/lambda/inference.py), por isso a ordem das colunas importa.
"""

import pandas as pd

from alignfit.keypoints import COLUNAS_COORDENADAS


def atributos_por_video(frames: pd.DataFrame, chave: str = "video") -> pd.DataFrame:
    """Retorna um DataFrame com uma linha por vídeo (índice = `chave`)."""
    grupos = frames.groupby(chave)[COLUNAS_COORDENADAS]
    return pd.concat(
        [
            grupos.mean(),
            grupos.std().fillna(0),
            grupos.min(),
            grupos.max(),
        ],
        axis=1,
    )
