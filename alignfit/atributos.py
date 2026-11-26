"""Transforma os keypoints de cada frame em um vetor de atributos por vídeo.

Este módulo é usado no treino (pipeline/) e deve ser usado também em
produção (iac-infra/lambda), para que o modelo receba exatamente os mesmos
atributos que viu no treino.

Dois conjuntos de atributos:

* **Coordenadas** (`atributos_coordenadas`): média, desvio, mínimo e máximo
  de cada uma das 34 coordenadas brutas -> 136 atributos. Era o único
  conjunto usado até a versão `modelo-coordenadas` (Seção 4.4 do TCC).

* **Biomecânicos** (`atributos_biomecanicos`): ângulos articulares
  (joelho, quadril, cotovelo, ombro), inclinação do tronco, alturas e
  larguras relativas, todos normalizados pelo comprimento do tronco. Por
  serem medidas do corpo, e não posições na imagem, dependem menos da
  câmera, da distância e da pessoa. Cada medida é resumida por média,
  desvio, mínimo, máximo, percentis 10/90 e amplitude (p90 - p10).

Antes de calcular, `selecionar_pessoa_principal` mantém só a pessoa
principal do vídeo, descartando quem aparece ao fundo.
"""

import numpy as np
import pandas as pd

from alignfit.keypoints import COLUNAS_COORDENADAS, NOMES_KEYPOINTS

# ---------------------------------------------------------------------------
# Pessoa principal
# ---------------------------------------------------------------------------


def _ponto(frames, nome):
    return frames[[f"{nome}_x", f"{nome}_y"]].to_numpy(dtype=float)


def selecionar_pessoa_principal(frames: pd.DataFrame, chave: str = "video") -> pd.DataFrame:
    """Mantém, em cada vídeo, só a pessoa detectada em mais frames e com
    maior tamanho de tronco (frames x tamanho mediano do tronco).

    Em produção o Lambda já grava só uma pessoa por frame (person_id = 0),
    então esta função não altera nada lá."""
    if "person_id" not in frames.columns or frames["person_id"].nunique() <= 1:
        return frames
    ombros = (_ponto(frames, "ombro_esquerdo") + _ponto(frames, "ombro_direito")) / 2
    quadris = (_ponto(frames, "quadril_esquerdo") + _ponto(frames, "quadril_direito")) / 2
    tronco = pd.Series(np.linalg.norm(ombros - quadris, axis=1), index=frames.index)
    resumo = (
        frames.assign(_tronco=tronco)
        .groupby([chave, "person_id"])
        .agg(n=("_tronco", "size"), t=("_tronco", "median"))
        .reset_index()
    )
    resumo["score"] = resumo["n"] * resumo["t"]
    principal = resumo.loc[resumo.groupby(chave)["score"].idxmax(), [chave, "person_id"]]
    return frames.merge(principal, on=[chave, "person_id"])


# ---------------------------------------------------------------------------
# Atributos de coordenadas (versão anterior)
# ---------------------------------------------------------------------------


def atributos_coordenadas(frames: pd.DataFrame, chave: str = "video") -> pd.DataFrame:
    """136 atributos: média, desvio, mínimo e máximo das 34 coordenadas."""
    grupos = frames.groupby(chave)[COLUNAS_COORDENADAS]
    return pd.concat(
        [grupos.mean(), grupos.std().fillna(0), grupos.min(), grupos.max()],
        axis=1,
    )


# nome antigo, mantido para os scripts em experimentos/legado/
atributos_por_video = atributos_coordenadas

# ---------------------------------------------------------------------------
# Atributos biomecânicos
# ---------------------------------------------------------------------------


def _angulo(a, b, c):
    """Ângulo ABC em graus (vértice em b)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def medidas_por_frame(frames: pd.DataFrame) -> pd.DataFrame:
    """Medidas biomecânicas de cada frame (uma linha por frame)."""
    P = {k: _ponto(frames, k) for k in NOMES_KEYPOINTS}
    for k in NOMES_KEYPOINTS:  # (0,0) = ponto não detectado
        P[k][(P[k] == 0).all(1)] = np.nan

    ombros = (P["ombro_esquerdo"] + P["ombro_direito"]) / 2
    quadris = (P["quadril_esquerdo"] + P["quadril_direito"]) / 2
    tronco = np.linalg.norm(ombros - quadris, axis=1)
    tronco = np.where(tronco > 1e-3, tronco, np.nan)  # escala do corpo

    M = {}
    for lado, nome in (("esq", "esquerdo"), ("dir", "direito")):
        M[f"joelho_{lado}"] = _angulo(P[f"quadril_{nome}"], P[f"joelho_{nome}"], P[f"tornozelo_{nome}"])
        M[f"quadril_{lado}"] = _angulo(P[f"ombro_{nome}"], P[f"quadril_{nome}"], P[f"joelho_{nome}"])
        M[f"cotovelo_{lado}"] = _angulo(P[f"ombro_{nome}"], P[f"cotovelo_{nome}"], P[f"pulso_{nome}"])
        M[f"ombro_{lado}"] = _angulo(P[f"quadril_{nome}"], P[f"ombro_{nome}"], P[f"cotovelo_{nome}"])
        M[f"pulso_alt_{lado}"] = (P[f"ombro_{nome}"][:, 1] - P[f"pulso_{nome}"][:, 1]) / tronco
        M[f"cotov_alt_{lado}"] = (P[f"ombro_{nome}"][:, 1] - P[f"cotovelo_{nome}"][:, 1]) / tronco
        M[f"cotov_lat_{lado}"] = np.abs(P[f"cotovelo_{nome}"][:, 0] - ombros[:, 0]) / tronco
        M[f"pulso_lat_{lado}"] = np.abs(P[f"pulso_{nome}"][:, 0] - ombros[:, 0]) / tronco

    vetor_tronco = ombros - quadris
    M["tronco_incl"] = np.degrees(np.arctan2(np.abs(vetor_tronco[:, 0]), np.abs(vetor_tronco[:, 1])))
    M["quadril_alt"] = (quadris[:, 1] - np.nanmedian(quadris[:, 1])) / tronco
    larg_quadril = np.abs(P["quadril_esquerdo"][:, 0] - P["quadril_direito"][:, 0])
    M["joelhos_larg"] = np.abs(P["joelho_esquerdo"][:, 0] - P["joelho_direito"][:, 0]) / (larg_quadril + 1e-6)
    M["tornoz_larg"] = np.abs(P["tornozelo_esquerdo"][:, 0] - P["tornozelo_direito"][:, 0]) / (larg_quadril + 1e-6)
    M["joel_torn"] = M["joelhos_larg"] / (M["tornoz_larg"] + 1e-6)
    larg_ombro = np.abs(P["ombro_esquerdo"][:, 0] - P["ombro_direito"][:, 0])
    M["cotov_larg"] = np.abs(P["cotovelo_esquerdo"][:, 0] - P["cotovelo_direito"][:, 0]) / (larg_ombro + 1e-6)
    M["ombro_desloc"] = (ombros[:, 0] - np.nanmedian(ombros[:, 0])) / tronco

    M = pd.DataFrame(M, index=frames.index)
    # combina os lados (média, diferença, menor, maior): independe de o
    # usuário estar de frente, de costas ou espelhado
    for base in ["joelho", "quadril", "cotovelo", "ombro", "pulso_alt", "cotov_alt", "cotov_lat", "pulso_lat"]:
        e, d = M.pop(f"{base}_esq"), M.pop(f"{base}_dir")
        M[f"{base}_med"] = pd.concat([e, d], axis=1).mean(axis=1)
        M[f"{base}_dif"] = (e - d).abs()
        M[f"{base}_min"] = np.fmin(e, d)
        M[f"{base}_max"] = np.fmax(e, d)
    return M


def _p10(x):
    return x.quantile(0.1)


def _p90(x):
    return x.quantile(0.9)


def atributos_biomecanicos(frames: pd.DataFrame, chave: str = "video") -> pd.DataFrame:
    """Uma linha por vídeo com as medidas biomecânicas resumidas."""
    frames = frames.sort_values([chave, "frame"])
    M = medidas_por_frame(frames)
    grupos = M.groupby(frames[chave].values)
    partes = []
    for nome, funcao in [("mean", "mean"), ("std", "std"), ("min", "min"), ("max", "max"), ("p10", _p10), ("p90", _p90)]:
        parte = grupos.agg(funcao)
        parte.columns = [f"{c}__{nome}" for c in parte.columns]
        partes.append(parte)
    X = pd.concat(partes, axis=1)
    amplitude = pd.DataFrame({f"{c}__amp": X[f"{c}__p90"] - X[f"{c}__p10"] for c in M.columns})
    X = pd.concat([X, amplitude], axis=1)
    X.index.name = chave
    return X.replace([np.inf, -np.inf], np.nan)


def atributos_completos(frames: pd.DataFrame, chave: str = "video") -> pd.DataFrame:
    """Pessoa principal -> atributos biomecânicos + coordenadas.

    É a entrada dos modelos atuais. Use `X[modelo.colunas]` para garantir a
    mesma ordem de colunas do treino."""
    frames = selecionar_pessoa_principal(frames, chave)
    bio = atributos_biomecanicos(frames, chave)
    coord = atributos_coordenadas(frames, chave).loc[bio.index]
    coord.columns = [f"{c}__{estat}" for estat in ("mean", "std", "min", "max") for c in COLUNAS_COORDENADAS]
    return pd.concat([bio, coord], axis=1)
