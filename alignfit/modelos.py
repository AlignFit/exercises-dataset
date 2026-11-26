"""Modelos do AlignFit.

Cada modelo guarda a lista de colunas com que foi treinado e recebe
diretamente a tabela de `alignfit.atributos.atributos_completos`, de modo
que treino, inferência local e produção usam a mesma interface:

    X = atributos_completos(frames)
    modelo.predict(X)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

VERSAO_ATRIBUTOS = "biomecanico-v1"


class ClassificadorExercicio:
    """Estágio 2: identifica o exercício.

    Random Forest sobre atributos biomecânicos + coordenadas da pessoa
    principal. Treinado com todos os vídeos (equipe + externos)."""

    versao_atributos = VERSAO_ATRIBUTOS

    def __init__(self, n_arvores=500, semente=0):
        self.n_arvores = n_arvores
        self.semente = semente

    def fit(self, X: pd.DataFrame, exercicios):
        self.colunas = list(X.columns)
        self.rotulos = LabelEncoder().fit(exercicios)
        self.modelo = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(self.n_arvores, class_weight="balanced", random_state=self.semente),
        ).fit(X[self.colunas].values, self.rotulos.transform(exercicios))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict_proba(X[self.colunas].values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Nome do exercício de cada vídeo."""
        return self.rotulos.inverse_transform(self.predict_proba(X).argmax(1))

    @property
    def classes(self):
        return list(self.rotulos.classes_)


# Medidas biomecânicas relevantes para cada exercício (definidas a partir dos
# critérios de execução da Seção 4.2 do TCC, não ajustadas pelos dados).
MEDIDAS_RELEVANTES = {
    "agachamento": ["joelho", "quadril", "tronco", "joelhos_larg", "tornoz", "joel_torn", "quadril_alt"],
    "biceps": ["cotovelo", "tronco", "ombro", "cotov_alt", "cotov_lat", "pulso_alt", "ombro_desloc", "quadril"],
    "desenvolvimento": ["cotovelo", "ombro", "cotov_lat", "cotov_larg", "pulso_alt", "cotov_alt", "pulso_lat", "tronco"],
    "elevacao_lateral": ["ombro", "cotovelo", "pulso_alt", "cotov_alt", "pulso_lat", "cotov_lat", "tronco"],
}


class ModeloConformidade:
    """Estágio 3: diz se a execução está correta (Seção 8 do TCC).

    Em vez de aprender "correta x errada", aprende só o padrão da execução
    CORRETA: cada medida é padronizada pela média e desvio das execuções
    corretas, e a execução vira um "afastamento" (raiz da média dos
    quadrados das medidas padronizadas, limitadas a 6 desvios). Se o
    afastamento passa do limiar, a execução é considerada ERRADA.

    O limiar é o percentil `percentil_limiar` do afastamento das execuções
    corretas da equipe: com 75, cerca de 1 em cada 4 execuções corretas da
    própria base já fica acima dele (o modelo prefere alertar a mais).
    """

    versao_atributos = VERSAO_ATRIBUTOS

    def __init__(self, exercicio: str, percentil_limiar: float = 75, limite_desvios: float = 6):
        self.exercicio = exercicio
        self.percentil_limiar = percentil_limiar
        self.limite_desvios = limite_desvios

    def _colunas_relevantes(self, X):
        medidas = MEDIDAS_RELEVANTES[self.exercicio]
        return [c for c in X.columns if "__" in c and any(c.startswith(m) for m in medidas)
                and not c.endswith(("_x__mean", "_y__mean", "_x__std", "_y__std",
                                    "_x__min", "_y__min", "_x__max", "_y__max"))]

    def fit(self, X_corretas: pd.DataFrame, X_corretas_referencia: pd.DataFrame = None):
        """X_corretas: execuções corretas usadas para aprender o padrão
        (equipe + externos). X_corretas_referencia: corretas usadas para o
        limiar (só da equipe); se omitido, usa X_corretas."""
        self.colunas = self._colunas_relevantes(X_corretas)
        self.padrao = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.padrao.fit(X_corretas[self.colunas].values)
        ref = X_corretas if X_corretas_referencia is None else X_corretas_referencia
        self.limiar = float(np.percentile(self.afastamento(ref), self.percentil_limiar))
        return self

    def afastamento(self, X: pd.DataFrame) -> np.ndarray:
        """Quanto cada execução se afasta do padrão correto (0 = idêntica)."""
        Z = np.clip(np.abs(self.padrao.transform(X[self.colunas].values)), 0, self.limite_desvios)
        return np.sqrt((Z ** 2).mean(1))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """1 = ERRADA, 0 = CORRETA (mesma convenção dos modelos anteriores)."""
        return (self.afastamento(X) > self.limiar).astype(int)
