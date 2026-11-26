# =========================================================
# IMPORTS
# =========================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

# =========================================================
# CONFIGURAÇÕES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DATASET_PATH = PROJECT_ROOT / "datasets" / "raw-dataset-v2_20261108_163655.csv"

TARGET_EXERCISE = "elevacao_lateral"

MODEL_OUTPUT = PROJECT_ROOT / "models" / "modelo-de-classificacao-de-execucao-elevacao-lateral.pkl"

# =========================================================
# CARREGAR DATASET
# =========================================================

print("========================================")
print("Carregando dataset...")
print("========================================")

df = pd.read_csv(
    DATASET_PATH,
    sep=","
)

# =========================================================
# FILTRAR EXERCÍCIO
# =========================================================

df = df[
    df["tipo_exercicio"] == TARGET_EXERCISE
]

# Vídeos externos (pessoa == "externo") só têm execução "correta":
# usá-los aqui criaria um atalho (origem do vídeo -> rótulo).
# Eles entram apenas no classificador de exercício.
df = df[df["pessoa"] != "externo"]

print(f"\nExercício filtrado: {TARGET_EXERCISE}")

# =========================================================
# LABEL
# =========================================================

df["label"] = df["tipo_execucao"].map({
    "correta": 0,
    "errada": 1
})

# =========================================================
# FEATURES
# =========================================================

feature_columns = [

    col for col in df.columns

    if (
        col.endswith("_x")
        or col.endswith("_y")
    )
]

print(f"\nQuantidade de features: {len(feature_columns)}")

# =========================================================
# AGRUPAR POR VÍDEO
# =========================================================

X = []
y = []

grouped = df.groupby("video")

print("\n========================================")
print("Extraindo features por vídeo...")
print("========================================")

for video_name, group in grouped:

    features = []

    # MÉDIA
    features.extend(
        group[feature_columns]
        .mean()
        .tolist()
    )

    # DESVIO PADRÃO
    features.extend(
        group[feature_columns]
        .std()
        .fillna(0)
        .tolist()
    )

    # MÍNIMO
    features.extend(
        group[feature_columns]
        .min()
        .tolist()
    )

    # MÁXIMO
    features.extend(
        group[feature_columns]
        .max()
        .tolist()
    )

    X.append(features)

    y.append(
        group["label"].iloc[0]
    )

# =========================================================
# NUMPY
# =========================================================

X = np.array(X)
y = np.array(y)

print("\n========================================")
print("Shapes")
print("========================================")

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================================================
# SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# MODELO
# =========================================================

# Hiperparâmetros escolhidos por busca em grade com validação
# Leave-One-Person-Out (grupos = pessoa), métrica balanced_accuracy.
# LOPO: 0.653 ± 0.071 (igual ao anterior, mas modelo bem mais simples)
model = XGBClassifier(
    n_estimators=100,        # nº de árvores do boosting
    max_depth=2,             # árvores rasas: poucas amostras -> menor variância
    learning_rate=0.03,       # taxa de aprendizado (shrinkage)
    min_child_weight=1,      # peso mínimo por folha (evita folhas com 1-2 vídeos)
    subsample=0.8,          # 80% dos vídeos por árvore (boosting estocástico)
    colsample_bytree=0.8,   # 80% das features por árvore (reduz correlação)
    random_state=42
)

# =========================================================
# TREINAMENTO
# =========================================================

print("\n========================================")
print("Treinando modelo...")
print("========================================")

model.fit(X_train, y_train)

# =========================================================
# PREDIÇÃO
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# RESULTADOS
# =========================================================

print("\n========================================")
print("RESULTADOS")
print("========================================\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["correta", "errada"]
    )
)

# =========================================================
# SALVAR MODELO
# =========================================================

joblib.dump(
    model,
    MODEL_OUTPUT
)

# =========================================================
# FINALIZAÇÃO
# =========================================================

print("\n========================================")
print("MODELO SALVO!")
print("========================================")

print(f"Modelo salvo: {MODEL_OUTPUT}")

print("========================================")