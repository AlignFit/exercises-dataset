# =========================================================
# IMPORTS
# =========================================================

import pandas as pd
import numpy as np
import joblib

# =========================================================
# CONFIGURAÇÕES
# =========================================================

INPUT_CSV = r"..\\datasets\user-raw-dataset.csv"

# Exercício detectado anteriormente
DETECTED_EXERCISE = "desenvolvimento"

# =========================================================
# DEFINIR MODELO
# =========================================================

MODEL_PATHS = {

    "agachamento":
        r"..\models\modelo-de-classificacao-de-execucao-agachamento.pkl",

    "biceps":
        r"..\models\modelo-de-classificacao-de-execucao-biceps.pkl",

    "elevacaoLateral":
        r"..\models\modelo-de-classificacao-de-execucao-elevacao-lateral.pkl",

    "desenvolvimento":
        r"..\models\modelo-de-classificacao-de-execucao-desenvolvimento.pkl"
}

# =========================================================
# VALIDAR EXERCÍCIO
# =========================================================

if DETECTED_EXERCISE not in MODEL_PATHS:

    print("Exercício inválido!")
    exit()

# =========================================================
# CARREGAR MODELO
# =========================================================

MODEL_PATH = MODEL_PATHS[DETECTED_EXERCISE]

print("========================================")
print("Carregando modelo...")
print("========================================")

model = joblib.load(MODEL_PATH)

print(f"Modelo carregado: {MODEL_PATH}")

# =========================================================
# CARREGAR CSV
# =========================================================

print("\n========================================")
print("Carregando CSV...")
print("========================================")

df = pd.read_csv(INPUT_CSV)

print(f"Frames encontrados: {len(df)}")

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

# =========================================================
# AGRUPAR POR VÍDEO
# =========================================================

X = []

grouped = df.groupby("video")

# =========================================================
# EXTRAIR FEATURES
# =========================================================

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

# =========================================================
# NUMPY
# =========================================================

X = np.array(X)

print("\nX shape:", X.shape)

# =========================================================
# PREDIÇÃO
# =========================================================

print("\n========================================")
print("Analisando execução...")
print("========================================")

prediction = model.predict(X)[0]

# =========================================================
# RESULTADO
# =========================================================

if prediction == 0:

    execution_result = "CORRETA"

else:

    execution_result = "ERRADA"

# =========================================================
# FINALIZAÇÃO
# =========================================================

print("\n========================================")
print("RESULTADO FINAL")
print("========================================")

print(f"Exercício detectado: {DETECTED_EXERCISE}")

print(f"Execução: {execution_result}")

print("========================================")