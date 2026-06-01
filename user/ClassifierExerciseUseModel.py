# =========================================================
# IMPORTS
# =========================================================

import pandas as pd
import numpy as np
import joblib

# =========================================================
# CONFIGURAÇÕES
# =========================================================

# CSV gerado do vídeo do usuário
INPUT_CSV = "video_user_keypoints.csv"

# Modelo treinado
MODEL_PATH = "modelo02_exercicio.pkl"

# Encoder treinado
ENCODER_PATH = "label_encoder_exercicio.pkl"

# =========================================================
# CARREGAR MODELO
# =========================================================

print("========================================")
print("Carregando modelo...")
print("========================================")

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(ENCODER_PATH)

print("Modelo carregado com sucesso!")

# =========================================================
# CARREGAR CSV
# =========================================================

print("\n========================================")
print("Carregando CSV do usuário...")
print("========================================")

df = pd.read_csv(INPUT_CSV)

print(f"Registros encontrados: {len(df)}")

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

print(f"Quantidade de features: {len(feature_columns)}")

# =========================================================
# AGRUPAR POR VÍDEO
# =========================================================

X = []

grouped = df.groupby("video")

video_names = []

# =========================================================
# EXTRAIR FEATURES
# =========================================================

print("\n========================================")
print("Extraindo features...")
print("========================================")

for video_name, group in grouped:

    features = []

    # =====================================================
    # MÉDIA
    # =====================================================

    features.extend(
        group[feature_columns]
        .mean()
        .tolist()
    )

    # =====================================================
    # DESVIO PADRÃO
    # =====================================================

    features.extend(
        group[feature_columns]
        .std()
        .fillna(0)
        .tolist()
    )

    # =====================================================
    # MÍNIMO
    # =====================================================

    features.extend(
        group[feature_columns]
        .min()
        .tolist()
    )

    # =====================================================
    # MÁXIMO
    # =====================================================

    features.extend(
        group[feature_columns]
        .max()
        .tolist()
    )

    X.append(features)

    video_names.append(video_name)

# =========================================================
# NUMPY
# =========================================================

X = np.array(X)

print("\nX shape:", X.shape)

# =========================================================
# PREDIÇÃO
# =========================================================

print("\n========================================")
print("Classificando exercício...")
print("========================================")

predictions = model.predict(X)

# =========================================================
# CONVERTER LABELS
# =========================================================

exercise_predictions = label_encoder.inverse_transform(
    predictions
)

# =========================================================
# RESULTADOS
# =========================================================

print("\n========================================")
print("RESULTADOS")
print("========================================")

for idx, video_name in enumerate(video_names):

    predicted_exercise = exercise_predictions[idx]

    print(f"\nVídeo: {video_name}")

    print(f"Exercício detectado: {predicted_exercise}")

print("\n========================================")