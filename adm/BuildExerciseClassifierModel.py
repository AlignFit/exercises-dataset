# =========================================================
# IMPORTS
# =========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

# =========================================================
# CONFIGURAÇÕES
# =========================================================

DATASET_PATH = r"..\\datasets\raw-dataset.csv"

MODEL_OUTPUT = r"..\\models\modelo-de-classificacao"

LABEL_ENCODER_OUTPUT = r"..\\models\modelo-de-classificacao.pkl"

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
# DEBUG COLUNAS
# =========================================================

print("\nColunas encontradas:\n")
print(df.columns.tolist())

# =========================================================
# LABEL
# =========================================================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(
    df["tipo_exercicio"]
)

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

    # =====================================================
    # ADICIONAR FEATURES
    # =====================================================

    X.append(features)

    # =====================================================
    # LABEL DO VÍDEO
    # =====================================================

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

print("\n========================================")
print("Train/Test")
print("========================================")

print("Treino:", X_train.shape)
print("Teste:", X_test.shape)

# =========================================================
# MODELO
# =========================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
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
        target_names=label_encoder.classes_
    )
)

# =========================================================
# SALVAR MODELO
# =========================================================

joblib.dump(
    model,
    MODEL_OUTPUT
)

joblib.dump(
    label_encoder,
    LABEL_ENCODER_OUTPUT
)

# =========================================================
# FINALIZAÇÃO
# =========================================================

print("\n========================================")
print("MODELO SALVO!")
print("========================================")

print(f"Modelo salvo: {MODEL_OUTPUT}")
print(f"Encoder salvo: {LABEL_ENCODER_OUTPUT}")

print("\nClasses aprendidas:")

for idx, class_name in enumerate(label_encoder.classes_):

    print(f"{idx} -> {class_name}")

print("========================================")