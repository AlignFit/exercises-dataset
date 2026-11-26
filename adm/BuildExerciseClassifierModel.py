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

DATASET_PATH = r"..\\datasets\raw-dataset-v2_20261108_163655.csv"

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
# AUMENTO DE DADOS (só no treino)
# =========================================================
#
# Cada vídeo de TREINO ganha cópias sintéticas:
#   - espelho:     x -> 1 - x e troca lados esquerdo/direito
#   - subamostra:  frames pares e frames ímpares da sequência
#   - ruído:       2 cópias com escala/deslocamento/ruído leves
# O teste usa sempre o vídeo original. Keypoints ausentes (0,0)
# continuam (0,0).
#
# LOPO por pessoa (5 seeds, v2 + externos):
#   sem aumento 88,4% ± 1,5  ->  com aumento 92,6% ± 0,0 (bal. acc.)

USAR_AUMENTO = True

rng = np.random.default_rng(42)

df = df.sort_values(["video_id", "person_id", "frame"])


def trocar_lado(col):
    return (col.replace("esquerd", "TMP")
               .replace("direit", "esquerd")
               .replace("TMP", "direit"))


def espelhar(frames):
    m = frames.copy()
    for c in feature_columns:
        if c.endswith("_x"):
            m[c] = np.where(frames[c] == 0, 0, 1 - frames[c])
    m = m.rename(columns={c: trocar_lado(c) for c in feature_columns})
    return m[frames.columns]


def com_ruido(frames):
    n = frames.copy()
    escala = rng.uniform(0.95, 1.05)
    desloc = rng.normal(0, 0.01, 2)
    for c in feature_columns:
        v = n[c].values
        d = desloc[0] if c.endswith("_x") else desloc[1]
        n[c] = np.where(
            v != 0,
            (v - 0.5) * escala + 0.5 + d + rng.normal(0, 0.005, len(v)),
            0,
        )
    return n


def features_por_video(frames):
    # média, desvio, mínimo e máximo de cada coordenada (136 features)
    g = frames.groupby("video_id")
    return pd.concat([
        g[feature_columns].mean(),
        g[feature_columns].std().fillna(0),
        g[feature_columns].min(),
        g[feature_columns].max(),
    ], axis=1)


def variantes(frames):
    out = [features_por_video(frames)]
    if USAR_AUMENTO:
        out.append(features_por_video(espelhar(frames)))
        ordem = frames.groupby(["video_id", "person_id"]).cumcount()
        out.append(features_por_video(frames[ordem % 2 == 0]))
        out.append(features_por_video(frames[ordem % 2 == 1]))
        out.append(features_por_video(com_ruido(frames)))
        out.append(features_por_video(com_ruido(frames)))
    return out


print("\n========================================")
print("Extraindo features por vídeo...")
print("========================================")

label_por_video = df.groupby("video_id")["label"].first()

# =========================================================
# SPLIT (por vídeo, ANTES do aumento -> sem vazamento)
# =========================================================

train_ids, test_ids = train_test_split(
    label_por_video.index,
    test_size=0.2,
    random_state=42,
    stratify=label_por_video.values
)

X_test = features_por_video(df[df["video_id"].isin(test_ids)])
y_test = label_por_video.loc[X_test.index].values
X_test = X_test.values

treino = pd.concat(variantes(df[df["video_id"].isin(train_ids)]))
y_train = label_por_video.loc[treino.index].values
X_train = treino.values

print(f"Aumento de dados: {'ligado' if USAR_AUMENTO else 'desligado'}")
print(f"Vídeos de treino: {len(train_ids)} -> {len(X_train)} amostras")

print("\n========================================")
print("Train/Test")
print("========================================")

print("Treino:", X_train.shape)
print("Teste:", X_test.shape)

# =========================================================
# MODELO
# =========================================================

# Hiperparâmetros escolhidos por busca em grade com validação
# Leave-One-Person-Out (grupos = pessoa), métrica balanced_accuracy.
# LOPO: 0.833 ± 0.059 (antes: 0.809 ± 0.066 com max_depth=8, lr=0.03)
model = XGBClassifier(
    n_estimators=300,        # nº de árvores do boosting
    max_depth=2,             # árvores rasas: poucas amostras -> menor variância
    learning_rate=0.1,       # taxa de aprendizado (shrinkage)
    min_child_weight=3,      # peso mínimo por folha (evita folhas com 1-2 vídeos)
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