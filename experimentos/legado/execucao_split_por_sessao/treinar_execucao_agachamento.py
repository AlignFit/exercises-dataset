# LEGADO (Seções 6.3–6.5 do TCC): avaliação agrupada por sessão na base v1.
# Não gera modelos de produção.
# =========================================================
# BuildSquatExecutionClassifierModel.py
#
# Treina o classificador de qualidade de execução do
# AGACHAMENTO (correta vs errada).
#
# CORREÇÃO (ver Seção 9.2 do TCC): o split treino/teste agora
# é feito por SESSÃO DE GRAVAÇÃO, não por vídeo isolado.
# Nenhuma sessão do dataset mistura execução correta e errada,
# então um split aleatório por vídeo deixa vídeos da mesma
# sessão (mesma câmera/luz/pessoa/dia) em treino E teste,
# inflando artificialmente a acurácia. Além do split único,
# também reportamos uma validação cruzada agrupada por sessão
# (5 folds), que é a métrica mais confiável para o TCC.
# =========================================================

import re
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score

from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from alignfit.sessoes import derive_session_id

# =========================================================
# CONFIGURAÇÕES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

DATASET_PATH = PROJECT_ROOT / "dados" / "base-v1.csv"

TARGET_EXERCISE = "agachamento"

MODEL_OUTPUT = BASE_DIR / "saida" / "modelo-de-classificacao-de-execucao-agachamento.pkl"  # nunca sobrescreve modelos/
MODEL_OUTPUT.parent.mkdir(exist_ok=True)

RANDOM_STATE = 42

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
].copy()

print(f"\nExercício filtrado: {TARGET_EXERCISE}")

# =========================================================
# LABEL
# =========================================================

df["label"] = df["tipo_execucao"].map({
    "correta": 0,
    "errada": 1
})

# =========================================================
# SESSÃO DE GRAVAÇÃO (para o split anti-vazamento)
# =========================================================

video_to_path = (
    df.drop_duplicates("video")
    .set_index("video")["video_id"]
    .to_dict()
)

session_map = {
    v: derive_session_id(v, video_to_path.get(v, ""))
    for v in df["video"].unique()
}

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

print(f"\nQuantidade de features (colunas de keypoints): {len(feature_columns)}")

# =========================================================
# AGRUPAR POR VÍDEO
# =========================================================

X = []
y = []
groups = []

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

    groups.append(
        session_map[video_name]
    )

# =========================================================
# NUMPY
# =========================================================

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

n_sessions = len(set(groups))

print("\n========================================")
print("Shapes")
print("========================================")

print("X shape:", X.shape)
print("y shape:", y.shape)
print(f"Sessões de gravação distintas: {n_sessions}")

# =========================================================
# SPLIT POR SESSÃO (não por vídeo isolado)
# =========================================================

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train = groups[train_idx]

overlap = set(groups[train_idx]) & set(groups[test_idx])
print(f"Sessões em treino: {len(set(groups[train_idx]))} | "
      f"em teste: {len(set(groups[test_idx]))} | "
      f"sobreposição: {len(overlap)} (deve ser 0)")

# =========================================================
# MODELO
# =========================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)

# =========================================================
# TREINAMENTO
# =========================================================

print("\n========================================")
print("Treinando modelo...")
print("========================================")

model.fit(X_train, y_train)

# =========================================================
# PREDIÇÃO E RESULTADOS (split único, sessões nunca vistas)
# =========================================================

y_pred = model.predict(X_test)

print("\n========================================")
print("RESULTADOS (split único, por sessão)")
print("========================================\n")

if len(set(y_test)) < 2:
    print("[AVISO] O conjunto de teste caiu 100% em uma única classe "
          "(isso é esperado, já que nenhuma sessão mistura execução "
          "correta e errada — ver Seção 9.2). O classification_report "
          "abaixo não é totalmente interpretável nesse caso; use a CV "
          "agrupada por sessão logo mais abaixo como métrica principal.")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["correta", "errada"],
        zero_division=0,
    )
)

# =========================================================
# VALIDAÇÃO CRUZADA AGRUPADA POR SESSÃO (5 folds)
# métrica principal reportada no TCC (Seção 9.2)
# =========================================================

print("========================================")
print("Validação cruzada agrupada por sessão (5 folds)")
print("========================================\n")

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_accuracies = []
cv_f1_scores = []

for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
    fold_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )
    fold_model.fit(X[tr_idx], y[tr_idx])
    fold_pred = fold_model.predict(X[te_idx])

    fold_acc = accuracy_score(y[te_idx], fold_pred)
    fold_f1 = f1_score(y[te_idx], fold_pred, zero_division=0)

    cv_accuracies.append(fold_acc)
    cv_f1_scores.append(fold_f1)

    print(f"  Fold {fold_i}: accuracy={fold_acc:.2%}  f1(errada)={fold_f1:.2%}  "
          f"(sessões de teste: {len(set(groups[te_idx]))}, vídeos de teste: {len(te_idx)})")

cv_accuracies = np.array(cv_accuracies)
cv_f1_scores = np.array(cv_f1_scores)

print(f"\nCV por sessão -> Accuracy: {cv_accuracies.mean():.2%} +/- {cv_accuracies.std():.2%}")
print(f"CV por sessão -> F1 (errada): {cv_f1_scores.mean():.2%} +/- {cv_f1_scores.std():.2%}")

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
