# =========================================================
# IMPORTS
# =========================================================

from ultralytics import YOLO

import os
import cv2
import pandas as pd

# =========================================================
# CONFIGURAÇÕES
# =========================================================

# Vídeo enviado pelo usuário
VIDEO_PATH = r"video_path"

# Modelo YOLO Pose
MODEL_PATH = "yolov8n-pose.pt"

# CSV temporário que será usado pelo modelo
OUTPUT_CSV = "video_user_keypoints.csv"

# Processar 1 frame a cada N frames
FRAME_SKIP = 5

# =========================================================
# KEYPOINTS
# =========================================================

KEYPOINT_NAMES = {
    0: "nariz",
    1: "olho_esquerdo",
    2: "olho_direito",
    3: "orelha_esquerda",
    4: "orelha_direita",
    5: "ombro_esquerdo",
    6: "ombro_direito",
    7: "cotovelo_esquerdo",
    8: "cotovelo_direito",
    9: "pulso_esquerdo",
    10: "pulso_direito",
    11: "quadril_esquerdo",
    12: "quadril_direito",
    13: "joelho_esquerdo",
    14: "joelho_direito",
    15: "tornozelo_esquerdo",
    16: "tornozelo_direito"
}

# =========================================================
# VALIDAR VÍDEO
# =========================================================

if not os.path.exists(VIDEO_PATH):

    print("Vídeo não encontrado!")
    exit()

# =========================================================
# CARREGAR MODELO
# =========================================================

print("========================================")
print("Carregando YOLO Pose...")
print("========================================")

model = YOLO(MODEL_PATH)

print("Modelo carregado com sucesso!")

# =========================================================
# CAPTURA DE VÍDEO
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

video_name = os.path.basename(VIDEO_PATH)

# =========================================================
# DATASET TEMPORÁRIO
# =========================================================

all_data = []

frame_id = 0

print("\n========================================")
print("Processando vídeo do usuário...")
print("========================================")

# =========================================================
# PROCESSAR FRAMES
# =========================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # =====================================================
    # PULAR FRAMES
    # =====================================================

    if frame_id % FRAME_SKIP != 0:

        frame_id += 1
        continue

    height, width = frame.shape[:2]

    # =====================================================
    # INFERÊNCIA YOLO
    # =====================================================

    results = model(frame, verbose=False)

    result = results[0]

    # =====================================================
    # VERIFICAR KEYPOINTS
    # =====================================================

    if result.keypoints is not None and len(result.keypoints.xy) > 0:

        keypoints = result.keypoints.xy.cpu().numpy()

        # =================================================
        # PERCORRER PESSOAS
        # =================================================

        for person_id, person_keypoints in enumerate(keypoints):

            row = {

                "video": video_name,
                "frame": frame_id,
                "person_id": person_id
            }

            # =============================================
            # KEYPOINTS NORMALIZADOS
            # =============================================

            for kp_id, (x, y) in enumerate(person_keypoints):

                body_part = KEYPOINT_NAMES[kp_id]

                x_norm = float(x / width)
                y_norm = float(y / height)

                row[f"{body_part}_x"] = x_norm
                row[f"{body_part}_y"] = y_norm

            # =============================================
            # ADICIONAR AO DATASET
            # =============================================

            all_data.append(row)

    frame_id += 1

# =========================================================
# FINALIZAR VÍDEO
# =========================================================

cap.release()

# =========================================================
# CRIAR DATAFRAME
# =========================================================

df = pd.DataFrame(all_data)

# =========================================================
# SALVAR CSV
# =========================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)

# =========================================================
# FINALIZAÇÃO
# =========================================================

print("\n========================================")
print("CSV GERADO COM SUCESSO!")
print("========================================")

print(f"Vídeo processado: {video_name}")

print(f"Frames processados: {len(df)}")

print(f"\nCSV salvo em:")
print(OUTPUT_CSV)

print("========================================")