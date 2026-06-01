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

TARGET_FOLDER = r"folder_path"

# Modelo YOLO Pose
MODEL_PATH = "yolov8n-pose.pt"

# CSV final
OUTPUT_CSV = "dataset_exercicios.csv"

# Processar 1 frame a cada N frames
FRAME_SKIP = 5

# =========================================================
# INFORMAR ERRO MANUALMENTE
# =========================================================

# Se for correto:
# ERROR_TYPE = "nenhum"
#
# Se for errado:
# ERROR_TYPE = "joelho_para_dentro"

ERROR_TYPE = "nunhum"

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
# EXTRAIR METADADOS DA PASTA
# =========================================================

folder_parts = TARGET_FOLDER.replace("\\", "/").split("/")

EXERCISE_TYPE = folder_parts[-3]

execution_folder = folder_parts[-2]

# =========================================================
# DEFINIR EXECUÇÃO
# =========================================================

if execution_folder.lower() == "correto":

    EXECUTION_TYPE = "correta"

    ERROR_TYPE = "nenhum"

else:

    EXECUTION_TYPE = "errada"

# =========================================================
# CARREGAR MODELO
# =========================================================

print("========================================")
print("Carregando YOLO Pose...")
print("========================================")

model = YOLO(MODEL_PATH)

print("Modelo carregado com sucesso!")

# =========================================================
# CARREGAR CSV ANTIGO
# =========================================================

processed_videos = set()

if os.path.exists(OUTPUT_CSV):

    print("\nCSV existente encontrado!")

    old_df = pd.read_csv(OUTPUT_CSV)

    if "video_id" in old_df.columns:

        processed_videos = set(old_df["video_id"].unique())

        print(f"Vídeos já processados: {len(processed_videos)}")

else:

    old_df = pd.DataFrame()

# =========================================================
# LISTAR VÍDEOS
# =========================================================

video_files = [
    file for file in os.listdir(TARGET_FOLDER)
    if file.endswith((".mp4", ".avi", ".mov", ".mkv"))
]

print(f"\nVídeos encontrados: {len(video_files)}")

# =========================================================
# DATASET NOVO
# =========================================================

all_data = []

# =========================================================
# PROCESSAR VÍDEOS
# =========================================================

for video_name in video_files:

    video_path = os.path.join(TARGET_FOLDER, video_name)

    # =====================================================
    # PULAR VÍDEOS JÁ PROCESSADOS
    # =====================================================

    if video_path in processed_videos:

        print(f"\nPulando vídeo já processado:")
        print(video_name)

        continue

    print("\n========================================")
    print(f"Processando vídeo:")
    print(video_name)
    print("========================================")

    cap = cv2.VideoCapture(video_path)

    frame_id = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        # =================================================
        # PULAR FRAMES
        # =================================================

        if frame_id % FRAME_SKIP != 0:

            frame_id += 1
            continue

        height, width = frame.shape[:2]

        # =================================================
        # INFERÊNCIA YOLO
        # =================================================

        results = model(frame, verbose=False)

        result = results[0]

        # =================================================
        # VERIFICAR KEYPOINTS
        # =================================================

        if result.keypoints is not None and len(result.keypoints.xy) > 0:

            keypoints = result.keypoints.xy.cpu().numpy()

            # =============================================
            # PERCORRER PESSOAS
            # =============================================

            for person_id, person_keypoints in enumerate(keypoints):

                row = {

                    # METADADOS
                    "video": video_name,
                    "video_id": video_path,
                    "frame": frame_id,
                    "person_id": person_id,

                    # LABELS
                    "tipo_exercicio": EXERCISE_TYPE,
                    "tipo_execucao": EXECUTION_TYPE,
                    "tipo_erro": ERROR_TYPE
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

    cap.release()

    print("Vídeo finalizado!")

# =========================================================
# CRIAR NOVO DATAFRAME
# =========================================================

new_df = pd.DataFrame(all_data)

# =========================================================
# CONCATENAR CSV
# =========================================================

if not old_df.empty:

    final_df = pd.concat(
        [old_df, new_df],
        ignore_index=True
    ).fillna(0)

else:

    final_df = new_df

# =========================================================
# SALVAR CSV
# =========================================================

final_df.to_csv(OUTPUT_CSV, index=False)

# =========================================================
# FINALIZAÇÃO
# =========================================================

print("\n========================================")
print("DATASET GERADO COM SUCESSO!")
print("========================================")

print(f"Tipo exercício: {EXERCISE_TYPE}")
print(f"Tipo execução: {EXECUTION_TYPE}")
print(f"Tipo erro: {ERROR_TYPE}")

print(f"\nCSV salvo em: {OUTPUT_CSV}")

print(f"Novos registros: {len(new_df)}")
print(f"Total registros: {len(final_df)}")

print("========================================")