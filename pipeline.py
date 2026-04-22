import os
import argparse
import uuid
import pandas as pd
import boto3
from botocore.exceptions import ClientError

from src.extract_frames import extract_frames
from src.extract_keypoints import extract_keypoints
from src.set_labels import set_labels
from src.build_dataset import build_dataset


S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_KEY = "dataset/dataset.csv"


# =========================
# S3
# =========================
def baixar_dataset_s3(local_path):
    s3 = boto3.client("s3")

    try:
        s3.download_file(S3_BUCKET, S3_KEY, local_path)
        return True
    except ClientError:
        print("Nenhum dataset anterior encontrado no S3")
        return False


def subir_dataset_s3(local_path):
    s3 = boto3.client("s3")

    try:
        s3.upload_file(local_path, S3_BUCKET, S3_KEY)
        print("✅ Dataset enviado para o S3")
    except Exception as e:
        print(f"Erro ao subir dataset: {e}")


def subir_pasta_s3(local_folder, s3_prefix):
    s3 = boto3.client("s3")

    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_path = f"{s3_prefix}/{relative_path}"

            try:
                s3.upload_file(local_path, S3_BUCKET, s3_path)
            except Exception as e:
                print(f"Erro upload {file}: {e}")

    print(f"✅ Upload concluído: {s3_prefix}")


# =========================
# PROCESSAMENTO DE UM VÍDEO
# =========================
def process_video(video_path, classe, exercise, erro=None):
    print(f"\n▶ Processando vídeo: {video_path}")

    base_path = "dataset"
    exercise_path = os.path.join(base_path, exercise, classe)

    video_id = os.path.basename(video_path).replace(".mp4", "")
    run_id = str(uuid.uuid4())[:6]

    prefix = f"{video_id}_{run_id}"

    images_path = os.path.join(exercise_path, "images", video_id)
    keypoints_path = os.path.join(exercise_path, "keypoints", video_id)
    labels_path = os.path.join(exercise_path, "labels", video_id)

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(keypoints_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # 1. Frames
    extract_frames(
        video_path=video_path,
        output_folder=images_path,
        prefix=prefix
    )

    # 2. Keypoints
    extract_keypoints(
        image_folder=images_path,
        output_folder=keypoints_path,
        prefix=prefix
    )

    # 3. Labels
    set_labels(
        pasta_keypoints=keypoints_path,
        pasta_labels=labels_path,
        classe=classe,
        exercise=exercise,
        erro=erro,
        prefix=prefix
    )

    # 4. Dataset parcial
    df = build_dataset(
        pasta_keypoints=keypoints_path,
        pasta_labels=labels_path,
        prefix=prefix
    )

    # 5. Upload arquivos
    base_s3_path = f"data/{exercise}/{classe}/{video_id}"

    subir_pasta_s3(images_path, f"{base_s3_path}/images")
    subir_pasta_s3(keypoints_path, f"{base_s3_path}/keypoints")
    subir_pasta_s3(labels_path, f"{base_s3_path}/labels")

    return df


# =========================
# PROCESSAMENTO DE PASTA
# =========================
def process_folder(folder_path, classe, exercise, erro=None):
    print(f"\n📂 Processando pasta: {folder_path}")

    dfs = []

    for file in os.listdir(folder_path):
        if not file.endswith(".mp4"):
            continue

        video_path = os.path.join(folder_path, file)

        df_video = process_video(
            video_path=video_path,
            classe=classe,
            exercise=exercise,
            erro=erro
        )

        if not df_video.empty:
            dfs.append(df_video)

    if not dfs:
        print("⚠️ Nenhum dado gerado")
        return None

    df_final = pd.concat(dfs, ignore_index=True)
    return df_final


# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(video=None, folder=None, classe=None, exercise=None, erro=None):
    base_path = "dataset"
    os.makedirs(base_path, exist_ok=True)

    csv_path = os.path.join(base_path, "dataset.csv")

    # 1. Processamento
    if folder:
        df_novo = process_folder(folder, classe, exercise, erro)
    elif video:
        df_novo = process_video(video, classe, exercise, erro)
    else:
        raise ValueError("Informe --video ou --folder")

    if df_novo is None or df_novo.empty:
        print("⚠️ Nenhum dado novo para adicionar")
        return

    # 2. Baixar dataset antigo
    existe = baixar_dataset_s3(csv_path)

    if existe:
        df_antigo = pd.read_csv(csv_path)
        df_final = pd.concat([df_antigo, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    # 3. Limpeza
    df_final = df_final.drop_duplicates()

    # 4. Salvar
    df_final.to_csv(csv_path, index=False)

    # 5. Subir
    subir_dataset_s3(csv_path)

    print("\n✅ Pipeline finalizado com sucesso!")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=False)
    parser.add_argument("--folder", required=False)
    parser.add_argument("--classe", required=True)
    parser.add_argument("--exercise", required=True)
    parser.add_argument("--erro", required=False)

    args = parser.parse_args()

    run_pipeline(
        video=args.video,
        folder=args.folder,
        classe=args.classe,
        exercise=args.exercise,
        erro=args.erro
    )