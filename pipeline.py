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


def baixar_dataset_s3(local_path):
    s3 = boto3.client("s3")

    try:
        s3.download_file(S3_BUCKET, S3_KEY, local_path)
        return True
    except ClientError as e:
        print(f"Erro detalhado: {e}")
        return False


def subir_dataset_s3(local_path):
    s3 = boto3.client("s3")

    try:
        s3.upload_file(local_path, S3_BUCKET, S3_KEY)
    except Exception as e:
        print(f"Erro: {e}")

def subir_pasta_s3(local_folder, s3_prefix):
    s3 = boto3.client("s3")

    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)

            relative_path = os.path.relpath(local_path, local_folder)

            s3_path = f"{s3_prefix}/{relative_path}"

            try:
                s3.upload_file(local_path, S3_BUCKET, s3_path)
            except Exception as e:
                print(f"{e}")

    print("✅ Upload concluído")

def run_pipeline(video_path, classe, exercise, erro=None):
    print("\n\nPipeline iniciada:")
    print(f"   Video path: {video_path}")
    print(f"   Classe: {classe}")
    print(f"   Exercício: {exercise}")
    print(f"   Erro: {erro}")

    base_path = "dataset"

    exercise_path = os.path.join(base_path, exercise, classe)

    images_path = os.path.join(exercise_path, "images")
    keypoints_path = os.path.join(exercise_path, "keypoints")
    labels_path = os.path.join(exercise_path, "labels")

    print("\n\nCriando diretórios:")
    print(images_path)
    print(keypoints_path)
    print(labels_path)

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(keypoints_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    run_id = str(uuid.uuid4())[:8]
    
    extract_frames(
        video_path=video_path,
        output_folder=images_path,
        prefix=run_id
    )

    extract_keypoints(
        image_folder=images_path,
        output_folder=keypoints_path,
        prefix=run_id
    )

    set_labels(
        pasta_keypoints=keypoints_path,
        pasta_labels=labels_path,
        classe=classe,
        exercise=exercise,
        erro=erro
    )

    df_novo = build_dataset(
        pasta_keypoints=keypoints_path,
        pasta_labels=labels_path
    )

    print(f"\n\nLinhas novas geradas: {len(df_novo)}")

    csv_path = os.path.join(base_path, "dataset.csv")

    existe = baixar_dataset_s3(csv_path)

    if existe:
        print("📎 Lendo dataset antigo...")
        df_antigo = pd.read_csv(csv_path)
        print(f"Linhas antigas: {len(df_antigo)}")

        df_final = pd.concat([df_antigo, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    df_final = df_final.drop_duplicates()

    df_final.to_csv(csv_path, index=False)

    subir_dataset_s3(csv_path)

    base_s3_path = f"data/{exercise}/{classe}"

    subir_pasta_s3(images_path, f"{base_s3_path}/images")
    subir_pasta_s3(keypoints_path, f"{base_s3_path}/keypoints")
    subir_pasta_s3(labels_path, f"{base_s3_path}/labels")

    print("\nPipeline finalizado")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--classe", required=True)
    parser.add_argument("--exercise", required=True)
    parser.add_argument("--erro", required=False)

    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        classe=args.classe,
        exercise=args.exercise,
        erro=args.erro
    )
