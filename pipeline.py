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
        print("Dataset antigo baixado do S3")
        return True
    except ClientError:
        print("Nenhum dataset anterior encontrado no S3")
        return False


def subir_dataset_s3(local_path):
    s3 = boto3.client("s3")
    s3.upload_file(local_path, S3_BUCKET, S3_KEY)
    print("Dataset enviado para o S3")


def run_pipeline(video_path, classe, exercise, erro=None):
    base_path = "dataset"

    exercise_path = os.path.join(base_path, exercise, classe)

    images_path = os.path.join(exercise_path, "images")
    keypoints_path = os.path.join(exercise_path, "keypoints")
    labels_path = os.path.join(exercise_path, "labels")

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

    csv_path = os.path.join(base_path, "dataset.csv")

    existe = baixar_dataset_s3(csv_path)

    if existe:
        df_antigo = pd.read_csv(csv_path)
        df_final = pd.concat([df_antigo, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    df_final.to_csv(csv_path, index=False)

    subir_dataset_s3(csv_path)

    print(f"Dataset atualizado e sincronizado com S3")
