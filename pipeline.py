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
    print("\n🔽 Tentando baixar dataset do S3...")
    print(f"Bucket: {S3_BUCKET}")
    print(f"Key: {S3_KEY}")

    s3 = boto3.client("s3")

    try:
        s3.download_file(S3_BUCKET, S3_KEY, local_path)
        print("✅ Dataset antigo baixado do S3")
        return True
    except ClientError as e:
        print("⚠️ Nenhum dataset anterior encontrado ou erro no S3")
        print(f"Erro detalhado: {e}")
        return False


def subir_dataset_s3(local_path):
    print("\n🔼 Enviando dataset para o S3...")
    print(f"Arquivo local: {local_path}")
    print(f"Bucket: {S3_BUCKET}")
    print(f"Key: {S3_KEY}")

    s3 = boto3.client("s3")

    try:
        s3.upload_file(local_path, S3_BUCKET, S3_KEY)
        print("✅ Dataset enviado para o S3")
    except Exception as e:
        print("❌ Erro ao subir para o S3")
        print(f"Erro: {e}")


def run_pipeline(video_path, classe, exercise, erro=None):
    print("\n🚀 INICIANDO PIPELINE")
    print(f"Video path: {video_path}")
    print(f"Classe: {classe}")
    print(f"Exercício: {exercise}")
    print(f"Erro: {erro}")

    print(f"Arquivo de vídeo existe? {os.path.exists(video_path)}")

    base_path = "dataset"

    exercise_path = os.path.join(base_path, exercise, classe)

    images_path = os.path.join(exercise_path, "images")
    keypoints_path = os.path.join(exercise_path, "keypoints")
    labels_path = os.path.join(exercise_path, "labels")

    print("\n📁 Criando diretórios...")
    print(images_path)
    print(keypoints_path)
    print(labels_path)

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(keypoints_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    run_id = str(uuid.uuid4())[:8]
    print(f"\n🆔 Run ID: {run_id}")

    print("\n🎞️ Extraindo frames...")
    extract_frames(
        video_path=video_path,
        output_folder=images_path,
        prefix=run_id
    )

    print(f"Frames gerados: {len(os.listdir(images_path))}")

    print("\n🧠 Extraindo keypoints...")
    extract_keypoints(
        image_folder=images_path,
        output_folder=keypoints_path,
        prefix=run_id
    )

    print(f"Keypoints gerados: {len(os.listdir(keypoints_path))}")

    print("\n🏷️ Gerando labels...")
    set_labels(
        pasta_keypoints=keypoints_path,
        pasta_labels=labels_path,
        classe=classe,
        exercise=exercise,
        erro=erro
    )

    print(f"Labels gerados: {len(os.listdir(labels_path))}")

    print("\n📊 Montando dataset novo...")
    df_novo = build_dataset(
        pasta_keypoints=keypoints_path,
        pasta_labels=labels_path
    )

    print(f"Linhas novas geradas: {len(df_novo)}")

    csv_path = os.path.join(base_path, "dataset.csv")

    print("\n📥 Verificando dataset anterior...")
    existe = baixar_dataset_s3(csv_path)

    if existe:
        print("📎 Lendo dataset antigo...")
        df_antigo = pd.read_csv(csv_path)
        print(f"Linhas antigas: {len(df_antigo)}")

        df_final = pd.concat([df_antigo, df_novo], ignore_index=True)
    else:
        print("🆕 Criando novo dataset")
        df_final = df_novo

    print(f"📦 Total de linhas finais: {len(df_final)}")

    df_final = df_final.drop_duplicates()

    print("\n💾 Salvando dataset local...")
    df_final.to_csv(csv_path, index=False)

    print(f"Arquivo existe localmente? {os.path.exists(csv_path)}")

    subir_dataset_s3(csv_path)

    print("\n✅ PIPELINE FINALIZADO COM SUCESSO")

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
