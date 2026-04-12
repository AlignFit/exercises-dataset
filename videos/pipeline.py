import os
import argparse
import uuid
import pandas as pd

from src.extract_frames import extract_frames
from src.extract_keypoints import extract_keypoints
from src.set_labels import set_labels
from src.build_dataset import build_dataset


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

    if os.path.exists(csv_path):
        df_antigo = pd.read_csv(csv_path)
        df_final = pd.concat([df_antigo, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    df_final.to_csv(csv_path, index=False)

    print(f"Dataset atualizado em: {csv_path}")


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