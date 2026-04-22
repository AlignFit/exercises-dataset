import os
from extract_frames import extract_frames
from extract_keypoints import extract_keypoints
from set_labels import set_labels
from build_dataset import build_dataset

def process_folder(folder, classe, exercise, erro=None):
    base_input = os.path.join("data", "videos", folder)

    for video_name in os.listdir(base_input):
        if not video_name.endswith(".mp4"):
            continue

        video_path = os.path.join(base_input, video_name)
        prefix = video_name.replace(".mp4", "")

        print(f"Processando: {video_name}")

        # 1. Frames
        frames_path = f"data/frames/{folder}/{prefix}"
        extract_frames(video_path, frames_path, prefix=prefix)

        # 2. Keypoints
        keypoints_path = f"data/keypoints/{folder}/{prefix}"
        extract_keypoints(frames_path, keypoints_path, prefix=prefix)

        # 3. Labels
        labels_path = f"data/labels/{folder}/{prefix}"
        set_labels(
            keypoints_path,
            labels_path,
            classe,
            exercise,
            erro,
            prefix=prefix
        )

    # 4. Dataset final (junta tudo)
    df = build_dataset(
        pasta_keypoints=f"data/keypoints/{folder}",
        pasta_labels=f"data/labels/{folder}"
    )

    df.to_csv(f"data/dataset_{folder}.csv", index=False)
    print("Dataset final gerado!")