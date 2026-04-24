import cv2
import mediapipe as mp
import os
import json

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "pose_landmarker.task")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.PoseLandmarker.create_from_options(options)

LANDMARK_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner",
    "right_eye","right_eye_outer","left_ear","right_ear","mouth_left",
    "mouth_right","left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky","left_index",
    "right_index","left_thumb","right_thumb","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle","left_heel",
    "right_heel","left_foot_index","right_foot_index
]

def extract_keypoints(image_folder, output_folder, prefix=None):
    os.makedirs(output_folder, exist_ok=True)

    for image_name in os.listdir(image_folder):
        if not image_name.endswith(".png"):
            continue
        
        if prefix and not image_name.startswith(prefix):
            continue

        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Erro ao carregar: {image_name}")
            continue

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )

        result = detector.detect(mp_image)

        keypoints = {}

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            for i, lm in enumerate(landmarks):
                keypoints[LANDMARK_NAMES[i]] = {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                }
        else:
            print(f"Nenhum corpo detectado em: {image_name}")

        json_name = image_name.replace(".png", ".json")
        json_path = os.path.join(output_folder, json_name)

        with open(json_path, "w") as f:
            json.dump(keypoints, f, indent=4)

    print("Keypoints extraídos com sucesso!")

