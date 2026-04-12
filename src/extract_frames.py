import cv2
import os

def extract_frames(video_path, output_folder, prefix="", fps=5, resize=(640, 480)):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if count % frame_interval == 0:
            if resize:
                frame = cv2.resize(frame, resize)

            filename = os.path.join(
                output_folder,
                f"{prefix}_img_{saved_count:05d}.png"
            )

            cv2.imwrite(filename, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"Frames salvos: {saved_count}")