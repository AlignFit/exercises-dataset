# =========================================================
# extrair_keypoints_video.py
#
# Inferência local, passo 1: extrai os keypoints de UM vídeo
# (ex.: gravado pelo usuário) e salva num CSV temporário.
#
# Uso:
#   python inferencia/extrair_keypoints_video.py --video caminho/do/video.mp4
#   python inferencia/extrair_keypoints_video.py --video v.mp4 --saida dados/meu-video.csv
# =========================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd
from ultralytics import YOLO

from alignfit import caminhos
from alignfit.keypoints import extrair_keypoints_do_frame

PULAR_FRAMES = 5  # processa 1 a cada 5 frames (igual ao treino)


def extrair(video: Path) -> pd.DataFrame:
    modelo = YOLO(str(caminhos.YOLO_POSE))
    cap = cv2.VideoCapture(str(video))
    linhas, frame_id = [], 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % PULAR_FRAMES == 0:
            altura, largura = frame.shape[:2]
            resultado = modelo(frame, verbose=False)[0]
            for pessoa in extrair_keypoints_do_frame(resultado, largura, altura):
                linhas.append({"video": video.name, "frame": frame_id, **pessoa})
        frame_id += 1
    cap.release()
    return pd.DataFrame(linhas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--saida", type=Path, default=caminhos.DADOS / "video-usuario.csv")
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"Vídeo não encontrado: {args.video}")

    df = extrair(args.video)
    df.to_csv(args.saida, index=False)
    print(f"{len(df)} registros extraídos de {args.video.name} -> {args.saida}")


if __name__ == "__main__":
    main()
