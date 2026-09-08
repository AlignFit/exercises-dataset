# =========================================================
# BuildRawDatasetV2.py
#
# Versão atualizada do BuildRawDataset.py que extrai os
# metadados (pessoa, sessão, exercício, execução) direto
# do NOME DO ARQUIVO, sem precisar editar nada manualmente.
#
# Convenção de nome esperada:
#   {pessoa}_{sessao}_{exercicio}_{execucao}_{seq}.mp4
#
# Exemplos:
#   bruno_S01_agachamento_correta_01.mp4
#   bruno_S01_biceps_errada_02.mp4
#   caio_S02_elevacao_lateral_correta_01.mp4
#
# Uso:
#   py BuildRawDatasetV2.py
#
# Ou, pra apontar pra outra pasta de vídeos:
#   py BuildRawDatasetV2.py --pasta "..\videos\novas_gravacoes"
#
# O CSV é salvo em ..\datasets\ com timestamp no nome,
# nunca sobrescrevendo o anterior.
# =========================================================

from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

import os
import sys
import cv2
import pandas as pd

# =========================================================
# CONFIGURAÇÕES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Pasta padrão onde os vídeos novos ficam
DEFAULT_VIDEO_FOLDER = PROJECT_ROOT / "videos" / "videos_por_sessao"

# Modelo YOLO Pose
MODEL_PATH = PROJECT_ROOT / "utils" / "yolov8n-pose.pt"

# Pasta de saída dos CSVs
OUTPUT_DIR = PROJECT_ROOT / "datasets"

# Processar 1 frame a cada N frames
FRAME_SKIP = 5

# =========================================================
# ARGUMENTOS DE LINHA DE COMANDO (opcional)
# =========================================================

video_folder = DEFAULT_VIDEO_FOLDER

if "--pasta" in sys.argv:
    idx = sys.argv.index("--pasta")
    if idx + 1 < len(sys.argv):
        video_folder = Path(sys.argv[idx + 1])

# =========================================================
# EXERCÍCIOS VÁLIDOS (pra validação do nome do arquivo)
# =========================================================

EXERCICIOS_VALIDOS = {
    "agachamento",
    "biceps",
    "desenvolvimento",
    "elevacao_lateral",
}

EXECUCOES_VALIDAS = {
    "correta",
    "errada",
}

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
    16: "tornozelo_direito",
}

# =========================================================
# PARSER DO NOME DO ARQUIVO
# =========================================================

def parse_video_filename(filename):
    """
    Extrai pessoa, sessao, exercicio, execucao e sequencial
    do nome do arquivo, seguindo a convenção:

        {pessoa}_{sessao}_{exercicio}_{execucao}_{seq}.mp4

    Para exercícios com underscore no nome (elevacao_lateral),
    tenta as duas possibilidades.

    Retorna um dicionário ou None se o nome não seguir a convenção.
    """
    name = filename.rsplit(".", 1)[0]  # remove extensão
    parts = name.split("_")

    # Mínimo: pessoa + sessao + exercicio + execucao + seq = 5 partes
    # Com elevacao_lateral: 6 partes
    if len(parts) < 5:
        return None

    pessoa = parts[0]
    sessao = parts[1]

    # Tentar exercício composto (elevacao_lateral) primeiro
    if len(parts) >= 6:
        exercicio_composto = f"{parts[2]}_{parts[3]}"
        if exercicio_composto in EXERCICIOS_VALIDOS:
            exercicio = exercicio_composto
            execucao = parts[4]
            sequencial = parts[5]
        elif parts[2] in EXERCICIOS_VALIDOS:
            exercicio = parts[2]
            execucao = parts[3]
            sequencial = parts[4]
        else:
            return None
    elif len(parts) == 5:
        exercicio = parts[2]
        execucao = parts[3]
        sequencial = parts[4]
    else:
        return None

    # Validação
    if exercicio not in EXERCICIOS_VALIDOS:
        return None
    if execucao not in EXECUCOES_VALIDAS:
        return None

    return {
        "pessoa": pessoa,
        "sessao": f"{pessoa}_{sessao}",  # sessão é por pessoa
        "exercicio": exercicio,
        "execucao": execucao,
        "sequencial": sequencial,
    }


# =========================================================
# VALIDAR PASTA DE VÍDEOS
# =========================================================

print("========================================")
print("BuildRawDatasetV2")
print("========================================")
print(f"Pasta de vídeos: {video_folder}")

if not video_folder.exists():
    print(f"\n[ERRO] Pasta não encontrada: {video_folder}")
    print("Crie a pasta e coloque os vídeos com a convenção:")
    print("  {{pessoa}}_{{sessao}}_{{exercicio}}_{{execucao}}_{{seq}}.mp4")
    print("Exemplo: bruno_S01_agachamento_correta_01.mp4")
    sys.exit(1)

# =========================================================
# LISTAR E VALIDAR VÍDEOS
# =========================================================

video_extensions = (".mp4", ".avi", ".mov", ".mkv")

all_files = [
    f for f in os.listdir(video_folder)
    if f.lower().endswith(video_extensions)
]

print(f"Arquivos de vídeo encontrados: {len(all_files)}")

videos_validos = []
videos_invalidos = []

for f in sorted(all_files):
    parsed = parse_video_filename(f)
    if parsed is None:
        videos_invalidos.append(f)
    else:
        videos_validos.append((f, parsed))

if videos_invalidos:
    print(f"\n[AVISO] {len(videos_invalidos)} arquivo(s) com nome fora da convenção (serão ignorados):")
    for v in videos_invalidos:
        print(f"  {v}")
    print()
    print("Convenção esperada: {{pessoa}}_{{sessao}}_{{exercicio}}_{{execucao}}_{{seq}}.mp4")
    print("Exemplo: bruno_S01_biceps_correta_01.mp4")

if not videos_validos:
    print("\n[ERRO] Nenhum vídeo válido encontrado. Nada a processar.")
    sys.exit(1)

print(f"\nVídeos válidos a processar: {len(videos_validos)}")

# Resumo do que vai ser processado
resumo = {}
for _, parsed in videos_validos:
    key = (parsed["pessoa"], parsed["sessao"], parsed["exercicio"], parsed["execucao"])
    resumo[key] = resumo.get(key, 0) + 1

print("\nResumo:")
for (pessoa, sessao, exercicio, execucao), count in sorted(resumo.items()):
    print(f"  {pessoa} | {sessao} | {exercicio} | {execucao} | {count} vídeo(s)")

# =========================================================
# CARREGAR MODELO YOLO
# =========================================================

print("\n========================================")
print("Carregando YOLO Pose...")
print("========================================")

if not MODEL_PATH.exists():
    print(f"[ERRO] Modelo não encontrado: {MODEL_PATH}")
    sys.exit(1)

model = YOLO(str(MODEL_PATH))
print("Modelo carregado com sucesso!")

# =========================================================
# PROCESSAR VÍDEOS
# =========================================================

all_data = []

for video_idx, (video_name, parsed) in enumerate(videos_validos, start=1):

    video_path = os.path.join(str(video_folder), video_name)

    print(f"\n[{video_idx}/{len(videos_validos)}] {video_name}")
    print(f"  pessoa={parsed['pessoa']} sessao={parsed['sessao']} "
          f"exercicio={parsed['exercicio']} execucao={parsed['execucao']}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [AVISO] Não foi possível abrir o vídeo. Pulando.")
        continue

    frame_id = 0
    frames_processados = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        # Pular frames
        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        height, width = frame.shape[:2]

        # Inferência YOLO
        results = model(frame, verbose=False)
        result = results[0]

        # Verificar keypoints
        if result.keypoints is not None and len(result.keypoints.xy) > 0:

            keypoints = result.keypoints.xy.cpu().numpy()

            for person_id, person_keypoints in enumerate(keypoints):

                row = {
                    # METADADOS (extraídos do nome do arquivo)
                    "video": video_name,
                    "video_id": video_path,
                    "frame": frame_id,
                    "person_id": person_id,
                    "pessoa": parsed["pessoa"],
                    "sessao": parsed["sessao"],

                    # LABELS (extraídos do nome do arquivo)
                    "tipo_exercicio": parsed["exercicio"],
                    "tipo_execucao": parsed["execucao"],
                    "tipo_erro": "mal executado" if parsed["execucao"] == "errada" else "nenhum",
                }

                # Keypoints normalizados
                for kp_id, (x, y) in enumerate(person_keypoints):

                    body_part = KEYPOINT_NAMES[kp_id]

                    x_norm = float(x / width)
                    y_norm = float(y / height)

                    row[f"{body_part}_x"] = x_norm
                    row[f"{body_part}_y"] = y_norm

                all_data.append(row)
                frames_processados += 1

        frame_id += 1

    cap.release()
    print(f"  {frames_processados} frames extraídos")

# =========================================================
# CRIAR DATAFRAME
# =========================================================

new_df = pd.DataFrame(all_data)

if new_df.empty:
    print("\n[AVISO] Nenhum frame foi extraído de nenhum vídeo. CSV não será gerado.")
    sys.exit(0)

# =========================================================
# SALVAR CSV COM TIMESTAMP (nunca sobrescreve o anterior)
# =========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"raw-dataset-v2_{timestamp}.csv"
output_path = OUTPUT_DIR / output_filename

new_df.to_csv(output_path, index=False)

# =========================================================
# RESUMO FINAL
# =========================================================

print("\n========================================")
print("DATASET GERADO COM SUCESSO!")
print("========================================")

print(f"\nArquivo salvo: {output_path}")
print(f"Vídeos processados: {len(videos_validos)}")
print(f"Frames extraídos: {len(new_df)}")

print(f"\nColunas extras neste CSV (em relação ao raw-dataset.csv original):")
print(f"  'pessoa'  = identificador da pessoa que gravou")
print(f"  'sessao'  = identificador da sessão de gravação (pessoa + código)")

print(f"\nDistribuição:")
print(new_df.groupby(["pessoa", "sessao", "tipo_exercicio", "tipo_execucao"])["video"].nunique().to_string())

print("\n========================================")
