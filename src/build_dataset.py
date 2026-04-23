import os
import json
import pandas as pd
import re

def extrair_frame_num(nome_arquivo):
    match = re.search(r"_img_(\d+)", nome_arquivo)
    if match:
        return int(match.group(1))
    return None


def build_dataset(pasta_keypoints, pasta_labels, prefix=None):
    dados = []

    for root, _, files in os.walk(pasta_keypoints):
        for file in files:
            if not file.endswith(".json"):
                continue

            if prefix and not file.startswith(prefix):
                continue

            path_key = os.path.join(root, file)

            # 🔥 repetição = nome da pasta (video_id)
            repeticao_id = os.path.basename(root)

            # 🔥 frame = número do arquivo
            frame_seq = extrair_frame_num(file)

            # 🔥 caminho correto do label (mesma estrutura)
            relative_path = os.path.relpath(path_key, pasta_keypoints)
            path_label = os.path.join(pasta_labels, relative_path)

            if not os.path.exists(path_label):
                continue

            with open(path_key) as f:
                keypoints = json.load(f)

            with open(path_label) as f:
                label = json.load(f)

            linha = {}

            # keypoints
            for ponto, coords in keypoints.items():
                if not isinstance(coords, dict):
                    continue

                if "x" in coords and "y" in coords and "z" in coords:
                    linha[f"{ponto}_x"] = coords["x"]
                    linha[f"{ponto}_y"] = coords["y"]
                    linha[f"{ponto}_z"] = coords["z"]

            # 🔥 NOVOS CAMPOS
            linha["repeticao"] = repeticao_id
            linha["frame"] = frame_seq

            # labels antigos (mantidos)
            linha["classe"] = label["classe"]
            linha["erro"] = label.get("erro")
            linha["exercicio"] = label.get("exercicio")

            dados.append(linha)

    df = pd.DataFrame(dados)

    # 🔥 ordenação IMPORTANTE
    if not df.empty:
        df = df.sort_values(by=["repeticao", "frame"])

    return df