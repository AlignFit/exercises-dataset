import os
import json
import pandas as pd

def build_dataset(pasta_keypoints, pasta_labels, prefix=None):
    dados = []

    for file in os.listdir(pasta_keypoints):
        if not file.endswith(".json"):
            continue

        if prefix and not file.startswith(prefix):
            continue

        path_key = os.path.join(pasta_keypoints, file)
        path_label = os.path.join(pasta_labels, file)

        if not os.path.exists(path_label):
            continue  

        with open(path_key) as f:
            keypoints = json.load(f)

        with open(path_label) as f:
            label = json.load(f)

        linha = {}

        for ponto, coords in keypoints.items():
            if not isinstance(coords, dict):
                continue

            if "x" in coords and "y" in coords and "z" in coords:
                linha[f"{ponto}_x"] = coords["x"]
                linha[f"{ponto}_y"] = coords["y"]
                linha[f"{ponto}_z"] = coords["z"]

        linha["classe"] = label["classe"]
        linha["erro"] = label.get("erro")
        linha["exercicio"] = label.get("exercicio")

        dados.append(linha)

    df = pd.DataFrame(dados)
    return df