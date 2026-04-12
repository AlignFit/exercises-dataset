import os
import json

def set_labels(pasta_keypoints, pasta_labels, classe, exercise, erro=None, prefix=None):
    os.makedirs(pasta_labels, exist_ok=True)

    for file in os.listdir(pasta_keypoints):
        if not file.endswith(".json"):
            continue

        if prefix and not file.startswith(prefix):
            continue

        label = {
            "exercicio": exercise,
            "classe": classe,
            "erro": erro
        }

        path_label = os.path.join(pasta_labels, file)

        with open(path_label, "w") as f:
            json.dump(label, f, indent=4)

    print(f"Labels criados para {classe}")