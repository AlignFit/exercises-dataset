"""Os 17 pontos do corpo detectados pelo YOLOv8-Pose (padrão COCO)."""

NOMES_KEYPOINTS = [
    "nariz",
    "olho_esquerdo", "olho_direito",
    "orelha_esquerda", "orelha_direita",
    "ombro_esquerdo", "ombro_direito",
    "cotovelo_esquerdo", "cotovelo_direito",
    "pulso_esquerdo", "pulso_direito",
    "quadril_esquerdo", "quadril_direito",
    "joelho_esquerdo", "joelho_direito",
    "tornozelo_esquerdo", "tornozelo_direito",
]

# Colunas de coordenadas no CSV: nariz_x, nariz_y, olho_esquerdo_x, ...
COLUNAS_COORDENADAS = [f"{k}_{eixo}" for k in NOMES_KEYPOINTS for eixo in ("x", "y")]

EXERCICIOS = ["agachamento", "biceps", "desenvolvimento", "elevacao_lateral"]


def extrair_keypoints_do_frame(resultado_yolo, largura, altura):
    """Converte a saída do YOLO de um frame numa lista de dicionários
    (um por pessoa detectada) com as coordenadas normalizadas entre 0 e 1."""
    if resultado_yolo.keypoints is None or len(resultado_yolo.keypoints.xy) == 0:
        return []
    pessoas = []
    for pessoa_id, pontos in enumerate(resultado_yolo.keypoints.xy.cpu().numpy()):
        linha = {"person_id": pessoa_id}
        for nome, (x, y) in zip(NOMES_KEYPOINTS, pontos):
            linha[f"{nome}_x"] = float(x / largura)
            linha[f"{nome}_y"] = float(y / altura)
        pessoas.append(linha)
    return pessoas
