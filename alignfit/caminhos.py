"""Caminhos do projeto, independentes do sistema operacional e da pasta
de onde o script é executado."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

DADOS = RAIZ / "dados"
MODELOS = RAIZ / "modelos"
VIDEOS = RAIZ / "videos"
YOLO_POSE = RAIZ / "yolo" / "yolov8n-pose.pt"

# Base usada no treino dos modelos atuais (Seção 7 do TCC)
BASE_TREINO = DADOS / "base-v2.csv"

# Modelos gerados pelo pipeline e consumidos pela produção (iac-infra)
MODELO_EXERCICIO = MODELOS / "classificador-exercicio.pkl"
ROTULOS_EXERCICIO = MODELOS / "rotulos-exercicio.pkl"


def modelo_execucao(exercicio: str) -> Path:
    """Ex.: 'elevacao_lateral' -> modelos/execucao-elevacao-lateral.pkl"""
    return MODELOS / f"execucao-{exercicio.replace('_', '-')}.pkl"
