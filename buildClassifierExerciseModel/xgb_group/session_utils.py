# =========================================================
# session_utils.py
#
# Deriva um ID de sessão de gravação a partir do nome/caminho
# do vídeo. Usado pelos 4 scripts Build*ExecutionClassifierModel.py
# para dividir treino/teste por SESSÃO em vez de por VÍDEO
# (ver Seção 9.2 do TCC: nenhuma sessão mistura execução
# correta e errada, então um split por vídeo aleatório vaza
# informação de sessão entre treino e teste).
#
# Heurística (não é uma identidade de pessoa verificada):
#   - Pastas "execucaoDDMMAAAA"          -> mesma sessão
#   - Arquivos "WhatsApp Video AAAA-MM-DD at ...")  -> mesma
#     sessão se a data for igual
#   - Demais vídeos (ex.: base pública da internet) -> cada
#     vídeo é tratado como sua própria sessão
# =========================================================

import re

_WHATSAPP_DATE_RE = re.compile(r"WhatsApp Video (\d{4}-\d{2}-\d{2})")
_EXECUCAO_FOLDER_RE = re.compile(r"(execucao\d+)", re.IGNORECASE)


def derive_session_id(video_filename: str, video_path: str) -> str:
    """Deriva um ID de sessão/gravação a partir do nome/caminho do vídeo."""
    m = _EXECUCAO_FOLDER_RE.search(video_path)
    if m:
        return f"sessao_{m.group(1).lower()}"

    m = _WHATSAPP_DATE_RE.search(video_filename)
    if m:
        return f"whatsapp_{m.group(1)}"

    # base publica / nao identificado -> cada video e sua propria sessao
    return f"video_{video_filename}"
