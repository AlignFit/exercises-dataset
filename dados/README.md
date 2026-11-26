# dados/

Bases de keypoints extraídas pelo YOLOv8-Pose. Cada linha é **uma pessoa
detectada em um frame** de um vídeo.

| Arquivo | Vídeos | Frames | Uso | TCC |
|---|---|---|---|---|
| `base-v2.csv` | 240 | 16.056 | **base atual de treino** | Seção 7 |
| `base-v1.csv` | 537 | 19.868 | base antiga (notebooks e experimentos legados) | Seções 4–6 |
| `exemplo-video-usuario.csv` | 1 | — | exemplo para `inferencia/` | — |

## base-v2.csv

Gravações da equipe (com pessoa e sessão no nome do arquivo) + vídeos externos
da base pública.

| Exercício | Correta (equipe) | Errada (equipe) | Externos (só corretas) | Total |
|---|---|---|---|---|
| Agachamento | 12 | 12 | 29 | 53 |
| Rosca bíceps | 12 | 11 | 62 | 85 |
| Desenvolvimento | 12 | 12 | 17 | 41 |
| Elevação lateral | 12 | 12 | 37 | 61 |
| **Total** | **48** | **47** | **145** | **240** |

- Equipe: 3 pessoas (paulo 47, bruno 32, caio 16 vídeos), 11 sessões, cada
  sessão com execuções corretas **e** erradas (corrige o vazamento de sessão da
  base v1, Seção 6.3).
- Externos: `pessoa = "externo"`, uma sessão por vídeo. Usados só no
  classificador de exercício.

## base-v1.csv

537 vídeos (135 da base pública + 402 gravados pela equipe). Cada sessão de
gravação tinha só execuções corretas ou só erradas, o que causou o vazamento
descrito na Seção 6.3. Mantida para reproduzir as Tabelas 1–10.

## arquivo/

Versões intermediárias, mantidas por rastreabilidade:

| Arquivo | Conteúdo |
|---|---|
| `base-v2-somente-equipe.csv` | os 95 vídeos da equipe, sem externos |
| `base-v2-somente-equipe-original.csv` | mesma base, gerada antes (era `raw-dataset.csv`) |
| `base-v2-com-internet.csv` | base v2 + 55 vídeos de banco de imagens (`pessoa = "internet"`), não usada |

## Colunas

| Coluna | Descrição |
|---|---|
| `video` | nome do arquivo de vídeo |
| `video_id` | caminho do vídeo na máquina que gerou a base |
| `frame` | número do frame (1 a cada 5 é processado) |
| `person_id` | índice da pessoa detectada no frame (0 = primeira detecção) |
| `pessoa`, `sessao` | quem gravou e em qual sessão (só v2) |
| `tipo_exercicio` | agachamento, biceps, desenvolvimento, elevacao_lateral |
| `tipo_execucao` | correta, errada |
| `tipo_erro` | "nenhum" ou "mal executado" (o tipo específico de erro não é rotulado) |
| `<ponto>_x`, `<ponto>_y` | coordenadas normalizadas (0–1) dos 17 pontos; `0,0` = ponto não detectado |
