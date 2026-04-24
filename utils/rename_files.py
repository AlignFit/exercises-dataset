import os

# Prefixo que você quer usar
prefixo = "prefix"

# Pasta atual (onde o script está)
pasta = os.path.dirname(os.path.abspath(__file__))

# Lista de arquivos (ignorando o próprio script)
arquivos = [f for f in os.listdir(pasta) if os.path.isfile(os.path.join(pasta, f))]

# Ordena os arquivos (opcional, mas recomendado)
arquivos.sort()

# Renomeia
for i, nome_antigo in enumerate(arquivos, start=1):
    extensao = os.path.splitext(nome_antigo)[1]
    novo_nome = f"{prefixo}_{i:02d}{extensao}"

    caminho_antigo = os.path.join(pasta, nome_antigo)
    caminho_novo = os.path.join(pasta, novo_nome)

    os.rename(caminho_antigo, caminho_novo)
    print(f"{nome_antigo} -> {novo_nome}")