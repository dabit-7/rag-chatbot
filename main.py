import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import subprocess

# Configuração do Pinecone usando a nova abordagem
pc = Pinecone(
    api_key="???"  # Chave Pinecode
)

# Verifica se o índice já existe, senão ele cria
index_name = "chatbot-rag"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Obtem a referência ao índice
index = pc.Index(index_name)

# Carrega modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")  #modelo leve

def gerar_embedding(texto):
    """Gera embeddings para o texto."""
    return model.encode(texto).tolist()

def adicionar_ao_pinecone(id, texto, metadados=None):
    """Adiciona um item ao Pinecone."""
    embedding = gerar_embedding(texto)
    index.upsert([(id, embedding, metadados)])

def buscar_no_pinecone(query, top_k=5):
    """Busca os itens mais relevantes no Pinecone."""
    embedding = gerar_embedding(query)
    resultados = index.query(
        vector=embedding, 
        top_k=top_k, 
        include_metadata=True
    )
    return resultados["matches"]

def gerar_resposta(prompt, contexto):
    """Gera uma resposta usando o Ollama."""
    prompt_completo = f"""
    Use o seguinte contexto para responder a pergunta:
    {contexto}

    Pergunta: {prompt}
    Resposta:
    """
    processo = subprocess.run(
        ["ollama", "run", "llama2", prompt_completo], #llama2 ou mistral
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    if processo.returncode != 0:
        raise RuntimeError(f"Erro ao gerar resposta: {processo.stderr}")
    return processo.stdout.strip()

# Exemplo 1: Adicionar dados ao Pinecone
adicionar_ao_pinecone("1", "Waldemar Henrique nasceu em Belémn do Pará, no dia 15 de fevereiro de 1905 e morreu no dia 29 de março de 1995.")
adicionar_ao_pinecone("2", "Waldemar Henrique morreu aos 90 anos vítima de cancêr.")


# Exemplo 2: Lê um arquivo txt
#def ler_arquivo_txt(caminho_arquivo):
#   """Lê o conteúdo de um arquivo txt e retorna como uma lista de linhas."""
#    with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
#        return arquivo.readlines()

# Função para adicionar dados de um arquivo txt ao Pinecone
#def adicionar_dados_do_arquivo(caminho_arquivo):
#    """Lê o arquivo e adiciona seu conteúdo ao Pinecone."""
#    linhas = ler_arquivo_txt(caminho_arquivo)
#    for i, linha in enumerate(linhas):
#        # Use o índice i como ID e a linha como texto
#        adicionar_ao_pinecone(str(i+1), linha.strip())


# Exemplo: Adicionar dados de um arquivo txt ao Pinecone
#caminho_arquivo = "C:\Py\Alice.txt"  # Coloque o caminho do seu arquivo de texto
#adicionar_dados_do_arquivo(caminho_arquivo)

# Fluxo principal do chatbot
#consulta_usuario = "O que é preservação digital?"
#resultados = buscar_no_pinecone(consulta_usuario)

#-----------------------------

# Fluxo principal do chatbot
consulta_usuario = input("Digite sua pergunta: ")  # Permite que o usuário digite qualquer pergunta
resultados = buscar_no_pinecone(consulta_usuario)

# Extração de contexto relevante
contexto_relevante = "\n".join([f"{res['metadata']}" for res in resultados if "metadata" in res])


# Extração de contexto relevante
contexto_relevante = "\n".join([f"{res['metadata']}" for res in resultados if "metadata" in res])

# Gerar resposta final
resposta = gerar_resposta(consulta_usuario, contexto_relevante)
print("Chatbot:", resposta)
