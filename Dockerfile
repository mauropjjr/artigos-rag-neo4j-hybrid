# Usa uma imagem base Python slim (menor e otimizada)
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de requisitos e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o script principal e o README
COPY rag_neo4j_hybrid.py .
COPY README.md .

# Comando padrão a ser executado ao iniciar o container, 
# mas o docker-compose irá sobrescrever isso com o comando 'sleep' para manter o container ativo
# para a execução manual posterior.
CMD ["python", "rag_neo4j_hybrid.py"]