# 🧠 RAG Híbrido com Neo4j e LangChain

Este repositório contém um exemplo prático de uma arquitetura **Retrieval-Augmented Generation (RAG)** que utiliza o **Neo4j** como base de conhecimento híbrida (Gráfico + Vetorial).

## Por que RAG Híbrido?

A arquitetura RAG tradicional usa um banco vetorial para encontrar similaridade **semântica**. Este exemplo vai além, usando o poder do Graph Database (Neo4j) para enriquecer o contexto de recuperação com **relações estruturais**.

**Exemplo:**
1.  **Busca Semântica:** O LLM pergunta sobre um "projeto móvel".
2.  **Busca Estrutural (Grafo):** O Cypher do Neo4j recupera o nó do projeto *e* a relação **[:MANAGED_BY]** que aponta para o nome do gerente.

Isso fornece um contexto mais rico e preciso para o LLM.

## Pré-requisitos

1.  **Neo4j Desktop ou Servidor:** Uma instância do Neo4j em execução.
2.  **Chave OpenAI:** Uma chave API para os embeddings e o modelo GPT.
3.  **Python 3.x**

## Configuração

1.  Clone este repositório ou crie a estrutura de arquivos.
2.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
3.  Defina suas variáveis de ambiente ou edite o arquivo `rag_neo4j_hybrid.py` (linhas 15 a 17) com suas credenciais:
    ```bash
    # Exemplo de configuração via terminal (Linux/macOS)
    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USERNAME="neo4j"
    export NEO4J_PASSWORD="sua_senha"
    export OPENAI_API_KEY="sua_chave_openai"
    ```

## Execução

```bash
python rag_neo4j_hybrid.py
```

## Usando docker-compose
```bash
docker-compose exec rag-app python rag_neo4j_hybrid.py
```
