import os
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# --- 1. Configurações e Inicialização ---
# É necessário ter uma chave OpenAI para embeddings e LLM
# Defina suas credenciais do Neo4j e OpenAI (use variáveis de ambiente ou defina aqui)
# os.environ["NEO4J_URI"] = "bolt://localhost:7687"
# os.environ["NEO4J_USERNAME"] = "neo4j"
# os.environ["NEO4J_PASSWORD"] = "sua_senha"
# os.environ["OPENAI_API_KEY"] = "sua_chave_openai"

# Configurações do Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "sua_senha")

# Inicializa o LangChain com o Neo4j
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Inicializa os Embeddings (vetores) e o LLM
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# --- 2. Preparação do Grafo e Dados ---

def criar_grafo_conhecimento(graph: Neo4jGraph, embeddings_model):
    """Cria nós, relações e índices vetoriais no Neo4j."""

    print("Criando índice vetorial 'project_embeddings'...")
    # Cria o índice vetorial no Neo4j. O LangChain usa este índice para busca vetorial.
    # Dimensões do embedding model 'text-embedding-ada-002' são 1536.
    graph.query(
        """
        CREATE VECTOR INDEX project_embeddings IF NOT EXISTS
        FOR (n:Project) ON (n.embedding)
        OPTIONS {
            indexProvider: 'vector-1.0',
            indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
    )
    
    # Cria o índice de texto para busca híbrida
    graph.query(
        """
        CREATE TEXT INDEX project_text_index IF NOT EXISTS
        FOR (n:Project) ON (n.description)
        """
    )
    
    # Dados de exemplo para o grafo
    data = [
        ("Projeto Alpha", "Desenvolvimento de uma plataforma de e-commerce escalável usando Microsserviços e Kubernetes.", "Maria"),
        ("Projeto Beta", "Sistema de análise de dados em tempo real utilizando Apache Flink e visualizações avançadas.", "João"),
        ("Projeto Gamma", "Aplicação móvel nativa (iOS/Android) focada em UX e acessibilidade.", "Maria"),
        ("Projeto Delta", "Implementação de um pipeline de Machine Learning (MLOps) para predição de churn de clientes.", "Pedro"),
    ]

    print("Criando nós e relações...")
    for nome, descricao, gerente in data:
        # Gera o embedding (vetor) da descrição para busca semântica
        embedding_vetor = embeddings_model.embed_query(descricao)

        # 1. Cria o nó do Projeto com o vetor
        graph.query(
            f"""
            MERGE (p:Project {{name: '{nome}'}})
            SET p.description = '{descricao}', p.embedding = {embedding_vetor}
            """
        )

        # 2. Cria o nó do Gerente
        graph.query(
            f"""
            MERGE (g:Manager {{name: '{gerente}'}})
            """
        )

        # 3. Cria a Relação
        graph.query(
            f"""
            MATCH (p:Project {{name: '{nome}'}})
            MATCH (g:Manager {{name: '{gerente}'}})
            MERGE (p)-[:MANAGED_BY]->(g)
            """
        )

    print("Grafo de conhecimento carregado com sucesso!")
    # Garante que o índice foi populado antes da busca
    # graph.refresh_schema()

# --- 3. Função de Busca RAG Híbrida ---

def busca_hibrida_rag(question: str, graph: Neo4jGraph, llm_model):
    """
    Executa a busca RAG Híbrida.
    A LangChain usará o índice vetorial para encontrar projetos semelhantes (Busca Semântica)
    E o Cypher para incluir o gerente relacionado (Busca Estrutural) no contexto.
    """
    
    # 1. Template do Prompt para o LLM
    # O contexto incluirá o resultado da busca vetorial E as informações do grafo
    template = """
    Você é um assistente de IA especialista em dados e projetos. 
    Use APENAS o contexto fornecido abaixo para responder à pergunta.
    O contexto pode incluir descrições de projetos e seus gerentes relacionados.
    Se a resposta não puder ser encontrada no contexto, diga que você não tem informações suficientes.

    Contexto:
    {context}

    Pergunta: {input}
    Resposta:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "input"])

    # 2. Configuração do Retriever (Recuperador)
    # Define a configuração para buscar no Neo4j.
    # O 'top_k' define quantos resultados vetoriais buscar.
    # O 'text_node_property' define qual propriedade será usada para a busca semântica.
    retriever = Neo4jVector.from_existing_index(
        embedding=embeddings,
        index_name="project_embeddings",
        search_type="vector",
        node_label="Project",
        text_node_property="description",
        embedding_node_property="embedding",
        graph=graph,
        retrieval_query="""
            RETURN node.description + '\nGerente: ' + coalesce(head([(node)-[:MANAGED_BY]->(m) | m.name]), 'N/A') AS text, score, {source: node.name} AS metadata
        """
    ).as_retriever(search_kwargs={"k": 2})

    # 3. Criação da Chain RAG
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    qa_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n--- Buscando Resposta para: '{question}' ---")
    resultado = qa_chain.invoke(question)
    
    print("\n[RESPOSTA FINAL DO LLM]")
    print(resultado)
    
    return resultado

# --- 4. Execução Principal ---

if __name__ == "__main__":
    # 1. Inicializa o banco de dados (Cria índice e carrega dados)
    # ATENÇÃO: Isso irá limpar os dados Project e Manager existentes
    try:
        graph.query("MATCH (n) WHERE n:Project OR n:Manager DETACH DELETE n")
        criar_grafo_conhecimento(graph, embeddings)
    except Exception as e:
        print(f"Erro ao conectar ou inicializar o Neo4j. Verifique o servidor e credenciais. Erro: {e}")
        exit()

    # 2. Pergunta que se beneficia da Busca Semântica + Estrutural
    # A pergunta não menciona 'Maria', mas sim 'mobile' e 'ux',
    # o que acionará o vetor do Projeto Gamma (Busca Semântica).
    # O Cypher, então, adicionará a relação de quem gerencia esse projeto.
    pergunta_1 = "Quem é o gerente do projeto que envolve aplicações móveis e foco em experiência do usuário?"
    busca_hibrida_rag(pergunta_1, graph, llm)

    # 3. Pergunta que se beneficia da busca Semântica em um tema genérico
    pergunta_2 = "Quais projetos envolvem arquitetura de microsserviços?"
    busca_hibrida_rag(pergunta_2, graph, llm)