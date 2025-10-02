from langchain_ollama import OllamaEmbeddings, ChatOllama

EMBED_MODEL = "embeddinggemma"
LLM_MODEL = "llama3.2:1b"

llm = ChatOllama(model=LLM_MODEL)
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

def get_llm():
    return llm

def get_embeddings():
    return embeddings