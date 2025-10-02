import os
import chromadb
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from components.data_loader import load_and_chunk_csv
from components.llm import get_embeddings, get_llm

WEATHER_FILE_PATH = "weather.csv"
VECTOR_DB_DIR = "vector_db/chroma_db"


def init_vector_store():
    """
    Initialize the Vector DB ( Chroma DB ) and ingest the chuncked csv docs
    """
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        print("Vector DB already exists")
    else:
        print("Initializing vector DB...")
        documents = load_and_chunk_csv(WEATHER_FILE_PATH)
        embeddings = get_embeddings()
        
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        
        Chroma.from_documents(documents, embeddings, client=client)
        print("Vector DB initialized and saved.")


def get_weather_qa_chain():
    """
    Creates a conversational retrieval QA chain for weather related questions.
    """
    if not os.path.exists(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR):
        raise RuntimeError("Vector DB not found")
    
    llm = get_llm()
    embeddings = get_embeddings()
    
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    
    vector_store = Chroma(
        embedding_function=embeddings,
        client=client
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        chain_type="stuff"
    )