import os
import chromadb
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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


def get_weather_rag_chain():
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

    contextualize_q_system_prompt = """
        Given a chat history and the latest user question
        which might reference context in the chat history, formulate a standalone question
        which can be understood without the chat history. Do NOT answer the question,
        just reformulate it if needed and otherwise return it as is.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """
        You are an expert weather analyst who analyze weather data in all number formats.
        Dates are in numerical date format ( like 2019-01-01 ). Analyze the user question 
        and convert the natural language queries in question to numerical format 
        and then use the retrieved context to answer it.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        keep the answer concise and clear
        {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain