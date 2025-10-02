# weather-rag-app


### Prerequisites

1. Install Ollama ( to run llm locally )
2. Pull the required LLM models

```bash

ollama pull embeddinggemma
ollama pull llama3.2:1b

```

### Tech Stack

1. Langchain for RAG pipeline backend
2. Chrome Vector Database to store embeddings
3. Streamlit for Frontend


### Ingest the weather CSV data into Chroma DB Vector Database

```bash

uv run main.py

```

### Run the Streamlit app Frontend

```bash

uv run streamlit run app.py

```
