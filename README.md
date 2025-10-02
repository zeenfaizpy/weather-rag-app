# weather-rag-app


### Prerequisites

1. Install Ollama ( to run llm locally )
2. Pull the required LLM models
3. Install [UV](https://docs.astral.sh/uv/)

### Setup

```bash

ollama pull embeddinggemma
ollama pull llama3.2:1b


git clone https://github.com/zeenfaizpy/weather-rag-app
cd weather-rag-app
uv sync
uv run init_db.py
uv run streamlit run app.py
```

### Tech Stack

1. Langchain for RAG pipeline backend
2. Chrome Vector Database to store embeddings
3. Streamlit for Frontend


### Ingest the weather CSV data into Chroma DB Vector Database

```bash

uv run init_db.py

```

### Run the Streamlit app Frontend

```bash

uv run streamlit run app.py

```
