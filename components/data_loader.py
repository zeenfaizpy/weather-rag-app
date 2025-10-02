import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document


def load_and_chunk_csv(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The csv file was not found at: {file_path}")

    print(f"Loading document from {file_path}...")
    loader = CSVLoader(file_path)
    documents = loader.load()

    cleaned_documents = []
    for doc in documents:
        cleaned_documents.append(Document(page_content=doc.page_content, metadata=doc.metadata))

    return cleaned_documents