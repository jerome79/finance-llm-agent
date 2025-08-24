import os
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedder():
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # All local, no API key required
    return HuggingFaceEmbeddings(model_name=model_name)
