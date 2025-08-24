from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_embedder():
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={
            "normalize_embeddings": True,   # unit-length vectors
            "batch_size": 32,               # safe default
            "show_progress_bar": False      # optional
        }
    )
