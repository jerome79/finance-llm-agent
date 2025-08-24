import os
from langchain_community.vectorstores import Chroma

def get_chroma(persist_dir: str | None = None):
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", ".chroma")
    os.makedirs(persist_dir, exist_ok=True)
    return Chroma(persist_directory=persist_dir, embedding_function=None)  # embedding supplied at call site
