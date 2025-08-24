# scripts/test_retriever_direct.py
from pathlib import Path
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from app.embeddings import get_embedder

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Resolve persist dir exactly as chains.py does
env_dir = os.getenv("CHROMA_PERSIST_DIR") or ".chroma"
persist_dir = (ROOT / env_dir) if not Path(env_dir).is_absolute() else Path(env_dir)
persist_dir = str(persist_dir.resolve())

print("Using persist_dir:", persist_dir)

# Build the SAME VS & retriever that the modern chain uses
vs = Chroma(persist_directory=persist_dir, embedding_function=get_embedder())
retriever = vs.as_retriever(search_kwargs={"k": 4})  # same k as chains.py

q = "What is Uniswap?"
docs = retriever.get_relevant_documents(q)
print(f"Retriever returned: {len(docs)} docs for: {q}")
for i, d in enumerate(docs, 1):
    print(f"{i}.", d.metadata.get("source"), "page", d.metadata.get("page"))
