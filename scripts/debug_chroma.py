# scripts/debug_chroma.py
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from app.embeddings import get_embedder

# Resolve repo root and load .env explicitly
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Resolve persist_dir to an absolute path (same rule as chains/ingest)
env_dir = os.getenv("CHROMA_PERSIST_DIR") or ".chroma"
persist_dir = (ROOT / env_dir) if not Path(env_dir).is_absolute() else Path(env_dir)
persist_dir = persist_dir.resolve()

print("Repo ROOT           :", ROOT)
print("CHROMA_PERSIST_DIR  :", os.getenv("CHROMA_PERSIST_DIR"))
print("Resolved persist_dir:", persist_dir)

# Show what's inside the folder
if not persist_dir.exists():
    print("⚠️ Persist dir does not exist.")
else:
    print("Dir contents:", [p.name for p in persist_dir.iterdir()])

# Open Chroma with the SAME embedder as ingest/runtime
vs = Chroma(
    persist_directory=str(persist_dir),
    embedding_function=get_embedder(),
)

# Count docs
try:
    print("Doc count (approx):", vs._collection.count())
except Exception as e:
    print("Count unavailable:", e)
    got = vs.get(limit=3)
    print("Doc ids sample:", (got or {}).get("ids", []))

# Try a direct similarity search
docs = vs.similarity_search("What is Uniswap?", k=3)
print("Top docs returned:", len(docs))
for i, d in enumerate(docs, 1):
    print(f"{i}.", d.metadata.get("source"), "page", d.metadata.get("page"))
