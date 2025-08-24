from app.chains import build_retrieval_chain
import os
import sys, os
from pathlib import Path
from dotenv import load_dotenv
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

qa = build_retrieval_chain()
result = qa.invoke({"input": "What is uniswap?"})
answer = result.get("answer")
print(f"Answer : {answer}")
print("number of source : "+str(len(result["context"])))
for d in result["context"]:
    print("-", d.metadata.get("source", "unknown"))
