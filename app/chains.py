import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from app.embeddings import get_embedder
from app.prompts import SYSTEM_PROMPT
from app.llm import get_llm
from pathlib import Path
from dotenv import load_dotenv

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def build_retrieval_chain(k: int = 4, persist_dir: str | None = None):
    """
    Modern retrieval chain using LangChain 0.2+.

    Usage:
        qa = build_retrieval_chain()
        res = qa.invoke({"input": "What is 2+2?"})
        print(res["answer"])
        print(res["context"])  # List[Document]

    Returns:
        Runnable chain with structured outputs:
          {
            "input": str,
            "answer": str,
            "context": List[Document]
          }
    """
    # --- Resolve persist dir (absolute path, consistent across envs) ---
    # Resolve repo root and load .env explicitly
    ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(ROOT / ".env")

    # Resolve persist_dir to an absolute path (same rule as chains/ingest)
    env_dir = os.getenv("CHROMA_PERSIST_DIR") or ".chroma"
    persist_dir = (ROOT / env_dir) if not Path(env_dir).is_absolute() else Path(env_dir)
    persist_dir = str(persist_dir.resolve())
    # --- Vector store & retriever ---
    vs = Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embedder()
    )
    retriever = vs.as_retriever(search_kwargs={"k": k})

    # --- LLM ---
    llm = get_llm()

    # --- Prompt (docs → {context}, question → {input}) ---
    template = SYSTEM_PROMPT + "\n\nContext:\n{context}\n\nQuestion: {input}\nAnswer:"
    prompt = PromptTemplate.from_template(template)

    # --- Combine retrieved docs into a single LLM call ---
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # --- Retrieval pipeline ---
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )
    return chain
