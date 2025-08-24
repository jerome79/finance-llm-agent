from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from app.embeddings import get_embedder

# Repo root
ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """
    Load .env from repo root (idempotent).
    """
    load_dotenv(ROOT / ".env")


def resolve_persist_dir(env_var: str = "CHROMA_PERSIST_DIR") -> Path:
    """
    Resolve CHROMA_PERSIST_DIR to an absolute path.
    """
    env_dir = os.getenv(env_var) or ".chroma"
    p = Path(env_dir)
    if not p.is_absolute():
        p = (ROOT / p)
    return p.resolve()


def get_vectorstore(persist_dir: Path | str | None = None) -> Chroma:
    """
    Open a Chroma vector store with the configured embedder.
    """
    if persist_dir is None:
        persist_dir = resolve_persist_dir()
    p = Path(persist_dir).resolve()
    return Chroma(persist_directory=str(p), embedding_function=get_embedder())


def chroma_doc_count(persist_dir: Path | str | None = None) -> int:
    """
    Approximate document count from Chroma.
    """
    try:
        vs = get_vectorstore(persist_dir)
        return getattr(vs, "_collection").count()
    except Exception:
        try:
            got = get_vectorstore(persist_dir).get(limit=1)
            return len((got or {}).get("ids", []))
        except Exception:
            return 0


def normalize_source_meta(docs: List[Document]) -> None:
    """
    Normalize metadata['source'] to be relative to repo root (for display).
    """
    for d in docs:
        src = d.metadata.get("source")
        if not src:
            continue
        try:
            p = Path(src)
            if p.is_absolute():
                d.metadata["source"] = str(p.relative_to(ROOT)).replace("\\", "/")
            else:
                d.metadata["source"] = str(p).replace("\\", "/")
        except Exception:
            pass


def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Deterministic character splitter.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    out: List[Document] = []
    for d in docs:
        for ch in splitter.split_text(d.page_content):
            out.append(Document(page_content=ch, metadata=dict(d.metadata) if d.metadata else {}))
    return out


def docs_to_texts_metas(chunks: List[Document]) -> Tuple[list[str], list[dict], int]:
    """
    Convert Document[] -> (texts, metadatas) with strict sanitation.
    Returns: (texts, metadatas, skipped_count)
    - Only keeps items where page_content is a non-empty str.
    - Ensures metadata is JSON-safe dict.
    """
    texts: list[str] = []
    metas: list[dict] = []
    skipped = 0
    for d in chunks:
        # coerce metadata to JSON-safe
        md = dict(d.metadata or {})
        safe = {}
        for k, v in md.items():
            try:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    safe[k] = v
                else:
                    safe[k] = str(v)
            except Exception:
                safe[k] = str(v)

        # coerce content to clean str
        pc = getattr(d, "page_content", None)
        if isinstance(pc, bytes):
            try:
                pc = pc.decode("utf-8", errors="ignore")
            except Exception:
                pc = None
        if isinstance(pc, str):
            s = pc.strip()
            if s:
                texts.append(s)
                metas.append(safe)
            else:
                skipped += 1
        else:
            skipped += 1
    return texts, metas, skipped


def make_ids_from_texts(texts: list[str]) -> list[str]:
    """
    Deterministic IDs from normalized text (for dedupe at storage layer).
    """
    import hashlib

    out = []
    for t in texts:
        norm = " ".join(t.split())
        out.append(hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest())
    return out