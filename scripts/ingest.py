#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Your project helpers
from app.embeddings import get_embedder


def find_files(source_dir: Path, exts: List[str]) -> List[Path]:
    files = []
    for p in source_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def load_documents(paths: List[Path]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        try:
            if p.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(p))
                docs.extend(loader.load())  # includes per-page metadata
            elif p.suffix.lower() in {".txt", ".md"}:
                loader = TextLoader(str(p), encoding="utf-8")
                docs.extend(loader.load())
        except Exception as e:
            print(f"[ingest] WARN: failed to load {p}: {e}")
    return docs


def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs: List[Document] = []
    for d in docs:
        for ch in splitter.split_text(d.page_content):
            split_docs.append(Document(page_content=ch, metadata=dict(d.metadata)))
    return split_docs


def ingest():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest files into Chroma (repo-root absolute paths).")
    parser.add_argument("--source_dir", required=True, help="Relative or absolute path to folder with PDFs/TXT/MD")
    parser.add_argument("--persist_dir", default=os.getenv("CHROMA_PERSIST_DIR") or ".chroma",
                        help="Chroma persist dir (default: .chroma at repo root)")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    parser.add_argument("--reset", action="store_true", help="Delete existing DB at persist_dir before ingest")
    args = parser.parse_args()

    # Resolve repo root
    REPO_ROOT = Path(__file__).resolve().parents[1]
    persist_dir = Path(args.persist_dir)
    if not persist_dir.is_absolute():
        persist_dir = (REPO_ROOT / persist_dir).resolve()
    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = (REPO_ROOT / source_dir).resolve()

    print(f"[ingest] Source dir   : {source_dir}")
    print(f"[ingest] Persist dir  : {persist_dir}")
    print(f"[ingest] Chunk size   : {args.chunk_size}, overlap={args.chunk_overlap}")
    print(f"[ingest] Reset        : {args.reset}")

    # Reset DB if asked
    if args.reset and persist_dir.exists():
        import shutil
        shutil.rmtree(persist_dir, ignore_errors=True)
        print(f"[ingest] Reset complete: removed {persist_dir}")

    # Load files
    paths = find_files(source_dir, exts=[".pdf", ".txt", ".md"])
    if not paths:
        print(f"[ingest] No supported files found in {source_dir}")
        return
    raw_docs = load_documents(paths)
    if not raw_docs:
        print("[ingest] No documents loaded.")
        return
    print(f"[ingest] Loaded {len(raw_docs)} documents (pages).")

    # Split
    split_docs = split_documents(raw_docs, args.chunk_size, args.chunk_overlap)
    print(f"[ingest] Split into {len(split_docs)} chunks.")

    # Embed + persist
    embedder = get_embedder()
    if args.reset or not persist_dir.exists():
        vs = Chroma.from_documents(
            documents=split_docs,
            embedding=embedder,
            persist_directory=str(persist_dir),
        )
        vs.persist()
        print(f"[ingest] Created new DB at {persist_dir}")
    else:
        vs = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embedder,
        )
        vs.add_documents(split_docs)
        vs.persist()
        print(f"[ingest] Appended to existing DB at {persist_dir}")

    # Quick stat
    try:
        print(f"[ingest] Document count: {vs._collection.count()}")
    except Exception:
        pass

    print("[ingest] ✅ Done.")


if __name__ == "__main__":
    ingest()
