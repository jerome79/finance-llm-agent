# app/ui_streamlit.py
# --- make project imports & .env loading robust (MUST be first) ---
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]  # <repo root>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")  # explicit .env load
# -------------------------------------------------------------------

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

from app.chains import build_retrieval_chain
from app.embeddings import get_embedder
from app.llm import get_llm
from app.utils import (
    resolve_persist_dir,
    chroma_doc_count,
    normalize_source_meta,
    split_documents,
    docs_to_texts_metas,
    make_ids_from_texts,
)

# ----------------------------- ingest helpers -----------------------------

def ingest_chunks(chunks: List[Document], persist_dir: Path, reset: bool = False) -> tuple[int, int]:
    """
    Ingest sanitized texts/metas into Chroma.
    Returns (added_count, new_total_estimate).
    """
    embedder = get_embedder()
    persist_dir.mkdir(parents=True, exist_ok=True)

    if reset:
        for item in persist_dir.glob("*"):
            if item.is_file():
                try:
                    item.unlink()
                except Exception:
                    pass
            else:
                import shutil
                try:
                    shutil.rmtree(item, ignore_errors=True)
                except Exception:
                    pass

    texts, metadatas, skipped = docs_to_texts_metas(chunks)

    added = 0
    if reset or not any(persist_dir.iterdir()):
        if texts:
            vs = Chroma.from_texts(
                texts=texts,
                embedding=embedder,
                persist_directory=str(persist_dir),
                metadatas=metadatas,
                ids=make_ids_from_texts(texts),  # dedupe by content hash
            )
            vs.persist()
            added = len(texts)
        else:
            added = 0
    else:
        vs = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embedder,
        )
        if texts:
            ids = make_ids_from_texts(texts)
            # Attach IDs as 'hash' too, for convenience
            for i, _ in enumerate(texts):
                metadatas[i] = dict(metadatas[i])
                metadatas[i]["hash"] = ids[i]
            try:
                vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                added = len(texts)
            except Exception:
                # Handle ID collisions by enforcing uniqueness
                uniq = {}
                for i, id_ in enumerate(ids):
                    if id_ not in uniq:
                        uniq[id_] = (texts[i], metadatas[i])
                if uniq:
                    vs.add_texts(
                        texts=[t for (t, m) in uniq.values()],
                        metadatas=[m for (t, m) in uniq.values()],
                        ids=list(uniq.keys()),
                    )
                    added = len(uniq)
        vs.persist()

    # New total estimate
    try:
        total = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embedder,
        )._collection.count()
    except Exception:
        total = -1

    if skipped:
        st.info(f"Skipped {skipped} empty/non-text chunks during ingest.")

    return added, total


# ----------------------------- UI ---------------------------------

st.set_page_config(page_title="Finance LLM Agent (RAG)", layout="wide")
st.title("💼 Finance LLM Agent (RAG)")
persist_dir_abs = resolve_persist_dir()

# Build the modern retrieval chain once (answer/context shape)
if "qa" not in st.session_state:
    st.session_state.qa = build_retrieval_chain()

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    st.caption("LLM & retrieval configured via `.env` and code.")
    # Show active LLM class
    try:
        llm = get_llm()
        st.text(f"Active LLM: {llm.__class__.__name__}")
    except Exception as e:
        st.warning(f"LLM init warning: {e}")

    # Diagnostics
    st.markdown("**Diagnostics**")
    st.code(
        f"cwd: {os.getcwd()}\n"
        f"PERSIST: {persist_dir_abs}\n"
        f"Doc count: {chroma_doc_count(persist_dir_abs)}",
        language="text"
    )

# ----------------------------- Ingest UI --------------------------------
st.markdown("### 📥 Ingest & Rebuild")

tab_upload, tab_rebuild = st.tabs(["Upload & Ingest (append)", "Rebuild index (reset folder)"])

with tab_upload:
    st.caption("Upload PDF/TXT/MD files — they will be appended to the existing vector DB.")
    uploaded = st.file_uploader("Upload files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk size", 200, 4000, 1000, 50)
    with col2:
        chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 150, 25)

    if st.button("Ingest uploaded files", disabled=not uploaded):
        docs: List[Document] = []
        for uf in uploaded:
            try:
                suffix = Path(uf.name).suffix.lower()
                if suffix == ".pdf":
                    tmp_path = persist_dir_abs / f"_tmp_{uf.name}"
                    tmp_path.write_bytes(uf.getbuffer())
                    pages = PyPDFLoader(str(tmp_path)).load()
                    tmp_path.unlink(missing_ok=True)
                    docs.extend(pages)
                else:
                    text = uf.getvalue().decode("utf-8", errors="ignore")
                    docs.append(Document(page_content=text, metadata={"source": uf.name, "page": 1}))
            except Exception as e:
                st.error(f"Failed to load {uf.name}: {e}")

        if not docs:
            st.warning("No documents could be read.")
        else:
            normalize_source_meta(docs)
            chunks = split_documents(docs, int(chunk_size), int(chunk_overlap))
            with st.spinner("Embedding & updating vector store…"):
                added, new_total = ingest_chunks(chunks, persist_dir_abs, reset=False)
            st.success(f"Ingested {added} chunks. New count ≈ {new_total if new_total >= 0 else 'N/A'}")

            # Rebuild the chain so retriever sees fresh data
            st.session_state.qa = build_retrieval_chain()
            st.info("Retrieval chain refreshed.")

with tab_rebuild:
    st.caption("Reset the DB and parse all files from a folder (relative to repo root).")
    default_folder = "data/seed_docs"
    source_dir_str = st.text_input("Folder to (re)ingest", value=default_folder)
    col1, col2 = st.columns(2)
    with col1:
        r_chunk_size = st.number_input("Chunk size ", 200, 4000, 1000, 50, key="r_cs")
    with col2:
        r_chunk_overlap = st.number_input("Chunk overlap ", 0, 1000, 150, 25, key="r_co")

    if st.button("🔁 Rebuild index (reset + ingest folder)"):
        source_dir = Path(source_dir_str)
        if not source_dir.is_absolute():
            source_dir = (ROOT / source_dir).resolve()
        if not source_dir.exists():
            st.error(f"Folder not found: {source_dir}")
        else:
            # Discover files
            paths = [p for p in source_dir.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}]
            if not paths:
                st.warning(f"No supported files in {source_dir}")
            else:
                # Load docs
                docs: List[Document] = []
                progress = st.progress(0.0, text="Loading documents…")
                for i, pth in enumerate(paths, 1):
                    try:
                        if pth.suffix.lower() == ".pdf":
                            docs.extend(PyPDFLoader(str(pth)).load())
                        else:
                            docs.extend(TextLoader(str(pth), encoding="utf-8").load())
                    except Exception as e:
                        st.error(f"Failed to load {pth.name}: {e}")
                    if i % 5 == 0:
                        progress.progress(min(1.0, i / len(paths)), text=f"Loading {i}/{len(paths)} files…")
                progress.progress(1.0, text="Loaded files.")

                normalize_source_meta(docs)
                chunks = split_documents(docs, int(r_chunk_size), int(r_chunk_overlap))
                with st.spinner("Resetting DB and embedding…"):
                    added, new_total = ingest_chunks(chunks, persist_dir_abs, reset=True)
                st.success(f"Rebuilt index with {added} chunks. New count ≈ {new_total if new_total >= 0 else 'N/A'}")

                # Rebuild the chain so retriever sees the fresh DB
                st.session_state.qa = build_retrieval_chain()
                st.info("Retrieval chain refreshed.")

st.divider()

# ----------------------------- Q&A UI -----------------------------------
st.markdown("### ❓ Ask a question")
question = st.text_input(
    "Ask about your ingested financial documents",
    placeholder="e.g., What are Apple’s main revenue risks mentioned in the 10-K?"
)

if question:
    with st.spinner("Thinking…"):
        try:
            # Modern chain expects {"input": ...}
            result = st.session_state.qa.invoke({"input": question})
        except Exception as e:
            st.error(f"Chain invocation error: {e}")
            st.stop()

    st.subheader("Answer")
    st.write(result.get("answer", ""))

    st.subheader("🔎 Sources")
    docs = result.get("context", []) or []
    if not docs:
        st.info("No sources returned. Try re-ingesting, adjusting chunking, or increasing k.")
    else:
        for i, d in enumerate(docs, start=1):
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("source") or meta.get("file_path") or meta.get("path") or "unknown"
            page = meta.get("page") or meta.get("Page") or meta.get("page_number")
            st.markdown(f"**{i}. {src}** — page {page if page is not None else 'N/A'}")
            with st.expander("View excerpt"):
                st.write((getattr(d, "page_content", "") or "")[:1200])