# app/ui_streamlit.py
# --- make project imports & .env loading robust (MUST be first) ---
import sys, os
from pathlib import Path
from typing import List, Tuple, Set
import hashlib

ROOT = Path(__file__).resolve().parents[1]              # <repo root>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")                              # explicit .env load
# -------------------------------------------------------------------

import streamlit as st
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.chains import build_retrieval_chain
from app.embeddings import get_embedder
from app.llm import get_llm


# ----------------------------- helpers ---------------------------------
def resolve_persist_dir() -> Path:
    env_dir = os.getenv("CHROMA_PERSIST_DIR") or ".chroma"
    p = Path(env_dir)
    if not p.is_absolute():
        p = (ROOT / p)
    return p.resolve()

def chroma_open(persist_dir: Path) -> Chroma:
    return Chroma(persist_directory=str(persist_dir), embedding_function=get_embedder())

def chroma_doc_count(persist_dir: Path) -> int:
    try:
        vs = chroma_open(persist_dir)
        return getattr(vs, "_collection").count()
    except Exception:
        try:
            got = chroma_open(persist_dir).get(limit=1)
            return len((got or {}).get("ids", []))
        except Exception:
            return 0

def sanitize_chunks(chunks: List[Document]) -> Tuple[List[Document], int]:
    """
    Keep only chunks whose page_content is a non-empty str.
    Returns (clean_chunks, skipped_count).
    """
    clean = []
    skipped = 0
    for d in chunks:
        pc = getattr(d, "page_content", None)
        if isinstance(pc, bytes):
            try:
                pc = pc.decode("utf-8", errors="ignore")
                d.page_content = pc
            except Exception:
                pc = None
        if isinstance(pc, str):
            pc = pc.strip()
            if pc:
                d.page_content = pc
                clean.append(d)
            else:
                skipped += 1
        else:
            skipped += 1
    return clean, skipped

def hash_text(s: str) -> str:
    # Normalize whitespace to reduce trivial diffs
    normalized = " ".join((s or "").split())
    return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()

def stamp_hash_metadata(chunks: List[Document]) -> Tuple[List[Document], Set[str]]:
    """Add a 'hash' field to metadata; return (chunks, set_of_hashes)."""
    hashes = set()
    for d in chunks:
        h = hash_text(d.page_content)
        d.metadata = dict(d.metadata or {})
        d.metadata["hash"] = h
        hashes.add(h)
    return chunks, hashes

def filter_new_by_hash(persist_dir: Path, chunks: List[Document], hashes: Set[str]) -> List[Document]:
    """Deduplicate against existing DB by hash (metadata['hash'])."""
    vs = chroma_open(persist_dir)
    existing_hashes: Set[str] = set()
    try:
        # Query existing by metadata filter (fast): where hash in hashes
        # Accessing underlying collection for where-in filter
        col = getattr(vs, "_collection", None)
        if col is not None and hashes:
            # Chroma supports where={"hash": {"$in": [...]}}
            got = col.get(where={"hash": {"$in": list(hashes)}}, include=["metadatas", "ids"])
            for m in (got.get("metadatas") or []):
                if isinstance(m, dict) and "hash" in m:
                    existing_hashes.add(m["hash"])
    except Exception:
        pass

    if not existing_hashes:
        return chunks

    filtered = [d for d in chunks if d.metadata.get("hash") not in existing_hashes]
    return filtered

def coerce_metadata_to_jsonable(d: Document) -> None:
    """
    Ensure metadata is a plain dict with JSON-serializable scalars.
    """
    md = dict(d.metadata or {})
    safe = {}
    for k, v in md.items():
        try:
            # Keep only simple types; stringify the rest
            if isinstance(v, (str, int, float, bool)) or v is None:
                safe[k] = v
            else:
                safe[k] = str(v)
        except Exception:
            safe[k] = str(v)
    d.metadata = safe

def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    out = []
    for d in docs:
        for ch in splitter.split_text(d.page_content):
            out.append(Document(page_content=ch, metadata=dict(d.metadata) if d.metadata else {}))
    return out

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
                try: item.unlink()
                except Exception: pass
            else:
                import shutil
                try: shutil.rmtree(item, ignore_errors=True)
                except Exception: pass

    # Convert once to validated lists
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
            for i, t in enumerate(texts):
                h = ids[i]
                metadatas[i] = dict(metadatas[i])
                metadatas[i]["hash"] = h
            try:
                vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                added = len(texts)
            except Exception:
                # If any IDs collide, enforce uniqueness map
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

def normalize_source_meta(docs: List[Document]) -> None:
    # Make metadata['source'] nice (relative to repo root for display)
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

def docs_to_texts_metas(chunks: List[Document]) -> tuple[list[str], list[dict], int]:
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
    """Deterministic IDs from normalized text (for dedupe at storage layer)."""
    out = []
    for t in texts:
        norm = " ".join(t.split())
        out.append(hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest())
    return out

# ----------------------------------------------------------------------


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
    st.caption("Upload PDF/TXT/MD files — they will be **appended** to the existing vector DB.")
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
    st.caption("**Reset** the DB and parse all files from a folder (relative to repo root).")
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
            paths = []
            for p in source_dir.rglob("*"):
                if p.suffix.lower() in {".pdf", ".txt", ".md"}:
                    paths.append(p)
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
                        progress.progress(min(1.0, i/len(paths)), text=f"Loading {i}/{len(paths)} files…")
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
            src  = meta.get("source") or meta.get("file_path") or meta.get("path") or "unknown"
            page = meta.get("page") or meta.get("Page") or meta.get("page_number")
            st.markdown(f"**{i}. {src}** — page {page if page is not None else 'N/A'}")
            with st.expander("View excerpt"):
                st.write((getattr(d, "page_content", "") or "")[:1200])
