from pathlib import Path

from langchain.schema import Document

from app.utils import (
    resolve_persist_dir,
    split_documents,
    make_ids_from_texts,
    normalize_source_meta,
)


def test_resolve_persist_dir_absolute(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path))
    p = resolve_persist_dir()
    assert p.is_absolute()
    assert p == tmp_path.resolve()


def test_split_documents_basic():
    docs = [Document(page_content="Alpha\n\nBeta\nGamma", metadata={"source": "sample.txt"})]
    chunks = split_documents(docs, chunk_size=8, chunk_overlap=0)
    # Should produce at least 2 chunks for short size
    assert len(chunks) >= 2
    for ch in chunks:
        assert isinstance(ch.page_content, str)
        assert ch.page_content.strip()


def test_make_ids_deterministic():
    texts = ["hello world", "  hello    world  ", "HELLO world"]
    ids = make_ids_from_texts(texts)
    # normalization collapses whitespace; case differences produce different ids
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


def test_normalize_source_meta_behaves(tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("x")
    d = Document(page_content="x", metadata={"source": str(p)})
    normalize_source_meta([d])
    # Should be a string and present
    assert "source" in d.metadata
    assert isinstance(d.metadata["source"], str)