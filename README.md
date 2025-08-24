# 💼 Finance LLM Agent — Compliance-Aware RAG for Financial & DeFi Docs

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/langchain-0.2.x-green)](https://python.langchain.com)
[![Streamlit](https://img.shields.io/badge/streamlit-app-brightgreen)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Ask questions over SEC filings, earnings reports, and DeFi whitepapers — and get grounded, source-linked answers.**  
Built with **LangChain**, **Chroma**, **HuggingFace embeddings**, and **local LLMs** (Ollama by default).  

---

## ✨ Features

- **Modern RAG** with LangChain `create_retrieval_chain` → returns `{answer, context}`
- **Transparent answers**: every response is grounded in original docs
- **Streamlit UI**:
  - Upload PDFs / TXT / MD → append to knowledge base
  - Rebuild index from a folder
  - Duplicate-chunk deduper (hash-based)
  - Doc count badge + diagnostics
- **Local-first stack**:
  - HuggingFace sentence-transformers for embeddings
  - Ollama (Mistral by default) for inference → no API limits
- **Configurable** chunk size, overlap, retriever `k`

---

## 🧭 Architecture

```text
[User] ─ Streamlit UI
   │
   └──► Retrieval Chain (LangChain 0.2+)
           │
           ├──► Retriever → Chroma (local vector DB)
           │         └──► HuggingFace embeddings
           │
           └──► LLM (Ollama / Mistral / DeepSeek / OpenAI)
