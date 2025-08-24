# Finance LLM Agent — Compliance-Aware RAG for Financial & DeFi Docs
An AI-powered Retrieval-Augmented Generation (RAG) Agent designed for financial analysis.
This agent lets you query financial documents (e.g., 10-Ks, reports, market data) and get precise, context-grounded answers backed by citations.

## 1) Business Problem

Financial analysts, compliance teams, and quant researchers waste hours searching long documents (SEC 10-Ks, earnings transcripts, DeFi whitepapers). Manual review is slow, error-prone, and often misses critical risks or insights.

## 2) Purpose of the App

Provide instant, cited answers and executive summaries from financial/DeFi documents, enriched with real-time market context and sentiment—so decisions are faster and better informed.

## 3) Who Uses It (Target Client)

* Equity/crypto research analysts
* Compliance & risk teams in fintech/asset management
* Portfolio managers & quant researchers
* Pro traders and advanced individual investors

## 4) How It’s Used

* Upload an SEC filing / earnings transcript / whitepaper or pick a sample.
* Ask natural-language questions (e.g., “Main revenue risks?”, “Debt covenants?”).
* Get cited answers (with page/source), optional sentiment, and market snapshot.

## 5) Benefits

* 50–80% faster document review.
* Reduced compliance risk via source-linked citations.
* Richer insights by combining filings with market and sentiment context.

# Architecture

## High Level

1) Ingestion: PDFs/URLs → clean → chunk → embed → Chroma vector store

2) Serve: User query → retrieve top chunks → synthesize answer → citations

3) Enrichment (optional): Market data (e.g., yfinance) for quick context  , FinBERT sentiment on retrieved passages or news

4) UI: Streamlit front-end for upload, questions, and viewing answers + sources

# user journey (text)

User → Streamlit UI → RAG Chain (LangChain)
                     ↘ Retriever (Chroma, embeddings)
Docs (PDF/URL) → Ingestion → Chunk → Embeddings → Chroma
Optional: Market Data (yfinance) / Sentiment (FinBERT)


# 🔌 LLM Providers (Local & Cloud)

This app runs locally by default using **Ollama** (free, no API limits), and can switch to cloud providers with only `.env` changes.

## Default: Ollama (local)
1. Install: https://ollama.com
2. Pull a model:
   ```bash
   ollama pull mistral
   
What is a RAG Agent?

RAG = Retrieval-Augmented Generation.

Instead of relying only on what the LLM “remembers” from training, RAG augments the model with your own financial data.

## How it works:
User Question ---> Retriever ---> [Relevant Docs from Chroma Vector DB]
                                   |
                                   v
                         [Docs + Question] ---> LLM ---> Answer


Retriever: searches a vector database (Chroma) for the most relevant financial document chunks.

LLM: generates an answer using both its internal knowledge and the retrieved text.

Agent: coordinates the process, ensuring the answer is grounded in facts.

## Why use RAG in Finance?

Financial analysis requires accuracy, traceability, and timeliness.

RAG ensures answers come from real documents (e.g., annual reports, filings) → no hallucinations.

You can trace answers back to the source (citations to specific document pages).

Keeps the agent up-to-date without retraining a model (just add new documents to Chroma).

## RAG vs Other Agent Technologies
Approach	        | Description                                       |	✅ Pros	                                   | ⚠️ Cons
Plain LLM	        | Ask GPT/LLM directly                              |	Easy to set up                             |	Hallucinations, outdated data, not finance-specific
Fine-tuned LLM	    | Train model on finance data                       |Tailored to finance, reusable                 |	Expensive, slow to update, needs lots of data
**RAG Agent** 	    | LLM + retriever on financial docs                 |	Accurate, up-to-date, explainable, cheaper |	Requires vector DB, quality depends on document chunking
Tool-based Agent    |	LLM can call APIs/tools (e.g., Bloomberg, SQL)  |	Real-time data, powerful                   |	Complex orchestration, depends on API availability
Hybrid (RAG + Tools)|	RAG for static docs + APIs for live data        |	Best of both worlds	                       |    Most complex to build/maintain