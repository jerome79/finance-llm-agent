from typing import Optional

from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from app.prompts import SYSTEM_PROMPT
from app.llm import get_llm
from app.utils import load_env, resolve_persist_dir, get_vectorstore

def build_retrieval_chain(k: int = 4, persist_dir: Optional[str] = None):
    """
    Modern retrieval chain using LangChain 0.2+.

    Usage:
        qa = build_retrieval_chain()
        res = qa.invoke({"input": "What is 2+2?"})
        print(res["answer"])
        print(res["context"])  # List[Document]

    Returns a Runnable chain with outputs:
      {
        "input": str,
        "answer": str,
        "context": List[Document]
      }
    """
    # Ensure env is loaded and vectorstore resolved in a single place
    load_env()
    pdir = resolve_persist_dir() if persist_dir is None else persist_dir

    vs = get_vectorstore(pdir)
    retriever = vs.as_retriever(search_kwargs={"k": k})

    llm = get_llm()

    # Prompt (docs -> {context}, question -> {input})
    template = SYSTEM_PROMPT + "\n\nContext:\n{context}\n\nQuestion: {input}\nAnswer:"
    prompt = PromptTemplate.from_template(template)

    # Combine retrieved docs into a single LLM call
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    # Retrieval pipeline
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
    )
    return chain