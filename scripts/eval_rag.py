# Simple harness to measure retrieval precision/recall on a handcrafted eval set
# Fill 'eval_items' with {question, gold_contains} and check retrieved chunks

eval_items = [
    {"question": "What are the main revenue risks?", "gold_contains": ["Risk Factors", "revenue", "headwinds"]},
]

def evaluate(retriever):
    ok = 0
    for item in eval_items:
        docs = retriever.get_relevant_documents(item["question"])
        text = " ".join(d.page_content for d in docs).lower()
        hit = all(any(g.lower() in text for _ in [0]) for g in item["gold_contains"])
        ok += int(hit)
    print(f"Recall@k proxy: {ok}/{len(eval_items)}")

if __name__ == "__main__":
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from dotenv import load_dotenv, find_dotenv
    import os
    load_dotenv(find_dotenv())
    vs = Chroma(persist_directory=os.getenv("CHROMA_PERSIST_DIR", ".chroma"), embedding_function=OpenAIEmbeddings())
    evaluate(vs.as_retriever(search_kwargs={"k": 4}))
