import os, yaml
from pathlib import Path
from dotenv import load_dotenv
from app.chains import build_retrieval_chain

def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    qa = build_retrieval_chain()
    data = yaml.safe_load((root / "eval/golden_questions.yaml").read_text(encoding="utf-8"))

    total = 0
    covered = 0
    for case in data:
        q = case.get("query")
        hint = case.get("hint", "").lower()
        res = qa.invoke({"input": q})
        ctx = res.get("context", []) or []
        joined = " ".join([getattr(d, "page_content", "") or "" for d in ctx]).lower()
        hit = hint in joined if hint else (len(ctx) > 0)
        total += 1
        covered += int(hit)
        print(f"[Q] {q}\n  sources: {len(ctx)} | hint_hit: {hit}\n")

    print(f"Coverage: {covered}/{total} = {covered/total:.2%}")

if __name__ == "__main__":
    main()
