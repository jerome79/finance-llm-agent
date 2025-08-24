import os
from dotenv import load_dotenv
from app.llm import get_llm

def main():
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = (os.getenv("OLLAMA_MODEL") or os.getenv("MISTRAL_MODEL") or
             os.getenv("DEEPSEEK_MODEL") or os.getenv("HF_REPO_ID") or "unknown")
    print(f"🔍 Provider: {provider} | Model: {model}")
    llm = get_llm()
    print("✅ LLM initialized successfully.")
    q = "What is 2 + 2?"
    print(f"➡️  Query: {q}")
    ans = llm.invoke(q)
    print("📝 Response:", str(ans)[:500])

if __name__ == "__main__":
    main()
