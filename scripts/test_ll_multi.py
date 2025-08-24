"""
Run the same prompts across multiple providers and summarize:
- latency
- output length
- a tiny 'sanity' check function (customizable)

Providers controlled via .env:
  ENABLE_OLLAMA=1
  ENABLE_MISTRAL=0
  ENABLE_DEEPSEEK=0
  ENABLE_HF=0

Prompts are in PROMPTS at the bottom; add more as you like.
"""

import os, time, textwrap
from dotenv import load_dotenv

# --- provider helpers ---------------------------------------------------------

def use(provider_flag: str) -> bool:
    val = os.getenv(provider_flag, "0").strip()
    return val in ("1", "true", "True", "YES", "yes")

def get_ollama():
    from langchain_community.llms import Ollama
    model = os.getenv("OLLAMA_MODEL", "phi3")
    return Ollama(model=model, temperature=0.1)

def get_mistral():
    from langchain_mistralai import ChatMistralAI
    key = os.getenv("MISTRAL_API_KEY")
    if not key: raise RuntimeError("MISTRAL_API_KEY not set")
    model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    return ChatMistralAI(model=model, temperature=0.1, mistral_api_key=key)

def get_deepseek():
    from langchain_openai import ChatOpenAI
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key: raise RuntimeError("DEEPSEEK_API_KEY not set")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return ChatOpenAI(model=model, temperature=0.1, base_url=base, api_key=key)

def get_huggingface_textgen():
    # TEXT-GENERATION models only (e.g., microsoft/Phi-3-mini-4k-instruct)
    from huggingface_hub import InferenceClient
    repo = os.getenv("HF_REPO_ID", "microsoft/Phi-3-mini-4k-instruct")
    tok  = (os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HF_API_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
            or None)
    client = InferenceClient(model=repo, token=tok)
    class HFWrapper:
        def invoke(self, prompt: str):
            return client.text_generation(prompt, max_new_tokens=512, temperature=0.1, stream=False)
    return HFWrapper()

# --- runner -------------------------------------------------------------------

def run_one(llm, prompt: str):
    t0 = time.time()
    out = llm.invoke(prompt)
    dt = time.time() - t0
    txt = str(out)
    return {"latency_sec": dt, "text": txt, "len": len(txt)}

def sanity_checks(prompt: str, text: str) -> dict:
    """Customize lightweight checks per prompt."""
    p = prompt.lower()
    t = text.lower()
    checks = {}
    if "2 + 2" in p or "2+2" in p:
        checks["contains_4"] = "4" in t
    if "risk" in p and "revenue" in p:
        checks["mentions_risk"] = "risk" in t
        checks["mentions_revenue"] = "revenue" in t
    return checks

def print_block(title: str, body: str):
    print(f"\n=== {title} ===")
    print(textwrap.shorten(body.strip().replace("\n\n","\n"), width=1200, placeholder=" [...]"))

def main():
    load_dotenv()

    providers = []
    if use("ENABLE_OLLAMA"):    providers.append(("ollama", get_ollama))
    if use("ENABLE_MISTRAL"):   providers.append(("mistral", get_mistral))
    if use("ENABLE_DEEPSEEK"):  providers.append(("deepseek", get_deepseek))
    if use("ENABLE_HF"):        providers.append(("huggingface", get_huggingface_textgen))

    if not providers:
        print("No providers enabled. Set e.g. ENABLE_OLLAMA=1 in .env")
        return

    PROMPTS = [
        "What is 2 + 2?",
        "In one sentence, explain revenue concentration risk for a publicly-traded company.",
        "List 3 key risk factors typically found in an SEC 10-K filing.",
    ]

    for prov_name, factory in providers:
        print(f"\n###############################")
        print(f"# Provider: {prov_name}")
        print(f"###############################")
        try:
            llm = factory()
            for pr in PROMPTS:
                res = run_one(llm, pr)
                checks = sanity_checks(pr, res["text"])
                print_block(f"{prov_name} | Prompt: {pr}", res["text"])
                print(f"Latency: {res['latency_sec']:.2f}s | Output len: {res['len']} | Checks: {checks}")
        except Exception as e:
            print(f"ERROR with provider '{prov_name}': {e}")

if __name__ == "__main__":
    main()
