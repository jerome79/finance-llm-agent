import os

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        # Local models managed by Ollama (https://ollama.ai)
        from langchain_community.llms import Ollama
        model = os.getenv("OLLAMA_MODEL", "mistral")
        return Ollama(model=model, temperature=0.1)

    elif provider == "huggingface":
        # Direct HF Inference API client (no LC adapter)
        from huggingface_hub import InferenceClient
        repo_id = os.getenv("HF_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.2")
        # Try env var first, else fall back to cached login from `huggingface-cli login`
        token = (
                os.getenv("HUGGINGFACEHUB_API_TOKEN")
                or os.getenv("HF_API_TOKEN")
                or os.getenv("HUGGING_FACE_HUB_TOKEN")
                or None
        )

        # Do NOT raise if token is missing — let HF use the cached CLI login
        client = InferenceClient(model=repo_id, token=token)  # token can be None

        class TinyHFWrapper:
            def invoke(self, prompt: str):
                # Most instruct models accept a single prompt string
                return client.text_generation(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.1,
                    stream=False
                )

        return TinyHFWrapper()

    elif provider == "openai":
        # Standard OpenAI
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL", None)  # optional
        return ChatOpenAI(model=model, temperature=0.1, base_url=base_url)

    elif provider == "mistral":
        # Native Mistral API
        # pip install langchain-mistralai
        from langchain_mistralai import ChatMistralAI
        model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set")
        return ChatMistralAI(model=model, temperature=0.1, mistral_api_key=api_key)

    elif provider == "deepseek":
        # DeepSeek exposes an OpenAI-compatible API.
        # We reuse ChatOpenAI and point it to DeepSeek's base_url.
        from langchain_openai import ChatOpenAI
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # or deepseek-reasoner
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        # ChatOpenAI will pick up DEEPSEEK_API_KEY via OPENAI_API_KEY env unless we pass api_key explicitly.
        # We'll pass it to be explicit.
        return ChatOpenAI(model=model, temperature=0.1, base_url=base_url, api_key=api_key)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
