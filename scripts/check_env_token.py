# scripts/check_env_token.py
import os, reprlib
tok = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_API_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Token present:", bool(tok))
if tok:
    print("Length:", len(tok))
    print("Starts with:", tok[:4])
    print("Preview:", reprlib.repr(tok))  # shows hidden whitespace/quotes
