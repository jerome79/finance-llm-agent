# scripts/hf_token_whoami.py
import os
from huggingface_hub import whoami
tok = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_API_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Token present:", bool(tok))
print("whoami:", whoami(token=tok))
