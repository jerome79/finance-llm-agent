import os
from huggingface_hub import InferenceClient

# Pick a text-generation model (Option A you chose)
repo  = os.getenv("HF_REPO_ID", "gpt2")
# Use env token if present; if not, the client will try your cached CLI login
token = (
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HF_API_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or None
)

print("Repo:", repo, "| Token via env:", bool(token))

# IMPORTANT: pass model here (constructor), then call without provider/model kwargs
client = InferenceClient(model="gpt2", token=None)  # use cached login

try:
    out = client.text_generation(
        "What is 2 + 2?",
        max_new_tokens=16,
        temperature=0.1,
        stream=False,
    )
    print("OK:", out)
except Exception as e:
    print("ERR:", repr(e))
from huggingface_hub import InferenceClient
