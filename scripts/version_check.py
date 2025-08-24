# scripts/versions_check.py
import pkg_resources as pr
for p in ["langchain","langchain-community","huggingface-hub"]:
    try: print(f"{p}=={pr.get_distribution(p).version}")
    except: print(f"{p} not installed")
