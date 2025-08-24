from pathlib import Path
from dotenv import load_dotenv

def test_import_chain():
    load_dotenv()
    from app.chains import build_retrieval_chain  # noqa: F401
    assert True
