import os
from pathlib import Path
from dotenv import load_dotenv

def test_env_template_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / ".env.example").exists(), ".env.example is missing"

def test_readme_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").exists(), "README.md is missing"
