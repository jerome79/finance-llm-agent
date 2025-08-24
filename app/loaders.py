from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import os

def load_pdfs_from_dir(source_dir: str):
    docs = []
    abs_path = os.path.abspath(source_dir)
    print(f"Absolute path: {abs_path}")
    print(f"Directory exists: {os.path.isdir(abs_path)}")

    pdf_files = list(Path(abs_path).glob("*.pdf")) + list(Path(abs_path).glob("*.PDF"))
    print(f"Found PDF files: {pdf_files}")

    for p in pdf_files:
        print(f"Processing file: {p}")
        if not os.access(p, os.R_OK):
            print(f"Cannot read file: {p}")
            continue
        try:
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {p}: {e}")

    return docs
