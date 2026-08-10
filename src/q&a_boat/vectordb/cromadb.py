from langchain_chroma import Chroma
import os
import hashlib
import json
from pathlib import Path


def _get_file_hash(filepath: Path):
    """Returns a hash of the file's content to detect changes."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def store_chromadb(documents, model, pdf_path: Path, persist_dir="./chroma_db"):
    metadata_file = os.path.join(persist_dir, "source_metadata.json")
    current_hash = _get_file_hash(pdf_path)

    store_exists = os.path.exists(persist_dir) and os.listdir(persist_dir)
    hash_matches = False

    if store_exists and os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            saved_metadata = json.load(f)
        hash_matches = saved_metadata.get("hash") == current_hash

    if store_exists and hash_matches:
        print("PDF unchanged. Loading existing vector store...")
        return Chroma(persist_directory=persist_dir, embedding_function=model)
    else:
        print("New or changed PDF detected. Embedding now...")
        vectordb = Chroma.from_documents(
            documents=documents, embedding=model, persist_directory=persist_dir
        )
        os.makedirs(persist_dir, exist_ok=True)
        with open(metadata_file, "w") as f:
            json.dump({"hash": current_hash, "source": pdf_path}, f)
        return vectordb
