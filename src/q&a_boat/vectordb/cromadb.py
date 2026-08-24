from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
import os
import hashlib
import json
from pathlib import Path

_EMBED_BATCH_SIZE = 50  # Ollama times out when sent too many chunks at once


def _get_file_hash(filepath: Path) -> str:
    """
    Compute and return the MD5 hex digest of a file's raw bytes.

    Used to detect whether the uploaded PDF has changed since the last
    ingestion run.  If the hash matches the one stored in
    ``source_metadata.json``, the existing ChromaDB store is reused and no
    re-embedding is performed.

    Parameters
    ----------
    filepath : Path or str
        Absolute path to the file to hash.

    Returns
    -------
    str
        32-character lowercase hex string (e.g. ``"d41d8cd98f00b204e9800998ecf8427e"``).

    Note
    ----
    MD5 is used here for speed, not cryptographic security.  The goal is fast
    change detection, not tamper resistance.
    """
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _deduplicate(documents: list) -> list:
    """
    Remove documents with identical ``page_content`` before embedding.

    ``unstructured`` emits every PDF element as a separate ``Document``.
    Page headers, footers, and repeated boilerplate text appear verbatim on
    every page, so a 50-page document can easily produce 50 identical chunks.
    These inflate the vector store, waste embedding time, and crowd out
    meaningful content in retrieval results.

    Deduplication is performed by comparing an MD5 hash of each document's
    ``page_content``.  Only the first occurrence of each unique content hash
    is kept; subsequent duplicates are discarded.

    Parameters
    ----------
    documents : list[Document]
        All documents returned by the ingestion pipeline before embedding.

    Returns
    -------
    list[Document]
        Deduplicated documents in their original order (first-seen wins).

    Side effects
    ------------
    Prints the number of removed duplicates to stdout if any were found.
    """
    seen = set()
    unique = []
    for doc in documents:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(doc)
    removed = len(documents) - len(unique)
    if removed:
        print(f"Deduplication removed {removed} duplicate chunk(s).")
    return unique


def store_chromadb(
    documents: list | None,
    model,
    pdf_path: Path,
    persist_dir: str = "./chroma_db",
) -> Chroma:
    """
    Persist document chunks to ChromaDB or load an existing store.

    This function is the single point of contact between the ingestion pipeline
    and the vector database.  It decides whether to embed from scratch or load
    the existing store based on an MD5 hash comparison.

    Parameters
    ----------
    documents : list[Document] or None
        Chunked documents to embed.  Pass ``None`` when the caller has already
        determined that the PDF is unchanged and no re-embedding is needed —
        the function will load the existing store directly.

    model : OllamaEmbeddings
        The embedding model returned by ``embeddings.ollama_embeddings()``.
        ChromaDB calls this internally to convert text chunks and queries into
        vectors.

    pdf_path : Path or str
        Path to the source PDF.  Its MD5 hash is compared against the stored
        hash in ``source_metadata.json`` to decide whether re-embedding is
        required.

    persist_dir : str, optional
        Directory where ChromaDB stores its SQLite database and binary index
        files.  Defaults to ``"./chroma_db"`` relative to the working directory
        (``src/q&a_boat/`` when running ``streamlit run main.py``).

    Returns
    -------
    Chroma
        A loaded (or freshly created) ChromaDB vector store, ready for
        similarity and MMR queries.

    Embedding strategy
    ------------------
    Documents are embedded in batches of ``_EMBED_BATCH_SIZE`` (default 50).

    - The first batch creates the ChromaDB collection via ``from_documents``.
    - Subsequent batches are added incrementally via ``add_documents``.

    Batching is necessary because Ollama's ``/api/embed`` endpoint times out
    when sent thousands of chunks in a single HTTP request.  Reduce
    ``_EMBED_BATCH_SIZE`` to 25 if timeouts persist; raise to 100 on faster
    machines.

    After embedding, the PDF's MD5 hash is written to
    ``<persist_dir>/source_metadata.json`` so future runs can skip re-embedding
    when the same file is uploaded again.

    Metadata handling
    -----------------
    ``filter_complex_metadata`` strips non-serialisable metadata fields
    (e.g. nested objects) before passing documents to ChromaDB, which only
    accepts scalar metadata values (strings, ints, floats, booleans).
    """
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

    print("New or changed PDF detected. Embedding now...")
    documents = _deduplicate(filter_complex_metadata(documents))
    total = len(documents)
    print(f"Embedding {total} chunks in batches of {_EMBED_BATCH_SIZE}...")

    # First batch creates the ChromaDB collection
    vectordb = Chroma.from_documents(
        documents=documents[:_EMBED_BATCH_SIZE],
        embedding=model,
        persist_directory=persist_dir,
    )

    # Remaining batches added incrementally
    for i in range(_EMBED_BATCH_SIZE, total, _EMBED_BATCH_SIZE):
        batch = documents[i: i + _EMBED_BATCH_SIZE]
        print(f"  {min(i + _EMBED_BATCH_SIZE, total)}/{total} chunks embedded...")
        vectordb.add_documents(batch)

    os.makedirs(persist_dir, exist_ok=True)
    with open(metadata_file, "w") as f:
        json.dump({"hash": current_hash, "source": str(pdf_path)}, f)
    return vectordb
