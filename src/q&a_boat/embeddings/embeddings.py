from langchain_ollama import OllamaEmbeddings


def ollama_embeddings() -> OllamaEmbeddings:
    """
    Return a configured OllamaEmbeddings instance for the nomic-embed-text model.

    This embedding model converts text chunks into dense vectors (lists of
    numbers) that capture semantic meaning.  Vectors are stored in ChromaDB
    during indexing and reused at query time to find the chunks most similar
    to a user's question.

    Model: nomic-embed-text
        - Output dimension : 768
        - Max context      : 8 192 tokens
        - Size             : ~274 MB — runs comfortably on CPU
        - Pulled via       : ``ollama pull nomic-embed-text``

    Why nomic-embed-text?
        It is one of the highest-quality open-source embedding models that
        runs fully locally via Ollama.  It outperforms many larger models on
        retrieval benchmarks (MTEB) while staying small enough to run on a
        standard laptop.

    Note on num_ctx
        Setting ``num_ctx=8192`` on OllamaEmbeddings causes Ollama to call its
        ``/tokenize`` endpoint before each embed request.  On current Ollama
        versions this returns HTTP 400 and crashes the pipeline.  The parameter
        is intentionally omitted; Ollama's default context is sufficient for
        800-token chunks.

    Returns
    -------
    OllamaEmbeddings
        Ready-to-use embedding object passed to ChromaDB's ``embedding``
        parameter in ``vectordb/cromadb.py``.
    """
    return OllamaEmbeddings(model="nomic-embed-text")
