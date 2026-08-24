from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def splitting_text(documents: list[Document]) -> list[Document]:
    """
    Split a list of LangChain Documents into smaller, token-accurate chunks.

    Uses ``RecursiveCharacterTextSplitter.from_tiktoken_encoder`` so chunk
    boundaries are measured in **tokens** (the model's native unit) rather than
    characters.  Character-based splitting at 500 chars produces only ~100–125
    tokens per chunk — tiny relative to nomic-embed-text's 8 192-token context
    window and far more embedding calls than necessary.

    Parameters
    ----------
    documents : list[Document]
        Raw LangChain Document objects to be split.  Typically the text
        elements returned by ``ingestion/loader.py`` after filtering out
        Table and Image categories.

    Returns
    -------
    list[Document]
        Chunked documents with the same metadata as their parents.

    Chunking parameters
    -------------------
    chunk_size=800
        Maximum tokens per chunk (~600 words).  Large enough to keep a full
        paragraph — and its concluding sentence — in the same chunk, reducing
        mid-thought cuts that confuse retrieval.

    chunk_overlap=150
        Tokens shared between adjacent chunks (~8–10 sentences).  Ensures that
        a concept spanning a chunk boundary is readable from either side.  The
        previous value of 50 tokens (~2–3 sentences) was too thin for dense
        technical documents where a procedure step often references the one
        immediately above it.

    encoding_name="cl100k_base"
        The tiktoken encoding used by OpenAI's GPT-3/4 models and compatible
        with Ollama's nomic-embed-text.  Produces accurate token counts for
        both English prose and code.

    Note
    ----
    This function is also used for oversized table fragments in loader.py.
    In that case, chunk metadata is extended with ``chunk_index`` and
    ``is_table_fragment=True`` by the caller.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=800,
        chunk_overlap=150,
    )
    return text_splitter.split_documents(documents)
