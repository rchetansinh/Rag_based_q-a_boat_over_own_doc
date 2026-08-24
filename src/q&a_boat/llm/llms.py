from langchain_ollama import ChatOllama


def get_llm() -> ChatOllama:
    """
    Return a configured ChatOllama instance for answer generation.

    This is the language model that reads retrieved document chunks and
    produces the final answer shown to the user.  It is separate from the
    embedding model — embeddings convert text to vectors; the LLM reads
    text and writes text.

    Model: llama3.1 (8B parameter variant)
        - Pulled via : ``ollama pull llama3.1``
        - Min RAM    : 8 GB (CPU) · 6 GB VRAM (GPU)
        - Download   : ~4.7 GB

    Upgrading the model
        For better answer quality on complex technical documents, substitute a
        larger variant in the ``model`` argument:

        - ``llama3.1:70b``  — higher quality, requires ~40 GB RAM
        - ``mistral``       — faster, slightly lower quality
        - ``llama3.2``      — lighter, good for quick answers

    Returns
    -------
    ChatOllama
        Passed to ``qa_chain.answer_question()`` and used in the LCEL chain
        for both question condensing (Step 1) and answer generation (Step 3).
    """
    return ChatOllama(model="llama3.1")
