def build_retriever(vectorstore):
    """
    Build and return an MMR retriever from the ChromaDB vector store.

    This is the component that, given a user's question, searches the vector
    store and returns the most relevant document chunks.  It is called once
    per session (inside ``build_pipeline`` in ``main.py``) and reused for
    every subsequent question.

    Search strategy: MMR (Maximal Marginal Relevance)
    --------------------------------------------------
    MMR runs in two stages:

    Stage 1 — Candidate fetch
        ChromaDB embeds the query and retrieves the ``fetch_k=40`` most
        similar chunks by cosine similarity.  These are the raw candidates.

    Stage 2 — Re-ranking
        From those 40 candidates, MMR selects the final ``k=6`` chunks by
        scoring each candidate on two criteria simultaneously:
          - Relevance : how similar the chunk is to the query
          - Diversity : how different the chunk is from already-selected chunks

        This prevents all 6 slots from being filled by near-identical passages
        from the same paragraph, which is a common failure mode of plain
        similarity search.

    Parameters
    ----------
    vectorstore : Chroma
        A ChromaDB vector store instance returned by ``vectordb/cromadb.py``.
        Must already contain embedded document chunks.

    Returns
    -------
    VectorStoreRetriever
        A LangChain retriever object.  Call ``retriever.invoke(query)`` to
        get a list of ``Document`` objects for any query string.

    Configuration
    -------------
    k=6
        Number of chunks returned to the LLM.  Increased from the default of
        4 to provide richer context, especially for answers that span multiple
        sections of a long technical document.

    fetch_k=40
        Candidate pool size before MMR re-ranking.  A larger pool reduces the
        probability of missing a relevant chunk in a large PDF (e.g. a 200-page
        manual where the answer might sit in a less obvious section).
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 40},
    )
