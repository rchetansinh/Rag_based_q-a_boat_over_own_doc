from langchain_core.prompts import ChatPromptTemplate


def get_prompt() -> ChatPromptTemplate:
    """
    Return a basic RAG prompt template (reference / utility).

    This template is provided as a reusable starting point.  The active
    pipeline in ``qa_chain/qa_chain.py`` builds its prompts inline using
    ``ChatPromptTemplate.from_messages`` so that chat history can be injected
    via ``MessagesPlaceholder`` — something a single static template cannot do.

    Template variables
    ------------------
    {context}
        The concatenated text of retrieved document chunks, assembled by
        the retriever step in ``qa_chain.answer_question``.

    {question}
        The user's question (raw or condensed).

    Behaviour
    ---------
    - Instructs the LLM to answer **only** from the provided context.
    - Tells the LLM to say "I don't know" when the answer is absent, which
      prevents hallucination on out-of-scope questions.

    Returns
    -------
    ChatPromptTemplate
        A LangChain prompt template ready to be chained with an LLM via
        ``prompt | llm`` in LCEL syntax.
    """
    prompt = ChatPromptTemplate.from_template(
        """
    Answer the question using only the context below.
    If the answer isn't in the context, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """
    )
    return prompt
