"""
qa_chain.py — Answer generation with conversational memory.

Flow for each user question:
  1. If prior chat history exists, a small LLM call condenses the follow-up
     into a fully self-contained question (resolves pronouns like "it", "that",
     "the previous step") so ChromaDB retrieval works correctly.
  2. The condensed (or original) question is used to retrieve the top-k chunks
     from ChromaDB using MMR search.
  3. The retrieved chunks are injected into a structured prompt together with
     the chat history, and the LLM generates the final answer.

Why LCEL instead of ConversationalRetrievalChain?
  langchain.chains was removed in LangChain 1.x. This implementation uses only
  langchain_core (stable) so it won't break on future upgrades.
"""

import ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


def answer_question(query, retriever, llm, chat_history):
    """
    Answer a user question using retrieved document context and chat history.

    Parameters
    ----------
    query        : the user's current question (raw text)
    retriever    : MMR retriever built by build_retriever()
    llm          : ChatOllama instance from llms.get_llm()
    chat_history : list of HumanMessage / AIMessage objects from session state.
                   Pass an empty list [] for the very first question.

    Returns
    -------
    dict with keys:
        "answer"           — the LLM's answer string
        "source_documents" — list of Document objects used as context
    """

    # ------------------------------------------------------------------
    # Step 1: Question condensing
    # ------------------------------------------------------------------
    # Follow-up questions often contain pronouns that reference earlier turns:
    #   "What happens if I exceed it?"  →  no standalone meaning without context
    #
    # We send the conversation history + the follow-up to the LLM and ask it
    # to rewrite it as a standalone question before hitting ChromaDB.
    # Without this step, retrieval would search for "it" and return irrelevant chunks.
    #
    # Skip this step on the very first question (no history) to save one LLM call.
    if chat_history:
        condense_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            (
                "human",
                "Rephrase the follow-up question above as a fully standalone question "
                "that contains all necessary context from the conversation. "
                "Return only the rephrased question, nothing else.",
            ),
        ])
        standalone_query = (condense_prompt | llm | StrOutputParser()).invoke(
            {"chat_history": chat_history, "input": query}
        )
    else:
        standalone_query = query

    # ------------------------------------------------------------------
    # Step 2: Retrieve relevant chunks
    # ------------------------------------------------------------------
    # Uses the condensed (or original) query so ChromaDB finds the right content.
    docs = retriever.invoke(standalone_query)
    context = "\n\n".join(d.page_content for d in docs)

    # ------------------------------------------------------------------
    # Step 3: Generate the answer
    # ------------------------------------------------------------------
    # The system message provides:
    #   - Role definition ("precise technical assistant")
    #   - Strict grounding rule (answer ONLY from the context below)
    #   - Output format guidance (bullets, numbered steps, quote numbers)
    #   - Fallback instruction when the answer is absent from the document
    #
    # Chat history is injected via MessagesPlaceholder so the LLM can refer
    # back to earlier turns (e.g. "as I mentioned earlier, ...") without
    # re-reading the full document context every turn.
    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise technical assistant. "
            "Use ONLY the context below to answer — do not use outside knowledge.\n\n"
            "Guidelines:\n"
            "- Give a thorough, structured answer (use bullet points or numbered steps when helpful).\n"
            "- Quote or paraphrase specific details, numbers, or names directly from the context.\n"
            "- If a table or figure is referenced in the context, describe its key data.\n"
            "- If the answer is not in the context, say exactly: "
            "'This information is not available in the provided document.'\n\n"
            f"Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    answer = (answer_prompt | llm | StrOutputParser()).invoke(
        {"chat_history": chat_history, "input": query}
    )

    return {"answer": answer, "source_documents": docs}


def vision_model(path):
    """
    Describe an image using a local vision model (qwen2.5vl:7b via Ollama).

    Called during PDF ingestion for every image element found by unstructured.
    The returned description is stored as a text chunk and embedded so image
    content becomes searchable by the RAG pipeline.

    Returns a placeholder string on any error so ingestion continues uninterrupted.
    """
    try:
        response = ollama.chat(
            model="qwen2.5vl:7b",
            messages=[{
                "role": "user",
                "content": "Describe this image in detail, including any text, "
                           "numbers, charts, or diagrams visible.",
                "images": [path],
            }]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"[Image at {path} could not be processed: {e}]"
