"""
main.py — Streamlit entry point for the RAG Q&A bot.

Responsibilities
----------------
1. Render the PDF upload widget and persist the file to a temp path keyed by
   its MD5 hash. Using the hash as the filename means @st.cache_resource
   invalidates only when the file actually changes, not on every page rerun.

2. Build the ingestion pipeline (load → chunk → embed → store) on first run
   or when a new PDF is uploaded. Subsequent runs with the same file load the
   existing ChromaDB store instantly.

3. Manage conversational state across Streamlit reruns:
   - st.session_state.messages    : plain dicts for rendering chat bubbles
   - st.session_state.lc_history  : LangChain HumanMessage / AIMessage objects
                                    passed to qa_chain for question condensing
                                    and grounded answering
   - st.session_state.last_file_hash : detects PDF changes and resets history

4. For each user question, call qa_chain.answer_question() which internally:
   - Condenses follow-up questions into standalone queries (resolves pronouns)
   - Retrieves the top-6 diverse chunks via MMR (k=6, fetch_k=40)
   - Generates a structured answer grounded in those chunks

5. Render the answer, any HTML tables referenced in source documents, and any
   images extracted from the PDF.

Chat history window
-------------------
Only the last 3 turns (6 messages) are passed to the chain. This caps the
token overhead added by history while still covering realistic follow-up depth.
Older messages are kept in st.session_state.messages for display only.
"""

from ingestion import loader
from logger import get_logger
from chunking import chunker
from embeddings import embeddings
from vectordb import cromadb
from retrivel import retrivel
from llm import llms
from prompts import prompt
from qa_chain import qa_chain
import streamlit as st
import os
import tempfile
import hashlib
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv(
    dotenv_path=r"/Users/prathod/Projects/Rag_based_q-a_boat_over_own_doc/config/.env"
)

logger = get_logger(__name__)

st.title("RAG Q&A over your documents")

# -------------------------------------------------------------------------
# PDF Upload
# -------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if not uploaded_file:
    st.info("Please upload a PDF to get started.")
    st.stop()

# Write the uploaded bytes to a stable temp path keyed by content hash.
# This ensures @st.cache_resource sees the same path for the same file and a
# new path (cache miss) when a different file is uploaded.
file_bytes = uploaded_file.read()
file_hash = hashlib.md5(file_bytes).hexdigest()
tmp_path = os.path.join(tempfile.gettempdir(), f"rag_upload_{file_hash}.pdf")
if not os.path.exists(tmp_path):
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)


# -------------------------------------------------------------------------
# Pipeline: load → embed → store  (cached per file hash)
# -------------------------------------------------------------------------
@st.cache_resource
def build_pipeline(pdf_path: str, _file_hash: str):
    """
    Build and return (retriever, llm) for the given PDF.

    Decorated with @st.cache_resource so the expensive ingestion and embedding
    work runs only once per unique file. The leading underscore on _file_hash
    tells Streamlit not to hash that argument (it's already baked into pdf_path).

    Steps:
      1. Check whether the ChromaDB store already matches this PDF via MD5 hash.
      2. If not, load and chunk the PDF (text + tables + images).
      3. Embed chunks and persist to ./chroma_db.
      4. Return an MMR retriever and the ChatOllama LLM.
    """
    logger.info("Starting RAG workflow processing...")

    model = embeddings.ollama_embeddings()

    from vectordb.cromadb import _get_file_hash
    import json

    metadata_file = "./chroma_db/source_metadata.json"
    needs_embedding = True
    if os.path.exists(metadata_file):
        with open(metadata_file) as f:
            saved = json.load(f)
        needs_embedding = saved.get("hash") != _get_file_hash(pdf_path)

    if needs_embedding:
        final_split_documents = loader.load_pdf_documents(pdf_path)
        logger.info(final_split_documents)
    else:
        final_split_documents = None

    vectordb = cromadb.store_chromadb(final_split_documents, model, pdf_path)
    llm = llms.get_llm()

    # Build the MMR retriever once and reuse it for all questions in this session.
    retriever = retrivel.build_retriever(vectordb)
    return retriever, llm


with st.spinner("Indexing document... (first run only)"):
    retriever, llm = build_pipeline(tmp_path, file_hash)


# -------------------------------------------------------------------------
# Session state: reset on new file, initialise on first load
# -------------------------------------------------------------------------
# When the user uploads a different PDF, wipe the conversation so old answers
# about a previous document don't bleed into the new session.
if st.session_state.get("last_file_hash") != file_hash:
    st.session_state.last_file_hash = file_hash
    # Plain dicts — used only for rendering chat bubbles in the UI.
    st.session_state.messages = []
    # LangChain message objects — passed to the chain for question condensing.
    # Keeping these separate from display messages avoids string-parsing overhead.
    st.session_state.lc_history = []

st.success("Ready! Ask a question below.")


# -------------------------------------------------------------------------
# Render prior conversation (so history is visible after Streamlit reruns)
# -------------------------------------------------------------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# -------------------------------------------------------------------------
# Handle new user question
# -------------------------------------------------------------------------
query = st.chat_input("Ask about your document")
if query:
    # Show the user's message immediately
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        # Pass the last 6 messages (3 turns) as LangChain objects.
        # The chain uses these to:
        #   a) condense vague follow-ups ("explain that more") into standalone queries
        #   b) ground the answer in prior context so continuity feels natural
        result = qa_chain.answer_question(
            query,
            retriever,
            llm,
            st.session_state.lc_history,   # already capped at 6 messages
        )

    answer = result["answer"]

    # Render and persist the assistant's reply
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Update the LangChain history window.
    # Append this turn and keep only the last 6 messages (3 full turns).
    # Older history is discarded to avoid ballooning the LLM's input length.
    st.session_state.lc_history.append(HumanMessage(content=query))
    st.session_state.lc_history.append(AIMessage(content=answer))
    st.session_state.lc_history = st.session_state.lc_history[-6:]

    # -------------------------------------------------------------------------
    # Render source material (tables and images) below the answer
    # -------------------------------------------------------------------------
    for doc in result["source_documents"]:
        # Tables extracted by unstructured carry their HTML structure in metadata.
        # Rendering as raw HTML preserves column alignment and cell borders.
        html_table = doc.metadata.get("text_as_html")
        if html_table:
            st.markdown(html_table, unsafe_allow_html=True)

        # Images saved to disk during ingestion are rendered inline if still present.
        img_path = doc.metadata.get("source_image")
        if img_path and os.path.exists(img_path):
            st.image(img_path, caption=img_path)
