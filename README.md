# RAG Q&A Bot over Your Own Documents

A local, privacy-first Retrieval-Augmented Generation (RAG) application that lets you upload a PDF and ask questions about it through a Streamlit chat interface. All processing runs entirely on your machine via [Ollama](https://ollama.com/) — no data is sent to the cloud.

---

## Features

| Feature | Detail |
|---|---|
| **PDF upload via UI** | Drag-and-drop or browse to upload any PDF directly in the browser |
| **Smart ingestion** | `unstructured` `hi_res` strategy extracts text, tables (as HTML), and images as separate elements |
| **Vision model** | Images inside PDFs are described by `qwen2.5vl:7b` and indexed as searchable text |
| **Safe chunking** | Text split with `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (**800 tokens / 150 overlap**, token-accurate); tables kept whole unless they exceed 1 000 tokens |
| **Local embeddings** | `nomic-embed-text` via Ollama — runs fully offline |
| **Persistent vector store** | ChromaDB persisted to disk; MD5 hash-gated so re-embedding only triggers on a changed file |
| **Local LLM** | `llama3.1` via Ollama for answer generation |
| **Conversational memory** | Last 3 turns of chat history kept in session state; follow-up questions are condensed into standalone queries before retrieval |
| **Table rendering** | Answers referencing tables render the original HTML table inline |

---

## Architecture

```
User uploads PDF (Streamlit UI)
        │
        ▼
  Ingestion — UnstructuredPDFLoader (hi_res strategy)
        │
        ├── Text elements  ──► RecursiveCharacterTextSplitter.from_tiktoken_encoder
        │                      chunk_size=800 tokens, chunk_overlap=150 tokens
        │
        ├── Table elements ──► kept as HTML; chunked only if > 1 000 tokens
        │
        └── Image elements ──► qwen2.5vl:7b description → text chunk
        │
        ▼
  Deduplication — remove duplicate chunks by content hash
        │
        ▼
  Embeddings — nomic-embed-text (Ollama)
        │
        ▼
  ChromaDB — persisted to ./chroma_db, hash-gated re-indexing
        │
        ▼
  Question condensing (if chat history exists)
  ─ LLM rewrites follow-up into standalone query
        │
        ▼
  MMR Retriever — k=6 chunks from fetch_k=40 candidates
        │
        ▼
  LLM — llama3.1 (Ollama / ChatOllama)
  ─ structured prompt with context + chat history
        │
        ▼
  Streamlit chat UI — answer + inline HTML tables + images
```

---

## Prerequisites

### System

| Requirement | Version | How to install |
|---|---|---|
| Python | ≥ 3.11 | Managed via `uv` |
| [Ollama](https://ollama.com/) | ≥ 0.3.6 | Download from ollama.com — must be running (`ollama serve`) |
| `poppler` | any | `brew install poppler` (macOS) / `apt install poppler-utils` (Linux) |
| `tesseract` | any | `brew install tesseract` (macOS) / `apt install tesseract-ocr` (Linux) |

> `poppler` and `tesseract` are required by `unstructured`'s `hi_res` strategy to detect and extract tables and images. Missing either will cause ingestion to fail.

---

### Ollama Models

Pull all three models before running the app:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
ollama pull qwen2.5vl:7b
```

| Model | Role in pipeline | Required | Download size | Min RAM / VRAM |
|---|---|---|---|---|
| `llama3.1` (8B) | Answer generation — reads retrieved chunks and produces the final answer | **Yes** | ~4.7 GB | 8 GB RAM (CPU) · 6 GB VRAM (GPU) |
| `nomic-embed-text` | Embeddings — converts every text chunk into a vector; also used at query time | **Yes** | ~274 MB | ~500 MB RAM — CPU-only is fine |
| `qwen2.5vl:7b` | Image captioning — describes images found inside the PDF and indexes them as text | Optional | ~4.4 GB | 6 GB VRAM (GPU recommended) |

**Notes**

- If `qwen2.5vl:7b` is not pulled or fails (insufficient VRAM), image elements are skipped with a placeholder — text and table extraction continues normally.
- `llama3.1` defaults to the 8B parameter variant. For better answers at the cost of more RAM, substitute `llama3.1:70b` in `llm/llms.py` (requires ~40 GB RAM).

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Rag_based_q-a_boat_over_own_doc

# 2. Install Python dependencies
uv sync
```

---

## Configuration

### `config/.env`

```env
HF_TOKEN=hf_your_token_here   # optional — suppresses Hugging Face rate-limit warnings
```

> The PDF path no longer needs to be set here — upload your PDF directly in the Streamlit UI.

---

## Running the App

```bash
cd src/q\&a_boat
uv run streamlit run main.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`).

### First run with a new PDF

1. Use the **Upload a PDF** widget at the top of the page.
2. The app ingests the document (text → chunks, tables → HTML, images → vision description).
3. Embeddings are generated and persisted to `src/q&a_boat/chroma_db/`.
4. Type a question in the chat input and hit Enter.

### Subsequent runs with the same PDF

The app detects that the file has not changed (MD5 hash comparison) and loads the existing vector store instantly — no re-embedding.

### Switching to a different PDF

Upload a new file. The pipeline re-runs automatically, the vector store is rebuilt, and **chat history is reset** so answers from the previous document don't bleed in.

---

## Project Structure

```
Rag_based_q-a_boat_over_own_doc/
├── config/
│   └── .env                        # HF_TOKEN and other secrets (not committed)
├── data/                           # Drop source PDFs here for reference
├── src/
│   └── q&a_boat/
│       ├── main.py                 # Streamlit entry point + pipeline orchestration
│       ├── logger.py               # Structured logging (file + console)
│       ├── ingestion/
│       │   └── loader.py           # PDF loading, element categorisation, token safety
│       ├── chunking/
│       │   └── chunker.py          # RecursiveCharacterTextSplitter wrapper
│       ├── embeddings/
│       │   └── embeddings.py       # OllamaEmbeddings (nomic-embed-text)
│       ├── vectordb/
│       │   └── cromadb.py          # ChromaDB store with hash-gated re-indexing
│       ├── retrivel/
│       │   └── retrivel.py         # Retriever utility (unused in main pipeline)
│       ├── llm/
│       │   └── llms.py             # ChatOllama (llama3.1)
│       ├── prompts/
│       │   └── prompt.py           # ChatPromptTemplate definition (reference)
│       ├── qa_chain/
│       │   └── qa_chain.py         # Conversational chain + vision model helper
│       └── chroma_db/              # Persisted vector store (auto-generated)
└── pyproject.toml                  # Project metadata and dependencies
```

---

## Module Reference

### `main.py`

Streamlit entry point. Responsibilities:
- Renders the file uploader and chat input.
- Saves the uploaded PDF to a temp file named by its MD5 hash (`rag_upload_<hash>.pdf`), ensuring `@st.cache_resource` invalidates only when the file actually changes.
- Calls `build_pipeline(pdf_path, file_hash)` which returns `(retriever, llm)`.
- Maintains two session-state lists:
  - `st.session_state.messages` — plain dicts for rendering chat bubbles.
  - `st.session_state.lc_history` — `HumanMessage`/`AIMessage` objects (capped at 6 = last 3 turns) passed to the chain for question condensing and grounded answering.
- Resets both lists automatically when a new PDF is uploaded.

### `ingestion/loader.py`

`load_pdf_documents(pdf_path: str) -> list[Document]`

Loads the PDF with `UnstructuredPDFLoader` using:
- `strategy="hi_res"` — detects tables and images as distinct elements.
- `infer_table_structure=True` — preserves table HTML in metadata.
- `extract_images_in_pdf=True` — saves embedded images to disk for the vision model.

After loading, elements are categorised:

| Category | Treatment |
|---|---|
| Text | Split by `chunker.splitting_text` (800 tokens / 150 overlap) |
| Table | Kept as HTML; chunked if > 1 000 tokens |
| Image | Described by `qa_chain.vision_model`; truncated at 1 000 tokens |

Returns the combined list of `Document` objects ready for embedding.

### `chunking/chunker.py`

`splitting_text(documents) -> list[Document]`

Wraps `RecursiveCharacterTextSplitter.from_tiktoken_encoder` with `chunk_size=800` **tokens** and `chunk_overlap=150` **tokens** (`cl100k_base` encoding).

> **Why 800 / 150?** The previous 512/50 setting frequently split paragraphs mid-sentence. 800 tokens (~600 words) keeps full paragraphs intact. A 150-token overlap (~8–10 sentences) ensures that any concept spanning a chunk boundary is readable from either side — important for dense technical documents where a procedure step may reference the one just above it.

### `embeddings/embeddings.py`

`ollama_embeddings() -> OllamaEmbeddings`

Returns an `OllamaEmbeddings` instance using the `nomic-embed-text` model.

### `vectordb/cromadb.py`

`store_chromadb(documents, model, pdf_path, persist_dir="./chroma_db") -> Chroma`

- If the vector store already exists and the PDF hash matches `source_metadata.json`, loads and returns the existing store.
- Otherwise, deduplicates documents, embeds in batches of 50, persists the store, and writes the new hash to `source_metadata.json`.

`_deduplicate(documents) -> list[Document]`

Removes documents with identical `page_content` (by MD5 hash). Headers and footers repeat across every page in `unstructured` output — deduplication prevents them from crowding out real content in retrieval results.

### `llm/llms.py`

`get_llm() -> ChatOllama`

Returns a `ChatOllama` instance with `model="llama3.1"`.

### `qa_chain/qa_chain.py`

#### `build_retriever(vectorstore) -> VectorStoreRetriever`

Returns an MMR retriever configured with:
- `k=6` — number of chunks sent to the LLM (up from 4). More chunks = richer context for multi-section answers.
- `fetch_k=40` — candidate pool before MMR re-ranking (up from 20). A larger pool reduces the chance of missing relevant content in long documents.

#### `answer_question(query, retriever, llm, chat_history) -> dict`

Three-step LCEL pipeline:

**Step 1 — Question condensing** *(skipped on first question)*

Follow-up questions contain pronouns that ChromaDB cannot resolve:

```
User: "What is the torque spec?"         → answered
User: "What happens if I exceed it?"     → "it" has no meaning in isolation
```

If `chat_history` is non-empty, a small LLM call rewrites the question:

```
"What happens if I exceed it?"
    ↓  condense
"What happens if the torque exceeds 45 Nm?"
    ↓  retrieve
ChromaDB finds the correct chunks
```

**Step 2 — Retrieval**

Uses the condensed (or original) query to fetch chunks via the MMR retriever.

**Step 3 — Answer generation**

A structured system prompt instructs the LLM to:
- Answer **only** from the retrieved context (no hallucination)
- Use bullets or numbered steps when appropriate
- Quote specific numbers and names from the document
- Explicitly say *"This information is not available in the provided document"* when the answer is absent

Chat history is injected via `MessagesPlaceholder` so the LLM can refer back to earlier turns naturally.

Returns `{"answer": str, "source_documents": list[Document]}`.

#### `vision_model(path) -> str`

Calls `ollama.chat` with `qwen2.5vl:7b` to generate a text description of an image file. Returns a placeholder string on any error.

### `logger.py`

`get_logger(name) -> Logger`

Returns a module-level logger writing to both console and a rotating log file under `src/q&a_boat/logs/`.

---

## Dependencies

Managed by `uv` via `pyproject.toml`:

| Package | Purpose |
|---|---|
| `langchain` | Core LLM orchestration |
| `langchain-ollama` | Ollama LLM and embedding adapters |
| `langchain-chroma` | ChromaDB vector store integration |
| `langchain-community` | `UnstructuredPDFLoader` and utilities |
| `langchain-core` | Document / prompt / message abstractions (stable API used throughout) |
| `unstructured[pdf]` | Advanced PDF parsing (hi_res, tables, images) |
| `pypdf`, `pdfminer-six` | PDF utilities used internally by unstructured |
| `streamlit` | Web UI |
| `tiktoken` | Token counting for safe chunking of tables |
| `chromadb` | Vector store (via langchain-chroma) |
| `watchdog` | Streamlit file watcher |

---

## Improvements Applied (and Why)

### 1. Conversational memory with question condensing (`qa_chain.py`, `main.py`)

**Problem:** Every question was treated independently. Follow-up questions like *"explain that more"* or *"what happens if I exceed it?"* retrieved irrelevant chunks because ChromaDB has no awareness of prior turns.

**Solution (Option B — condense before retrieval):**
1. If chat history exists, an extra LLM call rewrites the follow-up as a self-contained question.
2. The rewritten question is used for retrieval — ChromaDB finds the right content.
3. The original question and history are both included in the final answer prompt for natural continuity.

**Why not Option A (append history to prompt)?** Appending history to the retrieval query doesn't resolve pronouns — ChromaDB still searches for "it". Condensing first produces correct retrieval; history in the answer prompt then produces correct generation.

**Implementation:** Pure LCEL (`langchain_core` only). `langchain.chains.ConversationalRetrievalChain` was removed in LangChain 1.x.

---

### 2. More retrieved chunks (`qa_chain.py`)

**Before:** `k=4, fetch_k=20`
**After:** `k=6, fetch_k=40`

| Parameter | Before | After | Effect |
|---|---|---|---|
| `fetch_k` | 20 | 40 | Double the candidate pool; less chance of missing relevant content in a large PDF |
| `k` | 4 | 6 | 50% more context sent to the LLM; reduces incomplete answers when the answer spans multiple sections |

---

### 3. Larger chunk size with more overlap (`chunking/chunker.py`)

**Before:** `chunk_size=512 tokens, chunk_overlap=50 tokens`
**After:** `chunk_size=800 tokens, chunk_overlap=150 tokens`

512-token chunks frequently cut paragraphs mid-sentence, especially in dense technical documents. 50-token overlap (~2–3 sentences) was too thin to bridge a concept that spans a chunk boundary.

800 tokens keeps full paragraphs together. 150-token overlap (~8–10 sentences) ensures both sides of every boundary have enough shared context for retrieval to work correctly.

> **Note:** Changing chunk size requires re-embedding. The app detects this automatically via the MD5 hash — uploading the same PDF after this change triggers a fresh embed.

---

### 4. Structured answer prompt (`qa_chain.py`)

**Before:** `"Answer based on the context below in detail with figure if it is there."`

**After:** A structured system message with explicit rules:
- Answer **only** from the provided context
- Use bullets or numbered steps
- Quote specific details, numbers, names
- Admit when the answer is not in the document (instead of hallucinating)

---

### 5. Token-accurate chunking (`chunking/chunker.py`)

`from_tiktoken_encoder` splits by **tokens** (the model's native unit) rather than characters. Character-based splitting at 500 chars produces ~100–125 token chunks — tiny relative to `nomic-embed-text`'s 8192-token context window.

---

### 6. Deduplication before embedding (`vectordb/cromadb.py`)

`unstructured` emits every PDF element separately. Headers and footers repeat verbatim across every page, producing tens of identical chunks. `_deduplicate()` removes these by content hash before embedding so they don't crowd out real content in retrieval results.

---

### 7. Batched embedding (`vectordb/cromadb.py`)

Ollama's `/api/embed` endpoint times out when sent thousands of chunks at once. Embedding in batches of 50 keeps each request within reliable limits. Tune `_EMBED_BATCH_SIZE` at the top of `cromadb.py` (lower to 25 on slow machines, raise to 100 on fast ones).

---

## Known Limitations

- `hi_res` strategy requires `poppler` and `tesseract` on `PATH`; missing either causes ingestion to fail.
- Only one PDF is active at a time; the vector store is rebuilt when a different file is uploaded.
- `qwen2.5vl:7b` requires sufficient VRAM; images are skipped with a placeholder message if the model fails.
- ChromaDB is persisted relative to the working directory (`./chroma_db`), so run the app from `src/q&a_boat/`.
- Chat history is stored in `st.session_state` — it resets on browser refresh or tab close.
