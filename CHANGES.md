# Project Changes & Decision Log

This document captures every change made to the RAG Q&A project, why it was required, what problem it solved, and what was learned. Use this as a reference before making future modifications.

---

## 1. PDF Upload via Streamlit UI

**File changed:** `src/q&a_boat/main.py`

### Problem
The PDF path was hardcoded in `config/.env` as `PDF_FILE`. Every time you wanted to query a different document, you had to stop the app, edit the `.env` file, and restart. There was no way for an end user to load their own document without touching configuration files.

### What was changed
- Added `st.file_uploader("Upload a PDF", type="pdf")` at the top of the Streamlit UI.
- The uploaded file bytes are saved to a temp file named `rag_upload_<md5hash>.pdf` using the file's MD5 hash as part of the filename.
- `st.stop()` is called if no file is uploaded, so the rest of the app does not render until a PDF is provided.
- `build_pipeline(pdf_path, file_hash)` now accepts the temp path and hash as arguments. The hash is passed as a Streamlit cache key (prefixed `_` to skip Streamlit's own hashing).

### Why MD5 hash in the filename
Uploading the same PDF twice reuses the cached vector store with zero re-embedding. Uploading a different PDF produces a different hash, so `@st.cache_resource` treats it as a new call and re-runs the pipeline automatically.

### Impact on `loader.py`
`load_pdf_documents()` was changed to accept `pdf_path: str = None` as a parameter instead of always reading from the `PDF_FILE` environment variable. The env var fallback is kept for backward compatibility.

---

## 2. HF_TOKEN Added to `.env`

**File changed:** `config/.env`

### Problem
At startup, transitive dependencies (`transformers`, pulled in by `unstructured`) make unauthenticated requests to the Hugging Face Hub to fetch model configs and tokenizer files. Without a token, Hugging Face rate-limits these requests and prints:

```
Warning: You are sending unauthenticated requests to the HF Hub.
```

### What was changed
Added `HF_TOKEN=` to `config/.env`. The app loads this file via `load_dotenv` at startup, so HuggingFace libraries pick it up automatically.

### How to get the token
1. Create a free account at huggingface.co
2. Go to Settings → Access Tokens → New token (type: read)
3. Paste the token after `HF_TOKEN=` in the `.env` file

The token is **optional** — the app works without it but may be slower on first run due to rate limiting.

---

## 3. Token-Accurate Chunking

**File changed:** `src/q&a_boat/chunking/chunker.py`

### Problem
The original splitter used:

```python
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
```

`RecursiveCharacterTextSplitter` measures `chunk_size` in **characters** by default (using Python's `len()`), not tokens. 500 characters is approximately 100–125 tokens for English text. This meant:

- Chunks were tiny and semantically incomplete.
- `nomic-embed-text` supports up to 8192 tokens — chunks were using less than 2% of that capacity.
- Far more embedding calls than necessary (one call per tiny chunk instead of per meaningful paragraph).
- The README incorrectly described these as "500 tokens" — a documentation bug that would confuse future developers.

### What was changed

```python
RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=512,
    chunk_overlap=50,
)
```

`from_tiktoken_encoder` uses `tiktoken` to count tokens, so `chunk_size=512` means exactly 512 tokens. This produces semantically richer chunks that better represent complete thoughts.

### Why `cl100k_base`
This is the same encoding used by the `tiktoken` token counter already present in `loader.py` for table safety checks — keeping the tokenizer consistent across the pipeline.

### Why 50 overlap instead of 100
At character level, 100/500 = 20% overlap. At token level, 50/512 ≈ 10% overlap, which is sufficient to preserve cross-chunk context without duplicating too much content.

---

## 4. `num_ctx=8192` Attempt — Reverted

**File changed:** `src/q&a_boat/embeddings/embeddings.py`

### What was attempted

```python
OllamaEmbeddings(model="nomic-embed-text", num_ctx=8192)
```

The intent was to explicitly unlock `nomic-embed-text`'s full 8192-token context window, preventing Ollama from silently truncating chunks longer than its internal default (potentially 2048 tokens on some versions).

### Why it was reverted

Setting `num_ctx` on `OllamaEmbeddings` caused Ollama to call its `/tokenize` REST endpoint to validate input length before embedding. On the current Ollama version in use, this endpoint returned HTTP 400 with a `connection reset by peer` error, crashing the entire pipeline:

```
ResponseError: Post "http://127.0.0.1:56603/tokenize": read tcp ...
connection reset by peer (status code: 400)
```

### Current state

```python
OllamaEmbeddings(model="nomic-embed-text")
```

Ollama's default context window is sufficient for 512-token chunks. Do not add `num_ctx` back without first verifying your Ollama version supports it on the embed API.

---

## 5. Deduplication Before Embedding

**File changed:** `src/q&a_boat/vectordb/cromadb.py`

### Problem
`unstructured` emits every detected PDF element as a separate `Document` object. Page headers, footers, watermarks, and section titles repeat verbatim across every page. A 100-page document could easily produce 100 identical "Page 1 of 100" chunks, all of which would be:

- Embedded (wasted compute time).
- Stored in ChromaDB (inflated vector store).
- Retrieved by similarity search (crowd out actual content, reducing answer quality).

In practice, deduplication removed **1618 duplicate chunks** from one test document.

### What was changed
Added `_deduplicate(documents)` in `cromadb.py`:

```python
def _deduplicate(documents):
    seen = set()
    unique = []
    for doc in documents:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(doc)
    return unique
```

Called before `filter_complex_metadata` and before embedding:

```python
documents = _deduplicate(filter_complex_metadata(documents))
```

### How it works
Each document's `page_content` is hashed with MD5. The first occurrence of each unique hash is kept; all subsequent duplicates are dropped. This is a pure in-memory operation — fast even for large document sets.

---

## 6. Batched Embedding

**File changed:** `src/q&a_boat/vectordb/cromadb.py`

### Problem
The original code passed all documents to ChromaDB in a single call:

```python
Chroma.from_documents(documents=all_documents, embedding=model, ...)
```

`langchain_ollama`'s `OllamaEmbeddings.embed_documents()` forwards the entire list to Ollama's `/api/embed` endpoint in one HTTP request. When the list is large (hundreds or thousands of chunks), Ollama times out trying to process the batch, returning:

```
ResponseError: Post "http://127.0.0.1:57955/tokenize": 
operation timed out (status code: 400)
```

### What was changed

Documents are now embedded in batches of 50:

```python
_EMBED_BATCH_SIZE = 50

# First batch creates the ChromaDB collection
vectordb = Chroma.from_documents(
    documents=documents[:_EMBED_BATCH_SIZE],
    embedding=model,
    persist_directory=persist_dir,
)

# Remaining batches added incrementally
for i in range(_EMBED_BATCH_SIZE, total, _EMBED_BATCH_SIZE):
    batch = documents[i : i + _EMBED_BATCH_SIZE]
    print(f"  {min(i + _EMBED_BATCH_SIZE, total)}/{total} chunks embedded...")
    vectordb.add_documents(batch)
```

### Why not asyncio/TaskGroup for this
`asyncio` is for I/O-bound concurrency. While Ollama calls are I/O-bound, Ollama itself processes embedding requests serially on the same hardware — concurrent HTTP requests would just queue at the server. Batching is simpler, reliable, and has no concurrency overhead.

### Tuning
`_EMBED_BATCH_SIZE = 50` is set at the top of `cromadb.py`.
- Reduce to **25** if timeouts still occur.
- Increase to **100** if your Ollama version handles larger payloads reliably.

---

## 7. MMR Retrieval

**File changed:** `src/q&a_boat/qa_chain/qa_chain.py`

### Problem
The original retrieval used plain cosine similarity:

```python
docs = vectorstore.similarity_search(query, k=4)
```

Cosine similarity returns the `k` chunks with the smallest angular distance to the query vector. In practice, for a dense technical document, this often returns 4 near-identical chunks from the same paragraph — giving the LLM highly redundant context that wastes its context window and produces narrow answers.

### What was changed

```python
docs = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=20)
```

MMR (Maximum Marginal Relevance) fetches `fetch_k=20` candidate chunks by similarity, then iteratively selects the next chunk that maximises both relevance to the query and diversity relative to already-selected chunks.

### Result
The 4 chunks returned span broader sections of the document, giving the LLM more complete context and producing more comprehensive answers — especially important for multi-part questions.

---

## 8. Why Vision Model Is Slow (No Change Made)

**Files discussed:** `src/q&a_boat/ingestion/loader.py`, `src/q&a_boat/qa_chain/qa_chain.py`

### Observation
Ingestion takes a long time. The breakdown is:

| Step | Speed | Reason |
|---|---|---|
| `hi_res` OCR (Tesseract) | Slow | CPU-intensive, runs on every page |
| `llama3.2-vision` per image | Very slow | Sequential 11B model calls, 10–60s each |
| Chunking | Fast | Pure Python string ops, milliseconds |
| Embedding (batched) | Moderate | Network calls to Ollama, now batched |

### Why no change was made yet
Ingestion is a **one-time cost per PDF** — the hash-gated caching in `cromadb.py` means the second run with the same file loads from ChromaDB instantly. The slowness is only felt once.

### How to skip the vision model (future option)
Add `process_images: bool = True` parameter to `load_pdf_documents()` and pass `extract_images_in_pdf=process_images` to `UnstructuredPDFLoader`. Expose it as a Streamlit checkbox so users can opt out for text-heavy documents.

---

---

## Built-in & Library Functions — Deep Reference

This section documents every external function and class used in the project. Each entry explains what the function does internally, why it was chosen, what parameters matter, and what to watch out for.

---

### `RecursiveCharacterTextSplitter` — `langchain_text_splitters`

**Used in:** `chunking/chunker.py`

**What it does:**
Splits a list of `Document` objects into smaller chunks by recursively trying a list of separators. It starts with the largest separator (e.g. `"\n\n"` for paragraphs), and if the resulting chunk is still too large, it moves to the next separator (e.g. `"\n"`, then `" "`, then `""`). This preserves natural text boundaries as much as possible.

**Two modes:**

| Constructor | Measures `chunk_size` by | Use when |
|---|---|---|
| `RecursiveCharacterTextSplitter(chunk_size=N)` | Characters (`len()`) | You want fast splitting without a tokenizer |
| `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=N)` | Tokens (via tiktoken) | Your embedding model has a token limit |

**We use `from_tiktoken_encoder` because:**
The embedding model (`nomic-embed-text`) has a context window measured in tokens, not characters. Using character-based splitting can produce chunks that either exceed the model's token limit (if text is dense) or are far too small (if text is sparse). Token-based splitting guarantees the chunk fits within model limits.

**Key parameters:**
- `encoding_name="cl100k_base"` — the tiktoken encoding. `cl100k_base` is used by GPT-4 and is a reasonable default for token counting even for non-OpenAI models.
- `chunk_size=512` — maximum tokens per chunk.
- `chunk_overlap=50` — number of tokens repeated at the start of each new chunk to preserve context across boundaries.

**`split_documents(documents)`:**
Takes a list of LangChain `Document` objects and returns a new list of `Document` objects. Each output document preserves the `metadata` from its source document (page number, source file, etc.).

---

### `OllamaEmbeddings` — `langchain_ollama`

**Used in:** `embeddings/embeddings.py`

**What it does:**
A LangChain embedding adapter for Ollama. Wraps the Ollama REST API (`/api/embed`) to convert text strings into dense float vectors. These vectors represent the semantic meaning of the text in a high-dimensional space.

**How it works internally:**
1. Accepts a list of text strings via `embed_documents(texts)`.
2. Makes a POST request to `http://localhost:11434/api/embed` with the model name and texts.
3. Ollama loads the model into memory (or uses a cached copy) and returns a list of float arrays.
4. Returns `list[list[float]]` — one vector per input text.

**Key parameters:**
- `model="nomic-embed-text"` — the embedding model to use. Must be pulled with `ollama pull nomic-embed-text` first.
- `num_ctx` — **do not set** on this version. Setting it triggers Ollama's `/tokenize` endpoint which returns HTTP 400 on current versions (see Change #4).

**`embed_documents(texts: list[str]) -> list[list[float]]`:**
Called by ChromaDB internally. You do not call this directly — pass the `OllamaEmbeddings` instance to `Chroma.from_documents(embedding=model)`.

**`embed_query(text: str) -> list[float]`:**
Called at query time to convert the user's question into a vector for similarity/MMR search.

---

### `UnstructuredPDFLoader` — `langchain_community.document_loaders`

**Used in:** `ingestion/loader.py`

**What it does:**
A LangChain wrapper around the `unstructured` library for PDF parsing. Unlike `PyPDFLoader` (which extracts plain text page by page), `UnstructuredPDFLoader` detects and categorises structural elements: paragraphs, titles, tables, images, headers, footers, and more.

**Key parameters:**

| Parameter | Value used | What it does |
|---|---|---|
| `mode="elements"` | `"elements"` | Returns one `Document` per detected element (not one per page) |
| `strategy="hi_res"` | `"hi_res"` | Uses Tesseract OCR + poppler for high-accuracy detection. Slower but finds tables and images. Requires `poppler` and `tesseract` installed |
| `extract_images_in_pdf` | `True` | Saves images found in the PDF to disk as `.jpg`/`.png` files |
| `infer_table_structure` | `True` | Converts detected tables to HTML (`<table>...</table>`) and stores in `doc.metadata["text_as_html"]` |

**Element categories returned in `doc.metadata["category"]`:**
- `"NarrativeText"` / `"Text"` — regular paragraphs
- `"Table"` — tabular data; HTML version in `metadata["text_as_html"]`
- `"Image"` — embedded images; file path in `metadata["image_path"]`
- `"Title"` — section headings
- `"Header"` / `"Footer"` — page headers/footers (these are the main source of duplicate chunks)
- `"ListItem"` — bullet/numbered list items

**`.load()` method:**
Returns `list[Document]`. Each document has `page_content` (the text of that element) and `metadata` (page number, coordinates, category, source file, etc.).

**Why `hi_res` is slow:**
Every page is rendered at high DPI and passed through Tesseract's OCR engine. This is necessary to detect table boundaries and image regions, but it is CPU-intensive. A 100-page document can take several minutes.

---

### `tiktoken.get_encoding` — `tiktoken`

**Used in:** `ingestion/loader.py`

**What it does:**
Returns a `Encoding` object for the specified tokenisation scheme. A tokenizer converts text into a sequence of integer token IDs — the fundamental unit used by transformer models.

```python
tokenizer = tiktoken.get_encoding("cl100k_base")
tokens = tokenizer.encode("Hello world")  # → [9906, 1917]
```

**`cl100k_base`:**
The tokenisation vocabulary used by GPT-3.5 and GPT-4. It is a byte-pair encoding (BPE) scheme with ~100,000 vocabulary entries. It handles Unicode, code, and multilingual text well. Used here as a proxy token counter for `nomic-embed-text` — the exact token counts may differ slightly but are close enough for safety checking.

**`tokenizer.encode(text, disallowed_special=())`:**
- `disallowed_special=()` — by default tiktoken raises an error on special tokens like `<|endoftext|>`. Passing an empty tuple disables this check, which is necessary when processing arbitrary PDF content that might contain these strings accidentally.
- Returns `list[int]` — the list of token IDs.
- `len(tokenizer.encode(text))` gives the exact token count used in `get_token_count()`.

---

### `Chroma` and `Chroma.from_documents` — `langchain_chroma`

**Used in:** `vectordb/cromadb.py`

**What it does:**
ChromaDB is an embedded vector database — it stores text chunks alongside their embedding vectors and enables fast approximate nearest-neighbour search.

**`Chroma.from_documents(documents, embedding, persist_directory)`:**
- Takes a list of `Document` objects and an embedding function.
- Calls `embedding.embed_documents([doc.page_content for doc in documents])` to get vectors.
- Stores both the text and vectors in a ChromaDB collection on disk at `persist_directory`.
- Returns a `Chroma` instance ready for querying.
- **Important:** This method sends all documents to the embedding function at once. For large document sets this causes Ollama to time out — which is why we batch manually (see Change #6).

**`Chroma(persist_directory, embedding_function)`:**
- Loads an existing ChromaDB collection from disk without re-embedding.
- Used on subsequent runs when the PDF hash matches — instant load, no Ollama calls.

**`vectordb.add_documents(documents)`:**
- Adds additional documents to an existing collection.
- Also calls the embedding function on the new batch.
- Used in the batching loop after the first batch creates the collection.

**`vectordb.similarity_search(query, k)`:**
- Embeds `query` using the embedding function.
- Returns the `k` documents whose vectors are closest (by cosine similarity) to the query vector.
- **Replaced by MMR search** (see Change #7).

**`vectordb.max_marginal_relevance_search(query, k, fetch_k)`:**
- Embeds `query`.
- Fetches `fetch_k` candidates by cosine similarity.
- Iteratively selects `k` documents that maximise `λ · relevance − (1−λ) · similarity_to_selected`. Default `λ=0.5` balances both equally.
- Returns a diverse set of `k` documents.

**Persistence:**
ChromaDB writes to disk automatically when `persist_directory` is set. The directory contains:
- `chroma.sqlite3` — the metadata and document text store.
- `<collection-id>/data_level0.bin` — the HNSW index (approximate nearest-neighbour structure).
- `<collection-id>/header.bin`, `length.bin`, `link_lists.bin` — HNSW index components.

---

### `filter_complex_metadata` — `langchain_community.vectorstores.utils`

**Used in:** `vectordb/cromadb.py`

**What it does:**
ChromaDB only accepts metadata values of types `str`, `int`, `float`, or `bool`. `unstructured` adds rich metadata to each `Document` including nested dicts, lists, and `CoordinateSystem` objects. Passing these to ChromaDB raises a `ValueError`.

`filter_complex_metadata(documents)` iterates over every document and drops any metadata key whose value is not a scalar type. It returns a cleaned list of documents safe to pass to ChromaDB.

**Why called before `_deduplicate`:**
Deduplication only needs `page_content` (not metadata), so order doesn't matter — but filtering first means deduplication operates on already-clean documents, avoiding any edge case where metadata types affect hashing.

---

### `ChatOllama` — `langchain_ollama`

**Used in:** `llm/llms.py`

**What it does:**
A LangChain chat model adapter for Ollama. Wraps Ollama's `/api/chat` endpoint and presents it as a standard LangChain `BaseChatModel`.

**How it works internally:**
1. `llm.invoke(prompt)` sends a POST to `http://localhost:11434/api/chat` with the model name and message.
2. Ollama streams tokens back; `invoke()` waits for completion and returns an `AIMessage`.
3. `response.content` contains the generated text string.

**Key parameters:**
- `model="llama3.1"` — the generation model. Must be pulled with `ollama pull llama3.1`.
- `temperature` (not set, defaults to 0.8) — controls randomness. Lower = more deterministic answers. For a Q&A bot, `temperature=0.1` or `0.0` would give more consistent factual answers.
- `num_ctx` (not set) — context window for the generation model. `llama3.1` defaults to 8192 tokens. Unlike the embed API, `num_ctx` is safe to set here if needed.

**`llm.invoke(prompt: str) -> AIMessage`:**
- Sends the prompt as a single human message.
- Returns `AIMessage(content="...")`.
- Access the text with `response.content`.

---

### `ollama.chat` — `ollama` (Python SDK)

**Used in:** `qa_chain/qa_chain.py` (vision model)

**What it does:**
Direct Python SDK call to Ollama's `/api/chat` endpoint. Used for the vision model because `langchain_ollama` does not support multimodal (image + text) inputs — the raw SDK does.

**How it works:**
```python
response = ollama.chat(
    model="llama3.2-vision",
    messages=[{
        "role": "user",
        "content": "Describe this image...",
        "images": ["/path/to/image.jpg"]   # local file paths or base64 strings
    }]
)
text = response["message"]["content"]
```

Ollama reads the image file, encodes it, and passes it alongside the text prompt to the vision model. The model returns a text description.

**Why wrapped in try/except:**
`llama3.2-vision` requires Ollama ≥ 0.3.6 and sufficient VRAM (≥ 8 GB GPU). If the model is not pulled or the hardware cannot load it, Ollama returns an error. The try/except returns a placeholder string so ingestion continues without crashing on image elements.

---

### `ChatPromptTemplate.from_template` — `langchain_core.prompts`

**Used in:** `prompts/prompt.py`

**What it does:**
Creates a structured prompt template with named placeholders. At invocation time, you fill in the placeholders with actual values.

```python
prompt = ChatPromptTemplate.from_template("""
Answer using only the context below.
Context: {context}
Question: {question}
""")
filled = prompt.format(context="...", question="What is X?")
```

**Note:** The `prompt.py` module exists but is not currently used in the QA chain — `qa_chain.py` builds the prompt as a plain f-string directly. `ChatPromptTemplate` is available if you want to switch to a structured chain (e.g. `LLMChain` or LCEL `prompt | llm`).

---

### `st.cache_resource` — `streamlit`

**Used in:** `main.py`

**What it does:**
Caches the return value of a function across Streamlit re-runs. Unlike `st.cache_data` (which caches serialisable data), `cache_resource` is designed for heavy, non-serialisable objects like database connections, ML models, and vector stores.

**How it works:**
- On the first call with a given set of arguments, the function runs and the result is stored in memory.
- On every subsequent Streamlit re-run (triggered by any user interaction), if the arguments are the same, the cached result is returned instantly without re-running the function.
- The cache lives for the lifetime of the Streamlit server process.

**Cache key:**
Streamlit hashes all function arguments to form the cache key. Parameters prefixed with `_` (e.g. `_file_hash`) are excluded from hashing — useful when the value is already a hash and Streamlit's own hashing would be redundant or incorrect.

**In our pipeline:**
`build_pipeline(pdf_path, _file_hash)` — `_file_hash` is the MD5 of the uploaded file. When the user uploads a different PDF, the hash changes → cache miss → pipeline re-runs. Same PDF → cache hit → instant load.

---

### `st.file_uploader` — `streamlit`

**Used in:** `main.py`

**What it does:**
Renders a drag-and-drop / browse file upload widget. Returns a `UploadedFile` object when a file is selected, or `None` if no file is uploaded.

**Key parameters:**
- `label` — the text shown above the widget.
- `type=["pdf"]` — restricts accepted file types. Streamlit validates the file extension client-side and server-side.

**`UploadedFile` object:**
- `.read()` — returns the file contents as `bytes`. Must be called before accessing other methods; can only be read once per widget interaction.
- `.name` — original filename.
- `.size` — file size in bytes.
- `.type` — MIME type (e.g. `"application/pdf"`).

**Why we save to a temp file:**
The `UploadedFile` object is in-memory and is recreated on every Streamlit re-run. The embedding pipeline and ChromaDB require a file path on disk — not an in-memory stream. Saving to `tempfile.gettempdir()` with a hash-based name gives a stable path that persists across re-runs.

---

### `hashlib.md5` — `hashlib` (Python standard library)

**Used in:** `main.py` (upload dedup), `vectordb/cromadb.py` (file change detection + chunk dedup)

**What it does:**
Computes an MD5 cryptographic hash. MD5 produces a 128-bit (16-byte) digest, typically represented as a 32-character hex string.

**Usage pattern:**
```python
# Hashing bytes (file content)
digest = hashlib.md5(file_bytes).hexdigest()

# Hashing a string
digest = hashlib.md5(text.encode()).hexdigest()

# Hashing a large file incrementally (used in _get_file_hash)
with open(filepath, "rb") as f:
    digest = hashlib.md5(f.read()).hexdigest()
```

**Used for three purposes in this project:**
1. **PDF change detection** (`_get_file_hash`) — if the hash of the current PDF matches the hash saved in `source_metadata.json`, skip re-embedding.
2. **Temp file naming** (`main.py`) — `rag_upload_<hash>.pdf` ensures the same PDF always maps to the same temp path, enabling `@st.cache_resource` to work correctly.
3. **Chunk deduplication** (`_deduplicate`) — hash each chunk's text; drop chunks whose hash was already seen.

**MD5 is not cryptographically secure** (collisions are theoretically possible) but it is fast and sufficient for content-change detection. Do not use it for password hashing or security-sensitive purposes.

---

### `load_dotenv` — `python-dotenv`

**Used in:** `main.py`

**What it does:**
Reads a `.env` file and loads each `KEY=VALUE` line into `os.environ`, making them accessible via `os.getenv("KEY")`. Does not overwrite variables already set in the environment.

```python
load_dotenv(dotenv_path="/path/to/config/.env")
value = os.getenv("HF_TOKEN")
```

**Why used:**
Keeps secrets and machine-specific paths out of source code. The `.env` file is listed in `.gitignore` so it is never committed.

**Load order:**
`load_dotenv` runs at the top of `main.py` before any other module uses `os.getenv`. This ensures `HF_TOKEN` is available when `unstructured` / `transformers` initialise and attempt to contact the HF Hub.

---

### `tempfile.gettempdir` — `tempfile` (Python standard library)

**Used in:** `main.py`

**What it does:**
Returns the path to the system's temporary directory as a string.
- macOS / Linux: typically `/tmp`
- Windows: typically `C:\Users\<user>\AppData\Local\Temp`

Files written here persist across Streamlit re-runs (they are not cleared between runs) but are deleted when the OS cleans temp directories (typically on reboot or by a scheduled cleanup).

**Why used:**
`st.file_uploader` gives us bytes in memory. ChromaDB and `UnstructuredPDFLoader` require a file path. Writing to `tempfile.gettempdir()` gives a stable, writable, cross-platform location without hard-coding a project directory path.

---

## Summary Table

| # | Change | File | Problem Solved |
|---|---|---|---|
| 1 | PDF upload via Streamlit UI | `main.py` | Hardcoded PDF path, no end-user upload |
| 2 | `HF_TOKEN` in `.env` | `config/.env` | HuggingFace rate-limit warnings |
| 3 | Token-accurate chunking | `chunker.py` | Character-based chunks were ~125 tokens instead of 512 |
| 4 | `num_ctx=8192` reverted | `embeddings.py` | Caused Ollama `/tokenize` 400 crash — do not re-add |
| 5 | Deduplication before embedding | `cromadb.py` | 1618 duplicate chunks wasting storage and retrieval |
| 6 | Batched embedding (50/batch) | `cromadb.py` | Ollama timeout on large single-batch embed requests |
| 7 | MMR retrieval | `qa_chain.py` | Similarity search returned redundant near-duplicate chunks |
| 8 | Vision model slowness | `loader.py` | Documented — one-time cost, skip toggle planned |
