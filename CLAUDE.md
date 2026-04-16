# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
streamlit run app.py
```

Pre-download models before first run (required for offline/smooth startup):

```bash
python predownload_model.py
```

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `streamlit`, `PyMuPDF` (imported as `fitz`), `scikit-learn`, `transformers`, `torch`, `sentence-transformers`, `numpy`, `chromadb`.

## Architecture

Single-file Streamlit app (`app.py`) implementing a local RAG pipeline for PDF question-answering with login-gated access. No backend server — everything runs in-process.

**Auth:** Hardcoded `USERS` dict at the top of `app.py` gates entry via `login_page()`. Sessions are tracked in `st.session_state.authenticated`.

**Pipeline stages:**

1. **Ingestion** — `extract_page_elements()` extracts text and tables per page via PyMuPDF. Tables are detected with `page.find_tables()`, converted to markdown (`table_to_markdown()`) for storage, and to natural-language summaries (`summarize_table()`) for semantic indexing. Text between tables is extracted by clipping page regions.

2. **Chunking** — `chunk_text()` handles text and table elements differently: text uses sentence-boundary splitting with overlap (`max_chunk_size=800`, `overlap=150`, min 80 chars); tables are stored whole or split by row with a repeated header (`max_table_chunk_size=2000`). Each chunk is a `Chunk` dataclass with `text`, `page`, `chunk_type` ("text"/"table"), `search_text` (summary for tables), and `doc_name`.

3. **Indexing (dual):**
   - **ChromaDB** (`chroma_db/` directory) — persistent vector store. `add_to_chroma()` embeds `search_text` (or `text` for non-table chunks) and stores metadata. Survives app restarts; loaded on startup via `load_chunks_from_chroma()`.
   - **TF-IDF** — rebuilt in-memory (`build_tfidf_index()`) from all chunks after every add/delete. Uses bigrams, English stopwords.

4. **Retrieval** — `retrieve()` fuses TF-IDF keyword ranking and ChromaDB semantic similarity using Reciprocal Rank Fusion (RRF, k=60). Returns top-k (default 6) `(score, Chunk)` tuples. ChromaDB is queried by embedding the question; results are matched back to the in-memory chunk list via `(doc_name, page, text[:50])` key.

5. **Generation** — `answer_with_context()` applies Qwen2.5-1.5B-Instruct's chat template with a finance-assistant system prompt. Chunks are greedily added to context until a 1000-token budget is exhausted. Generates up to 400 new tokens with `do_sample=False` and `repetition_penalty=1.1`. Tables appear in context with a `[Table]` label.

6. **Audit logging** — every Q&A is appended to `logs/qa_history.jsonl` with timestamp, question, answer, `why` summary, citations (doc, page, chunk_type, RRF score, snippet), and model metadata.

**Multi-document support:** Each chunk carries a `doc_name`. The sidebar lists indexed documents with per-doc chunk counts and individual delete buttons. `delete_doc_from_chroma()` removes by `doc_name`; the TF-IDF index is rebuilt from the remaining ChromaDB contents afterward.

**Models:**
- LLM: `Qwen/Qwen2.5-1.5B-Instruct` (`AutoModelForCausalLM`, `torch.float32`)
- Embedding: `all-MiniLM-L6-v2` (sentence-transformers)

Both loaded once via `@st.cache_resource`. `get_chroma_collection()` is also `@st.cache_resource`. Session state holds `chunks`, `tfidf_index`, `llm_pipe`, `semantic_model`, and `history`.
