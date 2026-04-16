This project is a local Financial RAG demo that lets you upload a PDF, ask questions, and get grounded answers with page citations.
It is intentionally compact and implemented mostly in one file, which makes it great for learning the full RAG pipeline end-to-end.

Concept
RAG here means:

Retrieve relevant parts of your PDF (using TF-IDF similarity).
Augment the LLM prompt with those retrieved chunks.
Generate an answer using a local model (flan-t5-base).
So instead of “model knows everything,” this app makes the model answer from your uploaded document context.

Core stack from requirements.txt:

streamlit (UI app framework)
PyMuPDF (PDF text extraction)
scikit-learn (TF-IDF + cosine similarity retrieval)
transformers + torch (local LLM inference)
Technical Internals
1) Ingestion: PDF -> clean page text
In app.py, extract_text_by_page() opens PDF bytes and normalizes whitespace.


app.py
Lines 28-36
def extract_text_by_page(file_bytes: bytes) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            cleaned = re.sub(r"\s+", " ", text).strip()
            if cleaned:
                pages.append((i + 1, cleaned))
    return pages
Input: raw uploaded PDF bytes
Output: list of (page_number, cleaned_text)

2) Chunking: page text -> retrieval units
chunk_text() creates overlapping character chunks and tags each with source page.


app.py
Lines 39-50
def chunk_text(pages: List[Tuple[int, str]], chunk_size: int = 900, overlap: int = 150) -> List[Chunk]:
    chunks: List[Chunk] = []
    step = max(1, chunk_size - overlap)
    for page_num, text in pages:
        if len(text) <= chunk_size:
            chunks.append(Chunk(text=text, page=page_num))
            continue
        for start in range(0, len(text), step):
            piece = text[start : start + chunk_size].strip()
            if len(piece) >= 120:
                chunks.append(Chunk(text=piece, page=page_num))
    return chunks
Input: list of page texts
Output: list of Chunk(text, page)

3) Indexing: chunks -> TF-IDF matrix
build_index() fits a TF-IDF vectorizer (unigram + bigram) over chunk text.


app.py
Lines 53-57
def build_index(chunks: List[Chunk]):
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    return {"vectorizer": vectorizer, "matrix": matrix}
Input: chunk list
Output: in-memory retrieval index {vectorizer, matrix}

4) Retrieval: question -> top-k relevant chunks
retrieve() vectorizes the question and ranks chunks with cosine similarity.


app.py
Lines 60-68
def retrieve(question: str, chunks: List[Chunk], index, k: int = 4):
    vectorizer: TfidfVectorizer = index["vectorizer"]
    matrix = index["matrix"]
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_k = min(k, len(chunks))
    ranked_ids = sims.argsort()[::-1][:top_k]
    results = [(float(sims[idx]), chunks[idx]) for idx in ranked_ids if sims[idx] > 0]
    return results
Input: user question + index
Output: ranked list of (similarity_score, Chunk)

5) Generation: retrieved context -> answer
answer_with_context() builds a constrained prompt and runs local HF generation.


app.py
Lines 79-101
def answer_with_context(question: str, retrieved: List[Tuple[float, Chunk]], llm_pipe) -> str:
    if not retrieved:
        return "I do not have enough information in the uploaded PDF."
    context_blocks = [f"[Page {chunk.page}] {chunk.text}" for _, chunk in retrieved]
    context = "\n\n".join(context_blocks)
    prompt = (
        "You are a financial document assistant. Use ONLY the context below.\n"
        "If the answer is not in the context, respond exactly: "
        "'I do not have enough information in the uploaded PDF.'\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer concisely and include page references like (Page X) when possible."
    )
    ...
Input: question + retrieved chunks + model
Output: final answer string

6) Traceability: “why” + audit log
The app creates explanation metadata and appends local JSONL logs.


app.py
Lines 112-117
def append_audit_log(entry: dict) -> None:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "qa_history.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")
This gives meeting/compliance-style traceability in logs/qa_history.jsonl.

Framework Implementation
Streamlit patterns used
Single-file app controller/view: main() handles UI events and pipeline actions.
Stateful session memory: st.session_state stores chunks, index, history, llm_pipe.
Resource caching: @st.cache_resource avoids reloading model every rerun.
Event-driven flow: button clicks (Index document, Get Answer) trigger pipeline steps.

app.py
Lines 120-166
def main():
    st.set_page_config(page_title="Local Financial PDF Q&A Demo", page_icon=":bar_chart:", layout="wide")
    ...
    if "llm_pipe" not in st.session_state:
        with st.spinner("Initializing local model..."):
            try:
                st.session_state.llm_pipe = load_local_llm(MODEL_ID)
            except Exception as exc:
                st.error(f"Model initialization failed. Pre-download model first. Details: {exc}")
                st.stop()
    ...
    if st.button("Get Answer", type="primary"):
        if not st.session_state.chunks or st.session_state.index is None:
            st.warning("Please upload and index a PDF first.")
Hugging Face local inference
load_local_llm() loads tokenizer/model from local cache if TRANSFORMERS_OFFLINE=1.
Uses seq2seq generation (AutoModelForSeq2SeqLM.generate) with deterministic settings (do_sample=False).
Operations & launch
run-demo.ps1 and run-demo.bat launch Streamlit inside Conda env fin-demo.
predownload_model.py prefetches model artifacts for smooth/offline startup.
End-to-End Data Flow
User
UploadPDF
ExtractTextByPage
ChunkText
BuildTfidfIndex
AskQuestion
RetrieveTopKChunks
BuildPromptWithContext
GenerateWithFlanT5
RenderAnswerWhyCitations
AppendJsonlAuditLog
Strengths / Limitations
Strengths

Simple, understandable RAG pipeline in one place.
Fully local inference option; good privacy/offline posture.
Built-in citation display and audit trail.
Minimal setup and easy demoability.
Limitations

Retrieval is lexical TF-IDF, not semantic embeddings.
No persistent vector DB (index rebuilt per upload/session).
Character chunking can cut sentence/table boundaries.
Prompt length cap (max_length=1024) may truncate context.
No automated evaluation/test harness for retrieval quality or groundedness.
Upgrade Roadmap (practical next steps)
Modularize code into ingestion.py, retrieval.py, generation.py, ui.py.
Semantic retrieval with embedding model + vector store (e.g., FAISS/Chroma).
Hybrid retrieval (BM25/TF-IDF + embeddings) + optional reranker.
Persistent indexing by document hash so uploads don’t always rebuild.
Evaluation layer (gold QA pairs, recall@k, answer faithfulness checks).
Config surface for model choice, chunk params, top-k, generation settings.
All plan to-dos are now completed.





Full Project Breakdown Plan

Goal

Provide a complete, beginner-to-intermediate walkthrough of how this local financial RAG app works: what it does conceptually, how each technical layer is implemented, and how Streamlit + Hugging Face + TF-IDF are wired together.

Scope Covered





End-to-end request flow from app launch to answer rendering.



RAG concepts as implemented here (ingestion, chunking, retrieval, generation, citations, audit logging).



Framework implementation details in Streamlit and Transformers.



Operational setup and run workflow.



Practical limitations and extension paths.

Code Anchors





Main application and pipeline: app.py



Dependencies and framework stack: requirements.txt



Model artifact bootstrap: predownload_model.py



Windows launchers: run-demo.ps1, run-demo.bat



Runtime trace log output: logs/qa_history.jsonl

Explanation Structure





Project purpose and architecture





Explain why this is a “local RAG demo” and which components replace cloud services.



Conceptual RAG mapping to code





Ingestion (extract_text_by_page), chunking (chunk_text), indexing (build_index), retrieval (retrieve), generation (answer_with_context).



Framework implementation details





Streamlit lifecycle and state (st.session_state, @st.cache_resource, UI events).



Transformers local model loading and generation parameters.



Data flow diagram





Show how PDF/question move through extraction -> retrieval -> LLM -> UI + audit log.



Technical trade-offs





Why TF-IDF is simple/fast, where it falls short vs embedding/vector DB RAG.



How to run and reason about operations





Launcher scripts, model predownload path, offline behavior.



Next-step upgrades





Modularization, semantic retrieval, persistent vector store, evaluation harness.

Architecture Diagram

flowchart TD
    user[User] --> upload[UploadPDF]
    upload --> extract[ExtractTextByPage]
    extract --> chunk[ChunkText]
    chunk --> index[BuildTfidfIndex]
    user --> ask[AskQuestion]
    ask --> retrieve[RetrieveTopKChunks]
    index --> retrieve
    retrieve --> prompt[BuildPromptWithContext]
    prompt --> generate[GenerateWithFlanT5]
    generate --> ui[RenderAnswerWhyCitations]
    generate --> audit[AppendJsonlAuditLog]

Deliverable Format





A clear narrative with sections: Concept, Technical Internals, Framework Implementation, Strengths/Limitations, Upgrade Roadmap.



Concrete references to code symbols and file paths so each idea is traceable to implementation.

