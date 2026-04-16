import json
import os
import re
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb


os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CHROMA_DIR = "chroma_db"


@dataclass
class Chunk:
    text: str
    page: int
    chunk_type: str = "text"  # "text" or "table"
    search_text: str = ""  # natural-language summary for indexing; falls back to text if empty
    doc_name: str = ""  # source PDF filename


# --------------- ChromaDB ---------------

@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name="rag_chunks",
        metadata={"hnsw:space": "cosine"},
    )


def add_to_chroma(collection, chunks: List[Chunk], doc_name: str, semantic_model):
    ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
    documents = [c.search_text or c.text for c in chunks]
    embeddings = semantic_model.encode(documents, show_progress_bar=False).tolist()
    metadatas = [
        {
            "text": c.text,
            "page": c.page,
            "chunk_type": c.chunk_type,
            "search_text": c.search_text,
            "doc_name": doc_name,
        }
        for c in chunks
    ]
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def load_chunks_from_chroma(collection) -> List[Chunk]:
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    chunks = []
    for meta in results["metadatas"]:
        chunks.append(Chunk(
            text=meta["text"],
            page=meta["page"],
            chunk_type=meta["chunk_type"],
            search_text=meta.get("search_text", ""),
            doc_name=meta.get("doc_name", ""),
        ))
    return chunks


def get_indexed_docs(collection) -> dict:
    """Returns {doc_name: chunk_count}."""
    if collection.count() == 0:
        return {}
    results = collection.get(include=["metadatas"])
    docs = {}
    for meta in results["metadatas"]:
        name = meta.get("doc_name", "unknown")
        docs[name] = docs.get(name, 0) + 1
    return docs


def delete_doc_from_chroma(collection, doc_name: str):
    if collection.count() == 0:
        return
    results = collection.get(include=["metadatas"])
    ids_to_delete = [
        rid for rid, meta in zip(results["ids"], results["metadatas"])
        if meta.get("doc_name") == doc_name
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)


# --------------- PDF Parsing ---------------

def table_to_markdown(table_data: list) -> str:
    if not table_data or not table_data[0]:
        return ""
    cleaned = [
        [re.sub(r"\s+", " ", str(cell)).strip() if cell is not None else ""
         for cell in row]
        for row in table_data
    ]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""
    header = "| " + " | ".join(cleaned[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(cleaned[0])) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in cleaned[1:]]
    return header + "\n" + separator + "\n" + "\n".join(rows)


def summarize_table(table_data: list) -> str:
    if not table_data or not table_data[0]:
        return ""
    cleaned = [
        [re.sub(r"\s+", " ", str(cell)).strip() if cell is not None else ""
         for cell in row]
        for row in table_data
    ]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""
    headers = cleaned[0]
    parts = ["Table with columns: " + ", ".join(h for h in headers if h) + "."]
    for row in cleaned[1:]:
        row_label = row[0] if row[0] else "Row"
        values = ", ".join(f"{headers[i]}: {row[i]}" for i in range(len(row)) if row[i] and i > 0)
        if values:
            parts.append(f"{row_label}: {values}.")
    return " ".join(parts)


def extract_page_elements(file_bytes: bytes):
    elements = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            page_num = i + 1
            page_rect = page.rect

            tables = page.find_tables()
            if tables.tables:
                sorted_tables = sorted(tables.tables, key=lambda t: t.bbox[1])
                y_cursor = page_rect.y0

                for table in sorted_tables:
                    tb = table.bbox
                    if tb[1] > y_cursor + 5:
                        clip = fitz.Rect(page_rect.x0, y_cursor, page_rect.x1, tb[1])
                        text = page.get_text("text", clip=clip)
                        cleaned = re.sub(r"\s+", " ", text).strip()
                        if cleaned:
                            elements.append((page_num, cleaned, "text", ""))

                    raw_data = table.extract()
                    md = table_to_markdown(raw_data)
                    summary = summarize_table(raw_data)
                    if md.strip():
                        elements.append((page_num, md, "table", summary))

                    y_cursor = tb[3]

                if y_cursor < page_rect.y1 - 5:
                    clip = fitz.Rect(page_rect.x0, y_cursor, page_rect.x1, page_rect.y1)
                    text = page.get_text("text", clip=clip)
                    cleaned = re.sub(r"\s+", " ", text).strip()
                    if cleaned:
                        elements.append((page_num, cleaned, "text", ""))
            else:
                text = page.get_text("text")
                cleaned = re.sub(r"\s+", " ", text).strip()
                if cleaned:
                    elements.append((page_num, cleaned, "text", ""))

    return elements


def chunk_text(elements, max_chunk_size=800, overlap=150, max_table_chunk_size=2000):
    chunks = []
    for page_num, content, elem_type, summary in elements:
        if elem_type == "table":
            if len(content) <= max_table_chunk_size:
                chunks.append(Chunk(text=content, page=page_num, chunk_type="table", search_text=summary))
            else:
                lines = content.split("\n")
                header = lines[0] + "\n" + lines[1] if len(lines) > 1 else lines[0]
                current_block = header
                for line in lines[2:]:
                    if len(current_block) + len(line) + 1 > max_table_chunk_size:
                        chunks.append(Chunk(text=current_block, page=page_num, chunk_type="table", search_text=summary))
                        current_block = header + "\n" + line
                    else:
                        current_block += "\n" + line
                if current_block.strip() and len(current_block) > len(header) + 5:
                    chunks.append(Chunk(text=current_block, page=page_num, chunk_type="table", search_text=summary))
        else:
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)
            current, current_len = [], 0
            for sent in sentences:
                if current_len + len(sent) > max_chunk_size and current:
                    chunks.append(Chunk(text=" ".join(current).strip(), page=page_num))
                    overlap_sents = []
                    acc = 0
                    for s in reversed(current):
                        if acc + len(s) > overlap:
                            break
                        overlap_sents.insert(0, s)
                        acc += len(s)
                    current = overlap_sents
                    current_len = acc
                current.append(sent)
                current_len += len(sent)
            if current and current_len >= 80:
                chunks.append(Chunk(text=" ".join(current).strip(), page=page_num))
    return chunks


# --------------- Indexing & Retrieval ---------------

def build_tfidf_index(chunks: List[Chunk]):
    if not chunks:
        return None
    texts = [c.search_text or c.text for c in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    return {"vectorizer": vectorizer, "matrix": tfidf_matrix}


def retrieve(question: str, chunks: List[Chunk], tfidf_index, collection, semantic_model, k: int = 6):
    if not chunks:
        return []

    # 1. Keyword Search (TF-IDF)
    vectorizer = tfidf_index["vectorizer"]
    tfidf_matrix = tfidf_index["matrix"]
    q_tfidf = vectorizer.transform([question])
    tfidf_sims = cosine_similarity(q_tfidf, tfidf_matrix).flatten()
    keyword_rank = np.argsort(-tfidf_sims)

    # 2. Semantic Search (ChromaDB)
    n_results = min(k * 2, len(chunks))
    chroma_results = collection.query(
        query_embeddings=semantic_model.encode([question]).tolist(),
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    # Build a map from (doc_name, page, text[:50]) -> chunk index for matching
    chunk_key = lambda c: (c.doc_name, c.page, c.text[:50])
    chunk_lookup = {chunk_key(c): i for i, c in enumerate(chunks)}

    # Convert ChromaDB results to ranked indices into chunks list
    semantic_ranked_indices = []
    for meta in chroma_results["metadatas"][0]:
        key = (meta.get("doc_name", ""), meta["page"], meta["text"][:50])
        idx = chunk_lookup.get(key)
        if idx is not None:
            semantic_ranked_indices.append(idx)

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    k_constant = 60

    for rank, idx in enumerate(keyword_rank):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k_constant + rank)

    for rank, idx in enumerate(semantic_ranked_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k_constant + rank)

    sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:k]

    return [(rrf_scores[i], chunks[i]) for i in sorted_indices]


# --------------- Models ---------------

@st.cache_resource
def load_local_llm(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# --------------- Generation ---------------

def answer_with_context(question, retrieved, llm_pipe):
    if not retrieved:
        return "I do not have enough information in the uploaded PDF."

    tokenizer = llm_pipe["tokenizer"]
    model = llm_pipe["model"]

    context_parts = []
    budget = 1000
    for _, chunk in retrieved:
        if chunk.chunk_type == "table":
            snippet = f"[{chunk.doc_name} — Page {chunk.page} - Table]\n{chunk.text}"
        else:
            snippet = f"[{chunk.doc_name} — Page {chunk.page}] {chunk.text}"
        if len(tokenizer.encode(snippet)) > budget:
            break
        context_parts.append(snippet)
        budget -= len(tokenizer.encode(snippet))

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced finance assistant. Answer the user's question "
                "using ONLY the context below. Cite page numbers as (Page X). "
                "Context may include markdown tables — read them carefully. "
                "If the answer is not in the context, say so."
            ),
         },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=400,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_why_summary(retrieved: List[Tuple[float, Chunk]]) -> str:
    if not retrieved:
        return "No supporting evidence was retrieved."
    sources = sorted({(chunk.doc_name, chunk.page) for _, chunk in retrieved})
    parts = [f"{doc} p.{page}" for doc, page in sources]
    return f"Answer is based on: {', '.join(parts)}."


def append_audit_log(entry: dict) -> None:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "qa_history.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def load_history_from_file() -> list:
    log_file = Path("logs/qa_history.jsonl")
    if not log_file.exists():
        return []
    entries = []
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return list(reversed(entries))


def delete_history_entry(timestamp: str, question: str) -> None:
    log_file = Path("logs/qa_history.jsonl")
    if not log_file.exists():
        return
    kept = []
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("timestamp") == timestamp and entry.get("question") == question:
                    continue
                kept.append(entry)
            except json.JSONDecodeError:
                pass
    with log_file.open("w", encoding="utf-8") as f:
        for entry in kept:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def clear_all_history() -> None:
    log_file = Path("logs/qa_history.jsonl")
    if log_file.exists():
        log_file.write_text("", encoding="utf-8")


# --------------- Auth ---------------

USERS = {
    os.environ.get("ADMIN_USERNAME", "admin"): os.environ.get("ADMIN_PASSWORD", "admin123"),
    os.environ.get("USER_USERNAME", "user"): os.environ.get("USER_PASSWORD", "password"),
}


def login_page():
    st.set_page_config(page_title="Login - Finance RAG Q&A", page_icon=":bar_chart:", layout="centered")
    st.title("Finance Knowledge RAG Q&A")
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", type="primary"):
        if username in USERS and USERS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")


# --------------- Main ---------------

def main():
    if not st.session_state.get("authenticated"):
        login_page()
        return

    st.set_page_config(page_title="Finance Knowledge RAG Q&A", page_icon=":bar_chart:", layout="wide")
    st.title("Finance Knowledge RAG Q&A")
    st.caption("Lightweight local RAG system that extracts answers from PDF documents and answers with citations")

    # Init models
    if "llm_pipe" not in st.session_state:
        with st.spinner("Initializing local model..."):
            try:
                st.session_state.llm_pipe = load_local_llm(MODEL_ID)
            except Exception as exc:
                st.error(f"Model initialization failed. Pre-download model first. Details: {exc}")
                st.stop()
    if "semantic_model" not in st.session_state:
        with st.spinner("Loading semantic search engine..."):
            st.session_state.semantic_model = load_semantic_model()

    collection = get_chroma_collection()

    # Load persisted chunks from ChromaDB on first run
    if "chunks" not in st.session_state:
        st.session_state.chunks = load_chunks_from_chroma(collection)
        if st.session_state.chunks:
            st.session_state.tfidf_index = build_tfidf_index(st.session_state.chunks)
        else:
            st.session_state.tfidf_index = None
    if "history" not in st.session_state:
        st.session_state.history = load_history_from_file()

    # --- Sidebar ---
    with st.sidebar:
        st.write(f"Logged in as **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

        st.header("1) Upload PDF")
        uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
        build_clicked = st.button("Index document", use_container_width=True)

        # Show indexed documents
        st.header("Indexed Documents")
        indexed_docs = get_indexed_docs(collection)
        if indexed_docs:
            for doc_name, count in indexed_docs.items():
                col1, col2 = st.columns([3, 1])
                col1.write(f"**{doc_name}** ({count} chunks)")
                if col2.button("X", key=f"del_{doc_name}"):
                    delete_doc_from_chroma(collection, doc_name)
                    st.session_state.chunks = load_chunks_from_chroma(collection)
                    st.session_state.tfidf_index = build_tfidf_index(st.session_state.chunks) if st.session_state.chunks else None
                    st.rerun()
            if st.button("Clear all documents"):
                for doc_name in indexed_docs:
                    delete_doc_from_chroma(collection, doc_name)
                st.session_state.chunks = []
                st.session_state.tfidf_index = None
                st.rerun()
        else:
            st.write("No documents indexed yet.")

    # --- Indexing ---
    if build_clicked:
        if not uploaded:
            st.warning("Please upload a PDF first.")
        else:
            doc_name = uploaded.name
            if doc_name in get_indexed_docs(collection):
                st.warning(f"'{doc_name}' is already indexed. Delete it first to re-index.")
            else:
                with st.spinner("Extracting and indexing document..."):
                    start_time = time.time()
                    pdf_bytes = uploaded.read()
                    elements = extract_page_elements(pdf_bytes)
                    chunks = chunk_text(elements)
                    if not chunks:
                        st.error("No readable text found in this PDF.")
                    else:
                        for c in chunks:
                            c.doc_name = doc_name
                        add_to_chroma(collection, chunks, doc_name, st.session_state.semantic_model)
                        # Reload all chunks and rebuild TF-IDF
                        st.session_state.chunks = load_chunks_from_chroma(collection)
                        st.session_state.tfidf_index = build_tfidf_index(st.session_state.chunks)
                        elapsed = time.time() - start_time
                        table_count = sum(1 for c in chunks if c.chunk_type == "table")
                        text_count = len(chunks) - table_count
                        st.success(f"Indexed {len(chunks)} chunks ({text_count} text, {table_count} table) from '{doc_name}'. Time: {elapsed:.2f}s")

    # --- Q&A ---
    st.header("2) Ask Questions")
    question = st.text_input("Ask a question about your indexed documents")

    if st.button("Get Answer", type="primary"):
        if not st.session_state.chunks or st.session_state.tfidf_index is None:
            st.warning("Please upload and index a PDF first.")
        elif not question.strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                retrieved = retrieve(
                    question=question,
                    chunks=st.session_state.chunks,
                    tfidf_index=st.session_state.tfidf_index,
                    collection=collection,
                    semantic_model=st.session_state.semantic_model,
                    k=6,
                )
                answer = answer_with_context(question, retrieved, st.session_state.llm_pipe)
                why_summary = build_why_summary(retrieved)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                history_entry = {
                    "timestamp": timestamp,
                    "question": question,
                    "answer": answer,
                    "why": why_summary,
                    "citations": [
                        {
                            "doc_name": chunk.doc_name,
                            "page": chunk.page,
                            "chunk_type": chunk.chunk_type,
                            "similarity": round(score, 4),
                            "snippet": chunk.text[:400],
                        }
                        for score, chunk in retrieved
                    ],
                    "models": {"retrieval_model": "hybrid-rrf"},
                }
                st.session_state.history.insert(0, history_entry)
                append_audit_log(history_entry)

            st.subheader("Answer")
            st.write(answer)
            st.subheader("Why")
            st.write(why_summary)

            st.subheader("Citations")
            for rank, (score, chunk) in enumerate(retrieved, start=1):
                type_badge = "TABLE" if chunk.chunk_type == "table" else "Text"
                st.markdown(f"**{rank}. {chunk.doc_name} — Page {chunk.page}** [{type_badge}] (similarity: {score:.3f})")
                st.write(chunk.text[:600] + ("..." if len(chunk.text) > 600 else ""))

    # --- History ---
    st.header("3) Question History (Local Audit Trail)")
    st.caption("Saved to `logs/qa_history.jsonl` for traceability. History persists across sessions.")
    if not st.session_state.history:
        st.write("No questions asked yet.")
    else:
        if st.button("Clear all history"):
            clear_all_history()
            st.session_state.history = []
            st.rerun()
        for item in st.session_state.history:
            with st.expander(f"{item['timestamp']} - {item['question']}"):
                st.markdown("**Answer**")
                st.write(item["answer"])
                st.markdown("**Why**")
                st.write(item["why"])
                st.markdown("**Cited Pages**")
                st.write(", ".join(f"{c.get('doc_name', '')} p.{c['page']}" for c in item["citations"]))
                if st.button("Delete", key=f"del_hist_{item['timestamp']}_{item['question'][:20]}"):
                    delete_history_entry(item["timestamp"], item["question"])
                    st.session_state.history = load_history_from_file()
                    st.rerun()

    st.divider()
    st.caption(
        "This is a local RAG system using semantic retrieval and local Hugging Face generation."
    )


if __name__ == "__main__":
    main()
