# Lab4.py  — Streamlit page: ChromaDB-backed Course Info Chatbot (RAG)

import os
import re
import sys
import glob
from typing import List, Dict

import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader

# --- Make Chroma use pysqlite3 (esp. helpful on Python 3.12 environments) ---
import pysqlite3  # noqa: F401
sys.modules['sqlite3'] = pysqlite3
sys.modules['sqlite3.dbapi2'] = pysqlite3

import chromadb

# =========================
# Configuration
# =========================
PDF_DIR = "/workspaces/document-qa/PDFs"                 # <-- place your 7 PDFs here
CHROMA_PATH = "./Chroma_for_lab"   # persistent store location
COLLECTION_NAME = "Lab4Collection"
EMBED_MODEL = "text-embedding-3-small"

# expects .streamlit/secrets.toml:
# [openai]
# api_key = "sk-..."
OPENAI_API_KEY = st.secrets["openai"]["api_key"].strip()

# You can switch the chat model here
DEFAULT_CHAT_MODEL = "gpt-5-mini"   # or "gpt-4o-mini"

# =========================
# Page UI
# =========================
st.title("Lab 4 — ChromaDB Course Info Chatbot (RAG)")
st.caption("This bot uses your course PDFs as a knowledge base via ChromaDB.")

# One OpenAI client for the app
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)
openai_client = st.session_state.openai_client

# =========================
# Utilities
# =========================
def read_pdf_text(path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n".join(parts).strip()
    except Exception as e:
        st.warning(f"Failed to read {os.path.basename(path)}: {e}")
        return ""

def list_pdf_files(pdf_dir: str, limit: int = 7) -> List[str]:
    files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    return files[:limit]

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-embed texts with OpenAI."""
    if not texts:
        return []
    resp = openai_client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [d.embedding for d in resp.data]

# =========================
# 2) Core function per spec — create Chroma & ingest once
# =========================
def create_or_load_lab4_vectorDB():
    """
    - Construct 'Lab4Collection' in persistent Chroma store
    - Use OpenAI embeddings
    - Read up to 7 PDFs from ./pdfs and add to collection
    - Use filename as id, store metadata
    - Store handle in st.session_state.Lab4_vectorDB
    - Only run once per session (caller ensures the guard)
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # Skip ingest if already populated
    try:
        count = collection.count()
    except Exception:
        try:
            _ = collection.get(limit=1)
            count = 1
        except Exception:
            count = 0

    if count == 0:
        pdf_paths = list_pdf_files(PDF_DIR, limit=7)
        if not pdf_paths:
            st.error(f"No PDFs found in {PDF_DIR}. Add your 7 PDFs there.")
        else:
            docs: List[str] = []
            ids: List[str] = []
            metadatas: List[Dict] = []

            for p in pdf_paths:
                text = read_pdf_text(p)
                if not text:
                    continue
                fname = os.path.basename(p)
                docs.append(text)
                ids.append(fname)  # key = filename
                metadatas.append({"filename": fname, "source_path": p})

            if docs:
                embs = embed_texts(docs)
                collection.add(
                    documents=docs,
                    ids=ids,
                    metadatas=metadatas,
                    embeddings=embs,
                )
                st.success(f"Ingested {len(docs)} PDF(s) into '{COLLECTION_NAME}'.")
            else:
                st.error("No readable text extracted from the provided PDFs.")

    st.session_state.Lab4_vectorDB = {"client": chroma_client, "collection": collection}

# Call the function exactly once per app run
if "Lab4_vectorDB" not in st.session_state:
    create_or_load_lab4_vectorDB()

collection = st.session_state.Lab4_vectorDB["collection"]

# =========================
# 5–6) Chatbot with RAG
# =========================
def retrieve_context(query: str, k: int = 3):
    """Return top-k doc snippets + metadata for a query using embeddings."""
    # Try embeddings-based query first; fall back to query_texts for older Chroma
    try:
        q_emb = openai_client.embeddings.create(input=query, model=EMBED_MODEL).data[0].embedding
        res = collection.query(query_embeddings=[q_emb], n_results=k)
    except TypeError:
        res = collection.query(query_texts=[query], n_results=k)

    docs = res.get("documents", [[]])[0] if res.get("documents") else []
    metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    ids   = res.get("ids", [[]])[0] if res.get("ids") else []
    return docs, metas, ids

SYSTEM_PROMPT = (
    "You are a helpful course assistant. Explain clearly and simply. "
    "Cite which course PDFs you used if any. If you do not have enough information, say so."
)

def build_rag_prompt(user_q: str, ctx_docs: List[str], ctx_metas: List[Dict]) -> List[Dict]:
    """Construct messages for the chat completion using retrieved context."""
    # Join top documents with simple separators and attributions
    context_blobs = []
    for i, d in enumerate(ctx_docs):
        fname = (ctx_metas[i].get("filename") if i < len(ctx_metas) else None) or f"doc_{i+1}"
        # keep context reasonably bounded
        snippet = d[:4000]  # trim to reduce token usage while keeping substance
        context_blobs.append(f"[SOURCE: {fname}]\n{snippet}")

    context_text = "\n\n".join(context_blobs).strip()
    rag_note = (
        "Knowledge source: Course PDFs (retrieved via ChromaDB)." if context_text else
        "No relevant course PDF context retrieved; answering from general knowledge."
    )

    sys = {"role": "system", "content": SYSTEM_PROMPT}
    if context_text:
        ctx = {"role": "system", "content": f"Use the following course context to answer:\n\n{context_text}"}
        usr = {"role": "user", "content": f"{user_q}\n\n{rag_note}"}
        return [sys, ctx, usr]
    else:
        usr = {"role": "user", "content": f"{user_q}\n\n{rag_note}"}
        return [sys, usr]

def stream_completion(messages: List[Dict], model: str):
    """Yield streaming tokens from OpenAI."""
    resp = openai_client.chat.completions.create(model=model, messages=messages, stream=True)
    for ch in resp:
        d = ch.choices[0].delta
        if d and getattr(d, "content", None):
            yield d.content

# Sidebar options
with st.sidebar:
    model_choice = st.selectbox("Model", [DEFAULT_CHAT_MODEL, "gpt-4o-mini", "gpt-4o"], index=0)
    top_k = st.slider("Retrieval: top-k documents", 1, 5, 3)

# Session chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about the course, and I’ll use the PDFs when helpful."}
    ]

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_q = st.chat_input("Ask about Generative AI, Text Mining, assignments, etc.")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Retrieve RAG context
    docs, metas, ids = retrieve_context(user_q, k=top_k)

    # Build messages and stream answer
    rag_msgs = build_rag_prompt(user_q, docs, metas)
    with st.chat_message("assistant"):
        answer = st.write_stream(stream_completion(rag_msgs, model_choice)) or ""
        # Show sources if used
        if metas:
            names = [ (m.get("filename") if isinstance(m, dict) else None) or ids[i] for i, m in enumerate(metas) ]
            st.caption("Sources: " + ", ".join(names))
    st.session_state.messages.append({"role": "assistant", "content": answer})
