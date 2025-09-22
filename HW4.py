# HW4.py — Streamlit page: iSchool Student Orgs Chatbot (RAG over HTML)
# ---------------------------------------------------------------
# Chunking: For each HTML doc, we create exactly 2 mini-docs 
# Method: split by character count near a paragraph boundary.
# Why this method? It's simple, deterministic, guarantees 2 chunks, and keeps each
# chunk large enough for semantic retrieval while avoiding token limits and preserving
# coherence at paragraph boundaries better than naive mid-string splits.

import os
import re
import sys
import glob
from typing import List, Dict, Tuple

import streamlit as st
from openai import OpenAI

# HTML parsing
from bs4 import BeautifulSoup

# --- Make Chroma use pysqlite3 (needed on Streamlit Cloud / Python 3.12) ---
# If you see "unsupported sqlite3 version", this shim fixes it.
import pysqlite3  # noqa: F401
sys.modules['sqlite3'] = pysqlite3
sys.modules['sqlite3.dbapi2'] = pysqlite3

import chromadb

# =========================
# Configuration
# =========================
# Put your provided HTMLs here (unzipped). We will load *.html and *.htm.
HTML_DIR = "./iSchool_HTMLs"                 # <— copy *all* supplied HTML pages here
CHROMA_PATH = "./Chroma_HW4"                 # <— persistent vector store folder
COLLECTION_NAME = "HW4_iSchool_StudentOrgs"
EMBED_MODEL = "text-embedding-3-small"

# api_key = "sk-..."
OPENAI_API_KEY = st.secrets["openai"]["api_key"].strip()

# 3 model choices in the sidebar, per HW4
MODEL_CHOICES = ["gpt-5-mini", "gpt-4o-mini", "gpt-4o"]
DEFAULT_CHAT_MODEL = MODEL_CHOICES[0]

# =========================
# Page UI
# =========================
st.title("HW4 — iSchool Student Organizations Chatbot (RAG over HTML)")
st.caption("Answers iSchool student organization questions using your local HTML corpus + ChromaDB.")

# One OpenAI client for the app
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)
openai_client = st.session_state.openai_client

# =========================
# Utilities
# =========================
def read_html_text(path: str) -> str:
    """Extract visible text from an HTML file using BeautifulSoup."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style/head/nav elements to keep actual page content
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # Normalize whitespace
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()
    except Exception as e:
        st.warning(f"Failed to read {os.path.basename(path)}: {e}")
        return ""

def list_html_files(html_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(html_dir, "*.html")) + glob.glob(os.path.join(html_dir, "*.htm")))
    return files

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-embed texts with OpenAI."""
    if not texts:
        return []
    resp = openai_client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [d.embedding for d in resp.data]

# --------- Chunking Strategy (exactly 2 chunks per document) ----------
def split_on_para_boundaries(text: str) -> List[str]:
    """Split raw text into paragraphs (blank lines as separators)."""
    # Normalize Windows newlines, then split on blank lines
    text = text.replace("\r\n", "\n")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras

def chunk_text_into_two(text: str) -> Tuple[str, str]:
    """
    Create exactly TWO chunks per document.

    Method (explained as required):
    - We split the document into paragraphs.
    - We compute cumulative character lengths and find the midpoint target (len/2).
    - We choose a cut position at the paragraph boundary nearest to the midpoint.
      This preserves semantic coherence better than cutting in the middle of a paragraph
      and is deterministic and simple (good for HW). It keeps chunks reasonably balanced
      for embedding-based retrieval.

    If paragraphs are too few/uneven, we fallback to a simple char midpoint split.
    """
    if not text:
        return ("", "")
    paras = split_on_para_boundaries(text)
    total = len(text)
    if len(paras) >= 2:
        cum = 0
        target = total // 2
        # Find boundary nearest to target
        cut_index = 0
        for i, p in enumerate(paras):
            cum += len(p) + 2  # +2 for the two newlines we removed
            if cum >= target:
                cut_index = i + 1
                break
        left = "\n\n".join(paras[:cut_index]).strip()
        right = "\n\n".join(paras[cut_index:]).strip()
        if not left or not right:
            # Fallback if something went odd
            mid = max(1, total // 2)
            return (text[:mid].strip(), text[mid:].strip())
        return (left, right)
    else:
        # No clear paragraphs; fallback to midpoint split
        mid = max(1, total // 2)
        return (text[:mid].strip(), text[mid:].strip())

# =========================
# 2) Core function — create/load Chroma once
# =========================
def create_or_load_hw4_vectorDB():
    """
    Policy for HW4:
    - If CHROMA_PATH is missing or empty, build the vector DB from HTML docs.
    - If CHROMA_PATH exists AND contains a collection, **do not** re-ingest; just load.

    Ingestion details:
    - Use all *.html / *.htm in HTML_DIR
    - For each file, create exactly TWO chunks (see `chunk_text_into_two`)
    - Store metadata: {"filename": fname, "source_path": path, "chunk": 1 or 2}
    - IDs: use "{filename}#chunk{1|2}"
    """
    # Decide if we need to build from scratch
    need_build = (not os.path.exists(CHROMA_PATH)) or (os.path.exists(CHROMA_PATH) and not os.listdir(CHROMA_PATH))

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    if need_build:
        html_paths = list_html_files(HTML_DIR)
        if not html_paths:
            st.error(f"No HTML files found in {HTML_DIR}. Copy all supplied HTML pages there.")
        else:
            docs: List[str] = []
            ids: List[str] = []
            metadatas: List[Dict] = []

            for p in html_paths:
                raw = read_html_text(p)
                if not raw:
                    continue
                chunk1, chunk2 = chunk_text_into_two(raw)
                fname = os.path.basename(p)

                # Collect exactly two chunks
                for idx, chunk in enumerate((chunk1, chunk2), start=1):
                    if not chunk:
                        continue
                    docs.append(chunk)
                    ids.append(f"{fname}#chunk{idx}")
                    metadatas.append({"filename": fname, "source_path": p, "chunk": idx})

            if docs:
                embs = embed_texts(docs)
                collection.add(
                    documents=docs,
                    ids=ids,
                    metadatas=metadatas,
                    embeddings=embs,
                )
                st.success(f"Ingested {len(docs)} chunk(s) from {len(html_paths)} HTML file(s) into '{COLLECTION_NAME}'.")
            else:
                st.error("No readable text extracted from the provided HTML files.")

    # In all cases, keep a handle in session state
    st.session_state.HW4_vectorDB = {"client": chroma_client, "collection": collection}

# Call once per app run
if "HW4_vectorDB" not in st.session_state:
    create_or_load_hw4_vectorDB()

collection = st.session_state.HW4_vectorDB["collection"]

# =========================
# Retrieval + RAG Prompting
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

BASE_SYSTEM_PROMPT = (
    "You are a helpful iSchool assistant focused on student organizations, events, membership, "
    "leadership, and how to get involved. Prefer grounded answers from the provided HTML sources. "
    "If information is missing or unclear, say so and suggest where to find it."
)

def build_rag_messages(user_q: str, ctx_docs: List[str], ctx_metas: List[Dict], memory_pairs: List[Dict]) -> List[Dict]:
    """
    Construct chat messages for the chosen LLM.

    memory_pairs: last up to 5 Q&A items, each like:
      {"q": "...", "a": "..."}
    We inject them as an additional "system" message to give light conversational memory
    without forcing the model to imitate verbatim past responses.
    """
    # Join top documents with simple separators and attributions
    context_blobs = []
    for i, d in enumerate(ctx_docs):
        fname = (ctx_metas[i].get("filename") if i < len(ctx_metas) else None) or f"doc_{i+1}"
        # keep context bounded
        snippet = d[:4000]
        context_blobs.append(f"[SOURCE: {fname}]\n{snippet}")
    context_text = "\n\n".join(context_blobs).strip()

    # Build a lightweight memory context from prior 5 Q&A turns
    memory_text = ""
    if memory_pairs:
        lines = []
        for pair in memory_pairs[-5:]:
            lines.append(f"Q: {pair.get('q','')}\nA: {pair.get('a','')}")
        memory_text = "\n\n".join(lines).strip()

    sys = {"role": "system", "content": BASE_SYSTEM_PROMPT}

    msgs = [sys]

    if memory_text:
        msgs.append({"role": "system", "content": f"Conversation memory (last turns):\n\n{memory_text}"})

    if context_text:
        msgs.append({"role": "system", "content": f"Use the following context from iSchool HTML pages:\n\n{context_text}"})
        msgs.append({"role": "user", "content": f"{user_q}\n\n(Answer with citations of the used HTML file names if possible.)"})
    else:
        msgs.append({"role": "user", "content": f"{user_q}\n\n(No relevant HTML context retrieved; answer only if confident.)"})

    return msgs

def stream_completion(messages: List[Dict], model: str):
    """Yield streaming tokens from OpenAI."""
    resp = openai_client.chat.completions.create(model=model, messages=messages, stream=True)
    for ch in resp:
        d = ch.choices[0].delta
        if d and getattr(d, "content", None):
            yield d.content

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    model_choice = st.selectbox("Choose LLM", MODEL_CHOICES, index=0)
    top_k = st.slider("Retrieval: top-k chunks", 1, 5, 3)
    st.caption("Tip: If you see sqlite errors on Cloud, this page already applies the pysqlite3 shim.")

# =========================
# Session state for memory buffer + messages
# =========================
# rolling buffer of last 5 Q&A (for memory)
if "qa_memory" not in st.session_state:
    st.session_state.qa_memory = []  # list of {"q": str, "a": str}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about iSchool student organizations (joining, events, leadership, etc.). I’ll use the HTML corpus when helpful."}
    ]

# =========================
# Render chat history
# =========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================
# Chat input
# =========================
user_q = st.chat_input("Ask about iSchool student orgs (e.g., how to join, events, leadership roles)…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Retrieve RAG context
    docs, metas, ids = retrieve_context(user_q, k=top_k)

    # Build messages and stream answer
    rag_msgs = build_rag_messages(user_q, docs, metas, st.session_state.qa_memory)
    with st.chat_message("assistant"):
        st.caption(f"Model: **{model_choice}**")
        answer = st.write_stream(stream_completion(rag_msgs, model_choice)) or ""
        # Show sources if used
        if metas:
            names = [ (m.get("filename") if isinstance(m, dict) else None) or ids[i] for i, m in enumerate(metas) ]
            st.caption("Sources: " + ", ".join(dict.fromkeys(names)))  # de-dup while keeping order
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Update memory buffer (keep last 5)
    st.session_state.qa_memory.append({"q": user_q, "a": answer})
    st.session_state.qa_memory = st.session_state.qa_memory[-5:]