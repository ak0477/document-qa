import os
import re
import sys
import glob
from typing import List, Dict, Tuple

import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup

# Optional: load .env if you choose loadenv instead of Streamlit secrets
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Make Chroma use pysqlite3 (needed on Streamlit Cloud / Python 3.12) ---
import pysqlite3  # noqa: F401
sys.modules['sqlite3'] = pysqlite3
sys.modules['sqlite3.dbapi2'] = pysqlite3

import chromadb

# =========================
# Configuration
# =========================
st.set_page_config(page_title="HW5 — Short-Term Memory Chatbot", page_icon="🤖", layout="wide")

# Update these if your paths differ
HTML_DIR = "/workspaces/document-qa/iSchool_HTMLs"         # Directory for your HTML files
CHROMA_PATH = "/workspaces/document-qa/Chroma_HW4"         # Persistent vector store folder (reusing HW4 index)
COLLECTION_NAME = "HW4_iSchool_StudentOrgs"                # Reuse HW4 collection
EMBED_MODEL = "text-embedding-3-small"

# API key via Streamlit secrets or .env
OPENAI_API_KEY = (
    st.secrets.get("openai", {}).get("api_key")
    or os.getenv("OPENAI_API_KEY")
)
if not OPENAI_API_KEY:
    st.stop()  # Fail early with a clear error in the UI
openai_client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_CHOICES = ["gpt-5-mini", "gpt-4o-mini", "gpt-4o"]
DEFAULT_CHAT_MODEL = MODEL_CHOICES[0]

# =========================
# Page UI
# =========================
st.title("HW5 — Short-Term Memory Chatbot (RAG over HTML)")
st.caption("Enhancement of HW4: vector search → inject retrieved context into the LLM, with short-term memory.")

# =========================
# Utilities
# =========================
def read_html_text(path: str) -> str:
    """Extract visible text from an HTML file using BeautifulSoup."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{2,}", "\n", text)  # Normalize whitespace
        return text.strip()
    except Exception as e:
        st.warning(f"Failed to read {os.path.basename(path)}: {e}")
        return ""

def list_html_files(html_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(html_dir, "*.html")) + glob.glob(os.path.join(html_dir, "*.htm")))

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-embed texts with OpenAI."""
    if not texts:
        return []
    resp = openai_client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [d.embedding for d in resp.data]

# --------- Chunking Strategy (exactly 2 chunks per document) ----------
def split_on_para_boundaries(text: str) -> List[str]:
    """Split raw text into paragraphs (blank lines as separators)."""
    text = text.replace("\r\n", "\n")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras

def chunk_text_into_two(text: str):
    if not text:
        return ("", "")
    paras = split_on_para_boundaries(text)
    total = len(text)
    if len(paras) >= 2:
        cum = 0
        target = total // 2
        cut_index = 0
        for i, p in enumerate(paras):
            cum += len(p) + 2  # +2 for removed newlines
            if cum >= target:
                cut_index = i + 1
                break
        left = "\n\n".join(paras[:cut_index]).strip()
        right = "\n\n".join(paras[cut_index:]).strip()
        return (left, right)
    else:
        mid = max(1, total // 2)
        return (text[:mid].strip(), text[mid:].strip())

# =========================
# 1) Create/load Chroma once (reuse HW4 index)
# =========================
def create_or_load_vector_db():
    """Create/load Chroma vector DB. Reuses HW4 index if already built."""
    need_build = (not os.path.exists(CHROMA_PATH)) or (os.path.exists(CHROMA_PATH) and not os.listdir(CHROMA_PATH))

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    count = collection.count()
    st.info(f"Vector collection **{COLLECTION_NAME}** currently has **{count}** chunks.")

    if need_build:
        html_paths = list_html_files(HTML_DIR)
        if not html_paths:
            st.error(f"No HTML files found in {HTML_DIR}.")
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

                for idx, chunk in enumerate((chunk1, chunk2), start=1):
                    if chunk:
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
                st.success(f"Ingested {len(docs)} chunks from {len(html_paths)} HTML files.")
            else:
                st.error("No readable text extracted from the provided HTML files.")

    st.session_state.HW5_vectorDB = {"client": chroma_client, "collection": collection}
    return collection

if "HW5_vectorDB" not in st.session_state:
    create_or_load_vector_db()
collection = st.session_state.HW5_vectorDB["collection"]

# =========================
# 2) REQUIRED BY HW5: function that returns relevant text from vector search
# =========================
def get_relevant_club_info(query: str, k: int = 3) -> Dict[str, str]:
    """
    Takes a user 'query' and returns a dict with:
      - 'context_text': concatenated top-k snippets
      - 'sources_line': human-friendly list of filenames (for display)
    This is the ONLY place we do the vector search. The LLM never queries the DB.
    """
    try:
        q_emb = openai_client.embeddings.create(input=query, model=EMBED_MODEL).data[0].embedding
        res = collection.query(query_embeddings=[q_emb], n_results=k)
    except Exception as e:
        st.error(f"Error querying Chroma collection: {e}")
        return {"context_text": "", "sources_line": ""}

    docs = res.get("documents", [[]])[0] if res.get("documents") else []
    metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []

    # Build a readable context block and a sources line
    blocks = []
    source_names = []
    for i, d in enumerate(docs):
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        fname = meta.get("filename", f"doc_{i+1}")
        source_names.append(fname)
        snippet = (d or "")[:2000]
        blocks.append(f"[SOURCE: {fname}]\n{snippet}")

    context_text = "\n\n".join(blocks).strip()
    sources_line = ", ".join(dict.fromkeys(source_names))  # de-dup while preserving order
    return {"context_text": context_text, "sources_line": sources_line}

# =========================
# 3) LLM invocation USING the retrieved context (no tool/function calling)
# =========================
BASE_SYSTEM_PROMPT = (
    "You are a helpful iSchool assistant focused on student organizations, events, membership, "
    "leadership, and how to get involved. Prefer grounded answers from the provided HTML sources. "
    "If information is missing or unclear, say so and suggest where to find it."
)

def build_messages(user_q: str, context_text: str, memory_pairs: List[Dict]) -> List[Dict]:
    msgs: List[Dict] = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

    # Short-term memory: include last N QA pairs (e.g., 5)
    if memory_pairs:
        mem_lines = []
        for pair in memory_pairs[-5:]:
            mem_lines.append(f"Q: {pair['q']}\nA: {pair['a']}")
        if mem_lines:
            msgs.append({"role": "system", "content": "Recent conversation (short-term memory):\n\n" + "\n\n".join(mem_lines)})

    if context_text:
        msgs.append({"role": "system", "content": "Use the following retrieved context from iSchool HTML pages:\n\n" + context_text})
        msgs.append({"role": "user", "content": f"{user_q}\n\n(Please cite sources by filename when possible.)"})
    else:
        msgs.append({"role": "user", "content": f"{user_q}\n\n(No relevant context retrieved; answer only if confident.)"})

    return msgs

def stream_answer(messages: List[Dict], model: str):
    resp = openai_client.chat.completions.create(model=model, messages=messages, stream=True)
    for ch in resp:
        d = ch.choices[0].delta
        if d and getattr(d, "content", None):
            yield d.content

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    model_choice = st.selectbox("Choose LLM", MODEL_CHOICES, index=0)
    top_k = st.slider("Retrieval: top-k chunks", 1, 5, 3)
    st.caption("Tip: increase top-k if answers feel under-grounded.")

# =========================
# Session state (short-term memory + chat UI)
# =========================
if "qa_memory" not in st.session_state:
    st.session_state.qa_memory = []  # list of {"q": str, "a": str}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about iSchool student organizations (joining, events, leadership, etc.). I’ll search the HTML corpus and cite sources."}
    ]

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================
# Chat input loop
# =========================
user_q = st.chat_input("Ask about iSchool student orgs (e.g., how to join, events, leadership roles)…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # REQUIRED BY HW5: Do vector search FIRST via dedicated function
    search_out = get_relevant_club_info(user_q, k=top_k)
    context_text = search_out["context_text"]
    sources_line = search_out["sources_line"]

    # Then build messages & stream the model response
    rag_msgs = build_messages(user_q, context_text, st.session_state.qa_memory)
    with st.chat_message("assistant"):
        st.caption(f"Model: **{model_choice}**")
        answer = st.write_stream(stream_answer(rag_msgs, model=model_choice)) or ""
        if sources_line:
            st.caption("Sources: " + sources_line)

    # Track messages + short-term memory (keep last 5)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.qa_memory.append({"q": user_q, "a": answer})
    st.session_state.qa_memory = st.session_state.qa_memory[-5:]
