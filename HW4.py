import os
import re
import sys
import glob
from typing import List, Dict, Tuple

import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup

# --- Make Chroma use pysqlite3 (needed on Streamlit Cloud / Python 3.12) ---
import pysqlite3  # noqa: F401
sys.modules['sqlite3'] = pysqlite3
sys.modules['sqlite3.dbapi2'] = pysqlite3

import chromadb

# =========================
# Configuration
# =========================
HTML_DIR = "/workspaces/document-qa/iSchool_HTMLs"  # Directory for your HTML files
CHROMA_PATH = "/workspaces/document-qa/Chroma_HW4"  # Persistent vector store folder
COLLECTION_NAME = "HW4_iSchool_StudentOrgs"
EMBED_MODEL = "text-embedding-3-small"

OPENAI_API_KEY = st.secrets["openai"]["api_key"].strip()

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

        # Remove non-content elements
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
    text = text.replace("\r\n", "\n")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras

def chunk_text_into_two(text: str) -> Tuple[str, str]:
    if not text:
        return ("", "")
    paras = split_on_para_boundaries(text)
    total = len(text)
    if len(paras) >= 2:
        cum = 0
        target = total // 2
        cut_index = 0
        for i, p in enumerate(paras):
            cum += len(p) + 2  # +2 for the two newlines we removed
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
# 2) Core function — create/load Chroma once
# =========================
def create_or_load_hw4_vectorDB():
    """Create/load Chroma vector DB."""
    need_build = (not os.path.exists(CHROMA_PATH)) or (os.path.exists(CHROMA_PATH) and not os.listdir(CHROMA_PATH))

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # Log the number of documents in the collection
    count = collection.count()
    st.write(f"Collection has {count} documents.")  # This confirms that the collection is being loaded

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

            st.write(f"Documents to be added: {len(docs)}")  # Log number of documents to be embedded
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

    # Store collection in session state for later use
    st.session_state.HW4_vectorDB = {"client": chroma_client, "collection": collection}
    return collection

# Initialize the Chroma DB and collection when the app runs
if "HW4_vectorDB" not in st.session_state:
    create_or_load_hw4_vectorDB()

collection = st.session_state.HW4_vectorDB["collection"]

# =========================
# Retrieval + RAG Prompting
# =========================
def retrieve_context(query: str, k: int = 3):
    """Return top-k doc snippets + metadata for a query using embeddings."""
    # Ensure the collection is available in session state
    if "HW4_vectorDB" not in st.session_state:
        st.error("Chroma vector DB is not initialized!")
        return [], [], []

    # Get the collection from session state
    collection = st.session_state.HW4_vectorDB["collection"]

    try:
        # Generate query embeddings
        q_emb = openai_client.embeddings.create(input=query, model=EMBED_MODEL).data[0].embedding
        # Perform the query on the collection
        res = collection.query(query_embeddings=[q_emb], n_results=k)
    except Exception as e:
        st.error(f"Error querying Chroma collection: {e}")
        return [], [], []

    docs = res.get("documents", [[]])[0] if res.get("documents") else []
    metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    ids = res.get("ids", [[]])[0] if res.get("ids") else []

    return docs, metas, ids

# Base system prompt for the bot
BASE_SYSTEM_PROMPT = (
    "You are a helpful iSchool assistant focused on student organizations, events, membership, "
    "leadership, and how to get involved. Prefer grounded answers from the provided HTML sources. "
    "If information is missing or unclear, say so and suggest where to find it."
)

def build_rag_messages(user_q: str, ctx_docs: List[str], ctx_metas: List[Dict], memory_pairs: List[Dict]) -> List[Dict]:
    """Construct chat messages for the chosen LLM."""
    context_blobs = []
    for i, d in enumerate(ctx_docs):
        fname = ctx_metas[i].get("filename", f"doc_{i+1}")
        snippet = d[:4000]  # Trim document if it's too long
        context_blobs.append(f"[SOURCE: {fname}]\n{snippet}")
    
    context_text = "\n\n".join(context_blobs).strip()

    # Add memory if available
    memory_text = ""
    if memory_pairs:
        lines = []
        for pair in memory_pairs[-5:]:
            lines.append(f"Q: {pair['q']}\nA: {pair['a']}")
        memory_text = "\n\n".join(lines).strip()

    sys = {"role": "system", "content": BASE_SYSTEM_PROMPT}

    msgs = [sys]

    if memory_text:
        msgs.append({"role": "system", "content": f"Conversation memory (last turns):\n\n{memory_text}"})

    if context_text:
        msgs.append({"role": "system", "content": f"Use the following context from iSchool HTML pages:\n\n{context_text}"})
        msgs.append({"role": "user", "content": f"{user_q}\n\n(Answer with citations from the HTML files.)"})
    else:
        msgs.append({"role": "user", "content": f"{user_q}\n\n(No relevant context retrieved; answer only if confident.)"})

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

# =========================
# Session state for memory buffer + messages
# =========================
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
