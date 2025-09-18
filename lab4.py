import os
import glob
import sys
import re
from typing import List

import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader

# ---- Ensure sqlite3 uses pysqlite3 for Chroma (esp. on Py 3.12) ----
import pysqlite3  # noqa: F401
sys.modules['sqlite3'] = pysqlite3
sys.modules['sqlite3.dbapi2'] = pysqlite3

import chromadb

# =========================
# Config
# =========================
PDF_DIR = "/workspaces/document-qa/PDFs"                 # <-- put your 7 PDFs here
CHROMA_PATH = "./Chroma_for_lab"   # persistent store for Chroma
COLLECTION_NAME = "Lab4Collection"
EMBED_MODEL = "text-embedding-3-small"

# expects .streamlit/secrets.toml:
# [openai]
# api_key = "sk-..."
OPENAI_API_KEY = st.secrets["openai"]["api_key"].strip()

st.title("Lab 4 - ChromaDB (Build & Test Vector DB)")

# Single OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Utilities
# =========================
def read_pdf_text(path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(path)
        chunks = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                chunks.append(txt)
        return "\n".join(chunks).strip()
    except Exception as e:
        st.warning(f"Failed to read {os.path.basename(path)}: {e}")
        return ""

def list_pdf_files(pdf_dir: str, limit: int = 7) -> List[str]:
    files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    return files[:limit]

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-embed texts with OpenAI to save round trips."""
    if not texts:
        return []
    resp = openai_client.embeddings.create(
        input=texts,
        model=EMBED_MODEL
    )
    return [d.embedding for d in resp.data]

# =========================
# Core function (per spec)
# =========================
def create_or_load_lab4_vectorDB():
    """
    - Construct a ChromaDB collection named “Lab4Collection”
    - Use OpenAI embeddings (text-embedding-3-small)
    - Read 7 PDFs and convert to text
    - Use filename as key; store in metadatas as needed
    - Store the vector DB handle in st.session_state.Lab4_vectorDB
    - Create once per application run (reduce embedding cost)
    """
    # Persistent Chroma client + collection
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # If collection already has items, skip ingest to avoid duplicate embedding cost
    try:
        count = collection.count()
    except Exception:
        # Some older chroma versions may not have .count(); fall back
        try:
            _ = collection.get(limit=1)
            count = 1
        except Exception:
            count = 0

    if count == 0:
        # Ingest up to 7 PDFs
        pdf_paths = list_pdf_files(PDF_DIR, limit=7)
        if not pdf_paths:
            st.error(f"No PDFs found in {PDF_DIR}. Add your 7 PDFs there.")
        else:
            docs = []
            ids = []
            metadatas = []
            for p in pdf_paths:
                text = read_pdf_text(p)
                if not text:
                    continue
                fname = os.path.basename(p)
                docs.append(text)
                ids.append(fname)  # use filename as unique key
                metadatas.append({"filename": fname, "source_path": p})

            if docs:
                # Batch-embed and add to Chroma
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

    # Cache the handle in session_state
    st.session_state.Lab4_vectorDB = {
        "client": chroma_client,
        "collection": collection,
    }

# =========================
# Call the function once per run
# =========================
if "Lab4_vectorDB" not in st.session_state:
    create_or_load_lab4_vectorDB()
else:
    st.info("Vector DB already loaded from session cache.")

# =========================
# Test the vector DB (no chat yet)
# =========================
collection = st.session_state.Lab4_vectorDB["collection"]

TEST_QUERIES = ["Generative AI", "Text Mining", "Data Science Overview"]

st.subheader("Vector DB Test Queries")
for q in TEST_QUERIES:
    # embed the query
    q_emb = openai_client.embeddings.create(
        input=q,
        model=EMBED_MODEL
    ).data[0].embedding

    # run similarity search
    try:
        res = collection.query(query_embeddings=[q_emb], n_results=3)
    except TypeError:
        # Older Chroma might need 'query_texts' instead of embeddings. Try fallback:
        res = collection.query(query_texts=[q], n_results=3)

    ids = res.get("ids", [[]])[0] if res.get("ids") else []
    metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []

    st.write(f"**Query:** _{q}_")
    if ids:
        # Render ordered list of returned document filenames
        for idx, md in enumerate(metas, start=1):
            # Prefer metadata filename; fallback to id
            name = (md.get("filename") if isinstance(md, dict) else None) or ids[idx-1]
            st.write(f"{idx}. {name}")
    else:
        st.write("No results found.")
