"""
HW7 — Human-Centered News Info Bot (Allowed Libs Only)
=====================================================
- Ingest a CSV of news stories (no pandas)
- Answer:
   • "find the most interesting news"
   • "find news about <topic>"
- RAG-style retrieval over CSV rows via Chroma
- Transparent ranking with explanations
- Works with or without an OpenAI key
- Uses only: streamlit, openai, bs4 (optional), pysqlite3 shim, chromadb, stdlib

Secrets (optional):
Create /workspaces/document-qa/.streamlit/secrets.toml with:

[openai]
api_key = "sk-..."

Quickstart:
    streamlit run HW7.py
"""

import os
import re
import csv
import sys
import math
from datetime import datetime, timezone, date
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

# Optional OpenAI client (app runs without it)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# Optional (not required, allowed by your sample)
try:
    from bs4 import BeautifulSoup  # noqa: F401
except Exception:
    BeautifulSoup = None  # type: ignore

# --- Make Chroma use pysqlite3 (needed on Streamlit Cloud / Python 3.12) ---
try:
    import pysqlite3  # noqa: F401
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3
except Exception:
    pass

import chromadb

# =========================
# App Config
# =========================
st.set_page_config(page_title="HW7 — News Info Bot (Concise)", page_icon="📰", layout="wide")

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
VECTOR_PATH = os.path.join(os.getcwd(), ".newsbot_chroma")  # writable in Codespaces
COLLECTION_NAME = "hw7_news_csv"

# Suggested field names you can map to from your CSV
DEFAULT_TEXT_FIELDS = ["title", "summary", "description", "content", "body", "snippet"]
LEGAL_JURISDICTIONS = ["US", "EU", "UK", "CN", "IN", "JP", "CA", "AU", "BR"]

# Human-centered weights for “interestingness”
WEIGHTS = {"recency": 0.40, "relevance": 0.40, "authority": 0.10, "impact": 0.10}

SOURCE_AUTHORITY = {
    "Reuters": 0.95,
    "Bloomberg": 0.90,
    "Financial Times": 0.90,
    "AP": 0.88,
    "Wall Street Journal": 0.88,
}

IMPACT_KEYWORDS = [
    "lawsuit", "litigation", "settlement", "fine", "regulator", "antitrust", "merger",
    "acquisition", "DOJ", "FTC", "CMA", "EU", "GDPR", "privacy", "sanction", "ban", "compliance",
]

# =========================
# Helpers (secrets, dates, scoring)
# =========================

def _secret(section: str, key: str) -> Optional[str]:
    """Safe secrets getter: returns None if secrets.toml is missing."""
    try:
        sect = st.secrets.get(section)  # may raise if secrets not loaded
        return (sect or {}).get(key) if isinstance(sect, dict) else None
    except Exception:
        return None

def get_openai_client() -> Optional["OpenAI"]:
    key = os.getenv("OPENAI_API_KEY") or _secret("openai", "api_key")
    if not key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

def normalize_date(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None

def recency_score(d: Optional[datetime]) -> float:
    if not d:
        return 0.3
    days = max(0.0, (datetime.now(timezone.utc) - d).total_seconds() / 86400.0)
    return 1.0 / (1.0 + days / 7.0)  # ~weekly half-life

def authority_score(source: str) -> float:
    if not source:
        return 0.6
    s = source.lower()
    for k, v in SOURCE_AUTHORITY.items():
        if k.lower() in s:
            return v
    return 0.6

def impact_score(text: str) -> float:
    tl = text.lower()
    hits = sum(1 for kw in IMPACT_KEYWORDS if kw in tl)
    return min(1.0, 0.25 + 0.15 * hits)

def simple_relevance(text: str, query: str) -> float:
    """Dependency-free relevance: token overlap ratio."""
    if not query:
        return 0.7
    tq = [w for w in re.findall(r"\w+", query.lower()) if w]
    if not tq:
        return 0.7
    tt = re.findall(r"\w+", text.lower())
    if not tt:
        return 0.0
    qs, ts = set(tq), set(tt)
    inter = len(qs.intersection(ts))
    return min(1.0, inter / max(1, len(qs)))

def interestingness(source: str, date_val: Optional[datetime], combined_text: str, query: str) -> Tuple[float, Dict[str, float]]:
    r = recency_score(date_val)
    a = authority_score(source or "")
    imp = impact_score(combined_text or "")
    rel = simple_relevance(combined_text or "", query or "")
    total = WEIGHTS["recency"]*r + WEIGHTS["relevance"]*rel + WEIGHTS["authority"]*a + WEIGHTS["impact"]*imp
    return float(total), {"recency": r, "relevance": rel, "authority": a, "impact": imp}

# =========================
# Chroma (vector store)
# =========================

def ensure_chroma() -> chromadb.Client:
    try:
        os.makedirs(VECTOR_PATH, exist_ok=True)
        return chromadb.PersistentClient(path=VECTOR_PATH)
    except Exception as e:
        st.warning(f"Chroma storage unavailable ({e}); using in-memory index for this session.")
        return chromadb.Client()  # ephemeral fallback

def get_or_create_collection(client: chromadb.Client) -> chromadb.Collection:
    return client.get_or_create_collection(name=COLLECTION_NAME)

def embed_texts(texts: List[str]) -> List[List[float]]:
    cli = get_openai_client()
    if not cli:
        # Deterministic, tiny vectors so the app still runs without keys
        embs: List[List[float]] = []
        for t in texts:
            seed = (hash(t) % 997) / 997.0
            embs.append([((seed + i*0.03125) % 1.0) for i in range(32)])
        return embs
    resp = cli.embeddings.create(model=DEFAULT_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def _sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata must be Bool/Int/Float/Str/SparseVector (no None)."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, (bool, int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out

def build_rag_from_rows(rows: List[Dict[str, Any]], text_fields: List[str]) -> chromadb.Collection:
    client = ensure_chroma()
    coll = get_or_create_collection(client)
    # Clear & rebuild (simple for HW)
    try:
        ids_all = coll.get()["ids"]
        if ids_all:
            coll.delete(ids=ids_all)
    except Exception:
        pass

    docs, ids, metas = [], [], []
    for i, row in enumerate(rows):
        parts: List[str] = []
        for f in text_fields:
            v = row.get(f)
            if v is not None:
                s = str(v).strip()
                if s:
                    parts.append(s)
        if not parts:
            continue
        text = "\n\n".join(parts)
        ids.append(f"row-{i}")
        metas.append(_sanitize_meta(row))
        docs.append(text)

    if docs:
        coll.add(documents=docs, metadatas=metas, ids=ids, embeddings=embed_texts(docs))
    return coll

def query_rag(coll: chromadb.Collection, query: str, k: int = 8) -> Dict[str, Any]:
    cli = get_openai_client()
    if cli:
        q_emb = cli.embeddings.create(model=DEFAULT_EMBED_MODEL, input=query or "news").data[0].embedding
        return coll.query(query_embeddings=[q_emb], n_results=k)

    # Fallback: naive keyword relevance across stored documents
    got = coll.get()
    corpus = got["documents"]
    metas = got["metadatas"]
    scores: List[Tuple[int, float]] = []
    for idx, doc in enumerate(corpus):
        scores.append((idx, simple_relevance(doc, query or "news")))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:k]
    return {
        "documents": [[corpus[i] for i, _ in top]],
        "metadatas": [[metas[i] for i, _ in top]],
        "ids": [[got["ids"][i] for i, _ in top]],
    }

# =========================
# CSV loading (no pandas)
# =========================

def load_csv_to_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: v for k, v in r.items()})
    return rows

def read_uploaded_csv(uploaded_file) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    uploaded_file.seek(0)
    reader = csv.DictReader(text.splitlines())
    for r in reader:
        rows.append({k: v for k, v in r.items()})
    return rows

# =========================
# Streamlit UI
# =========================

st.title("📰 HW7 — Human-Centered News Info Bot")
st.caption("RAG over your CSV with transparent, human-centered ranking. (No extra libraries.)")

with st.sidebar:
    st.header("Setup")
    uploaded = st.file_uploader("Upload news CSV", type=["csv"])

    # Try a couple of default locations for a sample file
    candidate_paths = [
        "/mnt/data/Example_news_info_for_testing.csv",
        os.path.join(os.getcwd(), "Example_news_info_for_testing.csv"),
    ]
    default_path = next((p for p in candidate_paths if os.path.exists(p)), "")
    use_example = st.toggle("Use built-in example CSV (if found)", value=bool(default_path) and not uploaded)

    st.subheader("Filters & Retrieval")
    enable_jurisdiction_filter = st.checkbox("Enable jurisdiction filter", value=True)
    allowed_juris = st.multiselect("Jurisdictions (if column exists)", LEGAL_JURISDICTIONS, default=["US", "EU", "UK"])
    min_date = st.date_input("Min publish date (optional)")
    top_k = st.slider("Top-k results", 3, 10, 6)
    show_explain = st.checkbox("Show scoring breakdown", value=True)

# Load rows
if uploaded is not None:
    data_rows = read_uploaded_csv(uploaded)
elif use_example and default_path:
    data_rows = load_csv_to_rows(default_path)
else:
    st.warning("Please upload a CSV (or place an example at one of the default paths).")
    st.stop()

if not data_rows:
    st.error("CSV appears empty or unreadable.")
    st.stop()

# Field mapping
st.subheader("Map CSV Columns")
all_cols = sorted(list({k for r in data_rows for k in r.keys()}))
text_fields = st.multiselect(
    "Text fields (used to search & rank)",
    options=all_cols,
    default=[c for c in all_cols if c and c.lower() in DEFAULT_TEXT_FIELDS][:2] or all_cols[:2],
)
url_field = st.selectbox("URL field (optional)", options=["<none>"] + all_cols, index=0)
source_field = st.selectbox("Source field (optional)", options=["<none>"] + all_cols, index=0)
date_field = st.selectbox("Date field (optional)", options=["<none>"] + all_cols, index=0)
juris_field = st.selectbox("Jurisdiction/Region field (optional)", options=["<none>"] + all_cols, index=0)

# Normalize row helpers
for r in data_rows:
    r["_date"] = normalize_date(r.get(date_field)) if (date_field != "<none>" and date_field in r) else None
    r["_source"] = r.get(source_field, "") if source_field != "<none>" else ""
    r["_url"] = r.get(url_field, "") if url_field != "<none>" else r.get("url", "")
    r["_juris"] = r.get(juris_field, "") if juris_field != "<none>" else ""

# Filters
if enable_jurisdiction_filter and juris_field != "<none>":
    data_rows = [r for r in data_rows if (r.get("_juris") in allowed_juris) or (not r.get("_juris"))]
if isinstance(min_date, date):
    data_rows = [r for r in data_rows if (r.get("_date") is None) or (r.get("_date").date() >= min_date)]

st.success(f"Loaded {len(data_rows):,} stories. Building vector index…")
coll = build_rag_from_rows(data_rows, text_fields)

# Query
st.subheader("Ask the bot")
query_in = st.text_input("Try: 'find the most interesting news' or 'find news about <topic>'", value="find the most interesting news")

if st.button("Search & Rank", type="primary"):
    topical = "" if "most interesting" in query_in.lower() else query_in

    # Candidate retrieval
    res = query_rag(coll, topical or "news", k=top_k*3)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    results: List[Dict[str, Any]] = []
    for doc, meta in zip(docs, metas):
        date_val = meta.get("_date")  # may already be ISO; normalize if not
        if isinstance(date_val, str):
            date_val = normalize_date(date_val)
        source = meta.get("_source", "")
        score, parts = interestingness(source, date_val, doc, topical)
        title = meta.get("title") or meta.get("headline") or (doc.split("\n")[0][:120] if doc else "Untitled")
        url = meta.get("_url", "") or meta.get("url", "")
        snippet = doc[:300] + ("…" if len(doc) > 300 else "")
        results.append({"score": float(score), "parts": parts, "title": title, "url": url, "source": source, "snippet": snippet})

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    st.markdown("### Ranked Results")
    for i, r in enumerate(ranked, start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {r['title']}**")
            if r["url"]:
                st.markdown(f"[Open story]({r['url']})  |  Source: _{r['source'] or 'N/A'}_")
            else:
                st.markdown(f"Source: _{r['source'] or 'N/A'}_")
            st.write(r["snippet"])
            if show_explain:
                st.caption(
                    f"Why ranked → recency: {r['parts']['recency']:.2f}, "
                    f"relevance: {r['parts']['relevance']:.2f}, "
                    f"authority: {r['parts']['authority']:.2f}, "
                    f"impact: {r['parts']['impact']:.2f}  (weights {WEIGHTS})"
                )

st.divider()
st.markdown("#### Notes")
st.write(
    "This concise build uses only your allowed libraries. It provides RAG over CSV with transparent ranking and "
    "jurisdiction/date filters. If no OpenAI key is set, the app uses deterministic fallback vectors so it still runs."
)
