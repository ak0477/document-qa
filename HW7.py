import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anthropic
import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI


st.set_page_config(
    page_title="HW7 — Legal News Intelligence Bot",
    page_icon="🗞️",
    layout="wide",
)


@dataclass
class ModelChoice:
    vendor: str
    tier: str
    model: str
    label: str
    is_budget: bool


LAW_KEYWORD_WEIGHTS: Dict[str, float] = {
    "litigation": 1.0,
    "lawsuit": 1.0,
    "regulator": 0.9,
    "regulation": 0.9,
    "merger": 0.85,
    "acquisition": 0.85,
    "antitrust": 1.0,
    "sanction": 0.95,
    "fine": 0.8,
    "settlement": 0.9,
    "compliance": 0.75,
    "cybersecurity": 0.7,
    "privacy": 0.7,
    "intellectual property": 0.85,
    "patent": 0.75,
    "trade secret": 0.9,
    "class action": 1.0,
    "employment": 0.55,
    "labor": 0.55,
    "tax": 0.6,
    "fraud": 0.9,
    "governance": 0.6,
    "esg": 0.5,
}

PREFERRED_TEXT_COLUMNS: Sequence[str] = (
    "title",
    "headline",
    "summary",
    "description",
    "content",
    "body",
)

PREFERRED_META_COLUMNS: Sequence[str] = (
    "topic",
    "sector",
    "industry",
    "region",
    "jurisdiction",
    "source",
)

EMBED_MODEL = "text-embedding-3-small"

MODEL_REGISTRY: List[ModelChoice] = [
    ModelChoice("OpenAI", "budget", "gpt-4o-mini", "OpenAI gpt-4o-mini (fast, lower cost)", True),
    ModelChoice("OpenAI", "premium", "gpt-4o", "OpenAI gpt-4o (enhanced reasoning)", False),
    ModelChoice("Anthropic", "budget", "claude-3-haiku-20240307", "Anthropic Claude 3 Haiku (budget)", True),
    ModelChoice("Anthropic", "premium", "claude-3-sonnet-20240229", "Anthropic Claude 3 Sonnet (premium)", False),
]


def get_model_options(vendor: Optional[str] = None) -> List[ModelChoice]:
    if vendor:
        return [m for m in MODEL_REGISTRY if m.vendor == vendor]
    return list(MODEL_REGISTRY)


def pick_model(vendor: str, tier: str) -> ModelChoice:
    for entry in MODEL_REGISTRY:
        if entry.vendor == vendor and entry.tier == tier:
            return entry
    raise ValueError(f"Unknown model selection for vendor={vendor}, tier={tier}")


def get_api_keys() -> Tuple[Optional[str], Optional[str]]:
    openai_key = (
        st.secrets.get("openai", {}).get("api_key")
        if st.secrets
        else None
    ) or os.getenv("OPENAI_API_KEY")
    anthropic_key = (
        st.secrets.get("anthropic", {}).get("api_key")
        if st.secrets
        else None
    ) or os.getenv("ANTHROPIC_API_KEY")
    return openai_key, anthropic_key


OPENAI_KEY, ANTHROPIC_KEY = get_api_keys()

if not OPENAI_KEY:
    st.warning("OpenAI API key is required to embed the news collection and run the bot.")
    st.stop()


openai_client = OpenAI(api_key=OPENAI_KEY)
anthropic_client: Optional[anthropic.Anthropic] = None
if ANTHROPIC_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def read_news_csv(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    df = df.replace({np.nan: None})
    df.columns = [c.strip() for c in df.columns]
    return df


def concatenate_text(row: Dict[str, Any]) -> str:
    sections: List[str] = []
    for col in PREFERRED_TEXT_COLUMNS:
        value = row.get(col)
        if value and isinstance(value, str):
            sections.append(f"{col.title()}: {value.strip()}")
    if not sections:
        fallback_parts = [str(v) for v in row.values() if isinstance(v, str)]
        sections = fallback_parts[:3]
    return "\n".join(sections).strip()


def summarize_text_snippet(text: str, max_chars: int = 320) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].strip() + "..."


def compute_law_impact_score(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    for keyword, weight in LAW_KEYWORD_WEIGHTS.items():
        if keyword in lowered:
            score += weight
    return score


def extract_date(row: Dict[str, Any]) -> Optional[datetime]:
    for candidate in ("date", "published", "publish_date", "time", "timestamp"):
        value = row.get(candidate)
        if not value:
            continue
        try:
            parsed = pd.to_datetime(value, utc=True)
        except Exception:
            continue
        if isinstance(parsed, pd.Series):
            parsed = parsed.iloc[0]
        if pd.isna(parsed):
            continue
        return parsed.to_pydatetime()
    return None


def compute_recency_score(published_at: Optional[datetime]) -> float:
    if not published_at:
        return 0.0
    now = datetime.utcnow()
    delta_days = max((now - published_at).days, 0)
    return 1.0 / math.log(delta_days + 2.0)


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    if not texts:
        return []
    response = openai_client.embeddings.create(input=list(texts), model=EMBED_MODEL)
    return [record.embedding for record in response.data]


def build_vector_store(df: pd.DataFrame):
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="hw7_news",
        metadata={"hnsw:space": "cosine"},
    )

    docs: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for row_index, row_series in df.iterrows():
        row = row_series.to_dict()
        document_text = concatenate_text(row)
        if not document_text:
            continue
        summary = summarize_text_snippet(document_text)
        published_at = extract_date(row)
        law_score = compute_law_impact_score(document_text)
        recency_score = compute_recency_score(published_at)
        base_score = law_score * 0.7 + recency_score * 0.3
        meta: Dict[str, Any] = {
            "row_index": int(row_index),
            "title": row.get("title") or row.get("headline") or summary.split("\n")[0],
            "summary": summary,
            "law_score": law_score,
            "recency_score": recency_score,
            "base_score": base_score,
            "published_at": published_at.isoformat() if published_at else None,
        }
        for meta_col in PREFERRED_META_COLUMNS:
            value = row.get(meta_col)
            if value:
                meta[meta_col] = value
        meta["raw_row"] = row
        meta["document_text"] = document_text

        docs.append(document_text)
        metadatas.append(meta)
        ids.append(f"row-{row_index}")

    embeddings = embed_texts(docs)
    if embeddings:
        collection.add(
            documents=docs,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )

    return collection, metadatas


def render_dataset_overview(df: pd.DataFrame):
    st.markdown("### Uploaded news dataset")
    st.caption("Only the first 1,000 rows are displayed for preview purposes.")
    st.dataframe(df.head(1000))


def format_candidate_entry(meta: Dict[str, Any]) -> str:
    title = meta.get("title") or "Untitled"
    source = meta.get("source")
    topic = meta.get("topic") or meta.get("sector") or meta.get("industry")
    published = meta.get("published_at")
    summary = meta.get("summary") or "(summary unavailable)"
    extras = []
    if topic:
        extras.append(f"Topic: {topic}")
    if source:
        extras.append(f"Source: {source}")
    if published:
        extras.append(f"Published: {published[:10]}")
    extras.append(f"Law impact score: {meta.get('law_score', 0):.2f}")
    extras.append(f"Recency score: {meta.get('recency_score', 0):.2f}")
    extras_text = " | ".join(extras)
    return f"{title}\n{extras_text}\nSummary: {summary}"


def clean_json_output(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        chunks = cleaned.split("```")
        if len(chunks) >= 2:
            cleaned = chunks[1]
    cleaned = cleaned.strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            return json.loads(cleaned.replace("'", '"'))
        except json.JSONDecodeError:
            return {"raw_output": text}


def call_openai_chat(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    response = openai_client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or "{}"


def call_anthropic_chat(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    if not anthropic_client:
        raise RuntimeError("Anthropic API key unavailable")
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=1500,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text_blocks = [block.text for block in response.content if block.type == "text"]
    combined = "\n".join(text_blocks)
    return combined


def generate_ranked_results(
    task_description: str,
    candidates: Sequence[Dict[str, Any]],
    vendor: str,
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    system_prompt = (
        "You are a news intelligence analyst embedded in a global law firm. "
        "Rank news for attorneys by considering potential legal risk, compliance impact, "
        "deal flow, and client advisory opportunities. Always justify rankings using the facts provided. "
        "Respond with a JSON object containing `ranked_items` (an array of objects with "
        "fields id, title, reason, risk_level from 1-5, opportunity from 1-5) and "
        "`analysis_summary` (a short paragraph)."
    )
    lines = [
        "Evaluate the following candidate stories and fulfill the task described.",
        f"Task: {task_description}",
        "Candidates:",
    ]
    for idx, meta in enumerate(candidates, start=1):
        lines.append(
            json.dumps(
                {
                    "candidate_id": f"{meta['row_index']}",
                    "title": meta.get("title"),
                    "summary": meta.get("summary"),
                    "law_score": meta.get("law_score"),
                    "recency_score": meta.get("recency_score"),
                    "published_at": meta.get("published_at"),
                    "topic": meta.get("topic") or meta.get("sector") or meta.get("industry"),
                },
                ensure_ascii=False,
            )
        )
    user_prompt = "\n".join(lines)

    if vendor == "OpenAI":
        raw = call_openai_chat(model, system_prompt, user_prompt, temperature)
    elif vendor == "Anthropic":
        raw = call_anthropic_chat(model, system_prompt, user_prompt, temperature)
    else:
        raise ValueError(f"Unsupported vendor: {vendor}")

    parsed = clean_json_output(raw)
    return parsed


def answer_question_with_rag(
    question: str,
    collection: chromadb.api.models.Collection.Collection,
    vendor: str,
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    n_results = min(6, max(collection.count(), 3))
    query_response = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    metadatas = query_response.get("metadatas", [[]])[0]
    documents = query_response.get("documents", [[]])[0]

    context_lines = []
    for meta, doc in zip(metadatas, documents):
        context_lines.append(
            json.dumps(
                {
                    "candidate_id": meta.get("row_index"),
                    "title": meta.get("title"),
                    "summary": meta.get("summary"),
                    "document": summarize_text_snippet(doc, 420),
                    "topic": meta.get("topic") or meta.get("sector") or meta.get("industry"),
                    "published_at": meta.get("published_at"),
                },
                ensure_ascii=False,
            )
        )

    system_prompt = (
        "You answer attorney questions using the supplied news context. "
        "Highlight legal risk, regulatory exposure, affected clients, and next steps. "
        "If the answer is uncertain, say so. Provide citations using the candidate_id field." 
    )
    user_prompt = "\n".join(
        [
            f"Question: {question}",
            "Context entries:",
            *context_lines,
            "Respond with a JSON object containing `answer` (markdown string) and "
            "`used_items` (array of candidate_id strings).",
        ]
    )

    if vendor == "OpenAI":
        raw = call_openai_chat(model, system_prompt, user_prompt, temperature)
    elif vendor == "Anthropic":
        raw = call_anthropic_chat(model, system_prompt, user_prompt, temperature)
    else:
        raise ValueError(f"Unsupported vendor: {vendor}")

    parsed = clean_json_output(raw)
    parsed["retrieved_metadatas"] = metadatas
    return parsed


def describe_architecture():
    st.markdown(
        """
        ### Architecture & Techniques
        * **Vector RAG pipeline** – uploaded CSV rows are embedded with `text-embedding-3-small` and loaded into an in-memory Chroma collection. Metadata stores titles, summaries, law-impact heuristics, and recency information.
        * **Domain heuristics** – before calling an LLM, the app scores every article with a lightweight legal-impact heuristic built from risk keywords and recency. This improves recall for law-firm-relevant items and keeps prompts focused.
        * **LLM ranking layer** – for "most interesting" and topical rankings, shortlisted candidates are sent to the selected LLM (OpenAI or Anthropic) with instructions to output structured JSON containing rank, reasoning, and quantified risk/opportunity ratings.
        * **Question answering** – free-form questions run a vector search over the collection, then the LLM generates an answer with citations to the candidate IDs used.
        * **Model comparison workflow** – the sidebar lets you pick a *budget* and a *premium* model from two vendors. The comparison pane renders their rankings side-by-side to make evaluation straightforward.
        """
    )

    st.markdown(
        """
        ### Evaluation Strategy
        * **Semantic spot-checks** – inspect the retrieved context snippets and the LLM's ranked rationales to ensure they align with the CSV data. Mismatches show up quickly because each response cites candidate IDs.
        * **Heuristic baseline vs. LLM output** – the app exposes the heuristic base scores so you can confirm that the LLM is improving on (or at least not contradicting) the deterministic ranking.
        * **A/B model comparison** – run the same task across the budget and premium models. Differences in ordering and rationale indicate sensitivity to reasoning depth. Consistency on high-signal stories builds confidence in the ranking quality.
        * **Topic-specific regression** – use the "news about a topic" task with repeated keywords (e.g., antitrust, cybersecurity) to validate that targeted retrieval remains stable across runs.
        """
    )

    st.markdown(
        """
        ### Model Selection Guidance
        * **OpenAI gpt-4o-mini** – lower cost, strong for rapid triage of large batches.
        * **OpenAI gpt-4o** – higher reasoning ability for nuanced legal analysis (M&A, regulatory interpretation).
        * **Anthropic Claude 3 Haiku** – fast baseline from a second vendor to diversify risk.
        * **Anthropic Claude 3 Sonnet** – more expensive Anthropic option for deeper qualitative scoring.
        """
    )


def describe_testing_guidance():
    st.markdown(
        """
        ### How to Validate Rankings
        1. Run "Most interesting news" with the budget model and review whether the top stories carry the highest legal impact scores. Spot-check the raw summaries via the dataset table.
        2. Switch to the premium model and re-run. If the order changes, read the rationale to verify whether deeper legal reasoning (e.g., multi-jurisdiction implications) justifies the change.
        3. For topic-specific queries, verify that the surfaced IDs contain the keyword or related legal concepts within their text. Compare vendor outputs to ensure coverage parity.
        4. Use the cited candidate IDs to trace back to the CSV rows; discrepancies reveal gaps in either the heuristic shortlist or the LLM explanation.
        """
    )


def render_comparison(results: Dict[str, Any], header: str):
    ranked_items = results.get("ranked_items", [])
    if not ranked_items:
        st.info("No ranked items returned.")
        return
    for idx, item in enumerate(ranked_items, start=1):
        with st.container():
            st.markdown(
                f"**{idx}. {item.get('title', 'Untitled')}** — Risk {item.get('risk_level', 'N/A')} | "
                f"Opportunity {item.get('opportunity', 'N/A')}"
            )
            st.markdown(item.get("reason", "(no explanation)"))
            st.caption(f"Candidate ID: {item.get('id')} | {header}")


def render_answer(answer_payload: Dict[str, Any]):
    st.markdown(answer_payload.get("answer", "(no answer)"))
    used_items = answer_payload.get("used_items", [])
    if used_items:
        st.caption(f"Cited candidate IDs: {', '.join(str(item) for item in used_items)}")


def main():
    st.title("HW7 — Legal News Intelligence Bot")
    st.caption(
        "Upload a CSV of news stories and triage them for a global law firm using RAG, ranking, and LLM comparisons."
    )

    available_vendors = sorted({choice.vendor for choice in MODEL_REGISTRY})
    with st.sidebar:
        st.header("Model configuration")
        primary_vendor = st.selectbox("Primary vendor", options=available_vendors)
        primary_tier = st.radio(
            "Primary model tier",
            options=["budget", "premium"],
            index=0,
            key="primary-tier",
        )
        secondary_vendor = st.selectbox(
            "Comparison vendor",
            options=[v for v in available_vendors if v != primary_vendor] or available_vendors,
            index=1 if len(available_vendors) > 1 else 0,
        )
        secondary_tier = st.radio(
            "Comparison tier",
            options=["budget", "premium"],
            index=1,
            key="secondary-tier",
        )
        temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    if secondary_vendor == primary_vendor and secondary_tier == primary_tier:
        st.sidebar.warning("Select a different vendor or tier for comparison to satisfy the assignment requirement.")

    primary_choice = pick_model(primary_vendor, primary_tier)
    comparison_choice = pick_model(secondary_vendor, secondary_tier)

    if primary_choice.vendor == "Anthropic" and not anthropic_client:
        st.sidebar.error("Anthropic API key missing. Provide it via Streamlit secrets or environment variables.")
    if comparison_choice.vendor == "Anthropic" and not anthropic_client:
        st.sidebar.error("Anthropic API key missing. Provide it via Streamlit secrets or environment variables.")

    st.markdown("### Step 1: Upload a CSV of news stories")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

    if not uploaded_file:
        st.info(
            "The CSV should include columns such as `title`, `summary`, `content`, `topic`, and `date`. "
            "You can export these from any news scraping workflow."
        )
        describe_architecture()
        describe_testing_guidance()
        return

    df = read_news_csv(uploaded_file)
    if df.empty:
        st.error("Uploaded CSV is empty or could not be parsed.")
        return

    collection, metadatas = build_vector_store(df)
    st.success(f"Loaded {len(metadatas)} stories into the vector index.")
    render_dataset_overview(df)

    st.markdown("### Step 2: Choose an action")
    task = st.radio(
        "Select task",
        options=[
            "Most interesting news",
            "News about a specific topic",
            "Ask a custom question",
        ],
    )

    base_candidates = sorted(metadatas, key=lambda m: m.get("base_score", 0), reverse=True)
    top_candidates = base_candidates[: min(12, len(base_candidates))]

    if task == "Most interesting news":
        st.markdown(
            "Most interesting = stories with high legal risk/opportunity, recent developments, or strategic relevance for law-firm clients."
        )
        if st.button("Rank stories"):
            with st.spinner(f"Ranking with {primary_choice.label}..."):
                primary_results = generate_ranked_results(
                    "Identify the most legally significant stories to brief the partnership.",
                    top_candidates,
                    primary_choice.vendor,
                    primary_choice.model,
                    temperature,
                )
            with st.spinner(f"Ranking with {comparison_choice.label}..."):
                comparison_results = generate_ranked_results(
                    "Identify the most legally significant stories to brief the partnership.",
                    top_candidates,
                    comparison_choice.vendor,
                    comparison_choice.model,
                    temperature,
                )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Primary model — {primary_choice.label}")
                render_comparison(primary_results, header=primary_choice.label)
            with col2:
                st.subheader(f"Comparison model — {comparison_choice.label}")
                render_comparison(comparison_results, header=comparison_choice.label)

            st.markdown("#### Model analyses")
            st.markdown("**Primary model summary:**")
            st.markdown(primary_results.get("analysis_summary", "(none)"))
            st.markdown("**Comparison model summary:**")
            st.markdown(comparison_results.get("analysis_summary", "(none)"))

    elif task == "News about a specific topic":
        topic_query = st.text_input("Enter a topic, sector, client, or jurisdiction", value="antitrust")
        if st.button("Find topic-specific stories") and topic_query:
            topical_candidates = sorted(
                metadatas,
                key=lambda m: (topic_query.lower() in (m.get("summary") or "").lower(), m.get("base_score", 0)),
                reverse=True,
            )[: min(12, len(metadatas))]

            with st.spinner(f"Ranking topic matches with {primary_choice.label}..."):
                primary_results = generate_ranked_results(
                    f"Rank the stories most relevant to the topic: {topic_query}.",
                    topical_candidates,
                    primary_choice.vendor,
                    primary_choice.model,
                    temperature,
                )
            with st.spinner(f"Ranking topic matches with {comparison_choice.label}..."):
                comparison_results = generate_ranked_results(
                    f"Rank the stories most relevant to the topic: {topic_query}.",
                    topical_candidates,
                    comparison_choice.vendor,
                    comparison_choice.model,
                    temperature,
                )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Primary model — {primary_choice.label}")
                render_comparison(primary_results, header=primary_choice.label)
            with col2:
                st.subheader(f"Comparison model — {comparison_choice.label}")
                render_comparison(comparison_results, header=comparison_choice.label)

            st.markdown("#### Topic analysis summaries")
            st.markdown("**Primary model summary:**")
            st.markdown(primary_results.get("analysis_summary", "(none)"))
            st.markdown("**Comparison model summary:**")
            st.markdown(comparison_results.get("analysis_summary", "(none)"))

    elif task == "Ask a custom question":
        question = st.text_input("Enter your question about the news", value="Which stories pose the biggest compliance risk this week?")
        if st.button("Answer question") and question:
            with st.spinner(f"Answering with {primary_choice.label}..."):
                answer_payload = answer_question_with_rag(
                    question,
                    collection,
                    primary_choice.vendor,
                    primary_choice.model,
                    temperature,
                )
            st.subheader(f"Answer from {primary_choice.label}")
            render_answer(answer_payload)

            comparison_disabled = comparison_choice.vendor == primary_choice.vendor and comparison_choice.model == primary_choice.model
            if not comparison_disabled:
                with st.spinner(f"Answering with {comparison_choice.label}..."):
                    comparison_answer = answer_question_with_rag(
                        question,
                        collection,
                        comparison_choice.vendor,
                        comparison_choice.model,
                        temperature,
                    )
                st.subheader(f"Answer from {comparison_choice.label}")
                render_answer(comparison_answer)

    st.markdown("---")
    describe_architecture()
    describe_testing_guidance()


if __name__ == "__main__":
    main()

