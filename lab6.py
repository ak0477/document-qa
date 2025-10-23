
import os
import json
import time
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="AI Fact-Checker + Citation Builder", page_icon="🕵️‍♂️", layout="centered")
st.title("🕵️‍♂️ AI Fact-Checker + Citation Builder")
st.caption("Uses OpenAI Responses API when available; gracefully degrades if newer features are missing.")

if "history" not in st.session_state:
    st.session_state.history = []

def _safe_progress(value: float, text: str = ""):
    try:
        st.progress(value, text=text)
    except TypeError:
        # Older Streamlit doesn't support text kwarg
        st.progress(value)

def fact_check_claim(claim: str, model: str = "gpt-5.1") -> dict:
    if not claim or not claim.strip():
        return {"claim": claim, "verdict": "INVALID_INPUT", "explanation": "Please provide a non-empty factual claim.", "sources": [], "confidence": 0.0}

    api_key = st.secrets["openai"]["api_key"].strip()
    if not api_key:
        return {"claim": claim, "verdict": "CONFIG_ERROR", "explanation": "OPENAI_API_KEY is not set.", "sources": [], "confidence": 0.0}

    if OpenAI is None:
        return {"claim": claim, "verdict": "MISSING_DEPENDENCY", "explanation": "OpenAI SDK not installed. Run: pip install --upgrade openai", "sources": [], "confidence": 0.0}

    client = OpenAI(api_key=api_key)

    system_instructions = (
        "You are an AI fact-checking agent. "
        "Return ONLY valid JSON with keys: {claim, verdict, explanation, sources, confidence}. "
        "verdict in {TRUE,FALSE,PARTIALLY_TRUE,UNDETERMINED}. "
        "sources is a list of {title,url,quote}. confidence is a float in [0,1]. "
        "Prefer high-quality, recent, diverse sources. If tools are unavailable, rely on general knowledge and say UNDETERMINED when unsure."
    )

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": f"Claim: {claim}\n\nRespond with ONLY the JSON object."}
    ]

    # 1) Try Responses API with JSON formatting & tools
    try:
        kwargs = dict(model=model, input=messages)

        # Try to request JSON output if supported
        try:
            kwargs["response_format"] = {"type": "json_object"}
        except Exception:
            pass

        # Try web_search tool if supported by runtime
        try:
            kwargs["tools"] = [{"type": "web_search"}]
        except Exception:
            pass

        resp = client.responses.create(**kwargs)

        # Extract text safely across SDK variants
        raw_text = None
        try:
            # Newer SDK shape
            raw_text = resp.output[0].content[0].text
        except Exception:
            # Fallback: some SDKs expose output_text
            raw_text = getattr(resp, "output_text", None)

        if not raw_text:
            raise ValueError("Empty response content from model.")

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Attempt to carve out JSON braces
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and start < end:
                data = json.loads(raw_text[start:end+1])
            else:
                raise

        # Normalize
        data.setdefault("claim", claim)
        data.setdefault("verdict", "UNDETERMINED")
        data.setdefault("explanation", "")
        data.setdefault("sources", [])
        try:
            data["confidence"] = float(data.get("confidence", 0.0))
        except Exception:
            data["confidence"] = 0.0
        data["confidence"] = max(0.0, min(1.0, data["confidence"]))
        if not isinstance(data.get("sources"), list):
            data["sources"] = []
        return data

    except Exception as e:
        # 2) Fallback: plain text completion asking for JSON
        try:
            # Some environments still support chat.completions with similar models
            from openai import OpenAI as _Client
            _ = _Client(api_key=api_key)
            # We keep using responses API shape, but if it's not available, return structured error
            return {"claim": claim, "verdict": "API_ERROR", "explanation": f"Responses API call failed: {e}", "sources": [], "confidence": 0.0}
        except Exception as inner:
            return {"claim": claim, "verdict": "API_ERROR", "explanation": f"OpenAI call failed: {e} | {inner}", "sources": [], "confidence": 0.0}

user_claim = st.text_input("Enter a factual claim:", placeholder="e.g., Is dark chocolate healthy?")
col1, col2 = st.columns([1, 1])
with col1:
    model_name = st.text_input("Model (Responses-enabled)", value="gpt-5.1")
with col2:
    run_btn = st.button("Check Fact", type="primary")

if run_btn:
    with st.spinner("Verifying..."):
        result = fact_check_claim(user_claim, model=model_name)
        st.session_state.history.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "claim": user_claim, "result": result})

if st.session_state.history:
    st.subheader("Result")
    st.json(st.session_state.history[-1]["result"])

    res = st.session_state.history[-1]["result"]
    sources = res.get("sources", [])
    if sources:
        st.markdown("**Sources**")
        for i, s in enumerate(sources, start=1):
            title = s.get("title") or f"Source {i}"
            url = s.get("url") or ""
            quote = s.get("quote") or ""
            if url:
                st.markdown(f"- [{title}]({url})")
            else:
                st.markdown(f"- {title}")
            if quote:
                st.caption(f"“{quote}”")

    try:
        conf = float(res.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    _safe_progress(min(max(conf, 0.0), 1.0), text=f"Confidence: {conf:.2f}")

# Sample buttons without experimental APIs
st.markdown("---")
st.markdown("**Try a sample claim:**")
tests = ["Is dark chocolate actually healthy?", "Can drinking coffee prevent cancer?", "Is Pluto still a planet?"]
for t in tests:
    if st.button(t):
        result = fact_check_claim(t, model=model_name)
        st.session_state.history.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "claim": t, "result": result})
        st.rerun()

with st.expander("Reflection + Discussion"):
    st.markdown("""
    - **Responses vs. regular chat:** Responses tends to be more structured when available.
    - **Sources:** Inspect for authority and diversity; don't rely on a single outlet.
    - **Limitations:** If tools aren't available in your runtime, verdicts may be UNDETERMINED.
    """)
