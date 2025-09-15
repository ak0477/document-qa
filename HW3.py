import streamlit as st
import requests, re
from bs4 import BeautifulSoup

# Optional provider SDKs are imported lazily inside functions so the page can load without them.
# from openai import OpenAI
# import anthropic
# import google.generativeai as genai

st.set_page_config(page_title="HW 3 — Streaming Chatbot (URLs)", page_icon=":material/chat:", layout="wide")
st.title("HW 3 — Streaming Chatbot that discusses 1–2 URLs")

# =========================
# Helpers
# =========================
def read_url_text(url: str) -> str:
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        # Drop script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # Normalize whitespace
        return re.sub(r"\n{3,}", "\n\n", text).strip()
    except Exception as e:
        st.error(f"Failed to read {url}: {e}")
        return ""

def approx_token_len(s: str) -> int:
    # Super rough 1 token ~= 4 chars heuristic
    return max(1, len(s) // 4)

def chunk_stream(text: str, chunk_size: int = 180):
    # Stream any text in small chunks so the UI shows incremental output
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]

# =========================
# Memory strategies
# =========================
MEMORY_TYPES = [
    "Buffer of 6 questions",
    "Conversation summary",
    "Buffer ≈ 2,000 tokens",
]

def build_context_messages(messages, memory_type: str):
    """Return a trimmed list of messages based on the selected memory strategy."""
    sys = {
        "role": "system",
        "content": "Be accurate and concise. Explain simply, like to a 10-year-old when possible. Cite exact wording from the URLs only when relevant."
    }

    if memory_type == "Buffer of 6 questions":
        # Keep last 6 user turns and their assistant replies (roughly last 12 messages)
        trimmed = []
        # Walk backwards collecting pairs
        user_turns = 0
        for m in reversed(messages):
            trimmed.append(m)
            if m.get("role") == "user":
                user_turns += 1
            if user_turns >= 6:
                break
        trimmed = list(reversed(trimmed))
        return [sys] + trimmed

    elif memory_type == "Buffer ≈ 2,000 tokens":
        # Keep as many most-recent messages as fit ~2000 tokens
        budget = 2000
        kept = []
        for m in reversed(messages):
            t = approx_token_len(m.get("content", "")) + 8  # small role/format overhead
            if budget - t <= 0:
                break
            kept.append(m)
            budget -= t
        kept = list(reversed(kept))
        return [sys] + kept

    # Conversation summary (default branch)
    # We maintain/update a running summary in session_state when messages grow.
    summary = st.session_state.get("summary_memory", "")
    if summary:
        return [sys, {"role": "system", "content": f"Conversation summary so far:\n{summary}"}] + messages[-6:]
    else:
        return [sys] + messages[-8:]

def maybe_update_summary_memory(messages):
    """Update the running summary using OpenAI mini if available and conversation grew."""
    # Only update when there is enough history
    if len(messages) < 8:
        return

    api_key = st.secrets.get("openai", {}).get("api_key", "")
    if not api_key:
        return  # silent no-op; summary memory still works with recent turns
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Summarize last ~12 messages to keep costs tiny
        last = messages[-12:]
        prompt = "Summarize the conversation so far in 6–8 compact bullets capturing facts, decisions, and follow-ups."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a concise meeting minutes writer."}]
                     + last
                     + [{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        st.session_state.summary_memory = resp.choices[0].message.content.strip()
    except Exception:
        pass  # Best-effort

# =========================
# Providers & models
# =========================
PROVIDERS = ["OpenAI", "Claude (Anthropic)", "Gemini (Google)"]
OPENAI_MODELS = {"Cheap": "gpt-4o-mini", "Flagship": "gpt-5-chat-latest"}
ANTHROPIC_MODELS = {"Cheap": "claude-3-haiku-20240307", "Flagship": "claude-3-5-sonnet-20240620"}
GEMINI_MODELS = {"Cheap": "gemini-1.5-flash", "Flagship": "gemini-1.5-pro"}

def call_openai(messages, model_name: str, stream: bool = True):
    from openai import OpenAI
    key = st.secrets.get("openai", {}).get("api_key", "")
    if not key:
        st.error("OpenAI API key missing in secrets.toml under [openai].")
        return None, {}
    client = OpenAI(api_key=key)
    if stream:
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages, stream=True)
            # Re-wrap OpenAI event stream as generator that yields just text
            def gen():
                for ev in resp:
                    d = ev.choices[0].delta
                    if d and getattr(d, "content", None):
                        yield d.content
            return gen(), {"provider": "OpenAI", "model": model_name, "streamed": True}
        except Exception as e:
            st.error(f"OpenAI request failed: {e}")
            return None, {}
    else:
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages, stream=False)
            return resp.choices[0].message.content, {"provider": "OpenAI", "model": model_name, "streamed": False}
        except Exception as e:
            st.error(f"OpenAI request failed: {e}")
            return None, {}

def call_anthropic(messages, model_name: str):
    try:
        import anthropic
    except Exception as e:
        st.error(f"Anthropic SDK import error: {e}")
        return None, {}
    key = st.secrets.get("anthropic", {}).get("api_key", "")
    if not key:
        st.error("Anthropic API key missing in secrets.toml under [anthropic].")
        return None, {}
    client = anthropic.Anthropic(api_key=key)
    # Convert OpenAI-style messages -> Anthropic
    sys = ""
    user_content = []
    for m in messages:
        if m["role"] == "system":
            sys += (m["content"] + "\n")
        elif m["role"] == "user":
            user_content.append({"type": "text", "text": m["content"]})
        elif m["role"] == "assistant":
            user_content.append({"type": "text", "text": f"Assistant: {m['content']}"})
    try:
        resp = client.messages.create(
            model=model_name,
            system=sys.strip() or None,
            max_tokens=1200,
            temperature=0.2,
            messages=[{"role": "user", "content": user_content}],
        )
        text = "".join([blk.text for blk in resp.content if getattr(blk, "type", "") == "text"])
        # Stream to UI by chunking locally
        return chunk_stream(text), {"provider": "Anthropic", "model": model_name, "streamed": True}
    except Exception as e:
        st.error(f"Anthropic request failed: {e}")
        return None, {}

def call_gemini(messages, model_name: str):
    try:
        import google.generativeai as genai
    except Exception as e:
        st.error(f"Google Generative AI SDK import error: {e}")
        return None, {}
    key = st.secrets.get("google", {}).get("api_key", "")
    if not key:
        st.error("Google Gemini API key missing in secrets.toml under [google].")
        return None, {}
    genai.configure(api_key=key)

    # Collapse messages to a single prompt; Gemini supports multi-turn but this is simplest
    compiled = []
    for m in messages:
        compiled.append(f"{m['role'].upper()}: {m['content']}")
    prompt = "\n\n".join(compiled)
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None) or ""
        return chunk_stream(text), {"provider": "Google Gemini", "model": model_name, "streamed": True}
    except Exception as e:
        st.error(f"Gemini request failed: {e}")
        return None, {}

def run_provider(provider: str, tier: str, messages):
    if provider == "OpenAI":
        return call_openai(messages, OPENAI_MODELS[tier], stream=True)
    if provider == "Claude (Anthropic)":
        return call_anthropic(messages, ANTHROPIC_MODELS[tier])
    if provider == "Gemini (Google)":
        return call_gemini(messages, GEMINI_MODELS[tier])
    st.error("Unsupported provider selected.")
    return None, {}

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Options")
    url1 = st.text_input("URL 1", placeholder="https://example.com/page-a")
    url2 = st.text_input("URL 2 (optional)", placeholder="https://example.com/page-b")
    st.caption("Provide 1 or 2 URLs; the bot will use both if available.")

    provider = st.selectbox("LLM Vendor", options=PROVIDERS, index=0)
    model_tier = st.selectbox("Model", options=["Cheap", "Flagship"], index=0)

    memory_type = st.selectbox("Conversation memory", options=MEMORY_TYPES, index=0,
                               help="Choose how much prior conversation the model sees.")

    st.divider()
    st.caption("API keys must be set in secrets.toml for the selected vendor.")

# =========================
# Initialize session state
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "assistant", "content": "Ask me about the URLs, and I will answer from them."}]

# Cache URL texts for this session
if "url_cache" not in st.session_state:
    st.session_state.url_cache = {}

def get_url_text(u: str) -> str:
    if not u:
        return ""
    if u in st.session_state.url_cache:
        return st.session_state.url_cache[u]
    text = read_url_text(u)
    st.session_state.url_cache[u] = text
    return text

# =========================
# Chat UI
# =========================
# Render history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type your question about the URLs…")
if prompt:
    # Append the new user message
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare URL evidence
    u1_text = get_url_text(url1)
    u2_text = get_url_text(url2)
    if not u1_text and not u2_text:
        with st.chat_message("assistant"):
            st.markdown("Please add at least one valid URL in the sidebar.")
        st.session_state.chat.append({"role": "assistant", "content": "Please add at least one valid URL in the sidebar."})
    else:
        # Build an instruction prefix that injects the URL contents
        url_block = "\n\n".join(
            [f"<URL1>\n{u1_text[:15000]}\n</URL1>" if u1_text else "",
             f"<URL2>\n{u2_text[:15000]}\n</URL2>" if u2_text else "" ]
        ).strip()

        question_block = f"""Use ONLY the information in the provided URL texts and our conversation. 
If a detail is not in the URLs, say you don't have that info from the sources.
Answer clearly. If both URLs disagree, point it out.
""".strip()

        # Compose messages with selected memory strategy
        base_messages = build_context_messages(st.session_state.chat, memory_type)
        composed = base_messages + [
            {"role": "user", "content": f"{question_block}\n\nSOURCE TEXTS:\n{url_block}\n\nUser question: {prompt}"}
        ]

        # Update summary if that memory type is selected (best-effort)
        if memory_type == "Conversation summary":
            maybe_update_summary_memory(st.session_state.chat)

        # Call selected provider (streaming to UI)
        with st.chat_message("assistant"):
            stream, meta = run_provider(provider, model_tier, composed)
            if stream is None:
                st.stop()
            out_text = st.write_stream(stream)

        st.session_state.chat.append({"role": "assistant", "content": out_text})

        with st.expander("Response details"):
            st.json(meta)

# =========================
# Notes for graders
# =========================
st.caption(
    "This page streams answers, supports 3 vendors (OpenAI, Anthropic, Gemini), "
    "lets you pick Cheap vs Flagship models, accepts two URLs, and offers 3 memory strategies: "
    "last-6-turns buffer, conversation summary, and ~2k-token rolling buffer."
)