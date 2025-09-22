import streamlit as st
import requests, re, importlib.util
from bs4 import BeautifulSoup

# =========================
# App config
# =========================
st.set_page_config(
    page_title="HW manager",
    page_icon=":material/home:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Shared utilities
# =========================
def read_url_content(url: str):
    """Read and extract visible text from a web page URL."""
    try:
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        # remove script/style
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        text = soup.get_text(separator="\n")
        return re.sub(r"\n{3,}", "\n\n", text).strip()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

def build_summary_prompt(page_text: str, summary_mode: str, language: str):
    """Prompt for URL summarization with output language and style."""
    return f"""
You are a concise, faithful summarizer.

Task:
1) Read the web page text between <doc> tags.
2) Produce a {summary_mode} summary.
3) The entire output must be written in {language}.
4) Do not add facts that are not supported by the text.

<doc>
{page_text}
</doc>
""".strip()

def chunk_stream(text: str, chunk_size: int = 180):
    """Yield text in chunks to simulate streaming in the UI."""
    text = text or ""
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]

def approx_token_len(s: str) -> int:
    return max(1, len(s) // 4)

# =========================
# Provider runners (HW2/HW3)
# =========================
OPENAI_MODELS = {
    "standard": "gpt-4o-mini",
    "advanced": "gpt-5-chat-latest",
}
ANTHROPIC_MODELS = {
    "standard": "claude-3-haiku-20240307",
    "advanced": "claude-3-5-sonnet-20240620",
}
GEMINI_MODELS = {
    "standard": "gemini-1.5-flash",
    "advanced": "gemini-1.5-pro",
}

def have_pkg(modname: str) -> bool:
    return importlib.util.find_spec(modname) is not None

HAVE_ANTHROPIC = have_pkg("anthropic")
HAVE_GEMINI    = have_pkg("google.generativeai")

def run_openai(prompt: str, advanced: bool):
    try:
        from openai import OpenAI
    except Exception as e:
        st.error(f"OpenAI SDK import error: {e}")
        return None, {}

    api_key = st.secrets.get("openai", {}).get("api_key", "")
    if not api_key:
        st.error("OpenAI key missing in secrets.toml under [openai].")
        return None, {}

    client = OpenAI(api_key=api_key)
    model = OPENAI_MODELS["advanced" if advanced else "standard"]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        meta = {
            "provider": "OpenAI",
            "model": model,
            "finish_reason": getattr(resp.choices[0], "finish_reason", None),
        }
        return text, meta
    except Exception as e:
        st.error(f"OpenAI request failed: {e}")
        return None, {}

def run_anthropic(prompt: str, advanced: bool):
    if not HAVE_ANTHROPIC:
        st.error("Anthropic SDK not installed. Add `anthropic` to requirements.txt.")
        return None, {}
    try:
        import anthropic
    except Exception as e:
        st.error(f"Anthropic SDK import error: {e}")
        return None, {}

    api_key = st.secrets.get("anthropic", {}).get("api_key", "")
    if not api_key:
        st.error("Anthropic key missing in secrets.toml under [anthropic].")
        return None, {}

    client = anthropic.Anthropic(api_key=api_key)
    model = ANTHROPIC_MODELS["advanced" if advanced else "standard"]

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1200,
            temperature=0.2,
            system="Always respond in the requested output language. Be concise and accurate.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join([blk.text for blk in resp.content if getattr(blk, "type", "") == "text"])
        meta = {"provider": "Anthropic", "model": model}
        return text, meta
    except Exception as e:
        st.error(f"Anthropic request failed: {e}")
        return None, {}

def run_gemini(prompt: str, advanced: bool):
    if not HAVE_GEMINI:
        st.error("Google Generative AI SDK not installed. Add `google-generativeai` to requirements.txt.")
        return None, {}
    try:
        import google.generativeai as genai
    except Exception as e:
        st.error(f"Google Generative AI SDK import error: {e}")
        return None, {}

    api_key = st.secrets.get("google", {}).get("api_key", "")
    if not api_key:
        st.error("Google Gemini key missing in secrets.toml under [google].")
        return None, {}

    try:
        genai.configure(api_key=api_key)
        model_name = GEMINI_MODELS["advanced" if advanced else "standard"]
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        meta = {"provider": "Google Gemini", "model": model_name}
        return text, meta
    except Exception as e:
        st.error(f"Gemini request failed: {e}")
        return None, {}

def run_selected_provider(prompt: str, provider_name: str, advanced: bool):
    if provider_name == "OpenAI":
        return run_openai(prompt, advanced)
    if provider_name == "Claude (Anthropic)":
        return run_anthropic(prompt, advanced)
    if provider_name == "Gemini (Google)":
        return run_gemini(prompt, advanced)
    st.error("Unsupported provider selected.")
    return None, {}

# =========================
# Page: HW1 — Document Q&A
# =========================
def page_hw1():
    st.title("HW 1 — Document Q&A")
    st.write(
        "Upload a plain text file and ask a question. The model will answer from your file. "
        "API key is read from secrets.toml."
    )

    openai_api_key = st.secrets.get("openai", {}).get("api_key", "")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to secrets.toml under [openai].", icon="🗝️")
        st.stop()

    from openai import OpenAI  # local import
    client = OpenAI(api_key=openai_api_key)

    uploaded = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))
    question = st.text_area(
        "Ask a question about the document",
        placeholder="Example: Summarize this document.",
        disabled=not uploaded,
    )

    if uploaded and question:
        doc_text = uploaded.read().decode(errors="ignore")
        messages = [
            {"role": "user", "content": f"Document:\n{doc_text}\n\nQuestion:\n{question}"}
        ]
        try:
            stream = client.chat.completions.create(
                model="gpt-5-chat-latest",
                messages=messages,
                stream=True,
            )
            st.subheader("Answer")
            st.write_stream(stream)
        except Exception as e:
            st.error(f"OpenAI request failed: {e}")

# =========================
# Page: HW2 — URL Summarizer (Multi-LLM)
# =========================
SUMMARY_MODES = [
    "TL;DR (3–5 sentences)",
    "Bullet points (5–8 bullets)",
    "Key takeaways (top 5)",
    "Who / What / When / Where / Why / How",
    "Outline with headings",
]
LANG_OPTIONS = ["English", "French", "Spanish", "German", "Hindi"]

def available_llm_providers():
    opts = ["OpenAI"]
    if HAVE_ANTHROPIC:
        opts.append("Claude (Anthropic)")
    if HAVE_GEMINI:
        opts.append("Gemini (Google)")
    return opts

def page_hw2():
    st.title("HW 2 — URL Summarizer for multiple LLMs")

    url = st.text_input("Enter a web page URL", placeholder="https://example.com/article")
    out_language = st.selectbox("Output language", options=LANG_OPTIONS, index=0)

    with st.sidebar:
        st.subheader("Summary Options")
        summary_type = st.selectbox("Type of summary", options=SUMMARY_MODES, index=0)

        st.subheader("Model Options")
        providers = available_llm_providers()
        provider = st.selectbox("LLM Provider", options=providers, index=0)
        use_advanced = st.checkbox("Use advanced model", value=True)
        if not HAVE_ANTHROPIC or not HAVE_GEMINI:
            st.caption("Tip: Install `anthropic` and `google-generativeai` to enable all vendors.")

    submit = st.button("Summarize URL", type="primary", disabled=not url)

    if submit:
        with st.spinner("Fetching and summarizing..."):
            page_text = read_url_content(url)
            if not page_text:
                st.stop()

            prompt = build_summary_prompt(page_text, summary_type, out_language)
            output, meta = run_selected_provider(prompt, provider, use_advanced)

            if output:
                st.subheader("Summary")
                st.write(output)
                with st.expander("Details"):
                    st.json(
                        {
                            "provider": meta.get("provider"),
                            "model": meta.get("model"),
                            "summary_type": summary_type,
                            "language": out_language,
                            "advanced_model": use_advanced,
                        }
                    )

# =========================
# Page: HW3 — Streaming Chatbot that discusses 1–2 URLs
# =========================
MEMORY_TYPES = [
    "Buffer of 6 questions",
    "Conversation summary",
    "Buffer ≈ 2,000 tokens",
]

def build_context_messages(messages, memory_type: str):
    sys = {
        "role": "system",
        "content": "Be accurate and concise. Explain simply, like to a 10-year-old when possible. Cite exact wording from the URLs only when relevant."
    }

    if memory_type == "Buffer of 6 questions":
        trimmed, user_turns = [], 0
        for m in reversed(messages):
            trimmed.append(m)
            if m.get("role") == "user":
                user_turns += 1
            if user_turns >= 6:
                break
        trimmed = list(reversed(trimmed))
        return [sys] + trimmed

    if memory_type == "Buffer ≈ 2,000 tokens":
        budget, kept = 2000, []
        for m in reversed(messages):
            t = approx_token_len(m.get("content", "")) + 8
            if budget - t <= 0:
                break
            kept.append(m); budget -= t
        kept = list(reversed(kept))
        return [sys] + kept

    # Conversation summary
    summary = st.session_state.get("summary_memory", "")
    if summary:
        return [sys, {"role": "system", "content": f"Conversation summary so far:\n{summary}"}] + messages[-6:]
    else:
        return [sys] + messages[-8:]

def maybe_update_summary_memory(messages):
    if len(messages) < 8:
        return
    api_key = st.secrets.get("openai", {}).get("api_key", "")
    if not api_key:
        return
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
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
        pass

def call_openai_stream(messages, model_name: str):
    from openai import OpenAI
    key = st.secrets.get("openai", {}).get("api_key", "")
    if not key:
        st.error("OpenAI API key missing in secrets.toml under [openai].")
        return None, {}
    client = OpenAI(api_key=key)
    try:
        resp = client.chat.completions.create(model=model_name, messages=messages, stream=True)
        def gen():
            for ev in resp:
                d = ev.choices[0].delta
                if d and getattr(d, "content", None):
                    yield d.content
        return gen(), {"provider": "OpenAI", "model": model_name, "streamed": True}
    except Exception as e:
        st.error(f"OpenAI request failed: {e}")
        return None, {}

def call_anthropic_stream(messages, model_name: str):
    if not HAVE_ANTHROPIC:
        st.error("Anthropic SDK not installed. Add `anthropic` to requirements.txt.")
        return None, {}
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
        return chunk_stream(text), {"provider": "Anthropic", "model": model_name, "streamed": True}
    except Exception as e:
        st.error(f"Anthropic request failed: {e}")
        return None, {}

def call_gemini_stream(messages, model_name: str):
    if not HAVE_GEMINI:
        st.error("Google Generative AI SDK not installed. Add `google-generativeai` to requirements.txt.")
        return None, {}
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

def page_hw3():
    st.title("HW 3 — URL Chatbot (Streaming, Multi-LLM)")

    # Sidebar controls
    with st.sidebar:
        st.subheader("Options")
        url1 = st.text_input("URL 1", placeholder="https://example.com/page-a")
        url2 = st.text_input("URL 2 (optional)", placeholder="https://example.com/page-b")
        st.caption("Provide 1 or 2 URLs; the bot will use both if available.")

        provider_opts = available_llm_providers()
        provider = st.selectbox("LLM Vendor", options=provider_opts, index=0)
        tier = st.selectbox("Model", options=["Cheap", "Flagship"], index=0)

        memory_type = st.selectbox(
            "Conversation memory", options=MEMORY_TYPES, index=0,
            help="Choose how much prior conversation the model sees."
        )
        if not HAVE_ANTHROPIC or not HAVE_GEMINI:
            st.caption("Tip: Install `anthropic` and `google-generativeai` to enable all vendors.")

    # Session state
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "content": "Ask me about the URLs, and I will answer from them."}]
    if "url_cache" not in st.session_state:
        st.session_state.url_cache = {}

    def get_url_text(u: str) -> str:
        if not u:
            return ""
        if u in st.session_state.url_cache:
            return st.session_state.url_cache[u]
        t = read_url_content(u)
        st.session_state.url_cache[u] = t or ""
        return st.session_state.url_cache[u]

    # Render history
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Type your question about the URLs…")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        u1_text = get_url_text(url1)
        u2_text = get_url_text(url2)
        if not u1_text and not u2_text:
            reply = "Please add at least one valid URL in the sidebar."
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})
            return

        # Source block (limit size to keep prompts sane)
        url_block = "\n\n".join(
            [
                f"<URL1>\n{(u1_text or '')[:15000]}\n</URL1>" if u1_text else "",
                f"<URL2>\n{(u2_text or '')[:15000]}\n</URL2>" if u2_text else "",
            ]
        ).strip()

        instruction = (
            "Use ONLY the information in the provided URL texts and our conversation. "
            "If a detail is not in the URLs, say you don't have that info from the sources. "
            "Answer clearly. If both URLs disagree, point it out."
        )

        base_messages = build_context_messages(st.session_state.chat, memory_type)
        composed = base_messages + [
            {"role": "user", "content": f"{instruction}\n\nSOURCE TEXTS:\n{url_block}\n\nUser question: {prompt}"}
        ]
        if memory_type == "Conversation summary":
            maybe_update_summary_memory(st.session_state.chat)

        # Route provider call
        with st.chat_message("assistant"):
            if provider == "OpenAI":
                model = OPENAI_MODELS["Flagship" if tier == "Flagship" else "Cheap"]
                stream, meta = call_openai_stream(composed, model)
            elif provider == "Claude (Anthropic)":
                model = ANTHROPIC_MODELS["advanced" if tier == "Flagship" else "standard"]
                stream, meta = call_anthropic_stream(composed, model)
            else:  # Gemini
                model = GEMINI_MODELS["advanced" if tier == "Flagship" else "standard"]
                stream, meta = call_gemini_stream(composed, model)

            if stream is None:
                st.stop()
            out_text = st.write_stream(stream)

        st.session_state.chat.append({"role": "assistant", "content": out_text})
        with st.expander("Response details"):
            st.json(meta)

# =========================
# Navigation
# =========================
hw1_page = st.Page(page_hw1, title="HW 1 — Document Q&A", icon=":material/looks_one:")
hw2_page = st.Page(page_hw2, title="HW 2 — URL Summarizer", icon=":material/looks_two:")
hw3_page = st.Page(page_hw3, title="HW 3 — URL Chatbot", icon=":material/looks_3:")
hw4_page = st.Page(page_hw4, title="HW 4 — iSchool Chatbot", icon=":material/looks_4:")
pg = st.navigation([hw1_page, hw2_page, hw3_page, hw4_page])
pg.run()
