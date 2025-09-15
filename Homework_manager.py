import streamlit as st
import requests
from bs4 import BeautifulSoup

# Lazy imports
# from openai import OpenAI
# import anthropic
# import google.generativeai as genai

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
        return soup.get_text(separator="\n")
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


# =========================
# Provider runners (HW2)
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
        text = "".join(
            [blk.text for blk in resp.content if getattr(blk, "type", "") == "text"]
        )
        meta = {"provider": "Anthropic", "model": model}
        return text, meta
    except Exception as e:
        st.error(f"Anthropic request failed: {e}")
        return None, {}


def run_gemini(prompt: str, advanced: bool):
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
# (uses OpenAI key in secrets)
# =========================
def page_hw1():
    from openai import OpenAI  # local import so app can load without SDK until needed

    st.title("HW 1 — Document Q&A")
    st.write(
        "Upload a plain text file and ask a question. The model will answer from your file. "
        "API key is read from secrets.toml."
    )

    openai_api_key = st.secrets.get("openai", {}).get("api_key", "")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to secrets.toml under [openai].", icon="🗝️")
        st.stop()

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
LLM_PROVIDERS = ["OpenAI", "Claude (Anthropic)", "Gemini (Google)"]


def page_hw2():
    st.title("HW 2 — URL Summarizer for multiple LLMs")

    # Top controls, as required
    url = st.text_input("Enter a web page URL", placeholder="https://example.com/article")
    out_language = st.selectbox("Output language", options=LANG_OPTIONS, index=0)

    # Sidebar controls
    with st.sidebar:
        st.subheader("Summary Options")
        summary_type = st.selectbox("Type of summary", options=SUMMARY_MODES, index=0)

        st.subheader("Model Options")
        provider = st.selectbox("LLM Provider", options=LLM_PROVIDERS, index=0)
        use_advanced = st.checkbox("Use advanced model", value=True)
        st.caption("Add provider API keys in secrets.toml.")

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
# Navigation
# =========================
# Streamlit experimental multipage API used in your labs
# At the bottom where you build pages:
from HW3 import *  # or: from HW3 import st  # if needed just to ensure import side effects

# Create a page object for HW3
def page_hw3():
    # The HW3 page defines its own layout & UI when imported; if you prefer,
    # rename the top-level title in HW3.py and wrap its content in a function.
    import HW3  # ensures the page runs (HW3.py already builds the page UI)

hw1_page = st.Page(page_hw1, title="HW 1 — Document Q&A", icon=":material/looks_one:")
hw2_page = st.Page(page_hw2, title="HW 2 — URL Summarizer", icon=":material/looks_two:")
hw3_page = st.Page(page_hw3, title="HW 3 — URL Chatbot", icon=":material/looks_3:")

pg = st.navigation([hw1_page, hw2_page, hw3_page])
pg.run()

