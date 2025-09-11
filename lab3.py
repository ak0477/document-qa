import time
import random
import streamlit as st
from openai import OpenAI

# --- App title ---
st.title("Lab 3 - My Chatbot")
st.write("Hi I am Bob")

# --- Greeting streamer ---
def response_generator():
    response = random.choice(
        [
            "Hello there!",
            "Hi, human!",
            "Do you need any help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

with st.chat_message("assistant"):
    st.write_stream(response_generator())

# --- Model picker ---
openAI_model = st.sidebar.selectbox("Which model?", ("mini", "regular"))
model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

# --- Secrets / API key (match your [openai] block) ---
openai_api_key = st.secrets["openai"]["api_key"].strip()
if not openai_api_key.startswith("sk-"):
    st.error("Invalid or missing OpenAI API key. Please check your secrets.toml.")
    st.stop()

# --- Init client once ---
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=openai_api_key)

# --- Chat state ---
if "messages" not in st.session_state:
    # Keep the full conversation for display only
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# --- Render existing messages (full history for UI) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Conversation buffer: only last K user turns + their assistant replies ---
def build_buffer(messages: list[dict], k: int = 2) -> list[dict]:
    """
    Return a list of messages containing only the last k user turns
    and the assistant responses to those turns (i.e., up to 2k messages).
    Preserves order (oldest -> newest) for the API call.
    """
    buf = []
    user_turns = 0

    # Walk backward and collect until we've got k user turns
    for m in reversed(messages):
        buf.append(m)
        if m.get("role") == "user":
            user_turns += 1
            if user_turns >= k:
                break

    # buf is reversed order now; restore chronological order
    buf.reverse()

    # Optional: include an optional system prompt at the start of buffer if desired
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant. Keep responses concise and clear."
    }

    # Ensure the very first message is the system one (if you want a system prompt)
    # and then the buffered convo after it.
    return [system_msg] + buf

# --- Wrap OpenAI stream so write_stream gets plain text chunks ---
def openai_stream(model: str, messages: list[dict]):
    client = st.session_state.client
    # Use only the conversation buffer for the API call
    payload = build_buffer(messages, k=2)
    resp = client.chat.completions.create(
        model=model,
        messages=payload,
        stream=True,
    )
    for chunk in resp:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content

# --- Input + completion ---
if prompt := st.chat_input("What is up?"):
    # Save full conversation for UI/history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Stream using only the buffer (last 2 user turns)
        full_text = st.write_stream(openai_stream(model_to_use, st.session_state.messages))

    # Save the assistant reply to the full conversation
    st.session_state.messages.append({"role": "assistant", "content": full_text})
