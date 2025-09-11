import time
import random
import streamlit as st
from openai import OpenAI

# --- App title ---
st.title("Lab 3 - My Chatbot")
st.write("Hi I am Bob")

# --- Greeting streamer (fix: keep yield inside the function) ---
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

# --- Secrets / API key (use one convention) ---
# Expecting: in .streamlit/secrets.toml put:
# OPENAI_API_KEY = "sk-..."
openai_api_key = st.secrets.get("openai", "")
if not openai_api_key:
    st.info("Please add your OpenAI API key to secrets.toml as OPENAI_API_KEY.", icon="🗝️")
    st.stop()

# --- Init client once ---
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=openai_api_key)

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# --- Render existing messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Helper: wrap OpenAI stream so write_stream gets plain text chunks ---
def openai_stream(model: str, messages: list[dict]):
    client = st.session_state.client
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    # Convert event stream to text chunks
    for chunk in resp:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content

# --- Input + completion ---
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_text = st.write_stream(openai_stream(model_to_use, st.session_state.messages))

    # Save the assistant reply
    st.session_state.messages.append({"role": "assistant", "content": full_text})
