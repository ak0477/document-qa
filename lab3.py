import streamlit as st
from openai import OpenAI
import re

st.title("Lab 3 - My Chatbot")

# --- Setup ---
openai_api_key = st.secrets["openai"]["api_key"].strip()
client = OpenAI(api_key=openai_api_key)
model_to_use = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
if "awaiting_info" not in st.session_state:
    st.session_state.awaiting_info = False

# --- Show chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Buffer for API ---
def buffer(messages, n=20):
    system_msg = {
        "role": "system",
        "content": "Explain simply, as if to a 10-year-old."
    }
    return [system_msg] + messages[-n:]

# --- Stream response ---
def stream_reply(messages):
    resp = client.chat.completions.create(
        model=model_to_use,
        messages=buffer(messages),
        stream=True,
    )
    for chunk in resp:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content

# --- Handle input ---
user_input = st.chat_input("Type your question here")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.awaiting_info:
        if re.match(r"^\s*yes\s*$", user_input, re.I):
            st.session_state.messages.append({"role": "user", "content": "Give more info about your last answer."})
            with st.chat_message("assistant"):
                text = st.write_stream(stream_reply(st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": text})
            st.session_state.messages.append({"role": "assistant", "content": "DO YOU WANT MORE INFO (yes/no)?"})
            st.session_state.awaiting_info = True
        elif re.match(r"^\s*no\s*$", user_input, re.I):
            reply = "Okay! What else can I help you with?"
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.awaiting_info = False
        else:
            reply = "Please reply with yes or no."
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    else:
        with st.chat_message("assistant"):
            text = st.write_stream(stream_reply(st.session_state.messages))
        st.session_state.messages.append({"role": "assistant", "content": text})
        st.session_state.messages.append({"role": "assistant", "content": "DO YOU WANT MORE INFO (yes/no)?"})
        st.session_state.awaiting_info = True
