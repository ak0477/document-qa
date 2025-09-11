import streamlit as st
from openai import OpenAI
import re

st.title("Lab 3 - My Chatbot")

# Setup
client = OpenAI(api_key=st.secrets["openai"]["api_key"].strip())
model = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
if "awaiting_info" not in st.session_state:
    st.session_state.awaiting_info = False

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def buffer(msgs, n=20):
    sys = {"role": "system", "content": "Explain simply, like to a 10-year-old."}
    return [sys] + msgs[-n:]

def stream_reply(msgs):
    resp = client.chat.completions.create(model=model, messages=buffer(msgs), stream=True)
    for ch in resp:
        d = ch.choices[0].delta
        if d and getattr(d, "content", None):
            yield d.content

def ask_more():
    prompt = "DO YOU WANT MORE INFO (yes/no)?"
    with st.chat_message("assistant"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "assistant", "content": prompt})
    st.session_state.awaiting_info = True

yes_re = re.compile(r"^\s*(y|yes|yeah|yep|sure|ok|okay)\s*$", re.I)
no_re  = re.compile(r"^\s*(n|no|nope|nah)\s*$", re.I)

user = st.chat_input("Type your question")

if user:
    st.session_state.messages.append({"role": "user", "content": user})
    with st.chat_message("user"):
        st.markdown(user)

    if st.session_state.awaiting_info:
        if yes_re.match(user):
            st.session_state.messages.append({"role": "user", "content": "Give more info about your last answer."})
            with st.chat_message("assistant"):
                text = st.write_stream(stream_reply(st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": text})
            ask_more()
        elif no_re.match(user):
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
        ask_more()
