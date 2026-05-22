import streamlit as st
from openai import OpenAI
import os

st.set_page_config(page_title="MitrAI", page_icon="🤝", layout="centered")

st.title("🤝 MitrAI")
st.caption("Palco's AI Companion")

# Sidebar
with st.sidebar:
    st.header("👨‍💼 Staff / Admin")
    api_key = st.secrets.get("OPENAI_API_KEY") or st.text_input("OpenAI API Key", type="password", value="")
   
    if api_key:
        client = OpenAI(api_key=api_key)
        st.success("✅ API Connected")
    else:
        st.warning("API Key Required")

# Simple Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! 👋\n\nI'm **MitrAI** – Palco Team's AI Companion.\nAsk me anything about company, products, HR or personal issues."
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if api_key:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                    temperature=0.7
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Please enter API Key")