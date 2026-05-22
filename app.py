import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="MitrAI", page_icon="🤝", layout="centered")

st.title("🤝 MitrAI")
st.caption("Palco's AI Companion | Work + Life + Company Help")

# Sidebar
with st.sidebar:
    st.header("👨‍💼 Staff / Admin")
    is_admin = st.checkbox("🛠️ Admin Mode - Manual Reply", value=False)

# Initialize OpenAI Client
api_key = st.secrets.get("OPENAI_API_KEY") or st.text_input("OpenAI API Key", type="password", value="")

if api_key:
    client = OpenAI(api_key=api_key)
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.warning("API Key Required")
    client = None

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! 👋\n\nI'm **MitrAI** – Palco Team's AI Companion.\nAsk me anything about company, products, HR, or personal issues."
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
            if client:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                    temperature=0.7
                )
                ai_answer = response.choices[0].message.content
                st.markdown(ai_answer)
                st.session_state.messages.append({"role": "assistant", "content": ai_answer})

                # Admin Manual Reply
                if is_admin:
                    st.divider()
                    st.subheader("🛠️ Admin Manual Reply")
                    manual_reply = st.text_area("If AI response is not good, write your own reply:", height=130)
                    if st.button("📤 Send Manual Reply"):
                        if manual_reply.strip():
                            st.session_state.messages.append({"role": "assistant", "content": manual_reply})
                            st.success("✅ Manual Reply Sent!")
                            st.rerun()
            else:
                st.error("Please enter API Key")