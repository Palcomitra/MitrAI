import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import GoogleDriveLoader, WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

st.set_page_config(page_title="MitrAI", page_icon="🤝", layout="centered")

st.title("🤝 MitrAI")
st.caption("Palco's AI Companion | Drive + Website")

# Sidebar
with st.sidebar:
    st.header("👨‍💼 Staff / Admin")
    api_key = st.text_input("OpenAI API Key", type="password", value="")
   
    if api_key:
        client = OpenAI(api_key=api_key)
        st.success("✅ API Connected")
    else:
        st.warning("Enter your API Key")
    
    st.divider()
    is_admin = st.checkbox("🛠️ Admin Mode (Manual Reply)", value=False)

# Load Knowledge Base
@st.cache_resource
def load_knowledge_base():
    all_texts = []
    try:
        loader = GoogleDriveLoader(folder_id="0ADFKVoP1n82mUk9PVA", token_path="token.json", credentials_path="credentials.json", recursive=True)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        all_texts.extend(splitter.split_documents(docs))
    except:
        pass

    # Websites
    try:
        urls = ["https://www.palco.co.in/", "https://www.palcostore.com/"]
        web_docs = WebBaseLoader(urls).load()
        all_texts.extend(splitter.split_documents(web_docs))
    except:
        pass

    if all_texts:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return Chroma.from_documents(all_texts, embeddings, persist_directory="chroma_db")
    return None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_knowledge_base()

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "✅ MitrAI ready hai!\nAdmin Mode में manual reply भी कर सकते हो।"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.vectorstore and api_key:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(prompt)
                context = "\n\n".join([d.page_content[:700] for d in docs])

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}\n\nHelpful answer:"}],
                    temperature=0.7
                )
                ai_answer = response.choices[0].message.content
                st.markdown(ai_answer)
                st.session_state.messages.append({"role": "assistant", "content": ai_answer})

                # === ADMIN MANUAL REPLY ===
                if is_admin:
                    st.divider()
                    st.write("**🛠️ Admin Manual Reply**")
                    manual = st.text_area("AI का जवाब सही नहीं लगा तो अपना जवाब लिखो:", height=120, key="manual")
                    if st.button("Send Manual Reply"):
                        if manual.strip():
                            st.session_state.messages.append({"role": "assistant", "content": manual})
                            st.success("✅ Manual reply भेज दिया गया")
                            st.rerun()