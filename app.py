import os
import streamlit as st
from openai import OpenAI

from langchain_community.document_loaders import GoogleDriveLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


FOLDER_ID = "0ADFKVoP1n82mUk9PVA"
CHROMA_DIR = "chroma_db"


st.set_page_config(page_title="MitrAI", page_icon="🤝", layout="centered")

st.title("🤝 MitrAI")
st.caption("Always Here to Help")


with st.sidebar:
    st.header("Staff / Admin")
    api_key = st.text_input("OpenAI API Key", type="password")

    if api_key:
        client = OpenAI(api_key=api_key)
        st.success("API Connected")
    else:
        client = None
        st.warning("Enter OpenAI API Key")

    st.divider()

    if st.button("Refresh Google Drive Knowledge"):
        if os.path.exists(CHROMA_DIR):
            import shutil
            shutil.rmtree(CHROMA_DIR)
        st.cache_resource.clear()
        st.success("Knowledge refreshed. Ask a question again.")


@st.cache_resource
def load_knowledge_base(openai_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    if os.path.exists(CHROMA_DIR):
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

    loader = GoogleDriveLoader(
        folder_id=FOLDER_ID,
        token_path="token.json",
        credentials_path="credentials.json",
        recursive=True,
        file_types=["document", "pdf", "text/plain"]
    )

    docs = loader.load()

    if not docs:
        st.error("No documents found in Google Drive folder.")
        return None

    st.info(f"Loaded {len(docs)} documents from Google Drive")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        st.error("Documents found, but no readable text extracted.")
        return None

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    st.success(f"Knowledge base created with {len(chunks)} chunks")
    return vectorstore


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I am MitrAI. I am always here to help you with work, learning, guidance, and everyday questions."
    }]


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("Please enter OpenAI API Key first.")
        else:
            with st.spinner("Checking Palco documents..."):
                vectorstore = load_knowledge_base(api_key)

                if vectorstore:
                    retriever = vectorstore.as_retriever(
                        search_kwargs={"k": 5}
                    )

                    relevant_docs = retriever.invoke(prompt)

                    context = "\n\n".join(
                        [doc.page_content for doc in relevant_docs]
                    )

                    final_prompt = f"""
You are MitrAI, Palco's internal AI assistant.

Use the company document context below when it is relevant.
If the answer is not available in the documents, clearly say that it is not found in the available company documents and then give a general helpful answer if appropriate.

Company Document Context:
{context}

User Question:
{prompt}

Answer in simple, clear English.
"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "user", "content": final_prompt}
                        ],
                        temperature=0.4,
                        max_tokens=900
                    )

                    answer = response.choices[0].message.content
                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                else:
                    st.error("Knowledge base could not be loaded.")
