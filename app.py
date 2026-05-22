import os
import shutil
import streamlit as st
from openai import OpenAI

from langchain_community.document_loaders import GoogleDriveLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


FOLDER_ID = "0ADFKVoP1n82mUk9PVA"
CHROMA_DIR = "chroma_db"


st.set_page_config(
    page_title="MitrAI",
    page_icon="🤝",
    layout="centered"
)


st.title("🤝 MitrAI")
st.caption("Always Here to Help")


try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
except Exception:
    api_key = None
    client = None


with st.sidebar:
    st.header("Settings")

    if api_key:
        st.success("AI Connected")
    else:
        st.error("OpenAI API Key not found in Streamlit Secrets")

    st.divider()

    if st.button("Refresh Knowledge"):
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)

        st.cache_resource.clear()
        st.success("Knowledge refreshed. Please ask your question again.")


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
        st.error("No documents found in the knowledge folder.")
        return None

    st.info(f"Loaded {len(docs)} documents from knowledge folder.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        st.error("Documents found, but readable text could not be extracted.")
        return None

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    st.success(f"Knowledge base created with {len(chunks)} chunks.")
    return vectorstore


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hello! I am **MitrAI**.\n\n"
            "I am always here to help you with work, learning, guidance, "
            "and everyday questions."
        )
    }]


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("AI connection is not configured.")
        else:
            with st.spinner("Thinking..."):
                vectorstore = load_knowledge_base(api_key)

                context = ""

                if vectorstore:
                    retriever = vectorstore.as_retriever(
                        search_kwargs={"k": 5}
                    )

                    relevant_docs = retriever.invoke(prompt)

                    context = "\n\n".join(
                        [doc.page_content for doc in relevant_docs]
                    )

                final_prompt = f"""
You are MitrAI, a friendly and practical AI companion.

Your tagline is: Always Here to Help.

Your role:
- Help users with work-related questions.
- Help users with everyday guidance.
- Give clear, practical, calm, and respectful answers.
- If company or document context is available, use it.
- If the answer is not found in the available documents, clearly say that it is not found in the available knowledge base and then give a general helpful answer.
- For health, legal, financial, or emotional crisis topics, give safe general guidance and suggest consulting a qualified professional when needed.
- Do not mention Palco unless the user asks about Palco or the available documents contain Palco-related context.

Available Knowledge Context:
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