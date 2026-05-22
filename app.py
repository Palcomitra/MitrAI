import os
import io
import json
import shutil
import streamlit as st
from openai import OpenAI

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


FOLDER_ID = "0ADFKVoP1n82mUk9PVA"
CHROMA_DIR = "chroma_db"

st.set_page_config(page_title="MitrAI", page_icon="🤝", layout="centered")

st.title("🤝 MitrAI")
st.caption("Always Here to Help")


api_key = st.secrets.get("OPENAI_API_KEY", None)
google_service_account = st.secrets.get("GOOGLE_SERVICE_ACCOUNT", None)

client = OpenAI(api_key=api_key) if api_key else None


with st.sidebar:
    st.header("Settings")

    if api_key:
        st.success("AI Connected")
    else:
        st.error("OpenAI API Key missing")

    if google_service_account:
        st.success("Google Drive Connected")
    else:
        st.error("Google Service Account missing")

    st.divider()

    if st.button("Refresh Knowledge"):
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        st.cache_resource.clear()
        st.success("Knowledge refreshed. Ask again.")


def get_drive_service():
    service_account_info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT"])

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )

    return build("drive", "v3", credentials=credentials)


def list_drive_files(service, folder_id):
    all_files = []

    query = f"'{folder_id}' in parents and trashed = false"

    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    files = results.get("files", [])

    for file in files:
        mime_type = file.get("mimeType")

        if mime_type == "application/vnd.google-apps.folder":
            all_files.extend(list_drive_files(service, file["id"]))
        else:
            all_files.append(file)

    return all_files


def read_google_doc(service, file_id):
    request = service.files().export_media(
        fileId=file_id,
        mimeType="text/plain"
    )

    file_data = io.BytesIO()
    downloader = MediaIoBaseDownload(file_data, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    return file_data.getvalue().decode("utf-8", errors="ignore")


def read_pdf(service, file_id):
    request = service.files().get_media(fileId=file_id)

    file_data = io.BytesIO()
    downloader = MediaIoBaseDownload(file_data, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    file_data.seek(0)
    reader = PdfReader(file_data)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def read_text_file(service, file_id):
    request = service.files().get_media(fileId=file_id)

    file_data = io.BytesIO()
    downloader = MediaIoBaseDownload(file_data, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    return file_data.getvalue().decode("utf-8", errors="ignore")


@st.cache_resource
def load_knowledge_base(openai_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    if os.path.exists(CHROMA_DIR):
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

    service = get_drive_service()
    files = list_drive_files(service, FOLDER_ID)

    documents = []

    for file in files:
        file_id = file["id"]
        name = file["name"]
        mime_type = file["mimeType"]

        try:
            text = ""

            if mime_type == "application/vnd.google-apps.document":
                text = read_google_doc(service, file_id)

            elif mime_type == "application/pdf":
                text = read_pdf(service, file_id)

            elif mime_type == "text/plain":
                text = read_text_file(service, file_id)

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": name}
                    )
                )

        except Exception as e:
            st.warning(f"Could not read file: {name}")

    if not documents:
        st.error("No readable documents found in Google Drive.")
        return None

    

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    
    return vectorstore


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hello! I am **MitrAI**.\n\n"
            "Always here to help you with work, learning, guidance, "
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
        elif not google_service_account:
            st.error("Google Drive connection is not configured.")
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
                        [
                            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                            for doc in relevant_docs
                        ]
                    )

                final_prompt = f"""
You are MitrAI, a friendly and practical AI companion.

Your tagline is: Always Here to Help.

Rules:
- Help users with work-related questions and everyday guidance.
- Use the available knowledge context when relevant.
- If the answer is not available in the knowledge base, say that clearly.
- Do not mention any company name unless the user asks or the context requires it.
- Give simple, clear, practical answers.
- For medical, legal, financial, or emotional crisis questions, give safe general guidance and suggest professional help when needed.

Knowledge Context:
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