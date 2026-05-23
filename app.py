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

from supabase import create_client


FOLDER_ID = "0ADFKVoP1n82mUk9PVA"
CHROMA_DIR = "chroma_db"


st.set_page_config(page_title="MitrAI", page_icon="🤝", layout="centered")


# ================== Hide Streamlit UI ==================

hide_streamlit_style = """
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

[data-testid="stToolbar"] {
    display: none;
}

[data-testid="stDecoration"] {
    display: none;
}

[data-testid="stStatusWidget"] {
    visibility: hidden;
}

</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# ================== Secrets & Clients ==================

api_key = st.secrets.get("OPENAI_API_KEY", None)
google_service_account = st.secrets.get("GOOGLE_SERVICE_ACCOUNT", None)

supabase_url = st.secrets.get("SUPABASE_URL", None)
supabase_key = st.secrets.get("SUPABASE_KEY", None)

client = OpenAI(api_key=api_key) if api_key else None

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
else:
    supabase = None


# ================== Login Function ==================

def login_user(mobile, password):
    if not supabase:
        return None

    result = supabase.table("app_users") \
        .select("*") \
        .eq("mobile", mobile) \
        .eq("password", password) \
        .eq("status", "active") \
        .execute()

    if result.data:
        return result.data[0]

    return None


# ================== Admin Helper Functions ==================

def get_total_users():
    if not supabase:
        return 0

    try:
        result = supabase.table("app_users").select("*").execute()
        return len(result.data or [])
    except Exception:
        return 0


def get_total_chats():
    if not supabase:
        return 0

    try:
        result = supabase.table("chat_history").select("*").execute()
        return len(result.data or [])
    except Exception:
        return 0


def get_recent_chats(limit=20):
    if not supabase:
        return []

    try:
        result = supabase.table("chat_history") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        return result.data or []
    except Exception:
        try:
            result = supabase.table("chat_history") \
                .select("*") \
                .limit(limit) \
                .execute()
            return result.data or []
        except Exception:
            return []


def get_users():
    if not supabase:
        return []

    try:
        result = supabase.table("app_users") \
            .select("*") \
            .execute()
        return result.data or []
    except Exception:
        return []


# ================== Session State ==================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hello! I am **MitrAI**.\n\n"
            "Always here to help you with work, learning, guidance, "
            "and everyday questions."
        )
    }]


# ================== App Header ==================

st.title("🤝 MitrAI")
st.caption("Always Here to Help")


# ================== Login Screen ==================

if not st.session_state.logged_in:
    st.subheader("Login")

    mobile = st.text_input("Mobile Number")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(mobile, password)

        if user:
            st.session_state.logged_in = True
            st.session_state.user = user
            st.rerun()
        else:
            st.error("Invalid mobile number or password")

    st.stop()


# ================== Sidebar ==================

with st.sidebar:
    st.header("Settings")

    if st.session_state.user:
        st.write(f"Logged in as: {st.session_state.user.get('name')}")

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.messages = [{
                "role": "assistant",
                "content": (
                    "Hello! I am **MitrAI**.\n\n"
                    "Always here to help you with work, learning, guidance, "
                    "and everyday questions."
                )
            }]
            st.rerun()

    user_role = (st.session_state.user or {}).get("role", "staff")

    if user_role == "admin":
        st.divider()
        st.subheader("Admin Tools")

        if api_key:
            st.success("AI Connected")
        else:
            st.error("OpenAI API Key missing")

        if google_service_account:
            st.success("Google Drive Connected")
        else:
            st.error("Google Service Account missing")

        if st.button("Refresh Knowledge"):
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
            st.cache_resource.clear()
            st.success("Knowledge refreshed. Ask again.")

        st.divider()
        st.subheader("Admin Dashboard")

        total_users = get_total_users()
        total_chats = get_total_chats()

        st.metric("Total Users", total_users)
        st.metric("Total Chats", total_chats)

        with st.expander("Recent Chats", expanded=False):
            recent_chats = get_recent_chats(20)

            if recent_chats:
                for chat in recent_chats:
                    user_name = chat.get("user_name") or "Unknown User"
                    user_mobile = chat.get("user_mobile") or ""
                    question = chat.get("user_question") or ""
                    answer = chat.get("ai_answer") or ""

                    st.markdown(f"**{user_name}** {user_mobile}")
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer[:300]}...")
                    st.divider()
            else:
                st.info("No chat history found.")

        with st.expander("Users", expanded=False):
            users = get_users()

            if users:
                for u in users:
                    st.markdown(
                        f"**{u.get('name', 'Unknown')}**  \n"
                        f"Mobile: {u.get('mobile', '')}  \n"
                        f"Role: {u.get('role', '')}  \n"
                        f"Status: {u.get('status', '')}"
                    )
                    st.divider()
            else:
                st.info("No users found.")


# ================== Google Drive Functions ==================

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


# ================== Knowledge Base ==================

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

        except Exception:
            pass

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


# ================== Chat Display ==================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ================== Chat Logic ==================

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
- If the user asks a general question like products, blogs, company, policy, documents, or available information, assume they are asking about the available knowledge base unless they clearly mention another company or topic.
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

                if supabase and st.session_state.user:
                    supabase.table("chat_history").insert({
                        "user_email": st.session_state.user.get("email", ""),
                        "user_name": st.session_state.user.get("name"),
                        "user_mobile": st.session_state.user.get("mobile"),
                        "user_question": prompt,
                        "ai_answer": answer
                    }).execute()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
