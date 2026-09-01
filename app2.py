import os
import json
import re
import tempfile

import pandas as pd
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Back Office AI Router",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #f8fafc,
        #e2e8f0
    );
    color: #0f172a;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main Panels */

.panel {
    background: white;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow:
        0 10px 30px rgba(0,0,0,0.06);
}

/* File Title */

.file-title {
    font-size: 2rem;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 1rem;
    color: #0f172a;
}

/* Chat Input */

.stTextInput input {
    border-radius: 14px !important;
    border: 1px solid #cbd5e1 !important;
    padding: 12px !important;
    font-size: 16px !important;
}

/* Assistant Chat */

[data-testid="stChatMessage"] {
    background: white !important;
    border-radius: 16px !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0 !important;
    padding: 14px !important;
    margin-top: 12px !important;
}

/* User Message */

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-user"]
) {
    background: #dbeafe !important;
    color: #0f172a !important;
}

/* Assistant Message */

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-assistant"]
) {
    background: #ffffff !important;
    color: #0f172a !important;
}

/* Fix Assistant Text */

[data-testid="stChatMessageContent"] {
    color: #0f172a !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}

/* Buttons */

.stButton button,
.stDownloadButton button {

    background: #2563eb !important;
    color: white !important;

    border: none !important;

    border-radius: 12px !important;

    padding:
        0.6rem 1.2rem !important;

    font-weight: 600 !important;

    transition: 0.2s ease-in-out !important;
}

/* Button Hover */

.stButton button:hover,
.stDownloadButton button:hover {

    background: #1d4ed8 !important;
    color: white !important;

    transform: translateY(-1px);
}

/* Expander */

.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: #0f172a !important;
}

/* Metrics */

[data-testid="stMetricValue"] {
    color: #0f172a !important;
}

/* Progress Bar */

.stProgress > div > div > div > div {
    background-color: #2563eb !important;
}
/* Containers */

[data-testid="stVerticalBlock"] > div:has(
    div[data-testid="stChatMessage"]
) {
    gap: 0.8rem;
}

/* Text Inputs */

.stTextInput {
    margin-top: 1rem;
    margin-bottom: 1rem;
}

/* Cards */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border: 1px solid #dbe4ee !important;
    padding: 1rem !important;
    background: white !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# ENV VARIABLES
# =========================================================

load_dotenv()

groq_api_key = os.getenv(
    "GROQ_API_KEY"
)

if not groq_api_key:

    st.error(
        "GROQ_API_KEY missing in .env"
    )

    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key


# =========================================================
# ROUTING MAP
# =========================================================

ROUTING_MAP = {
    "Invoice":
        "Finance / Accounts Payable",

    "Purchase Order":
        "Procurement / Supply Chain",

    "Contract":
        "Legal / Compliance",

    "HR Document":
        "Human Resources",

    "Internal Memo":
        "Operations / Admin",

    "Financial Report":
        "Finance / Management"
}


def suggest_routing(
    doc_type: str
) -> str:

    return ROUTING_MAP.get(
        doc_type,
        "Back Office Review"
    )


# =========================================================
# CLEAN JSON
# =========================================================

def clean_llm_json(text: str) -> str:

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:
        return match.group(0)

    return text.strip()


# =========================================================
# LOAD EMBEDDINGS
# =========================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


# =========================================================
# LOAD CHAT MODEL
# =========================================================

@st.cache_resource
def load_chat_model():

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )


chat_llm = load_chat_model()


# =========================================================
# CLASSIFICATION CHAIN
# =========================================================

@st.cache_resource
def build_classification_chain():

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are an office document
classification assistant.

Classify into one of:

- Invoice
- Purchase Order
- Contract
- HR Document
- Internal Memo
- Financial Report

Return ONLY valid JSON.

Format:
{{
  "document_type": "...",
  "confidence": 0.0,
  "recommended_department": "...",
  "reasoning": "..."
}}

Document:
{document_text}
""")

    return prompt | llm


classification_chain = (
    build_classification_chain()
)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="panel">

<h1>
Back Office AI Router
</h1>

</div>
""", unsafe_allow_html=True)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, TXT",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)


# =========================================================
# STORAGE
# =========================================================

results_data = []


# =========================================================
# PROCESS FILES
# =========================================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} file(s) uploaded successfully"
    )

    for uploaded_file in uploaded_files:
        with st.container(border=True):
            file_name = uploaded_file.name

            file_extension = (
                file_name.rsplit(".", 1)[-1].lower()
                if "." in file_name
                else ""
            )

            st.markdown(
                f"""
                <div class="file-title">
                    {file_name}
                </div>
                """,
                unsafe_allow_html=True
            )
            temp_path = None

            try:

                # =============================================
                # SAVE TEMP FILE
                # =============================================

                file_bytes = (
                    uploaded_file.getvalue()
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{file_extension}"
                ) as tmp:

                    tmp.write(file_bytes)

                    temp_path = tmp.name

                # =============================================
                # DOCUMENT LOADER
                # =============================================

                if file_extension == "pdf":

                    loader = PyPDFLoader(
                        temp_path
                    )

                elif file_extension == "docx":

                    loader = Docx2txtLoader(
                        temp_path
                    )

                elif file_extension == "txt":

                    loader = TextLoader(
                        temp_path,
                        encoding="utf-8"
                    )

                else:

                    st.error(
                        "Unsupported file type"
                    )

                    continue

                docs = loader.load()

                # =============================================
                # SPLIT FOR CLASSIFICATION
                # =============================================

                splitter = (
                    RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                )

                split_docs = (
                    splitter.split_documents(
                        docs
                    )
                )

                text = "\n\n".join(
                    [
                        doc.page_content
                        for doc in split_docs
                    ]
                )[:6000]

                # =============================================
                # CLASSIFICATION
                # =============================================

                with st.spinner(
                    "Analyzing document..."
                ):

                    response = (
                        classification_chain.invoke(
                            {
                                "document_text": text
                            }
                        )
                    )

                cleaned = clean_llm_json(
                    response.content
                )

                result = json.loads(
                    cleaned
                )

                doc_type = str(
                    result.get(
                        "document_type",
                        "Unknown"
                    )
                )

                confidence = float(
                    result.get(
                        "confidence",
                        0
                    )
                )

                department = str(
                    result.get(
                        "recommended_department",
                        "Unknown"
                    )
                )

                reasoning = str(
                    result.get(
                        "reasoning",
                        "No reasoning provided."
                    )
                )

                routing = suggest_routing(
                    doc_type
                )

                # =============================================
                # RESULT CARD
                # =============================================

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"## {doc_type}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Department",
                            department
                        )

                    with col2:

                        st.metric(
                            "Routing",
                            routing
                        )

                    st.markdown(
                        "### Reasoning"
                    )

                    st.info(
                        reasoning
                    )

                    if confidence >= 0.85:

                        st.success(
                            f"""
    Confidence:
    {confidence:.2f} (High)
    """
                        )

                    elif confidence >= 0.60:

                        st.warning(
                            f"""
    Confidence:
    {confidence:.2f} (Medium)
    """
                        )

                    else:

                        st.error(
                            f"""
    Confidence:
    {confidence:.2f} (Low)
    """
                        )

                    st.progress(
                        confidence
                    )

                # =============================================
                # DOCUMENT RAG
                # =============================================

                st.markdown("""
                ### AI Document Assistant
                """)

                rag_splitter = (
                    RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                )

                rag_docs = (
                    rag_splitter.split_documents(
                        docs
                    )
                )

                vectorstore = (
                    FAISS.from_documents(
                        rag_docs,
                        embeddings
                    )
                )

                retriever = (
                    vectorstore.as_retriever(
                        search_kwargs={"k": 4}
                    )
                )

                # =============================================
                # AGENT FUNCTION
                # =============================================

                def agent(user_question):

                    retrieved_docs = (
                        retriever.invoke(
                            user_question
                        )
                    )

                    context = "\n\n".join(
                        [
                            doc.page_content
                            for doc in retrieved_docs
                        ]
                    )

                    prompt = f"""
    You are a helpful AI assistant.

    Answer ONLY from the
    provided context.

    If the answer is not found,
    say you could not find it.

    Context:
    {context}

    Question:
    {user_question}
    """

                    response = (
                        chat_llm.invoke(
                            prompt
                        )
                    )

                    return {
                        "answer":
                            response.content,

                        "sources":
                            retrieved_docs
                    }

                # =============================================
                # CHAT INPUT
                # =============================================

                user_question = st.text_input(
                    f"Ask anything about {file_name}",
                    key=f"chat_{file_name}",
                    placeholder="Summarize the document, find signatures, payment details..."
                )
                # =============================================
                # CHAT RESPONSE
                # =============================================

                if user_question:

                    with st.chat_message(
                        "user"
                    ):

                        st.markdown(
                            user_question
                        )

                    with st.spinner(
                        f"Searching {file_name}..."
                    ):

                        response = agent(
                            user_question
                        )

                    with st.chat_message(
                        "assistant"
                    ):

                        st.markdown(
                            response["answer"]
                        )

                        with st.expander(
                            "View Sources"
                        ):

                            for i, doc in enumerate(
                                response["sources"],
                                start=1
                            ):

                                st.markdown(
                                    f"### Source {i}"
                                )

                                st.code(
                                    doc.page_content[:800], language=None
                                )

                st.divider()

                # =============================================
                # STORE RESULTS
                # =============================================

                results_data.append({

                    "File Name":
                        file_name,

                    "Document Type":
                        doc_type,

                    "Department":
                        department,

                    "Routing Suggestion":
                        routing,

                    "Confidence":
                        round(confidence, 2),

                    "Reason":
                        reasoning
                })

            except Exception as e:

                st.error(
                    "Could not process document"
                )

                st.caption(
                    str(e)
                )

            finally:

                if (
                    temp_path
                    and os.path.exists(temp_path)
                ):

                    os.unlink(
                        temp_path
                    )


# =========================================================
# EXPORT RESULTS
# =========================================================

if results_data:

    st.markdown(
        "## Export Results"
    )

    df = pd.DataFrame(
        results_data
    )

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="document_results.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Back Office AI Router • AI Document Intelligence System"
)
