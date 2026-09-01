"""
Streamlit UI rendering helpers.

Keeping these separate from app.py / business logic makes the
layout easy to restyle without touching the processing pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.classification import ClassificationResult
from src.config import SUPPORTED_FILE_TYPES


NAV_CLASSIFICATION = "📋 Classification & Routing"
NAV_CHAT = "💬 Chat with Document"


def render_header() -> None:
    st.markdown(
        """
        <div class="app-hero">
        <h1>Back Office AI Router</h1>
        <p>Upload a document, get it classified and routed, then ask it questions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_nav() -> str:
    """
    Render the sidebar view switcher (Classification vs Chat).

    A sidebar radio (unlike st.tabs) keeps its selection across
    Streamlit reruns, so it doesn't snap back to the first view every
    time a chat message is sent.
    """
    st.sidebar.markdown("#### View")
    return st.sidebar.radio(
        "View",
        options=[NAV_CLASSIFICATION, NAV_CHAT],
        label_visibility="collapsed",
        key="nav_choice",
    )


def render_file_uploader():
    """Render the multi-file uploader and return the uploaded files."""
    return st.file_uploader(
        "Upload PDF, DOCX, TXT",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
    )


def render_file_title(file_name: str) -> None:
    st.markdown(
        f"""
        <div class="file-title">
            {file_name}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_classification_result(result: ClassificationResult) -> None:
    """Render the document-type / department / confidence result card."""
    with st.container(border=True):
        st.markdown(f"## {result.doc_type}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Department", result.department)

        with col2:
            st.metric("Routing", result.routing)

        st.markdown("### Reasoning")
        st.info(result.reasoning)

        if result.confidence >= 0.85:
            st.success(f"Confidence: {result.confidence:.2f} (High)")
        elif result.confidence >= 0.60:
            st.warning(f"Confidence: {result.confidence:.2f} (Medium)")
        else:
            st.error(f"Confidence: {result.confidence:.2f} (Low)")

        st.progress(result.confidence)


def render_chat_input(file_name: str, key: str) -> Optional[str]:
    """
    Render the chat input for asking questions about this document.

    Uses st.chat_input (not st.text_input) — it auto-clears after each
    submission, so the box is immediately ready for the next question
    instead of holding onto the previous one.
    """
    return st.chat_input(
        f"Ask anything about {file_name}",
        key=key,
    )


def render_chat_exchange(user_question: str, response: Dict[str, Any]) -> None:
    """Render a single user question + assistant answer + sources."""
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        st.markdown(response["answer"])

        with st.expander("View Sources"):
            for i, doc in enumerate(response["sources"], start=1):
                st.markdown(f"### Source {i}")
                st.code(doc.page_content[:800], language=None)


def render_chat_history(
    history: List[Tuple[str, Dict[str, Any]]]
) -> None:
    """Render every question/answer turn asked so far, in order."""
    for question, response in history:
        render_chat_exchange(question, response)


def render_export_button(results_data: List[Dict[str, Any]]) -> None:
    """Render the CSV export section, if there's anything to export."""
    if not results_data:
        return

    st.markdown("## Export Results")

    df = pd.DataFrame(results_data)
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="document_results.csv",
        mime="text/csv",
    )


def render_footer() -> None:
    st.markdown("---")
    st.caption("Back Office AI Router • AI Document Intelligence System")
