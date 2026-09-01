"""
Back Office AI Router — entrypoint.

Uploads office documents (PDF/DOCX/TXT), classifies each one with an
LLM, suggests which department it should be routed to, and lets the
user chat with each document via a RAG assistant.

Run with:  streamlit run app.py
"""

import streamlit as st

from src.classification import ClassificationParseError, classify_document
from src.config import configure_page, load_groq_api_key
from src.document_loader import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    compute_file_hash,
    get_file_extension,
    load_documents,
    save_temp_file,
    validate_documents_not_empty,
    validate_file_size,
)
from src.models import build_chat_agent, build_classification_chain, load_embeddings
from src.rag_engine import answer_question, build_retriever
from frontend.styles import apply_custom_styles
from frontend.ui_components import (
    NAV_CHAT,
    NAV_CLASSIFICATION,
    render_chat_exchange,
    render_chat_history,
    render_chat_input,
    render_classification_result,
    render_export_button,
    render_file_title,
    render_file_uploader,
    render_footer,
    render_header,
    render_sidebar_nav,
)

configure_page()
apply_custom_styles()
load_groq_api_key()

embeddings = load_embeddings()
chat_agent = build_chat_agent()
classification_chain = build_classification_chain()

render_header()

with st.sidebar:
    st.markdown("#### Upload")
    uploaded_files = render_file_uploader()
    st.markdown("---")
    nav_choice = render_sidebar_nav()

results_data = []

# Cache per-file processing (loading, classification, embedding) across
# Streamlit reruns — otherwise every chat question re-triggers the full
# pipeline for every uploaded file, which is what made large PDFs feel slow.
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}

# Per-file multi-turn chat history, keyed by file hash, so asking a new
# question doesn't lose earlier ones in the same session.
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

processed_cache = st.session_state.processed_files
chat_histories = st.session_state.chat_histories

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully")

    for uploaded_file in uploaded_files:
        with st.container(border=True):
            file_name = uploaded_file.name
            file_extension = get_file_extension(file_name)

            render_file_title(file_name)

            try:
                file_bytes = uploaded_file.getvalue()
                file_hash = compute_file_hash(file_bytes)

                if file_hash in processed_cache:
                    # Already loaded, classified, and embedded earlier in
                    # this session — reuse it instead of redoing the work.
                    cached = processed_cache[file_hash]
                    result = cached["result"]
                    retriever = cached["retriever"]

                else:
                    validate_file_size(file_bytes)

                    with save_temp_file(file_bytes, file_extension) as temp_path:
                        docs = load_documents(temp_path, file_extension)
                        validate_documents_not_empty(docs)

                        with st.spinner(
                            f"Processing {file_name} ({len(docs)} page(s))... "
                            f"this only happens once per document."
                        ):
                            # ---- Classification -----------------------
                            result = classify_document(docs, classification_chain)

                            # ---- Build RAG retriever (chunk + embed) --
                            retriever = build_retriever(docs, embeddings)

                    processed_cache[file_hash] = {
                        "result": result,
                        "retriever": retriever,
                    }

                if nav_choice == NAV_CLASSIFICATION:
                    render_classification_result(result)

                else:  # NAV_CHAT
                    history = chat_histories.setdefault(file_hash, [])

                    render_chat_history(history)

                    user_question = render_chat_input(
                        file_name, key=f"chat_input_{file_hash}"
                    )

                    if user_question:
                        with st.spinner(f"Searching {file_name}..."):
                            response = answer_question(
                                user_question, retriever, chat_agent
                            )

                        history.append((user_question, response))
                        render_chat_exchange(user_question, response)

                st.divider()
                # ---- Collect for export --------------------------
                results_data.append(
                    {
                        "File Name": file_name,
                        "Document Type": result.doc_type,
                        "Department": result.department,
                        "Routing Suggestion": result.routing,
                        "Confidence": round(result.confidence, 2),
                        "Reason": result.reasoning,
                    }
                )

            except UnsupportedFileTypeError:
                st.error("Unsupported file type")

            except FileTooLargeError as e:
                st.error(str(e))

            except EmptyFileError as e:
                st.error(str(e))

            except ClassificationParseError as e:
                st.error("The AI returned an unreadable classification for this document.")
                st.caption(str(e))

            except Exception as e:  # noqa: BLE001 - surface any processing error
                st.error("Could not process document")
                st.caption(str(e))

render_export_button(results_data)
render_footer()
