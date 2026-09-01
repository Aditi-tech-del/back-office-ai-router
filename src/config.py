"""
Application configuration.

Handles Streamlit page config and environment variable
loading/validation (e.g. GROQ_API_KEY).
"""

import os

import streamlit as st
from dotenv import load_dotenv

PAGE_TITLE = "Back Office AI Router"
PAGE_LAYOUT = "wide"

# Models
CLASSIFICATION_MODEL = "openai/gpt-oss-20b"
CHAT_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitting
CLASSIFICATION_CHUNK_SIZE = 500
CLASSIFICATION_CHUNK_OVERLAP = 50
CLASSIFICATION_TEXT_LIMIT = 6000

RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_RETRIEVER_K = 4

SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt"]

# --- Guardrails: file upload limits ---
MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- Guardrails: LLM output validation ---
CLASSIFICATION_MAX_RETRIES = 1  # extra attempts if the model returns bad JSON


def configure_page() -> None:
    """Set Streamlit page config. Must be called first, once."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout=PAGE_LAYOUT,
    )


def load_groq_api_key() -> str:
    """
    Load GROQ_API_KEY from the environment / .env file.

    Stops the app with an error message if the key is missing.
    """
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("GROQ_API_KEY missing in .env")
        st.stop()

    os.environ["GROQ_API_KEY"] = groq_api_key

    return groq_api_key
