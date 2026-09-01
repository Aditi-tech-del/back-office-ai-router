"""
LLM / embedding model loading, cached as Streamlit resources.
"""

import streamlit as st
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHAT_MODEL, CLASSIFICATION_MODEL, EMBEDDING_MODEL
from src.pii_middleware import build_pii_middleware

CLASSIFICATION_PROMPT = """
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
"""


@st.cache_resource
def load_embeddings() -> HuggingFaceEmbeddings:
    """Load (and cache) the sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def build_chat_agent():
    """
    Load (and cache) the RAG chat assistant, wrapped with
    PIIMiddleware so PII in the user's question and the retrieved
    document context is masked before it reaches the model, and any
    PII in the model's answer is masked again on the way out.
    """
    llm = ChatGroq(
        model=CHAT_MODEL,
        temperature=0,
    )

    return create_agent(
        model=llm,
        tools=[],
        middleware=build_pii_middleware(),
    )


@st.cache_resource
def build_classification_chain():
    """Build (and cache) the document-classification chain."""
    llm = ChatGroq(
        model=CLASSIFICATION_MODEL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)

    return prompt | llm
