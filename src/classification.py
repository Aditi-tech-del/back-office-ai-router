"""
Document classification: splitting, running the classification chain,
and parsing its JSON output into a structured result.
"""

import json
from dataclasses import dataclass
from typing import List

import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CLASSIFICATION_CHUNK_OVERLAP,
    CLASSIFICATION_CHUNK_SIZE,
    CLASSIFICATION_MAX_RETRIES,
    CLASSIFICATION_TEXT_LIMIT,
)
from src.routing import ROUTING_MAP, suggest_routing
from src.utils import clean_llm_json

KNOWN_DOC_TYPES = set(ROUTING_MAP.keys())

RETRY_SUFFIX = """

Your previous response was not valid JSON matching the required
format. Return ONLY the JSON object, with no extra text, no markdown
fences, and no commentary before or after it.
"""


class ClassificationParseError(ValueError):
    """Raised when the LLM's classification response can't be trusted,
    even after a retry."""


@dataclass
class ClassificationResult:
    doc_type: str
    confidence: float
    department: str
    reasoning: str
    routing: str


def _validate_and_build_result(raw: dict) -> ClassificationResult:
    """
    Turn a raw (already JSON-parsed) dict into a trustworthy
    ClassificationResult: unknown doc types fall back to "Unknown"
    instead of a routing lookup miss, and confidence is clamped to
    a sane [0, 1] range instead of trusted blindly.
    """
    doc_type = str(raw.get("document_type", "Unknown"))
    if doc_type not in KNOWN_DOC_TYPES:
        doc_type = "Unknown"

    try:
        confidence = float(raw.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    department = str(raw.get("recommended_department", "Unknown"))
    reasoning = str(raw.get("reasoning", "No reasoning provided."))

    routing = suggest_routing(doc_type)

    return ClassificationResult(
        doc_type=doc_type,
        confidence=confidence,
        department=department,
        reasoning=reasoning,
        routing=routing,
    )


def _truncate_for_classification(docs: List[Document]) -> str:
    """Split docs into small chunks and join into a size-limited string."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CLASSIFICATION_CHUNK_SIZE,
        chunk_overlap=CLASSIFICATION_CHUNK_OVERLAP,
    )

    split_docs = splitter.split_documents(docs)

    text = "\n\n".join(doc.page_content for doc in split_docs)

    return text[:CLASSIFICATION_TEXT_LIMIT]


def classify_document(
    docs: List[Document],
    classification_chain,
) -> ClassificationResult:
    """
    Run the classification chain on a document and return a
    structured, validated ClassificationResult (including routing
    suggestion).

    If the model's response isn't valid JSON, retries up to
    CLASSIFICATION_MAX_RETRIES times with a stricter prompt before
    giving up.
    """
    text = _truncate_for_classification(docs)
    document_text = text

    last_error: Exception = None

    for attempt in range(CLASSIFICATION_MAX_RETRIES + 1):
        with st.spinner(
            "Analyzing document..."
            if attempt == 0
            else "Re-analyzing document (retry)..."
        ):
            response = classification_chain.invoke(
                {"document_text": document_text}
            )

        try:
            cleaned = clean_llm_json(response.content)
            raw = json.loads(cleaned)

            if not isinstance(raw, dict):
                raise ValueError("Classification response was not a JSON object")

            return _validate_and_build_result(raw)

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            # Ask again more strictly, still grounded in the same document text
            document_text = text + RETRY_SUFFIX

    raise ClassificationParseError(
        f"Model did not return valid classification JSON after "
        f"{CLASSIFICATION_MAX_RETRIES + 1} attempt(s): {last_error}"
    )
