"""
Turn an uploaded Streamlit file into LangChain documents.
"""

import hashlib
import os
import tempfile
from contextlib import contextmanager
from typing import Iterator, List

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from src.config import MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file's extension isn't supported."""


class FileTooLargeError(ValueError):
    """Raised when an uploaded file exceeds MAX_FILE_SIZE_MB."""


class EmptyFileError(ValueError):
    """Raised when an uploaded file has no bytes, or no extractable text."""


def get_file_extension(file_name: str) -> str:
    """Return the lowercase extension of a file name, or '' if none."""
    if "." in file_name:
        return file_name.rsplit(".", 1)[-1].lower()
    return ""


def compute_file_hash(file_bytes: bytes) -> str:
    """
    Stable content hash for an uploaded file, used as a cache key so
    the same file (re-encountered across Streamlit reruns, e.g. every
    time a chat question is asked) doesn't get re-parsed, re-classified,
    and re-embedded from scratch.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def validate_file_size(file_bytes: bytes) -> None:
    """
    Reject empty files and files over MAX_FILE_SIZE_MB, before we ever
    write them to disk or hand them to a loader.
    """
    if not file_bytes:
        raise EmptyFileError("The uploaded file is empty (0 bytes).")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise FileTooLargeError(
            f"File is {size_mb:.1f}MB, which exceeds the "
            f"{MAX_FILE_SIZE_MB}MB limit."
        )


def validate_documents_not_empty(docs: List[Document]) -> None:
    """
    Reject documents that loaded successfully but contain no usable
    text (e.g. a corrupted PDF, or a scanned image PDF with no OCR
    layer) — this would otherwise silently produce a junk
    classification instead of a clear error.
    """
    combined_text = "".join(doc.page_content for doc in docs).strip()

    if not combined_text:
        raise EmptyFileError(
            "No extractable text was found in this document "
            "(it may be corrupted, empty, or a scanned image with no "
            "text layer)."
        )


@contextmanager
def save_temp_file(file_bytes: bytes, extension: str) -> Iterator[str]:
    """
    Write bytes to a temp file with the given extension and yield its
    path. The file is deleted on exit, even if an exception occurs.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{extension}",
        ) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        yield temp_path
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def load_documents(temp_path: str, file_extension: str) -> List[Document]:
    """
    Load a file on disk into LangChain Documents based on its
    extension. Raises UnsupportedFileTypeError for unknown types.
    """
    if file_extension == "pdf":
        loader = PyPDFLoader(temp_path)
    elif file_extension == "docx":
        loader = Docx2txtLoader(temp_path)
    elif file_extension == "txt":
        loader = TextLoader(temp_path, encoding="utf-8")
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {file_extension}"
        )

    return loader.load()
