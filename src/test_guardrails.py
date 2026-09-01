"""
Standalone checks for the file-upload and LLM-output guardrails.

Run with:  python3 test_guardrails.py

No API key needed — the LLM chain is faked, so this only tests our
own validation code, not Groq itself.
"""

import sys

from langchain_core.documents import Document

from document_loader import (
    EmptyFileError,
    FileTooLargeError,
    validate_documents_not_empty,
    validate_file_size,
)
from classification import (
    ClassificationParseError,
    classify_document,
    _validate_and_build_result,
)
from config import MAX_FILE_SIZE_BYTES

passed = 0
failed = 0


def check(label, condition):
    global passed, failed
    if condition:
        print(f"  PASS  {label}")
        passed += 1
    else:
        print(f"  FAIL  {label}")
        failed += 1


def expect_raises(label, exc_type, fn):
    try:
        fn()
        check(label, False)
    except exc_type:
        check(label, True)
    except Exception as e:  # wrong exception type
        check(f"{label} (raised {type(e).__name__} instead)", False)


# ---------------------------------------------------------------
print("\n== File upload validation ==")

expect_raises(
    "empty file (0 bytes) raises EmptyFileError",
    EmptyFileError,
    lambda: validate_file_size(b""),
)

expect_raises(
    "oversized file raises FileTooLargeError",
    FileTooLargeError,
    lambda: validate_file_size(b"x" * (MAX_FILE_SIZE_BYTES + 1)),
)

try:
    validate_file_size(b"normal small file content")
    check("normal-sized file passes", True)
except Exception:
    check("normal-sized file passes", False)

expect_raises(
    "doc with only whitespace text raises EmptyFileError",
    EmptyFileError,
    lambda: validate_documents_not_empty(
        [Document(page_content="   \n  "), Document(page_content="")]
    ),
)

try:
    validate_documents_not_empty([Document(page_content="Invoice #123, total $500")])
    check("doc with real text passes", True)
except Exception:
    check("doc with real text passes", False)


# ---------------------------------------------------------------
print("\n== LLM output validation ==")

result = _validate_and_build_result(
    {"document_type": "Made Up Type", "confidence": 0.9}
)
check("unknown doc_type falls back to 'Unknown'", result.doc_type == "Unknown")

result = _validate_and_build_result({"document_type": "Invoice", "confidence": 5.7})
check("confidence > 1 gets clamped to 1.0", result.confidence == 1.0)

result = _validate_and_build_result({"document_type": "Invoice", "confidence": -3})
check("confidence < 0 gets clamped to 0.0", result.confidence == 0.0)

result = _validate_and_build_result({"document_type": "Invoice", "confidence": "oops"})
check("non-numeric confidence defaults to 0.0", result.confidence == 0.0)

result = _validate_and_build_result({"document_type": "Contract", "confidence": 0.8})
check(
    "known doc_type still routes correctly",
    result.routing == "Legal / Compliance",
)


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeChainAlwaysBad:
    """Simulates a model that never returns valid JSON."""

    def invoke(self, _inputs):
        return FakeResponse("not json at all")


class FakeChainRecovers:
    """Simulates a model that fails once, then succeeds on retry."""

    def __init__(self):
        self.calls = 0

    def invoke(self, _inputs):
        self.calls += 1
        if self.calls == 1:
            return FakeResponse("garbage, not json")
        return FakeResponse(
            '{"document_type": "Invoice", "confidence": 0.95, '
            '"recommended_department": "Finance", "reasoning": "test"}'
        )


docs = [Document(page_content="Invoice total: $1,200 due Sept 30")]

expect_raises(
    "always-invalid model output raises ClassificationParseError after retries",
    ClassificationParseError,
    lambda: classify_document(docs, FakeChainAlwaysBad()),
)

fake_chain = FakeChainRecovers()
result = classify_document(docs, fake_chain)
check(
    "model that recovers on retry produces a valid result",
    result.doc_type == "Invoice" and fake_chain.calls == 2,
)


# ---------------------------------------------------------------
print("\n== PII middleware ==")

import os

os.environ.setdefault("GROQ_API_KEY", "dummy-test-key-for-structural-check")

from src.pii_middleware import PHONE_NUMBER_PATTERN

phone_cases = [
    ("my phone number is 9168578655", True),   # bare 10-digit run
    ("call 916-857-8655", True),               # dashed
    ("card: 4111111111111111", False),         # unformatted 16-digit card
    ("card: 4111 1111 1111 1111", False),      # 4-digit grouped card
    ("no numbers here at all", False),
]
for text, should_match in phone_cases:
    got_match = bool(PHONE_NUMBER_PATTERN.search(text))
    check(f"phone regex on {text!r} (expect match={should_match})", got_match == should_match)

try:
    from models import build_chat_agent

    agent = build_chat_agent.__wrapped__()  # bypass st.cache_resource
    node_names = set(agent.get_graph().nodes.keys())

    expected_hooks = {
        f"PIIMiddleware[{pii_type}].{hook}"
        for pii_type in ("email", "credit_card", "ip", "phone_number")
        for hook in ("before_model", "after_model")
    }
    check(
        "chat agent wires all 4 PII types with before/after hooks",
        expected_hooks.issubset(node_names),
    )
except Exception as e:
    check(f"chat agent builds without error (raised {type(e).__name__}: {e})", False)


# ---------------------------------------------------------------
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
