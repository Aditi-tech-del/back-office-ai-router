"""
PII detection/masking for the RAG chat assistant, using LangChain's
built-in PIIMiddleware.

Scope: applied ONLY to the RAG chat path (user questions + retrieved
document context + the model's answer) — NOT to document
classification, since classification needs to see real document
content (names, account numbers, etc.) to route it correctly.
"""

import re

from langchain.agents.middleware import PIIMiddleware

PII_STRATEGY = "mask"

# Not a built-in PIIMiddleware type, so we supply our own detector.
# Matches formatted numbers ("916-857-8655", "(916) 857-8655") AND
# bare 10-digit runs ("9168578655"). \b...\b on both ends means it
# only matches a digit run that is EXACTLY 10 digits long — it will
# not match a 10-digit substring inside a longer unbroken run (e.g. a
# 16-digit unformatted credit card number), since there's no word
# boundary in the middle of a longer digit sequence.
PHONE_NUMBER_PATTERN = re.compile(
    r"\b(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
)


def build_pii_middleware() -> list:
    """
    Middleware stack for the RAG chat agent: masks common PII types
    in the user's question / retrieved context (before_model) AND in
    the model's answer (after_model).

    Note: apply_to_output defaults to False in PIIMiddleware, so it
    must be set explicitly here or answers are never scanned.
    """
    return [
        PIIMiddleware(
            "email", strategy=PII_STRATEGY, apply_to_output=True
        ),
        PIIMiddleware(
            "credit_card", strategy=PII_STRATEGY, apply_to_output=True
        ),
        PIIMiddleware(
            "ip", strategy=PII_STRATEGY, apply_to_output=True
        ),
        PIIMiddleware(
            "phone_number",
            # detector must be a Callable or a plain regex string —
            # NOT a compiled re.Pattern (that raises
            # "'re.Pattern' object is not callable").
            detector=PHONE_NUMBER_PATTERN.pattern,
            strategy=PII_STRATEGY,
            apply_to_output=True,
        ),
    ]
