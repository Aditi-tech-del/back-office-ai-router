"""
General-purpose helpers.
"""

import re


def clean_llm_json(text: str) -> str:
    """
    Strip markdown code-fences from an LLM response and extract the
    first JSON object found in it, so it can be safely json.loads()'d.
    """
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        return match.group(0)

    return text.strip()
