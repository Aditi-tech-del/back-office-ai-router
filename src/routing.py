"""
Document-type -> department routing logic.
"""

ROUTING_MAP = {
    "Invoice": "Finance / Accounts Payable",
    "Purchase Order": "Procurement / Supply Chain",
    "Contract": "Legal / Compliance",
    "HR Document": "Human Resources",
    "Internal Memo": "Operations / Admin",
    "Financial Report": "Finance / Management",
}

DEFAULT_ROUTING = "Back Office Review"


def suggest_routing(doc_type: str) -> str:
    """Return the department a given document type should be routed to."""
    return ROUTING_MAP.get(doc_type, DEFAULT_ROUTING)
