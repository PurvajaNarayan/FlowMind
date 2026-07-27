"""Intent router (spec §7.0).

When evaluating on FlowVQA, route off the dataset's own `type` field — don't
waste a classification step on a label you already have. For free-form questions
with no label, fall back to a keyword heuristic (swap for an LLM call later).
"""

from __future__ import annotations

# Dataset `type` -> our three intents.
_TYPE_TO_INTENT = {
    "topological": "topological",
    "fact_retrieval": "content",
    "applied_scenario": "content",
    "flow_referential": "content",
    # code_request has no dataset `type`; it's triggered by intent, see below.
}

_CODE_KEYWORDS = ("code", "python", "implement", "function", "runnable", "program")
_TOPO_KEYWORDS = (
    "how many nodes", "how many edges", "shortest path", "in-degree",
    "predecessor", "successor", "directly follow", "directly precede",
)


def route_from_dataset_type(qa_type: str) -> str:
    """Fast path: map the FlowVQA `type` field to an intent."""
    return _TYPE_TO_INTENT.get(qa_type, "content")


def classify_intent_heuristic(question: str) -> str:
    """Fallback for unlabeled free-form questions (spec §7.0 fallback row)."""
    q = question.lower()
    if any(k in q for k in _CODE_KEYWORDS):
        return "code_request"
    if any(k in q for k in _TOPO_KEYWORDS):
        return "topological"
    return "content"


def route(question: str, qa_type: str | None = None) -> str:
    """Return one of 'topological' | 'content' | 'code_request'.
    Prefers the dataset label when present."""
    if qa_type is not None:
        return route_from_dataset_type(qa_type)
    return classify_intent_heuristic(question)
