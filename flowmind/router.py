"""Intent router (spec §7.0).

When evaluating on FlowVQA, route off the dataset's own `type` field — don't
waste a classification step on a label you already have. For free-form questions
with no label, classify the question type with a trained (no-LLM) model, and
fall back to a keyword heuristic if the model isn't available.

Classification stack, best available first:
  1. dataset `type` label            -> route_from_dataset_type
  2. trained TF-IDF + LogReg model   -> classify_type_model  (question_classifier.py)
  3. keyword heuristic               -> classify_intent_heuristic
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
    """Zero-dependency keyword fallback for unlabeled free-form questions."""
    q = question.lower()
    if any(k in q for k in _CODE_KEYWORDS):
        return "code_request"
    if any(k in q for k in _TOPO_KEYWORDS):
        return "topological"
    return "content"


# Lazily-loaded trained classifier; None once we know it's unavailable.
_model = None
_model_unavailable = False


def classify_type_model(question: str) -> str | None:
    """Predict one of the four FlowVQA question types with the trained model.
    Returns None if the model (or its deps) aren't available — callers fall
    back to the heuristic. Train it with tools/train_router.py."""
    global _model, _model_unavailable
    if _model is None and not _model_unavailable:
        try:
            from flowmind.question_classifier import (
                DEFAULT_MODEL_PATH,
                QuestionTypeClassifier,
            )

            if DEFAULT_MODEL_PATH.exists():
                _model = QuestionTypeClassifier.load()
            else:
                _model_unavailable = True
        except Exception:
            _model_unavailable = True
    return _model.predict(question) if _model is not None else None


def route(question: str, qa_type: str | None = None) -> str:
    """Return one of 'topological' | 'content' | 'code_request'.

    Uses the best available signal: the dataset label, else the trained
    question-type model, else the keyword heuristic. `code_request` has no
    dataset type, so it is detected up front by keyword before the model runs.
    """
    if qa_type is not None:
        return route_from_dataset_type(qa_type)

    # The trained model can't produce code_request (no such dataset type), so
    # catch explicit code asks first.
    q = question.lower()
    if any(k in q for k in _CODE_KEYWORDS):
        return "code_request"

    predicted_type = classify_type_model(question)
    if predicted_type is not None:
        return route_from_dataset_type(predicted_type)
    return classify_intent_heuristic(question)
