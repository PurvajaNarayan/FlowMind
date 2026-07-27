"""Lightweight test of the classifier pipeline — trains on a tiny in-memory set
so it doesn't need the full dataset or the saved model."""

import pytest

pytest.importorskip("sklearn")

from flowmind.question_classifier import QUESTION_TYPES, QuestionTypeClassifier

_TINY = [
    ("How many nodes exist in the given flowchart?", "topological"),
    ("How many edges exist in the given flowchart?", "topological"),
    ("What is the shortest path between A and B?", "topological"),
    ("What does the process node output?", "fact_retrieval"),
    ("What is the label of the start node?", "fact_retrieval"),
    ("If the input is negative, what happens next?", "applied_scenario"),
    ("Suppose the value is 5, which branch is taken?", "applied_scenario"),
    ("Which node is reached after the decision fails?", "flow_referential"),
    ("What comes after the initialization step?", "flow_referential"),
]


def test_train_predict_save_load(tmp_path):
    qs = [q for q, _ in _TINY]
    ys = [y for _, y in _TINY]
    model = QuestionTypeClassifier().train(qs, ys)

    assert model.predict("How many nodes exist?") == "topological"

    probs = model.predict_proba("How many edges exist?")
    assert set(probs) <= set(QUESTION_TYPES)
    assert abs(sum(probs.values()) - 1.0) < 1e-6

    path = tmp_path / "m.joblib"
    model.save(path)
    reloaded = QuestionTypeClassifier.load(path)
    assert reloaded.predict("How many nodes exist?") == "topological"
