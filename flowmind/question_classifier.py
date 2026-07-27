"""Trained question-type classifier (spec §7.0 fallback). [Owner: A]

A simple, no-LLM model: TF-IDF (word + char n-grams) -> Logistic Regression.
Maps an English question to one of the four FlowVQA types:
    fact_retrieval | applied_scenario | flow_referential | topological

Used by the router when a question has no dataset `type` label (free-form input).
When the label IS available (evaluating on FlowVQA), route off it directly —
don't spend a prediction on a label you already have.

Train / evaluate with tools/train_router.py; the fitted model is saved to
models/question_classifier.joblib.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

QUESTION_TYPES = ("fact_retrieval", "applied_scenario", "flow_referential", "topological")

DEFAULT_MODEL_PATH = Path("models/question_classifier.joblib")


def build_pipeline() -> Pipeline:
    """Word-level TF-IDF captures phrasing; char n-grams catch the templated
    topological patterns ('how many', 'in-degree', 'predecessor')."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
        )),
    ])
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


class QuestionTypeClassifier:
    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline or build_pipeline()

    def train(self, questions: list[str], types: list[str]) -> "QuestionTypeClassifier":
        self.pipeline.fit(questions, types)
        return self

    def predict(self, question: str) -> str:
        return self.pipeline.predict([question])[0]

    def predict_batch(self, questions: list[str]) -> list[str]:
        return list(self.pipeline.predict(questions))

    def predict_proba(self, question: str) -> dict[str, float]:
        probs = self.pipeline.predict_proba([question])[0]
        return dict(zip(self.pipeline.classes_, probs))

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "QuestionTypeClassifier":
        return cls(pipeline=joblib.load(Path(path)))
