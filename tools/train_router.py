"""Train the question-type classifier on train_full.json, evaluate on
test_full.json, and save the fitted model. [Owner: A]

    python tools/train_router.py
    python tools/train_router.py --train data/train_full.json --test data/test_full.json

No LLM involved — this is a TF-IDF + Logistic Regression model (see
flowmind/question_classifier.py). Prints overall accuracy, a per-class report,
and a confusion matrix, then writes models/question_classifier.joblib.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from sklearn.metrics import classification_report, confusion_matrix

from flowmind.question_classifier import (
    DEFAULT_MODEL_PATH,
    QUESTION_TYPES,
    QuestionTypeClassifier,
)


def load_qa(path: str) -> tuple[list[str], list[str]]:
    """Return (questions, types) for every QA pair in a FlowVQA split."""
    ds = json.load(open(path))
    questions, types = [], []
    for rec in ds.values():
        for qa in rec.get("qa", {}).values():
            questions.append(qa["Q"])
            types.append(qa["type"])
    return questions, types


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_full.json")
    ap.add_argument("--test", default="data/test_full.json")
    ap.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = ap.parse_args()

    Xtr, ytr = load_qa(args.train)
    Xte, yte = load_qa(args.test)
    print(f"train: {len(Xtr)} questions | test: {len(Xte)} questions")

    model = QuestionTypeClassifier().train(Xtr, ytr)
    pred = model.predict_batch(Xte)

    acc = sum(p == y for p, y in zip(pred, yte)) / len(yte)
    print(f"\ntest accuracy: {acc:.4f}\n")
    print(classification_report(yte, pred, digits=4))

    labels = list(QUESTION_TYPES)
    cm = confusion_matrix(yte, pred, labels=labels)
    print("confusion matrix (rows = true, cols = predicted):")
    print("            " + "  ".join(f"{l[:8]:>8}" for l in labels))
    for lab, row in zip(labels, cm):
        print(f"{lab[:10]:>10}  " + "  ".join(f"{v:>8}" for v in row))

    model.save(args.out)
    print(f"\nsaved model -> {args.out}")


if __name__ == "__main__":
    main()
