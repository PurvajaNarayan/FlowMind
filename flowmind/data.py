"""Dataset loader for FlowVQA (spec §6). [Owner: A]

Yields one (sample, question) unit at a time so the pipeline can iterate.
Keeps the raw record around so agents can reach `code` / `summary` / `mermaid`.

Layout (see data/README.md):
    data/train_full.json, data/test_full.json
    data/images/main/<key>.png        # top-down layout
    data/images/bottom_top/<key>.png  # directional-bias robustness set

Records come in three subsets by key prefix — `code` (has `code`),
`wiki`, `instruct` (metadata instead of `code`). All share `mermaid`/`qa`/`summary`.
There is NO image field in the JSON; images are joined by key -> <key>.png.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Image roots, relative to the dataset file's parent (data/).
_LAYOUTS = {"main": "images/main", "bottom_top": "images/bottom_top"}


@dataclass
class QAItem:
    sample_key: str          # "code00453"
    question_id: str         # "4"
    question: str
    answers: list[str]       # [A1, A2, A3] paraphrases — score against all (spec §6)
    qa_type: str             # fact_retrieval | applied_scenario | flow_referential | topological
    mermaid: str
    subset: str = "unknown"  # "code" | "wiki" | "instruct" (from the key prefix)
    code: str | None = None       # present for `code`/FloCo entries only
    summary: str | None = None
    image_path: str | None = None  # data/images/main/<key>.png


def _subset_of(key: str) -> str:
    return key.rstrip("0123456789") or "unknown"


def load_dataset(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def image_path_for(key: str, data_dir: str | Path = "data", layout: str = "main") -> Path:
    """Join a record key to its rendered flowchart image."""
    return Path(data_dir) / _LAYOUTS[layout] / f"{key}.png"


def iter_qa(
    dataset: dict,
    data_dir: str | Path = "data",
    layout: str = "main",
) -> Iterator[QAItem]:
    """Flatten the nested dataset into per-question units.

    `layout` picks which rendered image to attach ("main" top-down, or
    "bottom_top" for the directional-bias robustness ablation).
    """
    for key, rec in dataset.items():
        img = image_path_for(key, data_dir, layout)
        for qid, qa in rec.get("qa", {}).items():
            answers = [qa[k] for k in ("A1", "A2", "A3") if qa.get(k)]
            yield QAItem(
                sample_key=key,
                question_id=qid,
                question=qa["Q"],
                answers=answers,
                qa_type=qa["type"],
                mermaid=rec["mermaid"],
                subset=_subset_of(key),
                code=rec.get("code"),
                summary=rec.get("summary"),
                image_path=str(img) if img.exists() else None,
            )
