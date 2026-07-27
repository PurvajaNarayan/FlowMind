"""FROZEN CONTRACT — per-sample trace log (spec §12).

Every agent logs its input/output for each sample so Person C can do error
analysis for the report. Keep this format stable; C's analysis scripts depend
on it. One JSON object per line (JSONL) under runs/<run_name>.jsonl.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Trace:
    sample_key: str            # e.g. "code00453"
    question_id: str           # e.g. "4"
    intent: str                # "topological" | "content" | "code_request"
    branch: str                # which agent/tool handled it
    prediction: Any = None     # the produced answer / code / plan
    gold: Any = None           # reference answer(s) for scoring
    correct: bool | None = None
    revisions: int = 0         # Examiner loop count (spec §7.3, cap 2)
    steps: list[dict] = field(default_factory=list)  # per-agent io breadcrumbs

    def add_step(self, agent: str, inp: Any, out: Any) -> None:
        self.steps.append({"agent": agent, "input": inp, "output": out})


class TraceWriter:
    """Append-only JSONL writer. Usage:

        with TraceWriter("runs/m1.jsonl") as tw:
            tw.write(trace)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def __enter__(self) -> "TraceWriter":
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def write(self, trace: Trace) -> None:
        assert self._fh is not None, "use TraceWriter as a context manager"
        self._fh.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.close()


def read_traces(path: str | Path) -> list[dict]:
    """Load a run's traces for analysis."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
