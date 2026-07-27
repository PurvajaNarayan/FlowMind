"""Examiner agent (spec §7.3). [Owner: B]

Answers content questions (fact_retrieval / applied_scenario / flow_referential)
from the graph + node text, then self-checks against the reference answers.
Topological questions never reach here — the router sends them to graph_tool.

The revision loop is LOCAL to this branch: on `revise`, send a flag back to the
Reader with what looked wrong. Cap at 2 attempts per question.
"""

from __future__ import annotations

from dataclasses import dataclass

from flowmind.data import QAItem
from flowmind.schema import FlowGraph

MAX_REVISIONS = 2  # spec §7.3


@dataclass
class ExaminerResult:
    answer: str
    verdict: str            # "accept" | "revise"
    revise_reason: str | None = None  # sent back to Reader when verdict == "revise"
    revisions: int = 0


def answer(graph: FlowGraph, item: QAItem) -> ExaminerResult:
    """Generate a content answer and self-check it.

    TODO(B):
      1. Build a prompt from graph nodes/edges + question.
      2. Call the LLM to produce an answer.
      3. Score against item.answers (A1/A2/A3) — reuse eval.metrics.content_match.
      4. If it doesn't check out (e.g. referenced node not in graph), return
         verdict="revise" with a reason; the driver re-runs the Reader up to
         MAX_REVISIONS times.
    """
    raise NotImplementedError("Examiner not implemented yet — [Owner: B]")
