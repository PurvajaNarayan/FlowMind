"""Planner agent (spec §7.4). [Owner: C]

`code` subset:  graph -> (a) executable Python function, (b) markdown plan doc.
`wiki`/`instruct` subset:  markdown plan doc only; web search invoked only when a
step references something not resolvable from the flowchart text alone.

Correctness for the code subset is behavioral equivalence against the original
`code` field — no LLM judge (spec §8). See eval.metrics.behavioral_equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass

from flowmind.data import QAItem
from flowmind.schema import FlowGraph


@dataclass
class PlannerResult:
    plan_markdown: str
    code: str | None = None   # generated Python (code subset only)


def plan(graph: FlowGraph, item: QAItem, use_web_search: bool = False) -> PlannerResult:
    """Produce a plan doc and, for the code subset, runnable Python.

    TODO(C):
      1. Serialize the graph into an ordered step description.
      2. Prompt the LLM for a markdown plan.
      3. If item.code is not None (code subset), also prompt for a Python
         function and ensure it runs via exec/import (spec §7.4 requirement).
      4. For wiki/instruct, enable web search only on unresolved references.
    """
    raise NotImplementedError("Planner not implemented yet — [Owner: C]")
