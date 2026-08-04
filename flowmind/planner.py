"""Planner agent (spec §7.4). [Owner: C]

`code` subset:  graph -> (a) executable Python function, (b) markdown plan doc.
`wiki`/`instruct` subset: NOT implemented here (out of scope for this build --
no web-search dependency exists anywhere in the project yet). `plan()` raises
for any item without a `code` field rather than silently doing nothing.

Correctness for the code subset is behavioral equivalence against the original
`code` field — no LLM judge (spec §8). See eval.metrics.behavioral_equivalence.

The original `code` field is never shown to the LLM -- only its function name
and parameter names are (extracted via `_function_name_and_params`), so the
generation prompt can require a matching call signature (a hard requirement
for behavioral_equivalence to call both functions with the same inputs)
without making "regenerate the algorithm from the flowchart" circular.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass

from flowmind.data import QAItem
from flowmind.llm import LLMClient, get_client
from flowmind.schema import FlowGraph, NodeShape


@dataclass
class PlannerResult:
    plan_markdown: str
    code: str | None = None   # generated Python (code subset only)


PLAN_SYSTEM = (
    "You write a concise markdown plan describing the algorithm a flowchart "
    "implements, following the given step order. Use short numbered or "
    "bulleted steps. Do not include code."
)

CODE_SYSTEM = (
    "You write a single, self-contained Python function that implements the "
    "algorithm described by a flowchart. Output ONLY the function inside a "
    "fenced python code block, with no explanation before or after. The "
    "function must be named exactly as specified and accept exactly the "
    "parameters listed, in that order. Do not import any module, do not "
    "define helper classes, and do not include example calls or tests."
)

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def _ordered_steps(graph: FlowGraph) -> str:
    """Walk the graph from its start node, numbering steps in DFS order.

    Branch edges (Yes/No) are annotated inline. A revisited node (flowcharts
    have back-edges, e.g. a loop) is rendered as a reference to the step that
    already described it, rather than being re-expanded -- otherwise a loop
    would recurse forever.
    """
    if not graph.nodes:
        return "(empty flowchart)"

    outgoing: dict[str, list] = defaultdict(list)
    indegree: dict[str, int] = defaultdict(int)
    for e in graph.edges:
        outgoing[e.source].append(e)
        indegree[e.target] += 1

    starts = [n for n in graph.nodes if indegree.get(n.id, 0) == 0] or graph.nodes
    terminal_starts = [n for n in starts if n.shape == NodeShape.TERMINAL]
    start = (terminal_starts or starts)[0]

    lines: list[str] = []
    visited: dict[str, int] = {}

    def walk(node_id: str, incoming_label: str | None) -> None:
        if node_id in visited:
            lines.append(f"{len(lines) + 1}. (loop back to step {visited[node_id]})")
            return
        step_no = len(lines) + 1
        visited[node_id] = step_no
        node = graph.node(node_id)
        label = node.label if node else node_id
        prefix = f"[{incoming_label}] " if incoming_label else ""
        lines.append(f"{step_no}. {prefix}{label}")
        for e in outgoing.get(node_id, []):
            walk(e.target, e.label)

    walk(start.id, None)
    return "\n".join(lines)


def build_plan_prompt(steps_text: str) -> str:
    return f"Flowchart steps, in order:\n{steps_text}\n\nWrite the plan."


def _function_name_and_params(code: str) -> tuple[str, list[str]]:
    """Pull the original function's name and parameter names via ast.

    Only the signature is extracted, never the body -- see the module
    docstring for why showing the implementation would be circular.
    """
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name, [a.arg for a in node.args.args]
    raise ValueError("original code has no top-level function definition")


def build_code_prompt(steps_text: str, func_name: str, params: list[str]) -> str:
    return (
        f"Flowchart steps, in order:\n{steps_text}\n\n"
        f"Write a Python function named `{func_name}` with parameters "
        f"({', '.join(params)}) that implements exactly these steps.\n"
        "Output only the function in a fenced ```python code block."
    )


def _extract_code(reply: str) -> str:
    """Pull the function out of a fenced code block, or fall back to
    everything from the first top-level `def` onward if the model forgot the
    fence."""
    m = _CODE_FENCE_RE.search(reply or "")
    if m:
        return m.group(1).strip()
    m2 = re.search(r"^def\s+\w+\s*\(.*", reply or "", re.M)
    return reply[m2.start():].strip() if m2 else (reply or "").strip()


def plan(graph: FlowGraph, item: QAItem, client: LLMClient | None = None,
         max_new_tokens: int = 512) -> PlannerResult:
    """Produce a plan doc and, for the code subset, runnable Python.

    Only the `code` subset is implemented (see module docstring). Raises
    NotImplementedError for wiki/instruct items (item.code is None).
    """
    if item.code is None:
        raise NotImplementedError(
            "wiki/instruct plan-doc + web search is out of scope for this "
            "build -- see this module's docstring."
        )
    client = client or get_client()
    steps_text = _ordered_steps(graph)

    plan_reply = client.complete(build_plan_prompt(steps_text), system=PLAN_SYSTEM,
                                 max_new_tokens=max_new_tokens)

    func_name, params = _function_name_and_params(item.code)
    code_reply = client.complete(build_code_prompt(steps_text, func_name, params),
                                 system=CODE_SYSTEM, max_new_tokens=max_new_tokens)
    code = _extract_code(code_reply)

    return PlannerResult(plan_markdown=plan_reply.strip(), code=code)
