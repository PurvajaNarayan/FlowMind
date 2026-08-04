"""LLM fallback for topological questions the keyword parser can't dispatch.

`graph_tool.answer_topological` handles the FlowVQA-shaped questions
deterministically and in microseconds — that stays the primary path and keeps
the headline accuracy number reproducible (spec §2). This module only fires
when that parser returns `kind is None` (unrecognized phrasing, free-form user
questions, missing quotes, typos): it asks the LLM to pick ONE function and the
node labels, then runs the *same* deterministic compute in graph_tool. The LLM
chooses; it never counts. So the fallback widens coverage without letting model
variance touch an answer the parser could already give.

Kept out of graph_tool.py so that module stays LLM-free and importable without
the transformers stack, same as vlm_reader/llm defer their heavy imports.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from flowmind.graph_tool import (
    answer_topological,
    edge_count,
    is_direct_predecessor,
    is_direct_successor,
    max_indegree,
    max_outdegree,
    node_count,
    resolve_all,
    shortest_path_edges,
)
from flowmind.schema import FlowGraph

if TYPE_CHECKING:
    from flowmind.llm import LLMClient

# function name -> how many node labels it needs. This is the whitelist: the
# LLM's chosen function must be a key here or we reject the reply.
_FUNCTION_ARITY = {
    "node_count": 0,
    "edge_count": 0,
    "max_indegree": 0,
    "max_outdegree": 0,
    "shortest_path": 2,
    "direct_predecessor": 2,
    "direct_successor": 2,
}

_SYSTEM = (
    "You route a question about a flowchart to exactly ONE graph function. "
    "You do not answer the question or do any counting yourself — a "
    "deterministic tool computes the result. Reply with ONLY a JSON object."
)

_TOOL_DOC = """\
Available functions (pick exactly one):
  node_count()                      - total number of nodes
  edge_count()                      - total number of edges
  max_indegree()                    - highest number of incoming edges on any node
  max_outdegree()                   - highest number of outgoing edges on any node
  shortest_path(a, b)               - number of edges on the shortest path a -> b
  direct_predecessor(a, b)          - is a an immediate predecessor of b? (edge a -> b)
  direct_successor(a, b)            - is a an immediate successor of b?   (edge b -> a)

For the two-argument functions, `a` and `b` must be node labels copied verbatim
from the "Nodes" list below. Zero-argument functions take no labels."""


def _labels_block(graph: FlowGraph) -> str:
    seen: list[str] = []
    for n in graph.nodes:
        if n.label not in seen:
            seen.append(n.label)
    return "\n".join(f"  - {lab}" for lab in seen) or "  (none)"


def build_prompt(graph: FlowGraph, question: str) -> str:
    return (
        f"{_TOOL_DOC}\n\n"
        f"Nodes:\n{_labels_block(graph)}\n\n"
        f"Question: {question}\n\n"
        'Reply as JSON: {"function": "<name>", "labels": ["<a>", "<b>"]}\n'
        "Use an empty labels list for zero-argument functions."
    )


def _parse_reply(text: str) -> tuple[str, list[str]] | None:
    """Pull {function, labels} out of the model text, or None if unusable."""
    match = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate prose / code fences
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    fn = obj.get("function")
    labels = obj.get("labels", [])
    if fn not in _FUNCTION_ARITY or not isinstance(labels, list):
        return None
    labels = [str(x) for x in labels]
    if len(labels) != _FUNCTION_ARITY[fn]:
        return None
    return fn, labels


def _dispatch(graph: FlowGraph, fn: str, labels: list[str]):
    """Run the chosen function on resolved ids. Returns (pred, unresolved)."""
    if _FUNCTION_ARITY[fn] == 0:
        zero = {
            "node_count": node_count,
            "edge_count": edge_count,
            "max_indegree": max_indegree,
            "max_outdegree": max_outdegree,
        }
        return zero[fn](graph), False

    # Two-label functions: resolve each label to EVERY candidate id and
    # quantify over them exactly as answer_topological does, so duplicate
    # labels ("End" x2) keep the same semantics on this path.
    A, B = resolve_all(graph, labels[0]), resolve_all(graph, labels[1])
    if not (A and B):
        return None, True
    if fn == "shortest_path":
        lengths = [d for a in A for b in B
                   if (d := shortest_path_edges(graph, a, b)) is not None]
        return (min(lengths) if lengths else None), False
    if fn == "direct_predecessor":
        return ("Yes" if any(is_direct_predecessor(graph, a, b)
                             for a in A for b in B) else "No"), False
    if fn == "direct_successor":
        return ("Yes" if any(is_direct_successor(graph, a, b)
                             for a in A for b in B) else "No"), False
    return None, False  # unreachable given the whitelist


def answer_topological_llm(
    graph: FlowGraph, question: str, client: "LLMClient",
) -> tuple[str | None, object | None, bool]:
    """LLM-pick-then-deterministic-compute. Same return contract as
    graph_tool.answer_topological: (kind, predicted_answer, unresolved).

    kind is None if the model reply is unusable (bad JSON, unknown function,
    wrong number of labels) — caller treats that as unhandled, same as a
    deterministic miss.
    """
    reply = client.complete(build_prompt(graph, question), system=_SYSTEM)
    parsed = _parse_reply(reply)
    if parsed is None:
        return None, None, False
    fn, labels = parsed
    pred, unresolved = _dispatch(graph, fn, labels)
    return fn, pred, unresolved


def answer_topological_with_fallback(
    graph: FlowGraph, question: str, client: "LLMClient",
) -> tuple[str | None, object | None, bool]:
    """Deterministic parser first; LLM only when it returns kind is None."""
    kind, pred, unresolved = answer_topological(graph, question)
    if kind is not None:
        return kind, pred, unresolved
    return answer_topological_llm(graph, question, client)
