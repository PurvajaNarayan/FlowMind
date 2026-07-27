"""Scoring (spec §8). [Owner: C]

Four lanes:
  - graph_extraction: node/edge counts vs ground-truth mermaid (matters for VLM).
  - topological: exact match via graph_tool (should approach 100%).
  - content: best-of-3 vs A1/A2/A3 (embedding sim or LLM judge — spec §9).
  - behavioral_equivalence: run generated vs original Python on random inputs.
"""

from __future__ import annotations

from flowmind.reader.mermaid_reader import mermaid_to_graph
from flowmind.schema import FlowGraph


def graph_extraction_accuracy(pred: FlowGraph, gold_mermaid: str) -> dict:
    """Compare a (possibly VLM-produced) graph against the graph parsed from the
    ground-truth mermaid. Returns node/edge count match + exactness."""
    gold = mermaid_to_graph(gold_mermaid)
    return {
        "node_count_match": len(pred.nodes) == len(gold.nodes),
        "edge_count_match": len(pred.edges) == len(gold.edges),
        "pred_nodes": len(pred.nodes),
        "gold_nodes": len(gold.nodes),
        "pred_edges": len(pred.edges),
        "gold_edges": len(gold.edges),
    }


def topological_exact_match(prediction, gold: str) -> bool:
    """Exact match for graph-tool answers. gold comes from A1."""
    return str(prediction).strip() == str(gold).strip()


def content_match(prediction: str, references: list[str]) -> bool:
    """Best-of-3 match against A1/A2/A3.

    TODO(C): pick embedding similarity vs LLM judge and justify in the report
    (spec §9). Placeholder below is a naive containment check for wiring only.
    """
    p = prediction.strip().lower()
    return any(p and (p in r.lower() or r.lower() in p) for r in references)


def behavioral_equivalence(generated_code: str, original_code: str,
                           func_name: str, inputs: list[tuple]) -> float:
    """Run both functions on N random inputs, return fraction of matching outputs.

    TODO(C): sandbox the exec, generate type-appropriate random inputs, handle
    exceptions symmetrically (spec §8 code-behavioral row).
    """
    raise NotImplementedError("behavioral_equivalence not implemented — [Owner: C]")
