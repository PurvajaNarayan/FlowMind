"""FROZEN CONTRACT — the FlowGraph object every workstream shares.

The Reader (text or VLM) produces a FlowGraph. The router, graph tool, Examiner,
and Planner all consume it. Do NOT change these classes in a feature branch
without team sign-off: a change here ripples into all three workstreams.

Spec: §7.1 (Reader output), §7.2 (graph tool input).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeShape(str, Enum):
    """The four Mermaid node shapes FlowVQA uses (spec §7.1)."""

    TERMINAL = "terminal"      # ([ "Start"/"End" ])   rounded stadium
    IO = "io"                  # [/ "Input"/"Output" /] parallelogram
    PROCESS = "process"        # [ "..." ]              rectangle
    DECISION = "decision"      # { "..." }              diamond
    UNKNOWN = "unknown"        # shape not recognised by the parser


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    shape: NodeShape = NodeShape.UNKNOWN


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    label: str | None = None   # e.g. "Yes" / "No" on a decision branch


@dataclass
class FlowGraph:
    """A parsed flowchart. Backend-agnostic: text Reader and VLM Reader both
    return this exact type, so downstream code never knows which produced it."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    # Where this graph came from — useful for the ablation / error analysis.
    source: str = "unknown"   # "mermaid" | "vlm" | ...

    def node(self, node_id: str) -> Node | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def label_of(self, node_id: str) -> str | None:
        n = self.node(node_id)
        return n.label if n else None

    def find_by_label(self, text: str) -> list[Node]:
        """Loose label lookup — questions reference nodes by text, not id."""
        needle = text.strip().lower()
        return [n for n in self.nodes if needle in n.label.strip().lower()]

    def to_networkx(self):
        """Directed graph for the topological tools (spec §7.2)."""
        import networkx as nx

        g = nx.DiGraph()
        for n in self.nodes:
            g.add_node(n.id, label=n.label, shape=n.shape.value)
        for e in self.edges:
            g.add_edge(e.source, e.target, label=e.label)
        return g
