"""Text Reader: Mermaid script -> FlowGraph (spec §7.1).

Handles the four FlowVQA node shapes and both edge styles. This is a working
baseline parser for the scaffold; Owner A should harden it against the real
dataset (chained edges `A --> B --> C`, multi-line labels, subgraphs, etc.).
"""

from __future__ import annotations

import re

from flowmind.schema import Edge, FlowGraph, Node, NodeShape

# id + optional shape wrapper, e.g.  A(["Start"])  B[/"in"/]  C["do"]  D{"x?"}
# Order in the alternation matters: match the more specific wrappers first.
_SHAPE_PATTERNS = [
    (NodeShape.TERMINAL, re.compile(r'^\(\[(.*)\]\)$', re.S)),   # ([...])
    (NodeShape.IO, re.compile(r'^\[/(.*)/\]$', re.S)),           # [/.../]
    (NodeShape.DECISION, re.compile(r'^\{(.*)\}$', re.S)),       # {...}
    (NodeShape.PROCESS, re.compile(r'^\[(.*)\]$', re.S)),        # [...]
    (NodeShape.PROCESS, re.compile(r'^\((.*)\)$', re.S)),        # (...) rounded
]

_NODE_TOKEN = re.compile(r'^\s*([A-Za-z0-9_]+)\s*(.*)$', re.S)
_EDGE = re.compile(r'^(.*?)\s*--+>\s*(?:\|(.*?)\|\s*)?(.*)$', re.S)


def _clean_label(raw: str) -> str:
    return raw.strip().strip('"').strip("'").strip()


def _parse_token(token: str, nodes: dict[str, Node]) -> str | None:
    """Parse one side of an edge. Registers the node if a shape def is present.
    Returns the node id, or None if the token isn't a node reference."""
    token = token.strip()
    if not token:
        return None
    m = _NODE_TOKEN.match(token)
    if not m:
        return None
    node_id, rest = m.group(1), m.group(2).strip()

    if rest:  # has a shape wrapper -> (re)register with label + shape
        shape, label = NodeShape.UNKNOWN, node_id
        for candidate_shape, pat in _SHAPE_PATTERNS:
            sm = pat.match(rest)
            if sm:
                shape, label = candidate_shape, _clean_label(sm.group(1))
                break
        nodes[node_id] = Node(id=node_id, label=label, shape=shape)
    elif node_id not in nodes:  # bare reference before its definition
        nodes[node_id] = Node(id=node_id, label=node_id, shape=NodeShape.UNKNOWN)
    return node_id


def mermaid_to_graph(mermaid: str) -> FlowGraph:
    nodes: dict[str, Node] = {}
    edges: list[Edge] = []

    for line in mermaid.splitlines():
        line = line.strip()
        if not line or line.startswith(("flowchart", "graph", "%%", "subgraph", "end")):
            continue

        em = _EDGE.match(line)
        if em:
            src = _parse_token(em.group(1), nodes)
            label = _clean_label(em.group(2)) if em.group(2) else None
            dst = _parse_token(em.group(3), nodes)
            if src and dst:
                edges.append(Edge(source=src, target=dst, label=label))
        else:
            _parse_token(line, nodes)  # a standalone node definition

    return FlowGraph(nodes=list(nodes.values()), edges=edges, source="mermaid")
