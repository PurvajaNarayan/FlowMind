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

# Any FlowVQA edge operator (solid --> , dotted -.-> , thick ==> , longer runs),
# with an optional trailing |label|. Used with re.split so a single line with
# chained edges (A -->|Yes| B --> C) becomes tokens + per-edge labels.
_ARROW = re.compile(r'\s*[-.=]{2,}>\s*(?:\|([^|]*)\|\s*)?')

# Lines that define graph metadata, not nodes/edges.
_SKIP_PREFIXES = (
    "flowchart", "graph", "%%", "subgraph", "end", "direction",
    "classDef", "class ", "style ", "linkStyle", "click ",
)


def _clean_label(raw: str) -> str:
    """Remove the quote pair Mermaid wraps a label in, without eating quotes that
    belong to the label text.

    The previous implementation was `.strip('"').strip("'")`, which removes quote
    characters from each end independently. That truncated every label ending in
    an apostrophe -- 569 of them across 182 charts in train -- so
    `"Iterate over the list 'nums'"` became `Iterate over the list 'nums` and no
    longer matched the label as quoted in the question. That accounted for all 129
    questions the M1 harness had to skip as unresolvable.

    Stripping *balanced* apostrophes is not safe either: 9 labels genuinely start
    and end with one (e.g. `'sum' = 'sum' + 'i'`), so a balanced strip would
    corrupt them too. An apostrophe pair is therefore only treated as a wrapper
    when nothing between it is an apostrophe, which keeps `'Start'` -> `Start`
    for a VLM that quotes with single quotes.
    """
    s = raw.strip()
    while len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'" and "'" not in s[1:-1]:
        s = s[1:-1].strip()
    return s


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
        if not line or line.startswith(_SKIP_PREFIXES):
            continue

        # Split on every arrow. re.split with one capture group returns:
        #   [tok0, label0, tok1, label1, tok2, ...]
        # so labels[i] is the edge label between tokens[i] and tokens[i+1]
        # (None when that arrow had no |label|). One node line -> one token, no edges.
        parts = _ARROW.split(line)
        tokens, labels = parts[0::2], parts[1::2]
        ids = [_parse_token(t, nodes) for t in tokens]
        for i in range(len(ids) - 1):
            if ids[i] and ids[i + 1]:
                raw = labels[i] if i < len(labels) else None
                edges.append(
                    Edge(
                        source=ids[i],
                        target=ids[i + 1],
                        label=_clean_label(raw) if raw else None,
                    )
                )

    return FlowGraph(nodes=list(nodes.values()), edges=edges, source="mermaid")
