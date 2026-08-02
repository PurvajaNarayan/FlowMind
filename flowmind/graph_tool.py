"""Deterministic topological answers (spec §7.2). No LLM.

Every function takes a FlowGraph and returns a hard number/bool. This is the
lane that gives the project an arguable accuracy figure (spec §2).

The bottom section (`answer_topological` and its label-resolution helpers) is
the question-dispatch layer: given a topological question's raw text, figure
out which of the functions above to call and with which node ids. It lives
here rather than in tools/parser_coverage.py so both that coverage harness and
eval/ablation.py's full_pipeline can share one implementation.
"""

from __future__ import annotations

import re

import networkx as nx

from flowmind.schema import FlowGraph


def node_count(graph: FlowGraph) -> int:
    return len(graph.nodes)


def edge_count(graph: FlowGraph) -> int:
    """Number of distinct source->target edges.

    Parallel edges (e.g. a decision's Yes and No branch pointing at the same
    node) are counted once, matching FlowVQA's gold convention and staying
    consistent with to_networkx() (a DiGraph, which also collapses them).
    The full parallel edges remain available in graph.edges for label lookups.
    """
    return len({(e.source, e.target) for e in graph.edges})


def shortest_path_edges(graph: FlowGraph, a: str, b: str) -> int | None:
    """Number of edges on the shortest path from node id `a` to `b`.
    Returns None if unreachable."""
    g = graph.to_networkx()
    try:
        return nx.shortest_path_length(g, a, b)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _has_edge(graph: FlowGraph, src: str, dst: str) -> bool:
    return any(e.source == src and e.target == dst for e in graph.edges)


def is_direct_predecessor(graph: FlowGraph, x: str, y: str) -> bool:
    """True if x is a direct predecessor of y, i.e. edge x -> y exists.
    Matches the FlowVQA question 'Is node X direct predecessor of node Y?'."""
    return _has_edge(graph, x, y)


def is_direct_successor(graph: FlowGraph, x: str, y: str) -> bool:
    """True if x is a direct successor of y, i.e. edge y -> x exists.
    Matches the FlowVQA question 'Is node X direct successor of node Y?'."""
    return _has_edge(graph, y, x)


def max_indegree(graph: FlowGraph) -> int:
    """Highest number of incoming edges on any node.
    FlowVQA phrases this as 'What is the maximum indegree for the flowchart?'."""
    g = graph.to_networkx()
    return max((d for _, d in g.in_degree()), default=0)


def max_outdegree(graph: FlowGraph) -> int:
    """Highest number of outgoing edges on any node — the decision-branch counterpart
    of max_indegree. FlowVQA asks this as often as it asks about indegree, so the
    deterministic lane needs both.

    Like edge_count, this runs over the DiGraph, so a decision's parallel Yes/No
    edges into the same target count once. That matches FlowVQA's gold answers.
    """
    g = graph.to_networkx()
    return max((d for _, d in g.out_degree()), default=0)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _labels_in(q: str) -> list[str]:
    # FlowVQA quotes node labels with doubled double-quotes: ""like this.""
    doubled = re.findall(r'""(.*?)""', q)
    return [m.strip() for m in doubled] if doubled else re.findall(r'"([^"]+)"', q)


def resolve_all(graph: FlowGraph, label: str) -> list[str]:
    """Map a label from a question to EVERY node id it could mean.

    Returning a list rather than one id matters: charts routinely contain two
    nodes labelled "End" (also seen: duplicate process steps), and picking the
    first arbitrarily was wrong about half the time -- ambiguous adjacency
    questions scored 50% and 33% against 99.5%+ for unambiguous ones. Callers
    resolve the ambiguity with the semantics of their own question instead:
    "is X a predecessor of End" is true if X precedes ANY node labelled End.
    """
    lab = label.strip().rstrip(".").strip().lower()
    exact = [n.id for n in graph.nodes if n.label.strip().rstrip(".").strip().lower() == lab]
    if exact:
        return exact
    return [n.id for n in graph.find_by_label(label.rstrip("."))]


def resolve(graph: FlowGraph, label: str) -> str | None:
    """First candidate only. Kept for callers that genuinely want one id."""
    hits = resolve_all(graph, label)
    return hits[0] if hits else None


def answer_topological(graph: FlowGraph, question: str) -> tuple[str | None, object | None, bool]:
    """Parse a topological question's text and answer it deterministically.

    Returns (kind, predicted_answer, unresolved):
      - kind is None if the question doesn't match a recognized subtype.
      - unresolved is True only when a quoted node label couldn't be resolved
        to any node id (see resolve_all) -- a Reader/label-matching concern,
        not a wrong prediction. Callers should skip-and-not-score these, same
        as tools/parser_coverage.py always has.
      - predicted_answer can still be None with kind known and unresolved
        False, e.g. two validly-resolved nodes with no path between them;
        that's unscoreable, not wrong.
    """
    Q = _norm(question)
    labs = _labels_in(question)
    kind = pred = None
    unresolved = False

    if "how many nodes" in Q:
        kind, pred = "node_count", node_count(graph)
    # Degree questions before edge_count: they mention "incoming/outgoing
    # edges" in their preamble, and FlowVQA writes them as one word
    # ("indegree"/"outdegree"), not "in-degree".
    elif "outdegree" in Q or "out-degree" in Q or "out degree" in Q:
        kind, pred = "max_outdegree", max_outdegree(graph)
    elif "indegree" in Q or "in-degree" in Q or "in degree" in Q:
        kind, pred = "max_indegree", max_indegree(graph)
    elif "how many edges" in Q and "shortest path" not in Q:
        kind, pred = "edge_count", edge_count(graph)
    elif "shortest path" in Q and len(labs) >= 2:
        kind = "shortest_path"
        A, B = resolve_all(graph, labs[0]), resolve_all(graph, labs[1])
        if not (A and B):
            unresolved = True
        else:
            # Ambiguous label -> the shortest route between any pair of
            # candidates, which is what "the shortest path" asks for.
            lengths = [d for a in A for b in B
                       if (d := shortest_path_edges(graph, a, b)) is not None]
            pred = min(lengths) if lengths else None
    elif "predecessor" in Q and len(labs) >= 2:
        kind = "direct_predecessor"
        A, B = resolve_all(graph, labs[0]), resolve_all(graph, labs[1])
        if not (A and B):
            unresolved = True
        else:
            # "Is X a direct predecessor of End" holds if X precedes ANY node
            # labelled End, so quantify existentially over the candidates.
            pred = "Yes" if any(is_direct_predecessor(graph, a, b)
                                for a in A for b in B) else "No"
    elif "successor" in Q and len(labs) >= 2:
        kind = "direct_successor"
        A, B = resolve_all(graph, labs[0]), resolve_all(graph, labs[1])
        if not (A and B):
            unresolved = True
        else:
            pred = "Yes" if any(is_direct_successor(graph, a, b)
                                for a in A for b in B) else "No"

    return kind, pred, unresolved
