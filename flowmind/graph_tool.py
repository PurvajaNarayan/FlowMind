"""Deterministic topological answers (spec §7.2). No LLM.

Every function takes a FlowGraph and returns a hard number/bool. This is the
lane that gives the project an arguable accuracy figure (spec §2).
"""

from __future__ import annotations

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
