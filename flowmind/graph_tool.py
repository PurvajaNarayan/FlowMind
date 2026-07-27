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
    return len(graph.edges)


def shortest_path_edges(graph: FlowGraph, a: str, b: str) -> int | None:
    """Number of edges on the shortest path from node id `a` to `b`.
    Returns None if unreachable."""
    g = graph.to_networkx()
    try:
        return nx.shortest_path_length(g, a, b)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def is_direct_successor(graph: FlowGraph, a: str, b: str) -> bool:
    """True if there is an edge a -> b."""
    return any(e.source == a and e.target == b for e in graph.edges)


def is_direct_predecessor(graph: FlowGraph, a: str, b: str) -> bool:
    """True if there is an edge b -> a (i.e. a is a direct predecessor of b)."""
    return is_direct_successor(graph, b, a)


def max_indegree(graph: FlowGraph) -> int:
    g = graph.to_networkx()
    return max((d for _, d in g.in_degree()), default=0)
