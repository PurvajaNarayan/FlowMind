from flowmind import graph_tool as gt


def test_counts(graph):
    assert gt.node_count(graph) == 7
    assert gt.edge_count(graph) == 7


def test_adjacency(graph):
    assert gt.is_direct_successor(graph, "A", "B")
    assert not gt.is_direct_successor(graph, "A", "C")
    assert gt.is_direct_predecessor(graph, "B", "A")


def test_shortest_path(graph):
    # A -> B -> C -> D  => 3 edges
    assert gt.shortest_path_edges(graph, "A", "D") == 3
    assert gt.shortest_path_edges(graph, "G", "A") is None  # unreachable


def test_max_indegree(graph):
    # C is entered by B and by F  => in-degree 2, the max
    assert gt.max_indegree(graph) == 2
