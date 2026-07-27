from flowmind import graph_tool as gt


def test_counts(graph):
    assert gt.node_count(graph) == 7
    assert gt.edge_count(graph) == 7


def test_adjacency(graph):
    # edge A -> B exists in the fixture
    assert gt.is_direct_predecessor(graph, "A", "B")   # A precedes B
    assert gt.is_direct_successor(graph, "B", "A")      # B follows A
    assert not gt.is_direct_predecessor(graph, "A", "C")
    assert not gt.is_direct_successor(graph, "A", "B")  # A does not follow B


def test_shortest_path(graph):
    # A -> B -> C -> D  => 3 edges
    assert gt.shortest_path_edges(graph, "A", "D") == 3
    assert gt.shortest_path_edges(graph, "G", "A") is None  # unreachable


def test_max_indegree(graph):
    # C is entered by B and by F  => in-degree 2, the max
    assert gt.max_indegree(graph) == 2
