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


def test_max_outdegree(graph):
    # D branches to E (Yes) and F (No)  => out-degree 2, the max
    assert gt.max_outdegree(graph) == 2


def test_answer_topological_dispatches_each_subtype(graph):
    assert gt.answer_topological(graph, "How many nodes exist in the given flowchart?") \
        == ("node_count", 7, False)
    assert gt.answer_topological(graph, "How many edges exist in the given flowchart?") \
        == ("edge_count", 7, False)
    assert gt.answer_topological(
        graph, 'What is the shortest path in terms of edges between node "Start" and node "End"?'
    ) == ("shortest_path", 5, False)
    assert gt.answer_topological(
        graph, 'Is node "i = 0" a direct predecessor of node "arr[i] == i?"?'
    ) == ("direct_predecessor", "Yes", False)
    assert gt.answer_topological(
        graph, 'Is node "i = 0" a direct successor of node "arr[i] == i?"?'
    ) == ("direct_successor", "No", False)
    assert gt.answer_topological(graph, "What is the maximum indegree for the flowchart?") \
        == ("max_indegree", 2, False)
    assert gt.answer_topological(graph, "What is the maximum outdegree of the flowchart?") \
        == ("max_outdegree", 2, False)


def test_answer_topological_flags_unresolved_labels(graph):
    kind, pred, unresolved = gt.answer_topological(
        graph, 'What is the shortest path in terms of edges between node "Nonexistent" and node "Start"?'
    )
    assert kind == "shortest_path"
    assert pred is None
    assert unresolved is True


def test_answer_topological_unrecognized_question_returns_none_kind(graph):
    kind, pred, unresolved = gt.answer_topological(graph, "What color is the flowchart?")
    assert kind is None
    assert pred is None
    assert unresolved is False
