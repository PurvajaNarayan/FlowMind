from flowmind.schema import NodeShape


def test_node_and_edge_counts(graph):
    assert len(graph.nodes) == 7
    assert len(graph.edges) == 7
    assert graph.source == "mermaid"


def test_shapes_parsed(graph):
    shapes = {n.id: n.shape for n in graph.nodes}
    assert shapes["A"] == NodeShape.TERMINAL     # (["Start"])
    assert shapes["B"] == NodeShape.IO           # [/"Input"/]
    assert shapes["C"] == NodeShape.PROCESS      # ["i = 0"]
    assert shapes["D"] == NodeShape.DECISION     # {"arr[i] == i?"}


def test_labels_cleaned(graph):
    assert graph.label_of("A") == "Start"
    assert graph.label_of("D") == "arr[i] == i?"


def test_edge_labels(graph):
    labeled = {(e.source, e.target): e.label for e in graph.edges}
    assert labeled[("D", "E")] == "Yes"
    assert labeled[("D", "F")] == "No"


def test_find_by_label(graph):
    found = graph.find_by_label("start")
    assert [n.id for n in found] == ["A"]
