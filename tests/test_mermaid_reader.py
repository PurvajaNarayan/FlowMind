from flowmind.reader.mermaid_reader import mermaid_to_graph
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


def test_label_keeps_trailing_apostrophe():
    """A trailing apostrophe is part of the label, not a quote wrapper.

    569 labels across 182 train charts end in one; stripping it made them
    unmatchable against the label as quoted in the question.
    """
    g = mermaid_to_graph(
        'flowchart TD\n'
        '    A["Iterate over the list \'nums\'"] --> B[/"Output \'LEAP YEAR\'"/]\n'
    )
    assert g.node("A").label == "Iterate over the list 'nums'"
    assert g.node("B").label == "Output 'LEAP YEAR'"


def test_label_quote_wrappers_still_stripped():
    g = mermaid_to_graph(
        'flowchart TD\n'
        '    A(["Start"]) --> B["\'Quoted\'"]\n'
        '    B --> C["\'sum\' = \'sum\' + \'i\'"]\n'
    )
    assert g.node("A").label == "Start"
    # A balanced pair with nothing quoted inside is a wrapper...
    assert g.node("B").label == "Quoted"
    # ...but this label genuinely starts and ends with an apostrophe.
    assert g.node("C").label == "'sum' = 'sum' + 'i'"
