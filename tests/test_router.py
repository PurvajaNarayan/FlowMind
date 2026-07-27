from flowmind.router import (
    classify_intent_heuristic,
    route,
    route_from_dataset_type,
)


def test_route_from_dataset_type():
    assert route_from_dataset_type("topological") == "topological"
    assert route_from_dataset_type("fact_retrieval") == "content"
    assert route_from_dataset_type("applied_scenario") == "content"
    assert route_from_dataset_type("flow_referential") == "content"


def test_heuristic_fallback():
    assert classify_intent_heuristic("How many nodes exist?") == "topological"
    assert classify_intent_heuristic("Write runnable python code for this") == "code_request"
    assert classify_intent_heuristic("What does the decision check?") == "content"


def test_route_prefers_label():
    # even a code-looking question routes topological when the label says so
    assert route("write the code", qa_type="topological") == "topological"
