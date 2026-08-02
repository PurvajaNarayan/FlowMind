"""full_pipeline / run_ablation, exercised with no GPU and no weights.

Uses the same sample.json fixture graph as test_graph_tool.py (via the shared
`sample` conftest fixture) so expected values line up with those tests.
"""

import pytest

from flowmind.data import QAItem
from flowmind.eval.ablation import full_pipeline, run_ablation
from flowmind.llm import ScriptedClient


def make_item(mermaid, qa_type, question, answers):
    return QAItem(
        sample_key="code00453", question_id="1", question=question,
        answers=list(answers), qa_type=qa_type, mermaid=mermaid, subset="code",
    )


def test_full_pipeline_topological_routes_to_graph_tool_no_llm(sample):
    item = make_item(sample["mermaid"], "topological",
                     "How many nodes exist in the given flowchart?", ["7"])
    res = full_pipeline(item)  # no client passed -- graph_tool branch must not need one
    assert res.branch == "graph_tool"
    assert res.intent == "topological"
    assert res.answer == "7"
    assert res.correct is True
    assert res.revisions == 0


def test_full_pipeline_content_routes_to_examiner(sample):
    item = make_item(sample["mermaid"], "fact_retrieval",
                     "What does the flowchart output when a fixed point is found?",
                     ["It outputs the index i."])
    client = ScriptedClient("It outputs the index i.")
    res = full_pipeline(item, client=client)
    assert res.branch == "examiner"
    assert res.intent == "content"
    assert res.correct is True
    assert res.revisions == 0
    assert res.examiner_result is not None


def test_full_pipeline_examiner_revision_reflected_in_result(sample):
    item = make_item(sample["mermaid"], "fact_retrieval",
                     "What does the flowchart output when a fixed point is found?",
                     ["It outputs the index i."])
    client = ScriptedClient(["wrong", "It outputs the index i."])
    res = full_pipeline(item, client=client)
    assert res.correct is True
    assert res.revisions == 1


def test_full_pipeline_raises_for_code_request(monkeypatch, sample):
    import flowmind.eval.ablation as ablation_mod
    monkeypatch.setattr(ablation_mod, "route", lambda question, qa_type=None: "code_request")
    item = make_item(sample["mermaid"], "fact_retrieval", "irrelevant", ["irrelevant"])
    with pytest.raises(NotImplementedError):
        full_pipeline(item)


def test_run_ablation_pairs_baseline_and_pipeline_per_item(sample):
    topo_item = make_item(sample["mermaid"], "topological",
                          "How many nodes exist in the given flowchart?", ["7"])
    content_item = make_item(
        sample["mermaid"], "fact_retrieval",
        "What does the flowchart output when a fixed point is found?",
        ["It outputs the index i."],
    )
    # Consumption order: baseline(topo) -> [no LLM call for pipeline(topo),
    # graph_tool is deterministic] -> baseline(content) -> pipeline(content)'s
    # single examiner call (matches first try).
    client = ScriptedClient(["7", "It outputs the index i.", "It outputs the index i."])

    result = run_ablation([topo_item, content_item], client=client)
    summary = result["summary"]

    assert summary["n"] == 2
    assert summary["baseline_accuracy"] == 1.0
    assert summary["pipeline_accuracy"] == 1.0
    assert summary["by_type"]["topological"]["pipeline"] == 1.0
    assert summary["by_type"]["fact_retrieval"]["pipeline"] == 1.0
    assert summary["examiner_revisions_recovered_answer"] == 0

    rows = result["rows"]
    assert rows[0]["pipeline_branch"] == "graph_tool"
    assert rows[1]["pipeline_branch"] == "examiner"
