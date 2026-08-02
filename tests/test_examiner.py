"""Examiner agent, exercised with no GPU and no weights (ScriptedClient).

Mirrors tests/test_llm_baseline.py's approach: the revise loop and prompt
construction are asserted here without spending GPU time.
"""

from flowmind.data import QAItem
from flowmind.examiner import MAX_REVISIONS, answer, build_content_prompt
from flowmind.llm import ScriptedClient
from flowmind.schema import Edge, FlowGraph, Node, NodeShape


def make_item(answers=("7",)):
    return QAItem(
        sample_key="code00453", question_id="1",
        question="What does node B output?",
        answers=list(answers), qa_type="fact_retrieval",
        mermaid='flowchart TD\n    A(["Start"]) --> B["do it"]',
        subset="code",
    )


def make_graph():
    return FlowGraph(
        nodes=[Node(id="A", label="Start", shape=NodeShape.TERMINAL),
               Node(id="B", label="do it", shape=NodeShape.PROCESS)],
        edges=[Edge(source="A", target="B", label=None)],
        source="mermaid",
    )


def test_prompt_contains_graph_and_question():
    p = build_content_prompt(make_graph(), make_item())
    assert "Start" in p
    assert "do it" in p
    assert "A -> B" in p
    assert "What does node B output?" in p


def test_system_prompt_grounds_the_answer_in_the_graph():
    client = ScriptedClient("7")
    answer(make_graph(), make_item(answers=["7"]), client=client)
    system, _ = client.prompts[0]
    assert system and "only from the graph" in system.lower()


def test_accept_on_first_try():
    client = ScriptedClient("7")
    res = answer(make_graph(), make_item(answers=["7"]), client=client)
    assert res.verdict == "accept"
    assert res.revisions == 0
    assert res.answer == "7"
    assert len(client.prompts) == 1


def test_accept_after_one_revision():
    client = ScriptedClient(["wrong", "7"])
    res = answer(make_graph(), make_item(answers=["7"]), client=client)
    assert res.verdict == "accept"
    assert res.revisions == 1
    assert len(client.prompts) == 2
    _, second_prompt = client.prompts[1]
    assert "wrong" in second_prompt
    assert "rejected" in second_prompt


def test_revise_exhausted_after_max_revisions():
    client = ScriptedClient(["wrong"] * (MAX_REVISIONS + 5))
    res = answer(make_graph(), make_item(answers=["7"]), client=client)
    assert res.verdict == "revise"
    assert res.revisions == MAX_REVISIONS
    assert len(client.prompts) == MAX_REVISIONS + 1
    assert res.revise_reason is not None
