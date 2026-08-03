"""Examiner agent, exercised with no GPU and no weights (ScriptedClient).

Mirrors tests/test_llm_baseline.py's approach: the revise loop and prompt
construction are asserted here without spending GPU time.
"""

from flowmind.data import QAItem
from flowmind.examiner import (
    MAX_REVISIONS,
    answer,
    build_content_prompt,
    check_grounding,
)
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
    # "do it" is a node label, so the answer is grounded in the graph.
    client = ScriptedClient("It runs the do it step.")
    res = answer(make_graph(), make_item(), client=client)
    assert res.verdict == "accept"
    assert res.revisions == 0
    assert len(client.prompts) == 1


def test_accept_after_one_revision():
    # First reply cites a step that is not in the chart; second is grounded.
    client = ScriptedClient(['It performs "launch the rocket" first.',
                             "It runs the do it step."])
    res = answer(make_graph(), make_item(), client=client)
    assert res.verdict == "accept"
    assert res.revisions == 1
    assert len(client.prompts) == 2
    _, second_prompt = client.prompts[1]
    assert "launch the rocket" in second_prompt
    assert "rejected" in second_prompt


def test_revise_exhausted_after_max_revisions():
    client = ScriptedClient([""] * (MAX_REVISIONS + 5))
    res = answer(make_graph(), make_item(), client=client)
    assert res.verdict == "revise"
    assert res.revisions == MAX_REVISIONS
    assert len(client.prompts) == MAX_REVISIONS + 1
    assert res.revise_reason is not None


# --- the self-check must never look at the gold answers -------------------------

def test_gold_answers_never_reach_any_prompt():
    """The regression that made the ablation meaningless: references in the retry.

    A wrong-but-grounded answer must still be accepted, and no prompt may contain
    any reference answer.
    """
    secret = "MAGIC-REFERENCE-STRING"
    client = ScriptedClient("It runs the do it step.")
    item = make_item(answers=[secret])
    res = answer(make_graph(), item, client=client)
    assert res.verdict == "accept"          # grounded, even though it is wrong
    for _, prompt in client.prompts:
        assert secret not in prompt


def test_gold_absent_even_when_revisions_fire():
    secret = "MAGIC-REFERENCE-STRING"
    client = ScriptedClient([""] * (MAX_REVISIONS + 1))
    res = answer(make_graph(), make_item(answers=[secret]), client=client)
    assert res.verdict == "revise"
    for _, prompt in client.prompts:
        assert secret not in prompt
    assert secret not in (res.revise_reason or "")


def test_grounding_flags_empty_answer():
    assert check_grounding("", make_graph()) is not None
    assert check_grounding("   ", make_graph()) is not None


def test_grounding_flags_invented_citation():
    reason = check_grounding('It does "launch the rocket".', make_graph())
    assert reason and "launch the rocket" in reason


def test_grounding_accepts_a_real_citation():
    assert check_grounding('It runs "do it".', make_graph()) is None


def test_grounding_does_not_punish_terse_answers():
    """"Yes"/"0"/"Byte" share nothing with node labels and are often correct."""
    for terse in ("Yes", "No", "0", "Byte", "-1"):
        assert check_grounding(terse, make_graph()) is None


def test_grounding_flags_long_answers_about_another_chart():
    reason = check_grounding(
        "The recipe requires whisking eggs, melting butter and chilling dough.",
        make_graph())
    assert reason and "does not reference" in reason


# --- representation experiment ---------------------------------------------------

def test_mermaid_representation_shows_the_raw_script():
    client = ScriptedClient("It runs the do it step.")
    res = answer(make_graph(), make_item(), client=client, representation="mermaid")
    _, prompt = client.prompts[0]
    assert "flowchart TD" in prompt          # raw mermaid, as the baseline sees
    assert "Nodes:" not in prompt            # not the serialization
    assert res.representation == "mermaid"


def test_graph_representation_is_the_default():
    client = ScriptedClient("It runs the do it step.")
    res = answer(make_graph(), make_item(), client=client)
    _, prompt = client.prompts[0]
    assert "Nodes:" in prompt and "A -> B" in prompt
    assert res.representation == "graph"


def test_mermaid_system_prompt_matches_the_baseline_wording():
    """Only the revision loop should differ from the baseline, not the framing."""
    from flowmind.eval.ablation import SYSTEM as BASELINE_SYSTEM
    from flowmind.examiner import SYSTEM_MERMAID
    assert SYSTEM_MERMAID == BASELINE_SYSTEM


def test_unknown_representation_raises():
    import pytest
    with pytest.raises(ValueError):
        build_content_prompt(make_graph(), make_item(), representation="nope")
