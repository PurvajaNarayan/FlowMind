"""Baseline + LLM seam, exercised with no GPU and no weights.

The point of the ScriptedClient is that this whole path is testable in CI: the
prompt construction and the flow through single_pass_baseline are asserted here,
so a broken prompt is caught before anyone spends GPU time on a sweep.
"""

import pytest

from flowmind.data import QAItem
from flowmind.eval.ablation import build_prompt, single_pass_baseline
from flowmind.llm import LLMClient, ScriptedClient, get_client


def make_item(qa_type="fact_retrieval"):
    return QAItem(
        sample_key="code00453", question_id="1",
        question="What does the flowchart output when a fixed point is found?",
        answers=["It outputs the index i."], qa_type=qa_type,
        mermaid='flowchart TD\n    A(["Start"]) --> B["do it"]',
        subset="code",
    )


def test_scripted_client_satisfies_the_protocol():
    assert isinstance(ScriptedClient(), LLMClient)


def test_prompt_contains_mermaid_and_question():
    p = build_prompt(make_item())
    assert "flowchart TD" in p
    assert "fixed point is found" in p
    # A content question must not get the numeric instruction.
    assert "just the number" not in p


def test_topological_prompt_asks_for_a_bare_value():
    p = build_prompt(make_item(qa_type="topological"))
    assert "just the number" in p


def test_baseline_returns_the_model_reply_and_passes_a_system_prompt():
    client = ScriptedClient("It outputs the index i.")
    res = single_pass_baseline(make_item(), client=client)
    assert res.answer == "It outputs the index i."
    assert res.qa_type == "fact_retrieval"
    system, prompt = client.prompts[0]
    assert system and "only from the flowchart" in system
    assert "flowchart TD" in prompt


def test_baseline_strips_whitespace():
    res = single_pass_baseline(make_item(), client=ScriptedClient("  7  \n"))
    assert res.answer == "7"


def test_scripted_backend_selectable_by_env(monkeypatch):
    monkeypatch.setenv("FLOWMIND_LLM_BACKEND", "scripted")
    assert isinstance(get_client(fresh=True), ScriptedClient)


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("FLOWMIND_LLM_BACKEND", "nope")
    with pytest.raises(ValueError):
        get_client(fresh=True)
