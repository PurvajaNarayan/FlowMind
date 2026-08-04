"""Planner agent, exercised with no GPU and no weights (ScriptedClient).

Uses the same sample.json fixture (code00453 / find_fixed_point) as
test_graph_tool.py and test_ablation.py, via the shared `graph`/`sample`
conftest fixtures.
"""

import pytest

from flowmind.data import QAItem
from flowmind.llm import ScriptedClient
from flowmind.planner import (
    _extract_code,
    _function_name_and_params,
    _ordered_steps,
    build_code_prompt,
    build_plan_prompt,
    plan,
)


def make_item(sample, code=...):
    return QAItem(
        sample_key=sample["key"], question_id="1", question="irrelevant",
        answers=["irrelevant"], qa_type="topological", mermaid=sample["mermaid"],
        subset="code", code=(sample.get("code") if code is ... else code),
        summary=sample.get("summary"),
    )


def test_ordered_steps_starts_from_terminal_and_annotates_branches(graph):
    steps = _ordered_steps(graph)
    assert steps.startswith("1. Start")
    assert "[Yes]" in steps
    assert "[No]" in steps


def test_ordered_steps_marks_loop_back(graph):
    # F -> C is a back-edge in the fixture (i = i + 1 loops to i = 0).
    assert "loop back to step" in _ordered_steps(graph)


def test_function_name_and_params_from_sample(sample):
    name, params = _function_name_and_params(sample["code"])
    assert name == "find_fixed_point"
    assert params == ["arr", "n"]


def test_function_name_and_params_rejects_code_with_no_function():
    with pytest.raises(ValueError):
        _function_name_and_params("x = 1\n")


def test_extract_code_pulls_fenced_block():
    reply = "Here you go:\n```python\ndef f(x):\n    return x\n```\nDone."
    assert _extract_code(reply) == "def f(x):\n    return x"


def test_extract_code_falls_back_without_fence():
    reply = "Sure.\ndef f(x):\n    return x\n"
    assert _extract_code(reply) == "def f(x):\n    return x"


def test_plan_prompt_contains_ordered_steps(graph):
    assert "1. Start" in build_plan_prompt(_ordered_steps(graph))


def test_code_prompt_states_required_signature(graph):
    p = build_code_prompt(_ordered_steps(graph), "find_fixed_point", ["arr", "n"])
    assert "find_fixed_point" in p
    assert "arr, n" in p


def test_plan_runs_two_llm_calls_and_extracts_code(graph, sample):
    client = ScriptedClient([
        "1. Do the thing\n2. Return result",
        "```python\ndef find_fixed_point(arr, n):\n    return -1\n```",
    ])
    result = plan(graph, make_item(sample), client=client)

    assert "Do the thing" in result.plan_markdown
    assert result.code == "def find_fixed_point(arr, n):\n    return -1"
    assert len(client.prompts) == 2

    plan_system, plan_prompt = client.prompts[0]
    code_system, code_prompt = client.prompts[1]
    assert plan_system and "do not include code" in plan_system.lower()
    assert "1. Start" in plan_prompt
    assert "find_fixed_point" in code_prompt


def test_plan_raises_for_non_code_item(graph, sample):
    with pytest.raises(NotImplementedError):
        plan(graph, make_item(sample, code=None))
