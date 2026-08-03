"""Judge parsing and prompt construction — no GPU, no weights.

Verdict parsing is the fragile part: the score depends entirely on reading the
label correctly out of free text, so the edge cases are pinned here.
"""

from flowmind.judge import (
    CORRECT,
    INCORRECT,
    UNPARSED,
    Judge,
    build_judge_prompt,
    parse_verdict,
)
from flowmind.llm import ScriptedClient


def test_prompt_carries_question_all_references_and_candidate():
    p = build_judge_prompt("What does it output?",
                           ["The index i.", "It returns i.", "i, the index."],
                           "It outputs index i.")
    assert "What does it output?" in p
    for r in ("The index i.", "It returns i.", "i, the index."):
        assert r in p
    assert "It outputs index i." in p
    assert "VERDICT" in p


def test_parses_the_documented_format():
    v = parse_verdict("RATIONALE: Same meaning.\nVERDICT: CORRECT")
    assert v.label == CORRECT
    assert v.rationale == "Same meaning."
    assert v.is_correct


def test_parses_incorrect_and_tolerates_decoration():
    assert parse_verdict("VERDICT: INCORRECT").label == INCORRECT
    assert parse_verdict("**VERDICT:** CORRECT").label == CORRECT
    assert parse_verdict("verdict - incorrect").label == INCORRECT


def test_incorrect_wins_when_both_words_appear():
    # "INCORRECT" contains "correct", so substring order matters.
    assert parse_verdict("The answer is incorrect.").label == INCORRECT


def test_unreadable_reply_is_unparsed_not_incorrect():
    """A judge that fails to format must not be scored as the answer being wrong."""
    assert parse_verdict("I am not sure how to grade this.").label == UNPARSED
    assert parse_verdict("").label == UNPARSED


def test_empty_candidate_is_incorrect_without_calling_the_model():
    client = ScriptedClient("VERDICT: CORRECT")
    v = Judge(client=client).judge("q?", ["a"], "   ")
    assert v.label == INCORRECT
    assert client.prompts == []          # no call was made


def test_judge_uses_the_injected_client():
    client = ScriptedClient("RATIONALE: Matches reference 2.\nVERDICT: CORRECT")
    v = Judge(client=client).judge("What is output?", ["i", "the index i"], "index i")
    assert v.is_correct
    system, prompt = client.prompts[0]
    assert "grade answers" in system.lower()
    assert "index i" in prompt
