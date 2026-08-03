"""LLM judge for the three natural-language question types (spec §9). [Owner: A/C]

§9 left the judge open: LLM judge vs embedding similarity vs both. FlowVQA itself
settles it. The benchmark's own protocol (paper §3.2) gives three evaluator models
-- GPT-3.5, Llama-2 70B, Mixtral 8x7B -- the response, the question and all three
gold answers; each writes a chain-of-thought rationale then emits a binary
correct/incorrect label, and the score is a majority vote. The authors frame it as
"length-invariant paraphrase detection" and state explicitly that it surpasses
similarity metrics and rule-based matching.

So: LLM judge. Embedding similarity would produce a number comparable to nothing
in the literature, and it cannot tell "returns the index i" from "does not return
the index i" -- a distinction the flow_referential questions turn on constantly.

TWO DELIBERATE DEVIATIONS, both of which belong in the write-up
---------------------------------------------------------------
1. One judge, not three. Llama-2 70B and Mixtral 8x7B do not fit free hardware.
   The ensemble can be reinstated by running this with several --judge-model
   values and majority-voting the outputs; the single judge is the cheap default.

2. The judge is Mistral-7B-Instruct (Apache 2.0), NOT the model under test.
   Qwen3-8B scoring its own answers would be self-preference bias and is the first
   thing a reader would attack. What matters is *lineage*, not raw capability --
   note that DeepSeek's small distills are built on Qwen and Llama bases, so the
   Qwen ones would reintroduce exactly the problem. The judging task, paraphrase
   detection against three short references, is far easier than the generation
   being judged, so a 7B model is ample. See DEFAULT_JUDGE_MODEL below for the
   alternatives and why each is or isn't suitable.

Because of (1) and (2), our numbers are NOT like-for-like with the paper's 68.42%
for GPT-4V. Report them as our own measurement with the protocol cited.

The judge is checked rather than trusted: `tools/score_run.py` also runs it over
the topological questions, where exact match gives unambiguous ground truth, and
reports how often it agrees. That converts "we picked a judge" into "we picked a
judge and measured it".
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from flowmind.llm import LLMClient, LocalTransformersClient

# Apache 2.0, independent lineage from the Qwen3-8B under test.
DEFAULT_JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Judge alternatives, all settable via FLOWMIND_JUDGE_MODEL. The binding
# constraint is lineage: the judge must not share a base model with the system
# being graded, or "different model" stops meaning anything.
#
#   mistralai/Mistral-7B-Instruct-v0.3   default. Apache 2.0, ungated at time of
#                                        writing, genuinely separate lineage.
#                                        ~15GB download, ~5GB resident at 4-bit.
#   mistralai/Ministral-8B-Instruct-*    newer Mistral 3 line, also Apache 2.0,
#                                        reported to emit far fewer tokens --
#                                        attractive for a judge that only needs a
#                                        one-line rationale. Confirm the exact
#                                        repo id on HF before relying on it.
#   microsoft/Phi-4-mini-instruct        3.8B, MIT, smallest and cheapest option.
#                                        Needs attn_implementation="sdpa" because
#                                        it defaults to flash-attention, which a
#                                        Turing T4 cannot do (llm.py handles this).
#
# NOT suitable:
#   deepseek-ai/DeepSeek-R1-Distill-Qwen-*   distilled onto a QWEN base, so it
#       shares lineage with the model under test. Same-family self-preference is
#       exactly what picking a separate judge is meant to avoid.
#   deepseek-ai/DeepSeek-R1-Distill-Llama-8B  lineage is fine (Llama base, MIT),
#       but it is a reasoning model: it emits long chain-of-thought and would
#       exhaust max_new_tokens before reaching the VERDICT line, turning verdicts
#       into `unparsed`. Raise max_new_tokens a long way if trying it.
#   Qwen3-*   same family as the answering model.
#
# If a gated repo refuses to download, either accept its terms on the HF model
# page and set HF_TOKEN, or fall back to Phi-4-mini, which is ungated.

CORRECT, INCORRECT, UNPARSED = "correct", "incorrect", "unparsed"

JUDGE_SYSTEM = (
    "You grade answers to questions about flowcharts. You are given the question, "
    "several reference answers that are all correct paraphrases of each other, and "
    "a candidate answer. Decide whether the candidate conveys the same information "
    "as any one of the references.\n"
    "Judge meaning, not wording: the candidate may be shorter, longer or phrased "
    "differently and still be correct. But a candidate that negates, contradicts "
    "or omits the key fact is incorrect."
)

_VERDICT_RE = re.compile(r"VERDICT\s*[:\-]?\s*\**\s*(CORRECT|INCORRECT)", re.I)


@dataclass
class Verdict:
    label: str                  # correct | incorrect | unparsed
    rationale: str = ""
    raw: str = ""

    @property
    def is_correct(self) -> bool:
        return self.label == CORRECT


def build_judge_prompt(question: str, references: list[str], candidate: str) -> str:
    refs = "\n".join(f"  {i}. {r.strip()}" for i, r in enumerate(references, 1))
    return (
        f"Question:\n  {question.strip()}\n\n"
        f"Reference answers (all correct):\n{refs}\n\n"
        f"Candidate answer:\n  {candidate.strip()}\n\n"
        "Give one short sentence of reasoning, then the verdict on its own line, "
        "exactly in this format:\n"
        "RATIONALE: <one sentence>\n"
        "VERDICT: CORRECT or INCORRECT"
    )


def parse_verdict(text: str) -> Verdict:
    """Pull the label out of the reply.

    An unreadable reply is labelled `unparsed`, never silently counted as
    incorrect -- that would bias the score downward by exactly the judge's own
    formatting failures, which have nothing to do with the answer being graded.
    """
    m = _VERDICT_RE.search(text or "")
    label = UNPARSED
    if m:
        label = CORRECT if m.group(1).upper() == "CORRECT" else INCORRECT
    else:
        # Some models drop the label and just say it. Only accept an unambiguous
        # mention, so "not incorrect" style phrasings stay unparsed.
        low = (text or "").lower()
        has_c, has_i = "correct" in low, "incorrect" in low
        if has_i and not re.search(r"\bnot incorrect\b", low):
            label = INCORRECT
        elif has_c and not has_i:
            label = CORRECT

    rat = ""
    rm = re.search(r"RATIONALE\s*[:\-]?\s*(.+)", text or "", re.I)
    if rm:
        rat = rm.group(1).strip().splitlines()[0].strip()
    return Verdict(label=label, rationale=rat, raw=(text or "").strip())


class Judge:
    """Wraps a client so the judge model is independent of the answering model."""

    def __init__(self, client: LLMClient | None = None,
                 model_id: str | None = None, max_new_tokens: int = 200):
        self.model_id = model_id or os.environ.get("FLOWMIND_JUDGE_MODEL",
                                                   DEFAULT_JUDGE_MODEL)
        self.max_new_tokens = max_new_tokens
        # Built lazily and separately from the answering client so the two models
        # never have to be resident at the same time.
        self._client = client

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = LocalTransformersClient(model_id=self.model_id)
        return self._client

    def judge(self, question: str, references: list[str], candidate: str) -> Verdict:
        if not (candidate or "").strip():
            # Nothing to grade; an empty answer is wrong without spending a call.
            return Verdict(label=INCORRECT, rationale="empty candidate answer")
        prompt = build_judge_prompt(question, references, candidate)
        reply = self.client.complete(prompt, system=JUDGE_SYSTEM,
                                     max_new_tokens=self.max_new_tokens)
        return parse_verdict(reply)
