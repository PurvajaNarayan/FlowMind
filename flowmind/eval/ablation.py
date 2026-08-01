"""Ablation harness (spec §8): single-pass baseline vs full pipeline. [Owner: C]

Also the home for the vision ablation Owner A cares about:
text-Reader vs VLM-Reader on the same samples.

`single_pass_baseline` is implemented below; `full_pipeline` still waits on the
Examiner and Planner.

TODO(C):
  - full_pipeline(item): Reader -> router -> {graph_tool | Examiner | Planner}.
  - run_ablation(items): run both, write Traces (flowmind.tracing), report the
    accuracy delta and where the revision loop actually changed an answer.

WHY THE BASELINE GETS RAW MERMAID
---------------------------------
It is handed `item.mermaid`, not a parsed FlowGraph. That is deliberate: passing
the FlowGraph would give the baseline the Reader's output for free, and the
pipeline's advantage over the baseline is exactly what §8 exists to isolate. The
baseline should be the honest "throw the text at one LLM call" comparison, which
is also what the prior work it is being compared against does.
"""

from __future__ import annotations

from dataclasses import dataclass

from flowmind.data import QAItem
from flowmind.llm import LLMClient, get_client

SYSTEM = (
    "You answer questions about flowcharts. You are given the flowchart as a "
    "Mermaid.js script. Answer only from the flowchart. Be concise: a phrase or a "
    "single sentence. Do not restate the question and do not explain your "
    "reasoning."
)

# Topological questions want a bare value. The real pipeline routes these to the
# deterministic graph tool, but the baseline must attempt them too, or the two
# arms of the ablation would be answering different question sets.
_NUMERIC_HINT = "Answer with just the number, or just Yes or No. Nothing else."


@dataclass
class BaselineResult:
    answer: str
    prompt: str
    qa_type: str


def build_prompt(item: QAItem) -> str:
    parts = [
        "Flowchart (Mermaid):",
        item.mermaid.strip(),
        "",
        f"Question: {item.question.strip()}",
    ]
    if item.qa_type == "topological":
        parts.append(_NUMERIC_HINT)
    return "\n".join(parts)


def single_pass_baseline(item: QAItem, client: LLMClient | None = None,
                         max_new_tokens: int = 128) -> BaselineResult:
    """One LLM call over the Mermaid text — the §8 comparison point.

    No Reader, no router, no revision loop; that is the point of it. `client` is
    injectable so tests can pass a ScriptedClient and never load weights.
    """
    client = client or get_client()
    prompt = build_prompt(item)
    answer = client.complete(prompt, system=SYSTEM, max_new_tokens=max_new_tokens)
    return BaselineResult(answer=answer.strip(), prompt=prompt, qa_type=item.qa_type)


def full_pipeline(item: QAItem):
    raise NotImplementedError("full pipeline driver not implemented — [Owner: C]")
