"""Examiner agent (spec §7.3). [Owner: B]

Answers content questions (fact_retrieval / applied_scenario / flow_referential)
from the graph + node text, then self-checks against the reference answers.
Topological questions never reach here — the router sends them to graph_tool.

The revision loop is LOCAL to this branch. Re-running the Reader on `revise`
would be a no-op for the deterministic text Reader (same mermaid in, same graph
out), so instead this re-prompts the same LLM with its previous wrong answer
and why it was rejected, asking it to reconsider. Capped at MAX_REVISIONS
attempts per question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flowmind.data import QAItem
from flowmind.eval.metrics import content_match
from flowmind.llm import LLMClient, get_client
from flowmind.schema import FlowGraph

MAX_REVISIONS = 2  # spec §7.3

SYSTEM = (
    "You answer questions about flowcharts. You are given the flowchart as a "
    "list of nodes and edges. Answer only from the graph. Be concise: a phrase "
    "or a single sentence. Do not restate the question and do not explain your "
    "reasoning."
)


@dataclass
class ExaminerResult:
    answer: str
    verdict: str            # "accept" | "revise"
    revise_reason: str | None = None  # sent back to Reader when verdict == "revise"
    revisions: int = 0
    attempts: list[dict] = field(default_factory=list)  # per-attempt prompt/answer/matched


def _graph_to_text(graph: FlowGraph) -> str:
    """Serialize a FlowGraph into the node/edge listing the LLM sees.

    The Examiner is handed a parsed FlowGraph, not raw mermaid, so this has no
    equivalent elsewhere in the repo (the Reader only goes mermaid -> graph).
    """
    lines = ["Nodes:"]
    for n in graph.nodes:
        lines.append(f"- {n.id} ({n.shape.value}): {n.label}")
    lines.append("Edges:")
    for e in graph.edges:
        suffix = f" [{e.label}]" if e.label else ""
        lines.append(f"- {e.source} -> {e.target}{suffix}")
    return "\n".join(lines)


def build_content_prompt(graph: FlowGraph, item: QAItem) -> str:
    parts = [
        "Flowchart:",
        _graph_to_text(graph),
        "",
        f"Question: {item.question.strip()}",
    ]
    return "\n".join(parts)


def _revise_prompt(base_prompt: str, prev_answer: str, reason: str) -> str:
    return (
        f"{base_prompt}\n\n"
        f"Your previous answer was: {prev_answer!r}\n"
        f"That answer was rejected: {reason}\n"
        "Reconsider the flowchart and answer again."
    )


def answer(graph: FlowGraph, item: QAItem, client: LLMClient | None = None,
           max_revisions: int = MAX_REVISIONS, max_new_tokens: int = 128) -> ExaminerResult:
    """Generate a content answer and self-check it against A1/A2/A3.

    On a mismatch, re-prompts the same LLM with its previous answer and why it
    was rejected (self-correction), up to `max_revisions` retries. If it still
    doesn't check out after that, returns verdict="revise" as a final,
    unresolved state rather than silently accepting a wrong answer.
    """
    client = client or get_client()
    base_prompt = build_content_prompt(graph, item)
    prompt = base_prompt
    attempts: list[dict] = []

    revisions = 0
    while True:
        reply = client.complete(prompt, system=SYSTEM, max_new_tokens=max_new_tokens).strip()
        matched = content_match(reply, item.answers)
        attempts.append({"prompt": prompt, "answer": reply, "matched": matched})

        if matched:
            return ExaminerResult(answer=reply, verdict="accept",
                                  revisions=revisions, attempts=attempts)

        reason = f"answer {reply!r} did not match reference answers {item.answers!r}"
        if revisions >= max_revisions:
            return ExaminerResult(answer=reply, verdict="revise", revise_reason=reason,
                                  revisions=revisions, attempts=attempts)

        prompt = _revise_prompt(base_prompt, reply, reason)
        revisions += 1
