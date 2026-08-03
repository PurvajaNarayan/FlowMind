"""Examiner agent (spec §7.3). [Owner: B]

Answers content questions (fact_retrieval / applied_scenario / flow_referential)
from the graph + node text, then self-checks the answer. Topological questions
never reach here — the router sends them to graph_tool.

The revision loop is LOCAL to this branch. Re-running the Reader on `revise`
would be a no-op for the deterministic text Reader (same mermaid in, same graph
out), so instead this re-prompts the same LLM with its previous answer and why it
was rejected, asking it to reconsider. Capped at MAX_REVISIONS attempts.

THE SELF-CHECK MUST NOT SEE THE GOLD ANSWERS
--------------------------------------------
An earlier version triggered revision on `content_match(reply, item.answers)` and
interpolated `item.answers` into the retry prompt, so on a retry the model was
shown the reference answers and asked again. That invalidates the §8 ablation: the
pipeline arm would get three attempts with the answer key visible while the
baseline arm gets one blind attempt, so the delta is guaranteed positive and
measures leakage rather than architecture.

Revision is therefore triggered only by checks computable from the GRAPH alone --
which is also what §7.3's own example describes ("referenced node not found in
graph"):

  * an empty answer
  * an answer that quotes a phrase appearing nowhere in the flowchart
  * an answer with no lexical overlap with the flowchart at all

`verdict` now means "passed its own self-checks", NOT "is correct". Correctness is
a separate, after-the-fact judgement made by flowmind.judge via
tools/score_run.py, once the pipeline has committed to an answer. Keeping the two
apart is what makes the ablation meaningful -- and the gap between them is itself
a result, since it says whether the Examiner's self-assessment is calibrated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from flowmind.data import QAItem
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
    # "accept" = passed the graph-only self-checks. NOT a correctness claim;
    # see the module docstring.
    verdict: str
    revise_reason: str | None = None  # sent back to Reader when verdict == "revise"
    revisions: int = 0
    attempts: list[dict] = field(default_factory=list)  # per-attempt prompt/answer/reason


# Words too common to count as evidence that an answer is grounded in a chart.
_STOPWORDS = frozenset("""
a an the is are was were be been being do does did done to of in on at for from with by as
it its this that these those and or but not no if then else when what which who whom how
many much you your yours we our they their there here i me my he she his her
step steps node nodes edge edges flowchart chart diagram box boxes answer question
""".split())

# Quoted spans, in any of the quote styles a model reaches for. Three characters
# minimum so single letters and stray apostrophes are not treated as citations.
_QUOTED_RE = re.compile(
    r'"([^"]{3,})"' r"|'([^']{3,})'" r"|`([^`]{3,})`" r"|“([^”]{3,})”"
)


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


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", (text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def _quoted_spans(text: str) -> list[str]:
    return [next(g for g in m.groups() if g) for m in _QUOTED_RE.finditer(text or "")]


def check_grounding(reply: str, graph: FlowGraph) -> str | None:
    """Why this answer should be revised, judged against the graph alone.

    Returns a reason, or None if the answer passes. Never consults gold answers —
    see the module docstring for why that distinction decides whether the §8
    ablation means anything.
    """
    text = (reply or "").strip()
    if not text:
        return "the answer was empty"

    chart_text = " ".join(
        [n.label for n in graph.nodes] + [e.label or "" for e in graph.edges]
    ).lower()

    # A quoted phrase is the model citing the chart. If the citation is not in the
    # chart, it invented a step -- exactly §7.3's "referenced node not found".
    for span in _quoted_spans(text):
        if span.strip().lower() not in chart_text:
            return (f"it cites {span!r}, which does not appear anywhere in the "
                    f"flowchart")

    # No overlap at all suggests the answer is not about this chart. Only applied
    # to answers long enough that overlap should be expected: terse-but-correct
    # replies ("Yes", "0", "Byte") legitimately share nothing with node labels and
    # must not be punished for brevity.
    words = _content_words(text)
    if len(words) >= 3 and not (words & _content_words(chart_text)):
        return "it does not reference anything in the flowchart"

    return None


def _revise_prompt(base_prompt: str, prev_answer: str, reason: str) -> str:
    return (
        f"{base_prompt}\n\n"
        f"Your previous answer was: {prev_answer!r}\n"
        f"It was rejected because {reason}.\n"
        "Answer again using only the flowchart above."
    )


def answer(graph: FlowGraph, item: QAItem, client: LLMClient | None = None,
           max_revisions: int = MAX_REVISIONS, max_new_tokens: int = 128) -> ExaminerResult:
    """Generate a content answer and self-check it against the graph.

    On a failed check, re-prompts the same LLM with its previous answer and the
    reason it was rejected, up to `max_revisions` retries. If it still fails,
    returns verdict="revise" as a final unresolved state rather than presenting an
    ungrounded answer as though it had passed.

    `item` is used for the question text only. The gold answers in `item.answers`
    are deliberately never read here — correctness is judged afterwards, by
    flowmind.judge, once the pipeline has committed.
    """
    client = client or get_client()
    base_prompt = build_content_prompt(graph, item)
    prompt = base_prompt
    attempts: list[dict] = []

    revisions = 0
    while True:
        reply = client.complete(prompt, system=SYSTEM, max_new_tokens=max_new_tokens).strip()
        reason = check_grounding(reply, graph)
        attempts.append({"prompt": prompt, "answer": reply, "reason": reason})

        if reason is None:
            return ExaminerResult(answer=reply, verdict="accept",
                                  revisions=revisions, attempts=attempts)

        if revisions >= max_revisions:
            return ExaminerResult(answer=reply, verdict="revise", revise_reason=reason,
                                  revisions=revisions, attempts=attempts)

        prompt = _revise_prompt(base_prompt, reply, reason)
        revisions += 1
