"""Ablation harness (spec §8): single-pass baseline vs full pipeline. [Owner: C]

Also the home for the vision ablation Owner A cares about:
text-Reader vs VLM-Reader on the same samples.

Both `single_pass_baseline` and `full_pipeline` are implemented below.

WHY THE BASELINE GETS RAW MERMAID
---------------------------------
It is handed `item.mermaid`, not a parsed FlowGraph. That is deliberate: passing
the FlowGraph would give the baseline the Reader's output for free, and the
pipeline's advantage over the baseline is exactly what §8 exists to isolate. The
baseline should be the honest "throw the text at one LLM call" comparison, which
is also what the prior work it is being compared against does.

WHY code_request NEVER REACHES THE PLANNER HERE
------------------------------------------------
FlowVQA's own `qa_type` field is always one of `topological` / `fact_retrieval`
/ `applied_scenario` / `flow_referential` (spec §6); `route_from_dataset_type`
never maps to `code_request` (see router.py). So when evaluating on FlowVQA
itself, full_pipeline can only ever hit the graph_tool or Examiner branch — the
Planner (M3, not built) is structurally unreachable in this context, not just
untested.

I/O SEPARATION
---------------
Neither single_pass_baseline, full_pipeline, nor run_ablation write Traces or
print anything — that stays in the driver scripts (tools/run_baseline.py,
tools/run_ablation.py), matching how eval/* stays testable with ScriptedClient
and no filesystem side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from flowmind import examiner
from flowmind.data import QAItem
from flowmind.eval.metrics import content_match, topological_exact_match
from flowmind.examiner import ExaminerResult
from flowmind.graph_tool import answer_topological
from flowmind.llm import LLMClient, get_client
from flowmind.reader.mermaid_reader import mermaid_to_graph
from flowmind.router import route

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


@dataclass
class PipelineResult:
    answer: str | None
    branch: str            # "graph_tool" | "examiner"
    intent: str
    correct: bool | None = None
    revisions: int = 0
    examiner_result: ExaminerResult | None = None  # full detail, content branch only


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


def full_pipeline(item: QAItem, client: LLMClient | None = None) -> PipelineResult:
    """Reader -> router -> {graph_tool | Examiner} — the other §8 ablation arm.

    Raises NotImplementedError if routed to code_request; see the module
    docstring for why that never happens when evaluating on FlowVQA itself.
    """
    graph = mermaid_to_graph(item.mermaid)
    intent = route(item.question, qa_type=item.qa_type)

    if intent == "topological":
        _, pred, unresolved = answer_topological(graph, item.question)
        ans = None if (unresolved or pred is None) else str(pred)
        correct = (topological_exact_match(ans, item.answers[0])
                   if ans is not None else None)
        return PipelineResult(answer=ans, branch="graph_tool", intent=intent,
                              correct=correct)

    if intent == "content":
        res = examiner.answer(graph, item, client=client)
        return PipelineResult(answer=res.answer, branch="examiner", intent=intent,
                              correct=(res.verdict == "accept"), revisions=res.revisions,
                              examiner_result=res)

    raise NotImplementedError(
        f"intent {intent!r} routes to the Planner, which isn't built yet (M3). "
        "This shouldn't happen when qa_type is supplied for FlowVQA evaluation "
        "— see this module's docstring."
    )


def run_ablation(items: list[QAItem], client: LLMClient | None = None) -> dict:
    """Run single_pass_baseline and full_pipeline on the same items (spec §8).

    Scores both arms the same way per question type so the delta is apples to
    apples: exact match for topological, content_match for the other three
    (still the spec §9 placeholder — see eval.metrics.content_match's own
    caveat). Returns per-item rows plus a summary; writing Traces and printing
    a report is the driver script's job (tools/run_ablation.py).
    """
    client = client or get_client()
    rows = []
    for item in items:
        base = single_pass_baseline(item, client=client)
        pipe = full_pipeline(item, client=client)

        base_correct = (topological_exact_match(base.answer, item.answers[0])
                        if item.qa_type == "topological"
                        else content_match(base.answer, item.answers))

        rows.append({
            "sample_key": item.sample_key, "question_id": item.question_id,
            "qa_type": item.qa_type, "gold": item.answers,
            "baseline_answer": base.answer, "baseline_correct": base_correct,
            "pipeline_answer": pipe.answer, "pipeline_correct": pipe.correct,
            "pipeline_branch": pipe.branch, "pipeline_revisions": pipe.revisions,
        })

    def _rate(key: str, qa_type: str | None = None) -> float | None:
        pool = [r for r in rows if qa_type is None or r["qa_type"] == qa_type]
        pool = [r for r in pool if r[key] is not None]
        return (sum(bool(r[key]) for r in pool) / len(pool)) if pool else None

    summary = {
        "n": len(rows),
        "baseline_accuracy": _rate("baseline_correct"),
        "pipeline_accuracy": _rate("pipeline_correct"),
        "by_type": {
            t: {"baseline": _rate("baseline_correct", t),
                "pipeline": _rate("pipeline_correct", t)}
            for t in sorted({r["qa_type"] for r in rows})
        },
        # Items where the Examiner needed >=1 revision and still ended up
        # accepted -- the self-correction loop rescuing a wrong first answer.
        "examiner_revisions_recovered_answer": sum(
            1 for r in rows
            if r["pipeline_branch"] == "examiner"
            and r["pipeline_revisions"] > 0 and r["pipeline_correct"]
        ),
    }
    return {"rows": rows, "summary": summary}
