"""Run the Planner over a sample of code-subset flowcharts (spec §7.4 / M3).

    # dry run, no GPU and no weights -- checks the plumbing end to end
    python tools/run_planner.py --n 5 --backend scripted

    # real local run (Qwen3-4B, fp16, ~8GB -- see tools/run_examiner.py for why
    # not the llm.py default Qwen3-8B on non-CUDA backends)
    export FLOWMIND_LLM_MODEL=Qwen/Qwen3-4B
    python tools/run_planner.py --n 10 --backend local --save runs/planner.jsonl

One QAItem per chart is sampled -- the Planner only reads .code/.mermaid/
.subset/.sample_key, not .question, and a chart's several qa rows all share
the same code, so any one of them is enough.

SCORING: behavioral_equivalence against the original `code` field, on inputs
guessed by generate_inputs (both in flowmind/eval/metrics.py). Some `code`
entries are methods extracted out of class context (first param `self`) and
cannot be meaningfully tested with synthesized inputs -- generate_inputs
raises ValueError for these, reported here as "not applicable", not a
failure. A generated function that doesn't match the required name at all
(wrong name, syntax error, etc.) needs no special handling: it naturally
scores near 0.0, since every trial call fails while the original mostly
succeeds -- see behavioral_equivalence's symmetric-exception rule.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from flowmind import planner
from flowmind.data import iter_qa, load_dataset
from flowmind.eval.metrics import behavioral_equivalence, generate_inputs
from flowmind.reader.mermaid_reader import mermaid_to_graph
from flowmind.tracing import Trace, TraceWriter


def sample_code_charts(ds: dict, n: int, data_dir: str, seed: int = 0) -> list:
    """One QAItem per code-subset chart, seeded and shuffled for reproducibility."""
    by_chart = {}
    for item in iter_qa(ds, data_dir=data_dir):
        if item.subset == "code" and item.sample_key not in by_chart:
            by_chart[item.sample_key] = item
    charts = list(by_chart.values())
    random.Random(seed).shuffle(charts)
    return charts[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_full.json")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--backend", choices=["local", "scripted"], default=None,
                    help="overrides FLOWMIND_LLM_BACKEND; 'scripted' needs no GPU")
    ap.add_argument("--save", default="runs/planner.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--n-trials", type=int, default=20,
                    help="random inputs per item for behavioral_equivalence")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; fixed so a run is reproducible")
    args = ap.parse_args()

    if args.backend:
        import os
        os.environ["FLOWMIND_LLM_BACKEND"] = args.backend

    from flowmind.llm import get_client

    client = get_client()
    print(f"backend {type(client).__name__} | "
          f"model {getattr(client, 'model_id', 'n/a')}")

    ds = load_dataset(args.data)
    items = sample_code_charts(ds, args.n, args.data_dir, seed=args.seed)
    print(f"{len(items)} code-subset charts sampled\n")

    errors = 0
    not_applicable = 0
    scoring_errors = 0
    scores: list[float] = []

    with TraceWriter(args.save) as tw:
        for idx, item in enumerate(items, 1):
            try:
                graph = mermaid_to_graph(item.mermaid)
                result = planner.plan(graph, item, client=client,
                                      max_new_tokens=args.max_new_tokens)
            except Exception as exc:  # keep the sweep alive
                print(f"[{idx}/{len(items)}] {item.sample_key} ERROR "
                      f"{str(exc).splitlines()[0][:110]}")
                errors += 1
                continue

            score: float | None = None
            func_name: str | None = None
            note = ""
            try:
                func_name, _params = planner._function_name_and_params(item.code)
                inputs = generate_inputs(item.code, func_name, n=args.n_trials,
                                         seed=args.seed)
            except ValueError as exc:
                # generate_inputs' documented case: original looks like a
                # method (first param `self`), not a standalone function.
                not_applicable += 1
                note = f" (n/a: {exc})"
            except Exception as exc:
                # Dataset-quality issue, not a generated-code failure: e.g. a
                # snippet whose original code references a name only defined
                # outside the extracted function (seen in practice: a
                # NameError on a sentinel default argument). Not the thing
                # this milestone is scoring, so skip and keep the sweep alive
                # rather than crash the whole run over one bad record.
                scoring_errors += 1
                note = f" (scoring error: {type(exc).__name__}: {exc})"
            else:
                score = behavioral_equivalence(result.code or "", item.code,
                                               func_name, inputs)
                scores.append(score)

            trace = Trace(
                sample_key=item.sample_key, question_id=item.question_id,
                intent="code_request", branch="planner",
                prediction=result.code, correct=(score == 1.0 if score is not None else None),
            )
            trace.add_step("plan_doc", {}, {"plan_markdown": result.plan_markdown})
            trace.add_step("behavioral_equivalence", {"func_name": func_name}, {"score": score})
            tw.write(trace)

            score_str = f"{score:.2f}" if score is not None else "n/a"
            print(f"[{idx}/{len(items)}] {item.sample_key} score={score_str}{note}")
            print(f"  code: {(result.code or '')[:80]!r}")

    print(f"\n=== planner over {len(items) - errors} items ===")
    if scores:
        print(f"  mean behavioral equivalence : {sum(scores)/len(scores):.3f} "
              f"over {len(scores)} scoreable items")
    print(f"  not applicable (method, not a function): {not_applicable}")
    if scoring_errors:
        print(f"  scoring errors (dataset-quality issues, skipped): {scoring_errors}")
    if errors:
        print(f"  errors : {errors}")
    print(f"  traces -> {args.save}")


if __name__ == "__main__":
    main()
