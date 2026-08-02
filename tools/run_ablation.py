"""Run the full ablation: single-pass baseline vs full pipeline (spec §8 / M5).

Same sample set, both arms, scored the same way per question type. This is the
comparison the whole project is structured around — see flowmind.eval.ablation
for why the baseline gets raw mermaid while the pipeline gets the parsed graph,
and why code_request never reaches the Planner when qa_type comes from FlowVQA.

    # dry run, no GPU and no weights -- checks the plumbing end to end
    python tools/run_ablation.py --n 8 --backend scripted

    # real local run (Qwen3-4B, fp16, ~8GB -- fits comfortably in 18GB unified
    # memory; see tools/run_examiner.py for why not the llm.py default Qwen3-8B)
    export FLOWMIND_LLM_MODEL=Qwen/Qwen3-4B
    python tools/run_ablation.py --n 20 --backend local --save runs/ablation.jsonl

Reuses tools.run_baseline.stratified_items for sampling, across all four
question types (topological included -- the baseline has to attempt those
too, or the two arms would be scored on different question sets).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from flowmind.data import load_dataset
from flowmind.eval.ablation import run_ablation
from flowmind.tracing import Trace, TraceWriter
from tools.run_baseline import stratified_items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_full.json")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--backend", choices=["local", "scripted"], default=None,
                    help="overrides FLOWMIND_LLM_BACKEND; 'scripted' needs no GPU")
    ap.add_argument("--save", default="runs/ablation.jsonl")
    ap.add_argument("--max-per-chart", type=int, default=2,
                    help="cap questions drawn from one flowchart (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; fixed so a run is reproducible")
    args = ap.parse_args()

    if args.max_per_chart == 0:
        args.max_per_chart = None

    if args.backend:
        import os
        os.environ["FLOWMIND_LLM_BACKEND"] = args.backend

    from flowmind.llm import get_client

    client = get_client()
    print(f"backend {type(client).__name__} | "
          f"model {getattr(client, 'model_id', 'n/a')}")

    ds = load_dataset(args.data)
    items = stratified_items(ds, args.n, args.data_dir,
                             max_per_chart=args.max_per_chart, seed=args.seed)
    print(f"{len(items)} items over {len({i.sample_key for i in items})} charts\n")

    result = run_ablation(items, client=client)
    rows, summary = result["rows"], result["summary"]

    with TraceWriter(args.save) as tw:
        for row in rows:
            common = dict(sample_key=row["sample_key"], question_id=row["question_id"],
                          intent=row["qa_type"], gold=row["gold"])
            tw.write(Trace(**common, branch="single_pass",
                           prediction=row["baseline_answer"],
                           correct=row["baseline_correct"]))
            tw.write(Trace(**common, branch=row["pipeline_branch"],
                           prediction=row["pipeline_answer"],
                           correct=row["pipeline_correct"],
                           revisions=row["pipeline_revisions"]))

            b = "OK" if row["baseline_correct"] else "X "
            p = "OK" if row["pipeline_correct"] else "X "
            print(f"{row['sample_key']:<16} [{row['qa_type'][:16]:<16}] "
                  f"baseline={b} pipeline={p} ({row['pipeline_branch']}, "
                  f"rev={row['pipeline_revisions']})")

    def pct(x: float | None) -> str:
        return f"{100*x:.1f}%" if x is not None else "n/a"

    print(f"\n=== ablation over {summary['n']} items ===")
    print(f"  baseline accuracy : {pct(summary['baseline_accuracy'])}")
    print(f"  pipeline accuracy : {pct(summary['pipeline_accuracy'])}")
    print(f"  {'type':<20}{'baseline':>10}{'pipeline':>10}")
    for t, acc in sorted(summary["by_type"].items()):
        print(f"  {t:<20}{pct(acc['baseline']):>10}{pct(acc['pipeline']):>10}")
    print(f"  examiner revisions recovered a wrong first answer: "
          f"{summary['examiner_revisions_recovered_answer']}")
    print(f"  traces -> {args.save}")


if __name__ == "__main__":
    main()
