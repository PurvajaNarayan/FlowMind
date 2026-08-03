"""Run the Examiner over a stratified sample of content questions (spec §7.3 / M2).

Content questions only: fact_retrieval / applied_scenario / flow_referential.
Topological questions are excluded here — they're the graph_tool's job, already
covered by tools/parser_coverage.py.

    # dry run, no GPU and no weights -- checks the plumbing end to end
    python tools/run_examiner.py --n 8 --backend scripted

    # real local run (Qwen3-4B, fp16, ~8GB -- fits comfortably in 18GB unified
    # memory; the llm.py default Qwen3-8B needs ~16GB fp16 on non-CUDA backends
    # since bitsandbytes 4-bit only works on CUDA)
    export FLOWMIND_LLM_MODEL=Qwen/Qwen3-4B
    python tools/run_examiner.py --n 20 --backend local --save runs/examiner.jsonl

Reuses tools.run_baseline.stratified_items for sampling (same subset x
per-chart-cap x seeded-shuffle rationale — see that file's docstring), just
restricted to the three content types via its `qa_types` parameter.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from flowmind import examiner
from flowmind.data import load_dataset
from flowmind.reader.mermaid_reader import mermaid_to_graph
from flowmind.tracing import Trace, TraceWriter
from tools.run_baseline import stratified_items

CONTENT_TYPES = ("fact_retrieval", "applied_scenario", "flow_referential")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_full.json")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--backend", choices=["local", "scripted"], default=None,
                    help="overrides FLOWMIND_LLM_BACKEND; 'scripted' needs no GPU")
    ap.add_argument("--save", default="runs/examiner.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=128)
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
    items = stratified_items(ds, args.n, args.data_dir, qa_types=CONTENT_TYPES,
                             max_per_chart=args.max_per_chart, seed=args.seed)
    print(f"{len(items)} items over {len({i.sample_key for i in items})} charts")
    print(f"  by type   {dict(Counter(i.qa_type for i in items))}")
    print(f"  by subset {dict(Counter(i.subset for i in items))}\n")

    verdicts: Counter = Counter()
    revision_counts: Counter = Counter()
    errors = 0

    with TraceWriter(args.save) as tw:
        for idx, item in enumerate(items, 1):
            try:
                graph = mermaid_to_graph(item.mermaid)
                res = examiner.answer(graph, item, client=client,
                                      max_new_tokens=args.max_new_tokens)
            except Exception as exc:  # keep the sweep alive
                print(f"[{idx}/{len(items)}] {item.sample_key} ERROR "
                      f"{str(exc).splitlines()[0][:110]}")
                errors += 1
                continue

            verdicts[res.verdict] += 1
            revision_counts[res.revisions] += 1

            trace = Trace(
                sample_key=item.sample_key, question_id=item.question_id,
                intent="content", branch="examiner",
                prediction=res.answer, gold=item.answers,
                # correct stays None: "accept" only means the answer passed the
                # Examiner's graph-only self-checks, which is not a correctness
                # claim. Score these with tools/score_run.py (flowmind.judge).
                correct=None, revisions=res.revisions,
            )
            for i, attempt in enumerate(res.attempts):
                trace.add_step(f"examiner_attempt_{i}",
                               {"prompt": attempt["prompt"]},
                               {"answer": attempt["answer"],
                                "reject_reason": attempt["reason"]})
            tw.write(trace)

            mark = "ok" if res.verdict == "accept" else "REVISE"
            print(f"[{idx}/{len(items)}] {mark} {item.sample_key} "
                  f"[{item.qa_type[:16]:<16}] rev={res.revisions} {res.answer[:60]!r}")

    print(f"\n=== examiner over {len(items) - errors} items ===")
    total = sum(verdicts.values())
    if total:
        print("  self-check outcome (NOT accuracy -- see tools/score_run.py):")
        print(f"    accept : {verdicts['accept']}/{total} "
              f"({100*verdicts['accept']/total:.1f}%)")
        print(f"    revise : {verdicts['revise']}/{total} "
              f"({100*verdicts['revise']/total:.1f}%)")
        print(f"  revisions distribution: {dict(sorted(revision_counts.items()))}")
    if errors:
        print(f"  errors : {errors}")
    print(f"\n  answers recorded, unscored. To score them:")
    print(f"    python tools/score_run.py {args.save}")
    print(f"  traces -> {args.save}")


if __name__ == "__main__":
    main()
