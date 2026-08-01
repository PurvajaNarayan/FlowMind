"""Run the single-pass baseline over a stratified sample (spec §8). [Owner: A/C]

The comparison point for the whole project: one LLM call over the Mermaid text,
no Reader, no router, no revision loop.

    # dry run, no GPU and no weights -- checks the plumbing end to end
    python tools/run_baseline.py --n 8 --backend scripted

    # real run
    python tools/run_baseline.py --n 100 --save runs/baseline_qwen3_8b.jsonl

Sampling is stratified across all four question types (spec §3 asks for that
explicitly), so the sample is not dominated by the topological questions that
happen to be most numerous.

SCORING, AND WHAT IS DELIBERATELY NOT SCORED HERE
-------------------------------------------------
Topological answers are scored by exact match, which is sound.

fact_retrieval / applied_scenario / flow_referential are NOT scored. Choosing
between an embedding-similarity judge and an LLM judge is spec §9's open question
and a team decision; scoring them now with the placeholder containment check in
eval.metrics.content_match would silently answer it and produce a number nobody
should trust. Instead every prediction is written to the trace file, so the whole
run can be re-scored from disk the moment the judge is agreed -- the same trick
that let the VLM sweeps be re-scored without touching the GPU.

Pass --score-content to apply the placeholder anyway, for wiring checks only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from flowmind.data import iter_qa, load_dataset
from flowmind.eval.ablation import single_pass_baseline
from flowmind.eval.metrics import content_match, topological_exact_match
from flowmind.tracing import Trace, TraceWriter

QA_TYPES = ("topological", "fact_retrieval", "applied_scenario", "flow_referential")


def stratified_items(dataset: dict, n: int, data_dir: str):
    """Round-robin across question types so every type is represented."""
    buckets: dict[str, list] = defaultdict(list)
    for item in iter_qa(dataset, data_dir=data_dir):
        buckets[item.qa_type].append(item)

    out, i = [], 0
    while len(out) < n and any(i < len(buckets[t]) for t in QA_TYPES if t in buckets):
        for t in QA_TYPES:
            if t in buckets and i < len(buckets[t]) and len(out) < n:
                out.append(buckets[t][i])
        i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_full.json")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--backend", choices=["local", "scripted"], default=None,
                    help="overrides FLOWMIND_LLM_BACKEND; 'scripted' needs no GPU")
    ap.add_argument("--save", default="runs/baseline.jsonl")
    ap.add_argument("--score-content", action="store_true",
                    help="apply the placeholder content metric (wiring checks only)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    if args.backend:
        import os
        os.environ["FLOWMIND_LLM_BACKEND"] = args.backend

    from flowmind.llm import get_client

    client = get_client()
    print(f"backend {type(client).__name__} | "
          f"model {getattr(client, 'model_id', 'n/a')}")

    ds = load_dataset(args.data)
    items = stratified_items(ds, args.n, args.data_dir)
    print(f"{len(items)} items: {dict(Counter(i.qa_type for i in items))}\n")

    topo = [0, 0]
    content_seen = 0
    content_hits = 0
    errors = 0

    with TraceWriter(args.save) as tw:
        for idx, item in enumerate(items, 1):
            try:
                res = single_pass_baseline(item, client=client,
                                           max_new_tokens=args.max_new_tokens)
            except Exception as exc:  # keep the sweep alive
                print(f"[{idx}/{len(items)}] {item.sample_key} ERROR "
                      f"{str(exc).splitlines()[0][:110]}")
                errors += 1
                continue

            correct = None
            if item.qa_type == "topological":
                correct = topological_exact_match(res.answer, item.answers[0])
                topo[0] += correct
                topo[1] += 1
                mark = "OK" if correct else "X "
            else:
                content_seen += 1
                if args.score_content:
                    correct = content_match(res.answer, item.answers)
                    content_hits += correct
                    mark = "ok" if correct else "x "
                else:
                    mark = "--"

            trace = Trace(
                sample_key=item.sample_key, question_id=item.question_id,
                intent="baseline", branch="single_pass",
                prediction=res.answer, gold=item.answers, correct=correct,
            )
            # The prompt is kept so a re-score can tell a bad answer apart from a
            # bad prompt, and so the run is reproducible from disk alone.
            trace.add_step("single_pass", {"qa_type": item.qa_type,
                                           "prompt": res.prompt}, res.answer)
            tw.write(trace)

            print(f"[{idx}/{len(items)}] {mark} {item.sample_key} "
                  f"[{item.qa_type[:16]:<16}] {res.answer[:60]!r}")

    print(f"\n=== single-pass baseline over {len(items) - errors} items ===")
    if topo[1]:
        print(f"  topological exact match : {topo[0]}/{topo[1]} "
              f"({100*topo[0]/topo[1]:.1f}%)")
    if content_seen:
        if args.score_content:
            print(f"  content (PLACEHOLDER)   : {content_hits}/{content_seen} "
                  f"({100*content_hits/content_seen:.1f}%)  <- do not report this")
        else:
            print(f"  content questions       : {content_seen} recorded, unscored "
                  f"(judge is spec §9, still open)")
    if errors:
        print(f"  errors                  : {errors}")
    print(f"  traces -> {args.save}")


if __name__ == "__main__":
    main()
