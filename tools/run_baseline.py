"""Run the single-pass baseline over a stratified sample (spec §8). [Owner: A/C]

The comparison point for the whole project: one LLM call over the Mermaid text,
no Reader, no router, no revision loop.

    # dry run, no GPU and no weights -- checks the plumbing end to end
    python tools/run_baseline.py --n 8 --backend scripted

    # real run
    python tools/run_baseline.py --n 120 --save runs/baseline_qwen3_8b.jsonl

Sampling is stratified over (subset x question type) -- 12 cells -- with a cap on
how many questions come from any one flowchart, and a seeded shuffle inside each
cell. All three parts are needed; each was added after the previous sample turned
out to be misleading:

  question type   spec §3 asks for it, and without it the sample is dominated by
                  topological questions, which are the most numerous.
  subset          train_full.json is ordered code-first, so a type-only sample of
                  100 items came back 100% `code` -- and `code` is 261 of 1319
                  records, with wiki (651) and instruct (407) absent. The VLM sweep
                  had `code` at 0.854 edge-F1 against instruct's 0.619, so subsets
                  genuinely differ.
  per-chart cap   that same sample drew 100 questions from just 20 charts, so
                  answers were clustered and the effective n was far below 100.
  seeded shuffle  with a per-chart cap and no shuffle, selection follows question
                  position: a chart's topological questions run node_count,
                  edge_count, shortest_path, predecessor..., so the cap returned
                  counting questions only and dropped adjacency entirely -- losing
                  the split that carries the finding.

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
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from flowmind.data import iter_qa, load_dataset
from flowmind.eval.ablation import single_pass_baseline
from flowmind.eval.metrics import content_match, topological_exact_match
from flowmind.tracing import Trace, TraceWriter

QA_TYPES = ("topological", "fact_retrieval", "applied_scenario", "flow_referential")


SUBSETS = ("code", "instruct", "wiki")


def stratified_items(dataset: dict, n: int, data_dir: str,
                     qa_types: tuple[str, ...] = QA_TYPES,
                     max_per_chart: int | None = 2, seed: int = 0):
    """Round-robin over (subset x question type) so both dimensions are covered.

    `qa_types` restricts which question types are sampled — e.g. the Examiner
    runner passes only the three content types, reusing this same subset x
    per-chart-cap x seeded-shuffle logic instead of duplicating it.

    Stratifying on question type alone is not enough. train_full.json is ordered
    code-first, so taking the first items of each type produced a 100-item sample
    that was 100% `code` -- and `code` is only 261 of 1319 records, with wiki (651)
    and instruct (407) absent entirely. That matters: the VLM sweep showed `code`
    at 0.854 edge-F1 against instruct's 0.619, so subsets genuinely differ and a
    code-only sample cannot speak for the dataset.

    `max_per_chart` also caps how many questions come from any single flowchart.
    The earlier sample drew 100 questions from just 20 charts, so answers were
    clustered rather than independent and the effective sample size was well below
    the nominal one.
    """
    buckets: dict[tuple[str, str], list] = defaultdict(list)
    for item in iter_qa(dataset, data_dir=data_dir):
        buckets[(item.subset, item.qa_type)].append(item)

    # Shuffle within each cell, seeded. Taking items in dataset order biases the
    # sample by question position: a chart's topological questions are ordered
    # node_count, edge_count, shortest_path, predecessor..., so combining a
    # per-chart cap with positional selection returned counting questions only and
    # dropped adjacency entirely -- losing the 88.9%-vs-12.5% split that is the
    # most informative thing in the run. A fixed seed keeps it reproducible.
    rng = random.Random(seed)
    for pool in buckets.values():
        rng.shuffle(pool)

    cells = [(s, t) for s in SUBSETS for t in qa_types]
    per_chart: Counter = Counter()
    cursor = dict.fromkeys(cells, 0)
    out: list = []

    progressed = True
    while len(out) < n and progressed:
        progressed = False
        for cell in cells:
            if len(out) >= n:
                break
            pool = buckets.get(cell, [])
            i = cursor[cell]
            # Skip past charts that have already contributed their quota.
            while i < len(pool) and max_per_chart is not None \
                    and per_chart[pool[i].sample_key] >= max_per_chart:
                i += 1
            if i < len(pool):
                out.append(pool[i])
                per_chart[pool[i].sample_key] += 1
                cursor[cell] = i + 1
                progressed = True
            else:
                cursor[cell] = i
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
    print(f"{len(items)} items over {len({i.sample_key for i in items})} charts")
    print(f"  by type   {dict(Counter(i.qa_type for i in items))}")
    print(f"  by subset {dict(Counter(i.subset for i in items))}\n")

    topo = [0, 0]
    topo_by_subset: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    # Yes/No adjacency vs counting is the split that carries the finding: the
    # first-pass run scored 88.9% on adjacency and 12.5% on counting, i.e. the
    # model does local lookups fine and does not aggregate at all.
    topo_by_form: dict[str, list[int]] = defaultdict(lambda: [0, 0])
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
                topo_by_subset[item.subset][0] += correct
                topo_by_subset[item.subset][1] += 1
                form = ("adjacency" if str(item.answers[0]).strip().lower()
                        in ("yes", "no") else "counting")
                topo_by_form[form][0] += correct
                topo_by_form[form][1] += 1
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
        for form, (ok, t) in sorted(topo_by_form.items()):
            print(f"    {form:<20}: {ok}/{t} ({100*ok/t:.1f}%)")
        for sub, (ok, t) in sorted(topo_by_subset.items()):
            print(f"    subset {sub:<13}: {ok}/{t} ({100*ok/t:.1f}%)")
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
