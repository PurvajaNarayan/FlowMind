"""Score a saved run with the LLM judge (spec §9). [Owner: A/C]

Re-scores traces from disk, so the answers do not have to be regenerated:

    # dry run, no GPU
    python tools/score_run.py runs/baseline_qwen3_8b_v2.jsonl --backend scripted

    # real
    python tools/score_run.py runs/baseline_qwen3_8b_v2.jsonl --save runs/scored.jsonl

This is why run_baseline stores predictions and prompts rather than only a score:
the judge decision was still open when the baseline ran, and settling it later cost
no GPU time at all.

JUDGE VALIDATION
----------------
The topological questions are scored twice: once by exact match, which is
unambiguous, and once by the judge. Their agreement is the judge's measured
accuracy on questions where truth is known, and it is reported alongside the
content numbers. A judge that agrees 98% of the time earns trust on the
natural-language types; one that agrees 80% tells you how much noise those
numbers carry. Cheap, and it uses data already on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from flowmind.data import load_dataset
from flowmind.judge import CORRECT, UNPARSED, Judge
from flowmind.tracing import read_traces

CONTENT_TYPES = ("fact_retrieval", "applied_scenario", "flow_referential")


def index_questions(dataset: dict) -> dict[tuple[str, str], dict]:
    """(sample_key, question_id) -> qa record.

    The trace stores the prompt but not the bare question, and the judge needs the
    question itself (the FlowVQA protocol passes it alongside the references).
    Looking it up here is more robust than parsing it back out of the prompt.
    """
    out = {}
    for key, rec in dataset.items():
        for qid, qa in rec.get("qa", {}).items():
            out[(key, qid)] = qa
    return out


def pct(ok: int, total: int) -> str:
    return f"{ok}/{total} ({100*ok/total:.1f}%)" if total else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", help="JSONL written by tools/run_baseline.py")
    ap.add_argument("--data", default="data/train_full.json")
    ap.add_argument("--judge-model", default=None,
                    help="overrides FLOWMIND_JUDGE_MODEL")
    ap.add_argument("--backend", choices=["local", "scripted"], default=None)
    ap.add_argument("--save", default=None, help="write per-row verdicts here")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-validation", action="store_true",
                    help="skip judging the topological questions (loses the "
                         "agreement check, saves about a quarter of the calls)")
    args = ap.parse_args()

    if args.backend:
        os.environ["FLOWMIND_LLM_BACKEND"] = args.backend

    client = None
    if (args.backend or os.environ.get("FLOWMIND_LLM_BACKEND")) == "scripted":
        from flowmind.llm import get_client
        client = get_client()

    judge = Judge(client=client, model_id=args.judge_model)
    print(f"judge model: {judge.model_id if client is None else 'scripted'}")

    rows = read_traces(args.traces)
    if args.limit:
        rows = rows[: args.limit]
    questions = index_questions(load_dataset(args.data))

    content: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_subset: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    # An ablation trace file holds TWO rows per item -- one per arm, tagged by
    # `branch` (single_pass vs examiner/graph_tool). Pooling them would average
    # the two arms together and hide the very delta the ablation exists to show,
    # so content accuracy is also reported per branch.
    by_branch: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    agree = [0, 0]           # judge vs exact match on topological
    unparsed = 0
    out_rows = []

    for i, row in enumerate(rows, 1):
        key, qid = row["sample_key"], row["question_id"]
        qa = questions.get((key, qid))
        if qa is None:
            print(f"[{i}/{len(rows)}] {key} q{qid}: not found in {args.data}, skipping")
            continue

        qa_type = qa.get("type", "unknown")
        subset = key.rstrip("0123456789")
        prediction = row.get("prediction") or ""
        refs = row.get("gold") or [qa[k] for k in ("A1", "A2", "A3") if qa.get(k)]
        if isinstance(refs, str):
            refs = [refs]

        is_topo = qa_type == "topological"
        if is_topo and args.skip_validation:
            continue

        v = judge.judge(qa["Q"], refs, prediction)
        usable = v.label != UNPARSED
        if not usable:
            unparsed += 1

        if is_topo:
            # row["correct"] came from exact match in run_baseline: ground truth.
            # Only parseable verdicts count -- an unparsed reply happens to have
            # is_correct == False, which would be scored as "agreement" against
            # every exact-match failure and inflate the judge's apparent accuracy.
            truth = row.get("correct")
            if usable and truth is not None:
                agree[0] += (v.is_correct == bool(truth))
                agree[1] += 1
            mark = f"judge={v.label[:9]:<9} exact={row.get('correct')}"
        elif usable:
            # Unparsed replies stay out of the denominator too. Including them
            # would charge the answer for the judge's formatting failure.
            content[qa_type][0] += v.is_correct
            content[qa_type][1] += 1
            by_subset[subset][0] += v.is_correct
            by_subset[subset][1] += 1
            branch = row.get("branch") or "?"
            by_branch[branch][0] += v.is_correct
            by_branch[branch][1] += 1
            mark = f"judge={v.label}"
        else:
            mark = f"judge={v.label}"

        print(f"[{i}/{len(rows)}] {key} [{qa_type[:16]:<16}] {mark}"
              f"  {prediction[:44]!r}")
        out_rows.append({**row, "qa_type": qa_type, "subset": subset,
                         "judge_label": v.label, "judge_rationale": v.rationale})

    print(f"\n=== judged {len(out_rows)} rows from {args.traces} ===")
    tot = [0, 0]
    for t in CONTENT_TYPES:
        ok, n = content[t]
        tot[0] += ok
        tot[1] += n
        print(f"  {t:<18} {pct(ok, n)}")
    print(f"  {'CONTENT OVERALL':<18} {pct(*tot)}")
    if len(by_branch) > 1:
        # Ablation file: this is the comparison.
        print("\n  --- content accuracy by arm (the section 8 delta) ---")
        for b, (ok, n) in sorted(by_branch.items()):
            print(f"  {b:<18} {pct(ok, n)}")
        base = by_branch.get("single_pass")
        pipe = by_branch.get("examiner")
        if base and pipe and base[1] and pipe[1]:
            d = 100 * (pipe[0] / pipe[1] - base[0] / base[1])
            print(f"  {'delta':<18} {d:+.1f} points (pipeline - baseline)")
    if by_subset:
        print()
        for s, (ok, n) in sorted(by_subset.items()):
            print(f"  subset {s:<11} {pct(ok, n)}")

    if agree[1]:
        print(f"\n--- judge validation (topological, where exact match is truth) ---")
        print(f"  judge agrees with exact match: {pct(*agree)}")
        print("  treat the content numbers above as carrying this much noise")
    if unparsed:
        print(f"\n  unparsed judge replies: {unparsed} "
              f"(counted as neither correct nor incorrect)")

    if args.save and out_rows:
        p = Path(args.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for r in out_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  verdicts -> {p}")


if __name__ == "__main__":
    main()
