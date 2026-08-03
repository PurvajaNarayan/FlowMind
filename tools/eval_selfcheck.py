"""Do the Examiner's self-checks catch the errors the judge finds? [Owner: A]

    python tools/eval_selfcheck.py runs/scored_mermaid_phi.jsonl

Replays every self-check over a run that has already been judged, and compares the
two verdicts. No GPU, no model, no regeneration -- it reads answers and judgements
that are already on disk.

WHY
---
A revision loop is only worth having if its trigger correlates with being wrong.
The first version triggered on graph grounding alone, and measured against seven
answers the judge marked wrong it caught zero: every real error is *about* the
chart. The model names real steps and picks the wrong one. One observed failure was
answering with the correct answer to a DIFFERENT question about the same flowchart
-- perfectly grounded, completely wrong.

So the check gained question-aware parts, and this measures whether that helped.
Two numbers matter, and they trade off:

  catch rate       of answers the judge called WRONG, how many does the check
                   flag? This is the ceiling on how much the loop can ever help.
  false-alarm rate of answers the judge called RIGHT, how many does the check
                   flag? Every one of these sends a correct answer back for
                   revision, so a high rate makes the pipeline worse, not better.

A catch rate of 0% means the loop cannot help no matter how many retries it gets.
A false-alarm rate above the catch rate means it is actively harmful.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from flowmind.data import load_dataset
from flowmind.examiner import check_answers_question, check_grounding
from flowmind.reader.mermaid_reader import mermaid_to_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scored", help="JSONL written by tools/score_run.py --save")
    ap.add_argument("--data", default="data/train_full.json")
    ap.add_argument("--branch", default=None,
                    help="restrict to one ablation arm, e.g. examiner")
    ap.add_argument("--show", type=int, default=8, help="examples to print per group")
    args = ap.parse_args()

    ds = load_dataset(args.data)
    graphs: dict[str, object] = {}
    rows = [json.loads(l) for l in Path(args.scored).open(encoding="utf-8") if l.strip()]

    # judge verdict -> which checks fired
    tally: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    examples: dict[str, list] = defaultdict(list)

    for r in rows:
        if r.get("qa_type") == "topological":
            continue
        judged = r.get("judge_label")
        if judged not in ("correct", "incorrect"):
            continue                      # unparsed: no ground truth to compare to
        if args.branch and r.get("branch") != args.branch:
            continue

        key, qid = r["sample_key"], r["question_id"]
        rec = ds.get(key)
        qa = (rec or {}).get("qa", {}).get(qid)
        if not rec or not qa:
            continue
        if key not in graphs:
            graphs[key] = mermaid_to_graph(rec["mermaid"])
        graph = graphs[key]
        answer = r.get("prediction") or ""

        g_reason = check_grounding(answer, graph)
        q_reason = check_answers_question(answer, graph, qa["Q"])
        fired = "grounding" if g_reason else ("question-aware" if q_reason else "none")

        tally[judged][fired] += 1
        tally[judged]["total"] += 1
        if fired != "none" and len(examples[judged]) < args.show:
            examples[judged].append((key, qid, answer[:56], (g_reason or q_reason)[:78]))

    print(f"=== self-check vs judge over {sum(t['total'] for t in tally.values())} "
          f"content answers ===")
    print(f"    {args.scored}"
          + (f"   [branch={args.branch}]" if args.branch else "") + "\n")

    for judged, label in (("incorrect", "judge says WRONG  (flagging these is the point)"),
                          ("correct", "judge says RIGHT  (flagging these is harmful)")):
        t = tally[judged]
        n = t["total"]
        if not n:
            continue
        flagged = n - t["none"]
        print(f"  {label}")
        print(f"    n={n}  flagged {flagged} ({100*flagged/n:.1f}%)"
              f"   [grounding {t['grounding']}, question-aware {t['question-aware']}]")

    wrong, right = tally["incorrect"], tally["correct"]
    if wrong["total"] and right["total"]:
        catch = (wrong["total"] - wrong["none"]) / wrong["total"]
        alarm = (right["total"] - right["none"]) / right["total"]
        print(f"\n  catch rate       {100*catch:.1f}%   (upper bound on how much the "
              f"loop can help)")
        print(f"  false-alarm rate {100*alarm:.1f}%   (correct answers sent back for "
              f"revision)")
        if catch == 0:
            print("\n  The loop cannot help: it never fires on a wrong answer.")
        elif alarm >= catch:
            print("\n  The loop is likely HARMFUL: it flags correct answers at least "
                  "as often\n  as wrong ones, so revising costs more than it gains.")
        else:
            print(f"\n  Net favourable: catches {100*catch:.0f}% of errors against "
                  f"{100*alarm:.0f}% false alarms.")

    for judged in ("incorrect", "correct"):
        if examples[judged]:
            print(f"\n  --- flagged, judge said {judged} ---")
            for key, qid, ans, why in examples[judged]:
                print(f"    {key} q{qid}: {ans!r}")
                print(f"      {why}")


if __name__ == "__main__":
    main()
