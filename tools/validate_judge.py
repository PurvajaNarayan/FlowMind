"""Does the judge actually discriminate? (spec §9 due diligence). [Owner: A]

    python tools/validate_judge.py --n 30 --backend scripted     # plumbing check
    python tools/validate_judge.py --n 30                        # real
    python tools/validate_judge.py --n 30 --judge-model microsoft/Phi-4-mini-instruct

WHY THIS EXISTS
---------------
tools/score_run.py validates the judge by comparing it to exact match on the
topological questions, and both judges tried so far scored 30/30 there. That
turned out to certify almost nothing: topological answers are "9" and "Yes", so
agreeing on them is trivial, while the content questions are the hard case.

The ablation run made the gap visible. Two judges scored the same 90 content
answers 20 points apart, and the more lenient one returned 100.0% for a single
8B-model pass -- against a published SOTA of ~82.7% -- while saying "incorrect"
just twice in 90 rows. It also marked BOTH "7 steps." and "6 steps." correct for
the same question, which cannot be true of either.

A judge that rarely says "incorrect" carries no information, and nothing in the
previous validation would catch it. This measures both directions:

  true-positive rate   given a real reference answer, does it say correct?
                       Low means the judge is too strict and will understate every
                       system it grades.
  true-negative rate   given an answer that is definitely wrong, does it say
                       incorrect? Low means the judge is a rubber stamp and its
                       numbers are meaningless regardless of how good they look.

CONTROLS
--------
  gold        a reference answer, verbatim. Expect CORRECT.
  swapped     a reference answer belonging to a DIFFERENT question. Fluent,
              confident, well-formed and definitely wrong -- the hard negative,
              because no surface cue gives it away.
  negated     the reference answer with its meaning inverted. Catches judges
              scoring on topic overlap rather than on claim.
  renumbered  a reference answer with its digits changed, where it has any. Aimed
              squarely at the "7 steps"/"6 steps" failure.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from flowmind.data import load_dataset
from flowmind.judge import CORRECT, INCORRECT, UNPARSED, Judge

CONTENT_TYPES = ("fact_retrieval", "applied_scenario", "flow_referential")

# Control name -> the verdict a competent judge should return.
EXPECTED = {"gold": CORRECT, "swapped": INCORRECT,
            "negated": INCORRECT, "renumbered": INCORRECT}


def negate(text: str) -> str:
    """Invert the claim, preferring a natural edit over a clumsy wrapper."""
    subs = [
        (r"\bis\b", "is not"), (r"\bare\b", "are not"), (r"\bwas\b", "was not"),
        (r"\bwill\b", "will not"), (r"\bcan\b", "cannot"),
        (r"\breturns\b", "does not return"), (r"\boutputs\b", "does not output"),
        (r"\bincrements\b", "does not increment"), (r"\bsets\b", "does not set"),
    ]
    for pat, rep in subs:
        if re.search(pat, text, re.I):
            return re.sub(pat, rep, text, count=1, flags=re.I)
    # No verb we recognise: fall back to an explicit wrapper. Less natural, but
    # still unambiguously the opposite claim.
    return f"It is not true that {text[0].lower()}{text[1:]}" if text else text


def renumber(text: str) -> str | None:
    """Change every digit run to a different value. None if there are none."""
    if not re.search(r"\d", text):
        return None
    return re.sub(r"\d+", lambda m: str(int(m.group()) + 3), text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_full.json")
    ap.add_argument("--n", type=int, default=30, help="questions to probe")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--backend", choices=["local", "scripted"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    if args.backend:
        os.environ["FLOWMIND_LLM_BACKEND"] = args.backend

    client = None
    if os.environ.get("FLOWMIND_LLM_BACKEND") == "scripted":
        from flowmind.llm import get_client
        client = get_client()

    judge = Judge(client=client, model_id=args.judge_model)
    print(f"judge: {judge.model_id if client is None else 'scripted'}\n")

    # Collect content questions that carry at least one reference answer.
    pool = []
    for key, rec in load_dataset(args.data).items():
        for qid, qa in rec.get("qa", {}).items():
            if qa.get("type") in CONTENT_TYPES:
                refs = [qa[k] for k in ("A1", "A2", "A3") if qa.get(k)]
                if refs:
                    pool.append((key, qid, qa["Q"], refs))
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    probes = pool[: args.n]

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    unparsed: dict[str, int] = defaultdict(int)
    rows = []

    for i, (key, qid, question, refs) in enumerate(probes, 1):
        # The hard negative comes from a different question in the pool.
        other = rng.choice([p for p in pool if p[0] != key])[3][0]

        candidates = {
            "gold": refs[0],
            "swapped": other,
            "negated": negate(refs[0]),
            "renumbered": renumber(refs[0]),
        }
        for name, cand in candidates.items():
            if cand is None:            # renumbered with no digits present
                continue
            v = judge.judge(question, refs, cand)
            want = EXPECTED[name]
            if v.label == UNPARSED:
                unparsed[name] += 1
            else:
                stats[name][0] += (v.label == want)
                stats[name][1] += 1
            ok = "ok " if v.label == want else ("?? " if v.label == UNPARSED else "MISS")
            print(f"[{i}/{len(probes)}] {ok} {name:<11} want={want:<9} "
                  f"got={v.label:<9} {cand[:52]!r}")
            rows.append({"key": key, "question_id": qid, "control": name,
                         "candidate": cand, "expected": want, "got": v.label,
                         "rationale": v.rationale})

    print(f"\n=== judge discrimination over {len(probes)} questions ===")
    for name in ("gold", "swapped", "negated", "renumbered"):
        ok, n = stats[name]
        rate = f"{100*ok/n:.1f}%" if n else "n/a"
        label = "true-positive" if name == "gold" else "true-negative"
        print(f"  {name:<11} ({label}) {ok}/{n} ({rate})"
              + (f"   unparsed {unparsed[name]}" if unparsed[name] else ""))

    tpr_ok, tpr_n = stats["gold"]
    neg_ok = sum(stats[k][0] for k in ("swapped", "negated", "renumbered"))
    neg_n = sum(stats[k][1] for k in ("swapped", "negated", "renumbered"))
    print()
    if tpr_n and neg_n:
        tpr, tnr = tpr_ok / tpr_n, neg_ok / neg_n
        print(f"  TPR {100*tpr:.1f}%   TNR {100*tnr:.1f}%")
        # Balanced accuracy, so a judge cannot look good by always saying one thing.
        print(f"  balanced accuracy {50*(tpr+tnr):.1f}%")
        if tnr < 0.5:
            print("\n  WARNING: TNR below 50%. This judge is closer to a rubber "
                  "stamp than a\n  measurement -- content numbers scored with it "
                  "should not be reported.")
        if tpr < 0.8:
            print("\n  WARNING: TPR below 80%. This judge rejects genuine answers "
                  "and will\n  understate every system it grades.")

    if args.save and rows:
        p = Path(args.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  per-probe detail -> {p}")


if __name__ == "__main__":
    main()
