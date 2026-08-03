"""M1 coverage harness: validate the Mermaid parser + graph tool against
FlowVQA's own topological gold answers. [Owner: A]

This is the spec §8 "deterministic lane" metric. Re-run it whenever the parser
changes:

    python tools/parser_coverage.py data/train_full.json
    python tools/parser_coverage.py data/train_full.json --show-fails direct_successor

Note: shortest-path / predecessor / successor questions reference nodes by their
LABEL text, so this harness also exercises label resolution (a Reader concern).
Questions whose labels can't be resolved are reported separately, not counted as
wrong — that residual belongs to label matching, not the graph tool.

The actual question-dispatch logic (which subtype, which graph_tool function,
which node ids) lives in flowmind.graph_tool.answer_topological, shared with
eval/ablation.py's full_pipeline so both use one implementation.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from flowmind.graph_tool import answer_topological
from flowmind.reader.mermaid_reader import mermaid_to_graph


def evaluate(path: str, show_fails: str | None = None):
    ds = json.load(open(path))
    stats = defaultdict(lambda: [0, 0])   # kind -> [ok, total]
    fails = defaultdict(list)
    unresolved = 0

    for key, rec in ds.items():
        g = mermaid_to_graph(rec["mermaid"])
        for q in rec.get("qa", {}).values():
            if q.get("type") != "topological":
                continue
            gold = str(q["A1"]).strip()
            kind, pred, is_unresolved = answer_topological(g, q["Q"])

            if is_unresolved:
                unresolved += 1
                continue
            if kind is None or pred is None:
                continue
            ok = str(pred).strip().lower() == gold.lower()
            stats[kind][0] += ok
            stats[kind][1] += 1
            if not ok:
                fails[kind].append((key, str(pred), gold, q["Q"][:70]))

    print(f"{'subtype':<22}{'acc':>8}   ok/total")
    tot_ok = tot = 0
    for k, (ok, t) in sorted(stats.items()):
        tot_ok += ok
        tot += t
        print(f"{k:<22}{100*ok/t:>7.1f}%   {ok}/{t}")
    print("-" * 45)
    print(f"{'OVERALL':<22}{100*tot_ok/tot:>7.1f}%   {tot_ok}/{tot}")
    print(f"(label-unresolved questions skipped: {unresolved})")

    if show_fails and show_fails in fails:
        print(f"\n--- fails: {show_fails} ---")
        for key, pred, gold, q in fails[show_fails][:25]:
            print(f"  {key}: pred={pred!r} gold={gold!r}  | {q}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/train_full.json"
    sf = sys.argv[sys.argv.index("--show-fails") + 1] if "--show-fails" in sys.argv else None
    evaluate(path, sf)
