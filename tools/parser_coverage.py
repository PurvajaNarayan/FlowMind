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
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from flowmind.reader.mermaid_reader import mermaid_to_graph
from flowmind import graph_tool as gt


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def labels_in(q: str) -> list[str]:
    # FlowVQA quotes node labels with doubled double-quotes: ""like this.""
    doubled = re.findall(r'""(.*?)""', q)
    return [m.strip() for m in doubled] if doubled else re.findall(r'"([^"]+)"', q)


def resolve(graph, label: str) -> str | None:
    """Map a label string from a question back to a node id. Exact (period- and
    case-insensitive) match first, then loose substring."""
    lab = label.strip().rstrip(".").strip().lower()
    exact = [n.id for n in graph.nodes if n.label.strip().rstrip(".").strip().lower() == lab]
    if exact:
        return exact[0]
    hits = graph.find_by_label(label.rstrip("."))
    return hits[0].id if hits else None


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
            Q, gold = norm(q["Q"]), str(q["A1"]).strip()
            labs = labels_in(q["Q"])
            kind = pred = None

            if "how many nodes" in Q:
                kind, pred = "node_count", gt.node_count(g)
            elif "how many edges" in Q and "shortest path" not in Q:
                kind, pred = "edge_count", gt.edge_count(g)
            elif "in-degree" in Q or "in degree" in Q:
                kind, pred = "max_indegree", gt.max_indegree(g)
            elif "shortest path" in Q and len(labs) >= 2:
                kind = "shortest_path"
                a, b = resolve(g, labs[0]), resolve(g, labs[1])
                if not (a and b):
                    unresolved += 1
                    continue
                pred = gt.shortest_path_edges(g, a, b)
            elif "predecessor" in Q and len(labs) >= 2:
                kind = "direct_predecessor"
                a, b = resolve(g, labs[0]), resolve(g, labs[1])
                if not (a and b):
                    unresolved += 1
                    continue
                pred = "Yes" if gt.is_direct_predecessor(g, a, b) else "No"
            elif "successor" in Q and len(labs) >= 2:
                kind = "direct_successor"
                a, b = resolve(g, labs[0]), resolve(g, labs[1])
                if not (a and b):
                    unresolved += 1
                    continue
                pred = "Yes" if gt.is_direct_successor(g, a, b) else "No"

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
