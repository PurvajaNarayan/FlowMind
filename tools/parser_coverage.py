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


def resolve_all(graph, label: str) -> list[str]:
    """Map a label from a question to EVERY node id it could mean.

    Returning a list rather than one id matters: charts routinely contain two
    nodes labelled "End" (also seen: duplicate process steps), and picking the
    first arbitrarily was wrong about half the time -- ambiguous adjacency
    questions scored 50% and 33% against 99.5%+ for unambiguous ones. Callers
    resolve the ambiguity with the semantics of their own question instead:
    "is X a predecessor of End" is true if X precedes ANY node labelled End.
    """
    lab = label.strip().rstrip(".").strip().lower()
    exact = [n.id for n in graph.nodes if n.label.strip().rstrip(".").strip().lower() == lab]
    if exact:
        return exact
    return [n.id for n in graph.find_by_label(label.rstrip("."))]


def resolve(graph, label: str) -> str | None:
    """First candidate only. Kept for callers that genuinely want one id."""
    hits = resolve_all(graph, label)
    return hits[0] if hits else None


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
            # Degree questions before edge_count: they mention "incoming/outgoing
            # edges" in their preamble, and FlowVQA writes them as one word
            # ("indegree"/"outdegree"), not "in-degree".
            elif "outdegree" in Q or "out-degree" in Q or "out degree" in Q:
                kind, pred = "max_outdegree", gt.max_outdegree(g)
            elif "indegree" in Q or "in-degree" in Q or "in degree" in Q:
                kind, pred = "max_indegree", gt.max_indegree(g)
            elif "how many edges" in Q and "shortest path" not in Q:
                kind, pred = "edge_count", gt.edge_count(g)
            elif "shortest path" in Q and len(labs) >= 2:
                kind = "shortest_path"
                A, B = resolve_all(g, labs[0]), resolve_all(g, labs[1])
                if not (A and B):
                    unresolved += 1
                    continue
                # Ambiguous label -> the shortest route between any pair of
                # candidates, which is what "the shortest path" asks for.
                lengths = [d for a in A for b in B
                           if (d := gt.shortest_path_edges(g, a, b)) is not None]
                pred = min(lengths) if lengths else None
            elif "predecessor" in Q and len(labs) >= 2:
                kind = "direct_predecessor"
                A, B = resolve_all(g, labs[0]), resolve_all(g, labs[1])
                if not (A and B):
                    unresolved += 1
                    continue
                # "Is X a direct predecessor of End" holds if X precedes ANY node
                # labelled End, so quantify existentially over the candidates.
                pred = "Yes" if any(gt.is_direct_predecessor(g, a, b)
                                    for a in A for b in B) else "No"
            elif "successor" in Q and len(labs) >= 2:
                kind = "direct_successor"
                A, B = resolve_all(g, labs[0]), resolve_all(g, labs[1])
                if not (A and B):
                    unresolved += 1
                    continue
                pred = "Yes" if any(gt.is_direct_successor(g, a, b)
                                    for a in A for b in B) else "No"

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
