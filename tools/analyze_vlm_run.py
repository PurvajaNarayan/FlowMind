"""Structural analysis of a saved VLM extraction run. [Owner: A]

`tools/eval_vlm.py --save` reports node/edge COUNT match, which is a weak proxy:
a prediction can have the right counts and the wrong graph. Observed on
code00340 -- counts were off by one (17/16 nodes, 20/19 edges), which reads as
"nearly right", while a third of the edges were wrong and both loops were gone.

This script re-scores a saved run without touching the GPU, because eval_vlm
stores the predicted Mermaid text per sample. Run it after any sweep:

    python tools/analyze_vlm_run.py runs/vlm_zeroshot_main.jsonl
    python tools/analyze_vlm_run.py runs/vlm_zeroshot_main.jsonl --per-sample

What it measures, and why each one earns its place:

  label recall      fraction of gold node labels the model reproduced. This is
                    OCR/text fidelity, measured separately because it is the
                    part the model is good at and it should not be hidden inside
                    a structural score.
  shape accuracy    spec 7.1 requires all four Mermaid shapes. Counts say
                    nothing about shapes, and a run can score 50% on nodes while
                    emitting every node as a rectangle.
  edge P/R/F1       the real structural measure. Node ids are arbitrary, so edges
                    are compared as (source_label -> target_label) after
                    normalisation, not by id.
  cycle recall      flowcharts loop; a model that linearises a loop into a
                    straight line has misread the control flow entirely. This is
                    the single most diagnostic number here.
  scale split       metrics bucketed by the resize factor eval_vlm recorded,
                    which is what tells you whether a bigger GPU would help or
                    whether resolution is a red herring.

NOTE: the graph-comparison helpers here arguably belong in flowmind/eval/metrics.py,
which is Owner C's file. Kept in tools/ to avoid a cross-lane edit; move them with
C's sign-off.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import networkx as nx

from flowmind.reader.mermaid_reader import mermaid_to_graph
from flowmind.schema import FlowGraph

# A prediction with wildly more edges than gold is a degenerate generation (the
# model repeating itself until max_new_tokens), not a misread. code00521 produced
# 170 edges against 22. Averaging those in would corrupt every aggregate, so they
# are excluded from the metrics and reported separately.
DEGENERATE_EDGE_RATIO = 3.0
CYCLE_SCAN_LIMIT = 200      # simple_cycles can blow up; we only need "any / how many"


def norm(s: str) -> str:
    """Labels come from OCR, so compare them whitespace- and case-insensitively."""
    return re.sub(r"\s+", " ", s).strip().strip('."').lower()


def _labels(g: FlowGraph) -> dict[str, str]:
    return {n.id: norm(n.label) for n in g.nodes}


def edge_set(g: FlowGraph, with_labels: bool = False) -> set[tuple]:
    """Edges keyed by node LABEL, since ids are arbitrary between pred and gold."""
    lab = _labels(g)
    if with_labels:
        return {(lab[e.source], lab[e.target], norm(e.label or "")) for e in g.edges
                if e.source in lab and e.target in lab}
    return {(lab[e.source], lab[e.target]) for e in g.edges
            if e.source in lab and e.target in lab}


def count_cycles(g: FlowGraph) -> int:
    try:
        return len(list(islice(nx.simple_cycles(g.to_networkx()), CYCLE_SCAN_LIMIT)))
    except Exception:
        return 0


def prf(pred: set, gold: set) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    inter = len(pred & gold)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def analyze_row(row: dict) -> dict:
    P = mermaid_to_graph(row["pred_mermaid"])
    G = mermaid_to_graph(row["gold_mermaid"])

    gold_labels = set(_labels(G).values())
    pred_labels = set(_labels(P).values())
    label_recall = len(gold_labels & pred_labels) / len(gold_labels) if gold_labels else 0.0

    # Shape accuracy over nodes we can align by label. Coverage is reported too:
    # shape accuracy over 3 aligned nodes out of 30 would be meaningless alone.
    pred_shape = {norm(n.label): n.shape.value for n in P.nodes}
    aligned = correct = 0
    for n in G.nodes:
        key = norm(n.label)
        if key in pred_shape:
            aligned += 1
            correct += pred_shape[key] == n.shape.value

    ep, eg = edge_set(P), edge_set(G)
    p, r, f = prf(ep, eg)
    lp, lr, lf = prf(edge_set(P, True), edge_set(G, True))

    gold_cycles, pred_cycles = count_cycles(G), count_cycles(P)
    ratio = (len(P.edges) / len(G.edges)) if G.edges else float("inf")

    return {
        "key": row.get("key"),
        "subset": (row.get("key") or "").rstrip("0123456789"),
        "scale": row.get("scale", 1.0),
        "degenerate": ratio > DEGENERATE_EDGE_RATIO,
        "parse_failed": len(P.nodes) == 0,
        "label_recall": label_recall,
        "shape_aligned": aligned,
        "shape_correct": correct,
        "shape_total_gold": len(G.nodes),
        "edge_p": p, "edge_r": r, "edge_f1": f,
        "labeled_edge_f1": lf,
        "gold_cycles": gold_cycles, "pred_cycles": pred_cycles,
        "pred_shapes": Counter(n.shape.value for n in P.nodes),
        "gold_shapes": Counter(n.shape.value for n in G.nodes),
    }


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _bucket_report(rows: list[dict], label: str) -> None:
    if not rows:
        return
    cyc = [r for r in rows if r["gold_cycles"] > 0]
    cyc_rec = mean(r["pred_cycles"] > 0 for r in cyc) if cyc else float("nan")
    print(f"  {label:<28} n={len(rows):<3} "
          f"edge-F1 {mean(r['edge_f1'] for r in rows):.3f}   "
          f"label-recall {mean(r['label_recall'] for r in rows):.3f}   "
          f"cycle-recall {cyc_rec:.2f} ({len(cyc)} w/ loops)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="output of eval_vlm.py --save")
    ap.add_argument("--per-sample", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.jsonl).open(encoding="utf-8") if l.strip()]
    res = [analyze_row(r) for r in rows]

    degenerate = [r for r in res if r["degenerate"]]
    failed = [r for r in res if r["parse_failed"]]
    good = [r for r in res if not r["degenerate"] and not r["parse_failed"]]

    print(f"=== {args.jsonl}: {len(res)} samples "
          f"({len(good)} scored, {len(degenerate)} degenerate, {len(failed)} unparseable) ===\n")

    if args.per_sample:
        print(f"{'key':<16}{'scale':>6}{'edgeF1':>8}{'labelRec':>10}"
              f"{'shapes':>9}{'cyc g/p':>9}")
        for r in sorted(res, key=lambda r: r["edge_f1"]):
            sh = (f"{r['shape_correct']}/{r['shape_aligned']}"
                  if r["shape_aligned"] else "-")
            flag = "  DEGENERATE" if r["degenerate"] else ""
            print(f"{r['key']:<16}{r['scale']:>6.2f}{r['edge_f1']:>8.3f}"
                  f"{r['label_recall']:>10.3f}{sh:>9}"
                  f"{str(r['gold_cycles'])+'/'+str(r['pred_cycles']):>9}{flag}")
        print()

    if not good:
        print("nothing scoreable")
        return

    print("--- text fidelity ---")
    print(f"  node label recall          {mean(r['label_recall'] for r in good):.3f}")

    print("\n--- shapes (spec 7.1 requires all four) ---")
    al = sum(r["shape_aligned"] for r in good)
    co = sum(r["shape_correct"] for r in good)
    tg = sum(r["shape_total_gold"] for r in good)
    print(f"  correct shape, over label-aligned nodes   {co}/{al} "
          f"({100*co/al:.1f}%)" if al else "  no aligned nodes")
    print(f"  alignment coverage                        {al}/{tg} gold nodes")
    ps, gs = Counter(), Counter()
    for r in good:
        ps.update(r["pred_shapes"]); gs.update(r["gold_shapes"])
    print(f"  gold shape mix   {dict(gs)}")
    print(f"  pred shape mix   {dict(ps)}")

    print("\n--- structure (the real metric) ---")
    print(f"  edge precision  {mean(r['edge_p'] for r in good):.3f}")
    print(f"  edge recall     {mean(r['edge_r'] for r in good):.3f}")
    print(f"  edge F1         {mean(r['edge_f1'] for r in good):.3f}")
    print(f"  edge F1 incl. Yes/No labels  {mean(r['labeled_edge_f1'] for r in good):.3f}")

    print("\n--- control flow ---")
    withloops = [r for r in good if r["gold_cycles"] > 0]
    if withloops:
        rec = mean(r["pred_cycles"] > 0 for r in withloops)
        print(f"  gold charts containing a loop      {len(withloops)}/{len(good)}")
        print(f"  ... where prediction had any loop  "
              f"{sum(r['pred_cycles'] > 0 for r in withloops)}/{len(withloops)} "
              f"({100*rec:.1f}%)")
        print(f"  total cycles  gold {sum(r['gold_cycles'] for r in withloops)}  "
              f"pred {sum(r['pred_cycles'] for r in withloops)}")
    else:
        print("  no gold chart in this run contains a loop")

    print("\n--- does resolution matter? (the bigger-GPU question) ---")
    _bucket_report([r for r in good if r["scale"] >= 1.0], "full resolution")
    _bucket_report([r for r in good if 0.6 < r["scale"] < 1.0], "mildly downscaled")
    _bucket_report([r for r in good if r["scale"] <= 0.6], "heavily downscaled (<=0.6x)")

    print("\n--- by subset ---")
    for s in sorted({r["subset"] for r in good}):
        _bucket_report([r for r in good if r["subset"] == s], s)

    if degenerate:
        print("\n--- degenerate generations (excluded above) ---")
        for r in degenerate:
            print(f"  {r['key']}: predicted {sum(r['pred_shapes'].values())} nodes, "
                  f"likely repeated until max_new_tokens")


if __name__ == "__main__":
    main()
