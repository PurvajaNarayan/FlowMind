"""Zero-shot VLM extraction eval (spec §8 graph-extraction accuracy). [Owner: A]

Runs the Qwen3-VL Reader on a sample of flowchart images and compares the graph
it produces against the graph parsed from the ground-truth `mermaid` field.
This is the number that decides whether zero-shot is good enough or a LoRA
fine-tune is needed.

    pip install -r requirements-vlm.txt && python tools/download_vlm.py
    python tools/eval_vlm.py --n 30
    python tools/eval_vlm.py --n 30 --layout main --save runs/vlm_zeroshot.jsonl

Each record's VLM Mermaid output is saved (with --save) so it can double as
image->mermaid training pairs for fine-tuning later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowmind.data import image_path_for
from flowmind.eval.metrics import graph_extraction_accuracy
from flowmind.reader.mermaid_reader import mermaid_to_graph


def stratified_keys(ds: dict, n: int) -> list[str]:
    """Take up to n keys, spread across the code/wiki/instruct subsets."""
    from collections import defaultdict

    buckets = defaultdict(list)
    for k in ds:
        buckets[k.rstrip("0123456789")].append(k)
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in buckets.values()):
        for v in buckets.values():
            if i < len(v) and len(out) < n:
                out.append(v[i])
        i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_full.json")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--layout", default="main", choices=["main", "bottom_top"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--save", default=None, help="JSONL path to dump per-sample results")
    args = ap.parse_args()

    from flowmind.reader.vlm_reader import QwenVLExtractor

    ds = json.load(open(args.data))
    keys = stratified_keys(ds, args.n)
    ext = QwenVLExtractor()
    # Stated up front and stored per row: which prompt produced a run is the
    # difference between two runs being comparable and being confusing later.
    print(f"model {ext.model_id} | prompt {ext.prompt_version} | "
          f"pixel cap {ext.max_pixels / 1e6:.2f} MP")

    node_hits = edge_hits = both = total = 0
    errors = downscaled = 0
    rows = []
    for i, key in enumerate(keys, 1):
        img = image_path_for(key, args.data_dir, args.layout)
        if not img.exists():
            print(f"[{i}/{len(keys)}] {key}: image missing, skipping")
            continue
        gold_mermaid = ds[key]["mermaid"]
        try:
            pred_mermaid = ext.image_to_mermaid(str(img))
        except Exception as e:  # keep the sweep going on a single bad decode
            # CUDA OOM messages are a paragraph long; the first line is the useful bit.
            print(f"[{i}/{len(keys)}] {key}: ERROR {str(e).splitlines()[0][:120]}")
            errors += 1
            continue
        scale = getattr(ext, "last_scale", 1.0)
        pred_graph = mermaid_to_graph(pred_mermaid)
        m = graph_extraction_accuracy(pred_graph, gold_mermaid)
        total += 1
        node_hits += m["node_count_match"]
        edge_hits += m["edge_count_match"]
        both += m["node_count_match"] and m["edge_count_match"]
        if scale < 1.0:
            downscaled += 1
        print(f"[{i}/{len(keys)}] {key}: "
              f"nodes {m['pred_nodes']}/{m['gold_nodes']} "
              f"{'OK' if m['node_count_match'] else 'X'} | "
              f"edges {m['pred_edges']}/{m['gold_edges']} "
              f"{'OK' if m['edge_count_match'] else 'X'}"
              + (f" | resized x{scale:.2f}" if scale < 1.0 else ""))
        # `scale` matters for error analysis: a downscaled chart may have failed
        # because the resize made its labels unreadable, not because the model
        # cannot read flowcharts.
        rows.append({"key": key, "layout": args.layout, "scale": scale,
                     "prompt_version": ext.prompt_version, "model_id": ext.model_id,
                     "pred_mermaid": pred_mermaid, "gold_mermaid": gold_mermaid, **m})

    if total:
        print(f"\n=== zero-shot graph-extraction over {total} samples ===")
        print(f"  node-count match: {node_hits}/{total} ({100*node_hits/total:.1f}%)")
        print(f"  edge-count match: {edge_hits}/{total} ({100*edge_hits/total:.1f}%)")
        print(f"  both match:       {both}/{total} ({100*both/total:.1f}%)")
        print(f"  downscaled:       {downscaled}/{total} "
              f"(resized to fit the pixel cap — see vlm_reader docstring)")
    if errors:
        print(f"  FAILED outright:  {errors} (not counted above — report this "
              f"alongside the percentages, they are not accuracy on the full sample)")

    if args.save and rows:
        p = Path(args.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"saved {len(rows)} rows -> {p}")


if __name__ == "__main__":
    main()
