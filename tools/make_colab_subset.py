"""Build a small, upload-sized slice of FlowVQA for the Colab VLM run. [Owner: A]

The full image set is ~1.6GB across both layouts, which is a slow one-time upload
to Drive for no benefit: the zero-shot sweep in notebooks/flowmind_colab.ipynb runs
at --n 20-30. A 100-key slice is ~57MB and answers the same question.

    python tools/make_colab_subset.py                       # 100 keys -> ~/Desktop
    python tools/make_colab_subset.py --n 300 --out /tmp     # bigger slice

Upload the resulting zip to MyDrive/FlowMind/ and run cell 0.5b in the notebook;
it unzips into the layout flowmind.data expects.

IMPORTANT: keys are chosen with tools.eval_vlm.stratified_keys, the same function
eval_vlm.py uses to pick what to evaluate. That function is round-robin by index,
so its n=20 selection is a strict prefix of its n=100 selection -- meaning any
--n up to the slice size hits a staged image. Picking keys any other way (e.g.
random.sample) would leave eval_vlm.py printing "image missing, skipping" for
every record.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from tools.eval_vlm import stratified_keys

LAYOUTS = ("main", "bottom_top")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_full.json")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=100, help="number of flowchart keys")
    ap.add_argument("--out", default=str(Path.home() / "Desktop"),
                    help="directory to write the zip into")
    ap.add_argument("--name", default="flowvqa_colab_subset.zip")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ds = json.load(open(args.data))
    keys = stratified_keys(ds, args.n)

    # The prefix property is what makes `eval_vlm.py --n <smaller>` work against
    # this slice. Fail loudly if a future change to stratified_keys breaks it.
    if stratified_keys(ds, min(20, args.n)) != keys[: min(20, args.n)]:
        sys.exit("stratified_keys is no longer prefix-stable; eval_vlm --n would "
                 "select unstaged keys. Fix before relying on this subset.")

    print(f"{len(keys)} keys, spread: "
          f"{dict(Counter(k.rstrip('0123456789') for k in keys))}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / args.name

    copied, missing = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for layout in LAYOUTS:
            (stage / "images" / layout).mkdir(parents=True)
            for key in keys:
                src = data_dir / "images" / layout / f"{key}.png"
                if src.exists():
                    shutil.copy2(src, stage / "images" / layout / f"{key}.png")
                    copied += 1
                else:
                    missing.append(str(src))

        # Both splits: train drives every section, test is needed by train_router.py.
        for name in ("train_full.json", "test_full.json"):
            src = data_dir / name
            if src.exists():
                shutil.copy2(src, stage / name)
            else:
                missing.append(str(src))

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(stage))

    print(f"copied {copied} images across {len(LAYOUTS)} layouts")
    if missing:
        print(f"WARNING: {len(missing)} missing, e.g. {missing[:3]}")
    print(f"wrote {zip_path}  ({zip_path.stat().st_size / 1e6:.0f} MB)")
    print(f"\nNext: upload it to MyDrive/FlowMind/ and run cell 0.5b in "
          f"notebooks/flowmind_colab.ipynb")


if __name__ == "__main__":
    main()
