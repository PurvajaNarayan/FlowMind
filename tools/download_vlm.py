"""One-time download of the Qwen3-VL weights from Hugging Face. [Owner: A]

    pip install -r requirements-vlm.txt
    python tools/download_vlm.py                       # default model
    python tools/download_vlm.py --model Qwen/Qwen3-VL-2B-Instruct

Weights land in the HF cache (~/.cache/huggingface). For a gated repo, run
`huggingface-cli login` first. The model id here must match DEFAULT_MODEL_ID in
flowmind/reader/vlm_reader.py (or set FLOWMIND_VLM_MODEL).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowmind.reader.vlm_reader import DEFAULT_MODEL_ID
from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    args = ap.parse_args()

    print(f"downloading {args.model} ... (several GB, one time)")
    path = snapshot_download(repo_id=args.model)
    print(f"done -> {path}")


if __name__ == "__main__":
    main()
