"""VLM Reader (stretch, spec §7.1 / §9): flowchart IMAGE -> Mermaid -> FlowGraph.

Design decision (locked with the team): the VLM emits *Mermaid text*, which we
feed into the already-tested `mermaid_to_graph()`. This reuses the parser, gives
free ground-truth supervision (the dataset's own `mermaid` field), and makes the
VLM a drop-in Reader backend — downstream code can't tell it apart from text.

Backend: Qwen3-VL via Hugging Face `transformers` (fine-tune-ready). On Apple
Silicon it runs on the MPS device for inference; LoRA fine-tuning happens later
on a cloud GPU (see requirements-vlm.txt).

    pip install -r requirements-vlm.txt
    python tools/download_vlm.py            # one-time model download

Usage:
    from flowmind.reader.vlm_reader import QwenVLExtractor
    ext = QwenVLExtractor()                 # loads the model once (lazy)
    graph = ext.image_to_graph("data/images/main/code00453.png")
"""

from __future__ import annotations

import re
from functools import lru_cache

from flowmind.reader.mermaid_reader import mermaid_to_graph
from flowmind.schema import FlowGraph
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from PIL import Image

# Override via QwenVLExtractor(model_id=...) or the FLOWMIND_VLM_MODEL env var.
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

PROMPT = (
    "You are reading a flowchart image. Output ONLY the Mermaid.js flowchart "
    "script that reproduces it, starting with `flowchart TD`. Use ([...]) for "
    "start/end, [/.../] for input/output, [...] for process, {...} for decision, "
    "and -->|label| for labeled edges. No prose, no explanation, no code fences."
)

_FENCE_RE = re.compile(r"^```(?:mermaid)?\s*|\s*```$", re.MULTILINE)


def _clean_mermaid(text: str) -> str:
    """Strip markdown fences / stray prose the model may wrap around the script."""
    text = _FENCE_RE.sub("", text).strip()
    # Keep from the first `flowchart`/`graph` directive onward if present.
    m = re.search(r"(flowchart|graph)\s+\w+", text)
    return text[m.start():].strip() if m else text


def _pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class QwenVLExtractor:
    """Lazy-loaded Qwen3-VL wrapper. Instantiate once and reuse — model load is
    the expensive part."""

    def __init__(self, model_id: str | None = None, device: str | None = None,
                 max_new_tokens: int = 1024):
        import os

        self.model_id = model_id or os.environ.get("FLOWMIND_VLM_MODEL", DEFAULT_MODEL_ID)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        

        self.device = self.device or _pick_device()
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self.device).eval()

    def image_to_mermaid(self, image_path: str) -> str:
        """Run the VLM on one image and return cleaned Mermaid text."""
        

        self._ensure_loaded()
        img = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       do_sample=False)
        # Drop the prompt tokens; decode only the generated continuation.
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self._processor.decode(gen, skip_special_tokens=True)
        return _clean_mermaid(text)

    def image_to_graph(self, image_path: str) -> FlowGraph:
        """The Reader interface: image -> FlowGraph (source='vlm')."""
        graph = mermaid_to_graph(self.image_to_mermaid(image_path))
        graph.source = "vlm"
        return graph


@lru_cache(maxsize=1)
def _default_extractor() -> QwenVLExtractor:
    return QwenVLExtractor()


def image_to_mermaid(image_path: str) -> str:
    """Convenience wrapper using a shared, cached extractor."""
    return _default_extractor().image_to_mermaid(image_path)


def image_to_graph(image_path: str) -> FlowGraph:
    """Convenience wrapper using a shared, cached extractor."""
    return _default_extractor().image_to_graph(image_path)
