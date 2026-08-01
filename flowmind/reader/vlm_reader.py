"""VLM Reader (stretch, spec §7.1 / §9): flowchart IMAGE -> Mermaid -> FlowGraph.

Design decision (locked with the team): the VLM emits *Mermaid text*, which we
feed into the already-tested `mermaid_to_graph()`. This reuses the parser, gives
free ground-truth supervision (the dataset's own `mermaid` field), and makes the
VLM a drop-in Reader backend — downstream code can't tell it apart from text.

Backend: Qwen3-VL via Hugging Face `transformers` (fine-tune-ready). Runs on CUDA
(Colab T4) or Apple MPS; LoRA fine-tuning happens later on a cloud GPU.

    pip install -r requirements-vlm.txt
    python tools/download_vlm.py            # one-time model download

Usage:
    from flowmind.reader.vlm_reader import QwenVLExtractor
    ext = QwenVLExtractor()                 # loads the model once (lazy)
    graph = ext.image_to_graph("data/images/main/code00453.png")

RESOLUTION / MEMORY (read this before changing max_pixels)
---------------------------------------------------------
FlowVQA images are a fixed 1568px wide and grow *downwards* — a long wiki chart
reaches 12038px tall (18.9 MP). Qwen3-VL uses dynamic resolution with a spatial
compression ratio of 32, so visual tokens ~= pixels / (32*32):

    2.5 - 4.9 MP  ->  2400 - 4800 tokens  ->  runs fine on a 15GB T4
    7.6 - 18.9 MP ->  7400 - 18400 tokens ->  CUDA OOM (asks for 4-16 GiB)

Measured on a T4: everything at or below ~4.9 MP succeeded, everything at or
above 7.6 MP died. So we cap total pixels and downscale anything above it,
preserving aspect ratio. The cap is deliberately set just above the observed
safe ceiling rather than at Qwen's suggested 1280 tokens (1.3 MP), because these
images are dense text and over-shrinking makes node labels unreadable — and
label text is the one thing the model currently gets right.

Downscaling is lossy for the tallest charts, so `last_scale` records the factor
actually applied. Correlate it with accuracy before concluding that a failure is
the model misreading rather than the resize destroying the text.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from flowmind.reader.mermaid_reader import mermaid_to_graph
from flowmind.schema import FlowGraph

# Override via QwenVLExtractor(model_id=...) or the FLOWMIND_VLM_MODEL env var.
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# Qwen3-VL compresses by 32, so this is ~4800 visual tokens: just above the
# largest image observed to succeed on a T4. Override with FLOWMIND_VLM_MAX_PIXELS.
DEFAULT_MAX_PIXELS = 4800 * 32 * 32

# Never shrink below this on an OOM retry; past here the labels are gone anyway
# and a failure is more honest than a confidently-wrong graph.
MIN_PIXELS_FLOOR = 512 * 32 * 32

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


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _fit_pixels(img, max_pixels: int):
    """Downscale `img` so width*height <= max_pixels, preserving aspect ratio.

    Returns (image, scale). scale == 1.0 means untouched.
    """
    from PIL import Image

    w, h = img.size
    total = w * h
    if total <= max_pixels:
        return img, 1.0
    scale = (max_pixels / total) ** 0.5
    new = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new, Image.LANCZOS), scale


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, getattr(torch, "OutOfMemoryError", ())):
        return True
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class QwenVLExtractor:
    """Lazy-loaded Qwen3-VL wrapper. Instantiate once and reuse — model load is
    the expensive part.

    Heavy imports (torch, transformers, PIL) are deliberately kept inside methods
    so this module can be imported without requirements-vlm.txt installed —
    tools/download_vlm.py only needs DEFAULT_MODEL_ID, and the test suite should
    not need a 3GB torch wheel to collect.
    """

    def __init__(self, model_id: str | None = None, device: str | None = None,
                 max_new_tokens: int = 1024, max_pixels: int | None = None,
                 oom_retries: int = 2):
        self.model_id = model_id or os.environ.get("FLOWMIND_VLM_MODEL", DEFAULT_MODEL_ID)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_pixels = int(
            max_pixels or os.environ.get("FLOWMIND_VLM_MAX_PIXELS", DEFAULT_MAX_PIXELS)
        )
        self.oom_retries = oom_retries
        self.last_scale = 1.0      # resize factor applied to the last image
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Fragmentation is the difference between fitting and not fitting when
        # image sizes vary this much across a sweep. Must be set before the CUDA
        # context is created, i.e. before .to(device) below.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        self.device = self.device or _pick_device()
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=dtype     # `torch_dtype` is deprecated in transformers 5.x
        ).to(self.device).eval()

    def _free(self) -> None:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(self, image_path: str, max_pixels: int) -> str:
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        img, self.last_scale = _fit_pixels(img, max_pixels)

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

        try:
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                           do_sample=False)
            # Drop the prompt tokens; decode only the generated continuation.
            gen = out[0][inputs["input_ids"].shape[1]:]
            return _clean_mermaid(self._processor.decode(gen, skip_special_tokens=True))
        finally:
            # Without this, reserved-but-unallocated memory accumulates across a
            # sweep and later samples OOM even though earlier ones fit.
            del inputs
            self._free()

    def image_to_mermaid(self, image_path: str) -> str:
        """Run the VLM on one image and return cleaned Mermaid text.

        On CUDA OOM, halves the pixel cap and retries (up to `oom_retries`), so a
        single oversized chart degrades in quality instead of killing the sweep.
        """
        self._ensure_loaded()
        cap = self.max_pixels
        last_exc: BaseException | None = None

        for attempt in range(self.oom_retries + 1):
            try:
                return self._generate(image_path, cap)
            except Exception as exc:            # noqa: BLE001 - re-raised below
                if not _is_oom(exc) or cap <= MIN_PIXELS_FLOOR:
                    raise
                last_exc = exc
                self._free()
                cap = max(cap // 2, MIN_PIXELS_FLOOR)
                print(f"    OOM on attempt {attempt + 1}; retrying at "
                      f"{cap / 1e6:.1f} MP cap")

        raise RuntimeError(
            f"OOM after {self.oom_retries + 1} attempts down to {cap / 1e6:.1f} MP"
        ) from last_exc

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
