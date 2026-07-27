"""VLM Reader (stretch, spec §7.1 / §9): flowchart IMAGE -> Mermaid -> FlowGraph.

Design decision (locked with the team): the VLM emits *Mermaid text*, which we
feed into the already-tested `mermaid_to_graph()`. This reuses the parser, gives
free ground-truth supervision (the dataset's own `mermaid` field), and makes the
VLM a drop-in Reader backend — downstream code can't tell it apart from text.

Local inference target: Qwen3-VL-2B / 4B on Apple Silicon via MLX-VLM or Ollama.

Owner A checklist (see the plan in the project notes):
  0. De-risk data: confirm FlowVQA ships images; if not, render from `mermaid`
     with mermaid-cli (`mmdc`). See scripts/render_images.py (TODO).
  1. Zero-shot: prompt Qwen3-VL for Mermaid, parse, measure graph-extraction
     accuracy vs the ground-truth mermaid (spec §8). This gates fine-tuning.
  2. If zero-shot is weak, LoRA fine-tune on image->mermaid pairs (cloud GPU).
"""

from __future__ import annotations

from flowmind.reader.mermaid_reader import mermaid_to_graph
from flowmind.schema import FlowGraph

PROMPT = (
    "You are reading a flowchart image. Output ONLY the Mermaid.js flowchart "
    "script that reproduces it, starting with `flowchart TD`. Use ([...]) for "
    "start/end, [/.../] for input/output, [...] for process, {...} for decision, "
    "and -->|label| for labeled edges. No prose, no code fences."
)


def image_to_mermaid(image_path: str, model: str = "qwen3-vl:4b") -> str:
    """Run Qwen3-VL on the image and return raw Mermaid text.

    TODO(A): implement via MLX-VLM or Ollama. Sketch:

        from mlx_vlm import load, generate
        model_, processor = load("mlx-community/Qwen3-VL-4B-Instruct-4bit")
        return generate(model_, processor, PROMPT, image=image_path).strip()
    """
    raise NotImplementedError("VLM backend not wired up yet — see checklist above.")


def image_to_graph(image_path: str, model: str = "qwen3-vl:4b") -> FlowGraph:
    """The Reader interface: image -> FlowGraph (source='vlm')."""
    mermaid = image_to_mermaid(image_path, model=model)
    graph = mermaid_to_graph(mermaid)
    graph.source = "vlm"
    return graph
