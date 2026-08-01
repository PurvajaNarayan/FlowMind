"""Shared LLM seam. [infrastructure — created by A, used by B and C]

Every LLM-using component (Examiner §7.3, Planner §7.4, single-pass baseline §8,
and an LLM judge if §9 goes that way) talks to this interface and nothing else.
Swapping backends is then a one-line change here rather than an edit in three
workstreams, and tests can run without a GPU.

Backend choice: **Qwen3-8B**, Apache 2.0, open weights from the HF hub.
  - Apache 2.0 has no conditions to justify in the write-up, unlike Llama's and
    Gemma's custom licences.
  - Same family as the Qwen3-VL Reader already in the repo, so the transformers
    code path, chat-template handling and weight cache are shared.
  - ~5GB at 4-bit on a 15GB T4, leaving headroom. Qwen3-8B is reported to match
    Qwen2.5-14B, which matters most for the Planner's code generation.

    export FLOWMIND_LLM_MODEL=Qwen/Qwen3-4B   # fp16 fallback, ~8GB, no bitsandbytes
    export FLOWMIND_LLM_BACKEND=scripted      # tests: no torch, no weights

Note on comparability: an 8B open model will score below the published GPT-4
(68.42%) and TextFlow (~82.7%) figures. That is fine for the §8 ablation, whose
claim is a *delta* between single-pass and pipeline with the model held constant.
It does mean absolute numbers should not be compared to those baselines.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"

# Greedy by default: the deterministic-lane work showed how much easier analysis
# is when a re-run reproduces exactly (the VLM sweeps were bit-identical), and it
# removes seed variance from the ablation delta.
DEFAULT_MAX_NEW_TOKENS = 512


@runtime_checkable
class LLMClient(Protocol):
    """The only surface B and C should depend on."""

    def complete(self, prompt: str, system: str | None = None,
                 max_new_tokens: int | None = None) -> str:
        ...


class ScriptedClient:
    """Returns canned replies. For tests and for wiring work without a GPU.

    Records every prompt it saw on `.prompts`, which is how prompt-construction
    is asserted in tests without needing weights.
    """

    def __init__(self, replies: list[str] | str | None = None):
        if isinstance(replies, str):
            replies = [replies]
        self._replies = list(replies or ["(scripted reply)"])
        self._i = 0
        self.prompts: list[tuple[str | None, str]] = []

    def complete(self, prompt: str, system: str | None = None,
                 max_new_tokens: int | None = None) -> str:
        self.prompts.append((system, prompt))
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        return reply


class LocalTransformersClient:
    """Qwen3 (or any chat model) via transformers, loaded once and reused.

    Heavy imports stay inside methods so `import flowmind.llm` works without the
    VLM/LLM dependency stack installed — the same reason vlm_reader defers its
    imports.
    """

    def __init__(self, model_id: str | None = None, device: str | None = None,
                 load_in_4bit: bool = True,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
        self.model_id = model_id or os.environ.get("FLOWMIND_LLM_MODEL", DEFAULT_MODEL_ID)
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")

        kwargs: dict = {}
        if self.load_in_4bit and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                # fp16 compute, not bf16: a T4 is Turing and has no bf16 support.
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                kwargs["device_map"] = "auto"
            except Exception as exc:  # bitsandbytes missing or unusable
                print(f"[llm] 4-bit unavailable ({exc}); falling back to fp16")
                kwargs["dtype"] = torch.float16
        else:
            kwargs["dtype"] = (torch.float16 if self.device in ("cuda", "mps")
                               else torch.float32)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if "device_map" not in kwargs:
            self._model = self._model.to(self.device)
        self._model.eval()

    def _render(self, prompt: str, system: str | None) -> str:
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        # Qwen3 is a hybrid-reasoning model; thinking traces would burn the token
        # budget and are not wanted for a single-pass answer. The kwarg is
        # Qwen-specific, so fall back for any other model.
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def complete(self, prompt: str, system: str | None = None,
                 max_new_tokens: int | None = None) -> str:
        import torch

        self._ensure_loaded()
        text = self._render(prompt, system)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True).strip()


_cached: LLMClient | None = None


def get_client(fresh: bool = False) -> LLMClient:
    """Process-wide client. Model load is expensive, so it is cached.

    FLOWMIND_LLM_BACKEND=local (default) | scripted
    """
    global _cached
    if _cached is not None and not fresh:
        return _cached

    backend = os.environ.get("FLOWMIND_LLM_BACKEND", "local").lower()
    if backend == "scripted":
        client: LLMClient = ScriptedClient()
    elif backend == "local":
        client = LocalTransformersClient()
    else:
        raise ValueError(f"unknown FLOWMIND_LLM_BACKEND {backend!r}")

    if not fresh:
        _cached = client
    return client
