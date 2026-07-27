"""Ablation harness (spec §8): single-pass baseline vs full pipeline. [Owner: C]

Also the home for the vision ablation Owner A cares about:
text-Reader vs VLM-Reader on the same samples.

TODO(C):
  - single_pass_baseline(item): one LLM call over the mermaid text.
  - full_pipeline(item): Reader -> router -> {graph_tool | Examiner | Planner}.
  - run_ablation(items): run both, write Traces (flowmind.tracing), report the
    accuracy delta and where the revision loop actually changed an answer.
"""

from __future__ import annotations

from flowmind.data import QAItem


def single_pass_baseline(item: QAItem):
    raise NotImplementedError("single-pass baseline not implemented — [Owner: C]")


def full_pipeline(item: QAItem):
    raise NotImplementedError("full pipeline driver not implemented — [Owner: C]")
