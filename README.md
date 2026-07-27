# FlowMind

Multi-agent flowchart understanding, QA & code synthesis on [FlowVQA](https://github.com/flowvqa/flowvqa).

See [`flowmind_feature_spec.md`](./flowmind_feature_spec.md) for the full spec.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                     # deterministic pieces should pass today
```

## Repo layout

```
flowmind/
  schema.py         # FROZEN CONTRACT: FlowGraph / Node / Edge. Do not change without team sign-off.
  tracing.py        # FROZEN CONTRACT: per-sample trace log format (spec §12).
  data.py           # dataset loader / iterator            [Owner: A]
  router.py         # intent dispatch (spec §7.0)          [Owner: A]
  graph_tool.py     # deterministic topological answers    [Owner: A]
  reader/
    mermaid_reader.py  # mermaid text  -> FlowGraph         [Owner: A]
    vlm_reader.py      # image (Qwen3-VL) -> mermaid -> FlowGraph  [Owner: A, stretch]
  examiner.py       # content QA + revision loop (§7.3)    [Owner: B]
  planner.py        # code regen + plan doc (§7.4)         [Owner: C]
  eval/
    metrics.py      # scoring: graph/topo/content/code     [Owner: C]
    ablation.py     # single-pass vs full pipeline (§8)    [Owner: C]
tests/              # unit tests + fixtures
```

## How we work in parallel

Everyone codes against **two frozen contracts** and never against each other's internals:

1. **`flowmind/schema.py`** — the `FlowGraph` object the Reader produces and everyone
   downstream consumes. If you think it needs to change, raise it with the team first;
   a change here breaks all three workstreams.
2. **`flowmind/tracing.py`** — the trace record written for every sample. Person C's error
   analysis depends on this format being stable.

Because of these contracts:

- **B (Examiner)** develops against `FlowGraph` fixtures in `tests/fixtures/`, so does not
  wait on A's Reader or the VLM.
- **C (Planner + Eval)** develops against the same fixtures and a fake pipeline, so does not
  wait on A or B.
- **A** can swap the text Reader for the VLM Reader with zero downstream changes, because both
  return a `FlowGraph`.

## Ownership (maps to spec §10 milestones)

| Owner | Modules | Milestones |
|-------|---------|-----------|
| **A — Data & Reader** | `data.py`, `router.py`, `graph_tool.py`, `reader/*` (incl. VLM) | M0, M1 |
| **B — Examiner** | `examiner.py` | M2 |
| **C — Planner & Eval** | `planner.py`, `eval/*`, trace analysis | M3, M4, M5 |

## Branch / PR conventions

- Branch per feature: `a/mermaid-parser`, `b/examiner-loop`, `c/behavioral-eq`.
- Never edit `schema.py` / `tracing.py` in a feature PR without a heads-up to the team.
- Keep `pytest` green on `main`.
