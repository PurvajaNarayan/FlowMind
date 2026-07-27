# FlowMind

Multi-agent flowchart understanding, QA & code synthesis on [FlowVQA](https://github.com/flowvqa/flowvqa).

A Reader → (Graph tool | Examiner | Planner) pipeline that reads a flowchart,
routes each question by intent, and answers it — with a deterministic graph lane
for topological questions and a Qwen3-VL vision Reader as the stretch goal.
Full spec: [`flowmind_feature_spec.md`](./flowmind_feature_spec.md).

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                       # 14 tests, deterministic pieces — should be green
```

Then drop the dataset into `data/` (see [Data setup](#data-setup)) and run the
coverage harness to see the deterministic-lane accuracy:

```bash
python tools/parser_coverage.py data/train_full.json
```

---

## Data setup

Download FlowVQA into `data/` (not committed — see `.gitignore`). Layout:

```
data/
├── train_full.json          1319 records  (code 261 / instruct 407 / wiki 651)
├── test_full.json            953 records
├── images/
│   ├── main/<key>.png        top-down layout (flowchart TD) — default
│   └── bottom_top/<key>.png  same charts, bottom-top (directional-bias set)
└── raw/                      source CSVs (redundant with JSON)
```

Records join to images by key: `data/images/main/<key>.png`. There is **no image
field in the JSON**. Full schema notes: [`data/README.md`](./data/README.md).

---

## Repo layout

```
flowmind/
  schema.py         # FROZEN CONTRACT: FlowGraph / Node / Edge. Team sign-off to change.
  tracing.py        # FROZEN CONTRACT: per-sample trace log (spec §12).
  data.py           # dataset loader / iterator                         [Owner: A]
  router.py         # intent dispatch (spec §7.0)                       [Owner: A]
  graph_tool.py     # deterministic topological answers (spec §7.2)     [Owner: A]
  reader/
    mermaid_reader.py  # mermaid text -> FlowGraph (spec §7.1)          [Owner: A]
    vlm_reader.py      # image (Qwen3-VL) -> mermaid -> FlowGraph, stub [Owner: A]
  examiner.py       # content QA + revision loop (§7.3), stub           [Owner: B]
  planner.py        # code regen + plan doc (§7.4), stub                [Owner: C]
  eval/
    metrics.py      # scoring: graph/topo/content/code (§8)             [Owner: C]
    ablation.py     # single-pass vs full pipeline (§8), stub           [Owner: C]
tools/
  parser_coverage.py  # M1 metric: parser+graph tool vs FlowVQA gold    [Owner: A]
tests/              # unit tests + fixtures (tests/fixtures/sample.json)
data/               # dataset (gitignored; see data/README.md)
```

---

## How to run each piece

All examples assume `source .venv/bin/activate` first.

### Run the tests
```bash
pytest                       # everything
pytest tests/test_graph_tool.py -v
```

### Parser coverage / M1 metric (the deterministic-lane number, spec §8)
```bash
python tools/parser_coverage.py data/train_full.json
# inspect the failing cases for one subtype:
python tools/parser_coverage.py data/train_full.json --show-fails direct_successor
```

### Load the dataset
```python
from flowmind.data import load_dataset, iter_qa

ds = load_dataset("data/train_full.json")
for item in iter_qa(ds, layout="main"):     # layout="bottom_top" for the robustness set
    item.sample_key   # "code00453"
    item.qa_type      # fact_retrieval | applied_scenario | flow_referential | topological
    item.mermaid      # the flowchart script  -> text Reader
    item.image_path   # data/images/main/code00453.png  -> VLM Reader
    item.answers      # [A1, A2, A3] gold paraphrases
    item.subset       # code | wiki | instruct
    break
```

### Reader — Mermaid text → graph (spec §7.1)
```python
from flowmind.reader.mermaid_reader import mermaid_to_graph

g = mermaid_to_graph(ds["code00453"]["mermaid"])
g.nodes            # [Node(id="A", label="Start", shape=NodeShape.TERMINAL), ...]
g.edges            # [Edge(source="A", target="B", label=None), ...]
g.find_by_label("End")   # loose label lookup used by the graph tool
```

### Graph tool — deterministic topological answers (spec §7.2)
```python
from flowmind import graph_tool as gt

gt.node_count(g)                          # int
gt.edge_count(g)                          # int (dedupes parallel Yes/No edges)
gt.shortest_path_edges(g, "A", "D")       # int | None
gt.is_direct_predecessor(g, "A", "B")     # True  (edge A->B: A precedes B)
gt.is_direct_successor(g, "B", "A")       # True  (edge A->B: B follows A)
gt.max_indegree(g)                        # int
```

### Router — intent dispatch (spec §7.0)
```python
from flowmind.router import route

route("How many nodes exist?", qa_type="topological")   # "topological"
route("Write runnable python for this")                 # "code_request" (heuristic)
# When evaluating on FlowVQA, always pass qa_type — it's a free, correct label.
```

### Trace logging (spec §12 — shared, used by everyone)
```python
from flowmind.tracing import Trace, TraceWriter

with TraceWriter("runs/m1.jsonl") as tw:
    tw.write(Trace(sample_key="code00453", question_id="4",
                   intent="topological", branch="graph_tool",
                   prediction=7, gold="7", correct=True))
```

### VLM Reader (stretch — not wired yet)
```python
# Requires a local Qwen3-VL backend (MLX-VLM or Ollama). See vlm_reader.py checklist.
from flowmind.reader.vlm_reader import image_to_graph
g = image_to_graph("data/images/main/code00453.png")   # raises NotImplementedError today
```

---

## Current status

- **M0 done** — data loader, Mermaid parser, router, graph tool, all tested.
- **M1 in progress** — deterministic topological lane on all 1319 train records:

  | subtype | accuracy |
  |---|---|
  | node_count | 100.0% |
  | shortest_path | 99.6% |
  | direct_predecessor | 99.1% |
  | direct_successor | 98.4% |
  | edge_count | 97.1% |
  | **overall** | **98.8%** |

  Reproduce with `python tools/parser_coverage.py data/train_full.json`.
  Residual is dataset gold-label noise + label-resolution ambiguity (multiple
  nodes sharing a label, e.g. "End") — the next M1 task, a Reader concern.

- **M2–M5** — Examiner, Planner, eval, ablation: interfaces stubbed, ready to build.

---

## Working in parallel

Everyone codes against **two frozen contracts** and never against each other's internals:

1. **`flowmind/schema.py`** — the `FlowGraph` the Reader produces and everyone consumes.
2. **`flowmind/tracing.py`** — the trace record written per sample (Person C's analysis).

Because of these, **B** (Examiner) and **C** (Planner/eval) develop against the
`tests/fixtures/sample.json` `FlowGraph` and don't wait on A's Reader or the VLM.
A can swap the text Reader for the VLM Reader with zero downstream changes.

| Owner | Modules | Milestones |
|-------|---------|-----------|
| **A — Data & Reader** | `data.py`, `router.py`, `graph_tool.py`, `reader/*`, `tools/parser_coverage.py` | M0, M1 |
| **B — Examiner** | `examiner.py` | M2 |
| **C — Planner & Eval** | `planner.py`, `eval/*`, trace analysis | M3, M4, M5 |

### Conventions
- Branch per feature: `a/label-resolution`, `b/examiner-loop`, `c/behavioral-eq`.
- Don't edit `schema.py` / `tracing.py` in a feature PR without a team heads-up.
- Keep `pytest` green on `main`.
