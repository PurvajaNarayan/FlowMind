# FlowMind

Multi-agent flowchart understanding, QA & code synthesis on [FlowVQA](https://github.com/flowvqa/flowvqa).

A Reader → (Graph tool | Examiner | Planner) pipeline that reads a flowchart,
routes each question by intent, and answers it — with a deterministic graph lane
for topological questions and a Qwen3-VL vision Reader as the stretch goal.
Full spec: [`flowmind_feature_spec.md`](./flowmind_feature_spec.md).
All measured results, with methodology and caveats: [`RESULTS.md`](./RESULTS.md).

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                       # 16 tests, deterministic pieces — should be green
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
  question_classifier.py  # TF-IDF+LogReg question-type model (no LLM)  [Owner: A]
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
  train_router.py     # train/evaluate the question-type classifier     [Owner: A]
tests/              # unit tests + fixtures (tests/fixtures/sample.json)
data/               # dataset (gitignored; see data/README.md)
models/             # fitted models, e.g. question_classifier.joblib (gitignored)
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
gt.max_outdegree(g)                       # int
```

### Router — intent dispatch (spec §7.0)
```python
from flowmind.router import route

route("How many nodes exist?", qa_type="topological")   # "topological" (uses the label)
route("Which node follows the decision on No?")         # "content"  (trained model)
route("Write runnable python for this")                 # "code_request" (keyword)
# When evaluating on FlowVQA, always pass qa_type — it's a free, correct label.
```

### Question-type classifier — no-LLM model (spec §7.0 fallback)

A TF-IDF + Logistic Regression model that maps a question to one of the four
FlowVQA types. Train on `train_full.json`, evaluate on `test_full.json`:

```bash
python tools/train_router.py
# -> prints accuracy + per-class report + confusion matrix
# -> saves models/question_classifier.joblib
```

Latest result: **99.5% test accuracy** (topological 100%; the rest are the
natural-language types). Use it directly:

```python
from flowmind.question_classifier import QuestionTypeClassifier
clf = QuestionTypeClassifier.load()            # models/question_classifier.joblib
clf.predict("How many edges exist?")           # "topological"
clf.predict_proba("What does the node output?")  # {type: prob, ...}
```

The router loads this model automatically for unlabeled questions; if the
`.joblib` isn't present it falls back to the keyword heuristic, so retrain with
the command above after cloning (the model file is gitignored).

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
- **M1 done** — deterministic topological lane on all 1319 train records:

  | subtype | accuracy | scored |
  |---|---|---|
  | node_count | 100.0% | 1207/1207 |
  | max_outdegree | 100.0% | 101/101 |
  | shortest_path | 99.7% | 861/864 |
  | direct_predecessor | 99.6% | 807/810 |
  | direct_successor | 99.2% | 854/861 |
  | edge_count | 97.1% | 1172/1207 |
  | max_indegree | 95.1% | 427/449 |
  | **overall** | **98.7%** | **5429/5499** |

  Reproduce with `python tools/parser_coverage.py data/train_full.json`.
  Covers 5499 of the 5516 topological questions in train (**99.7%**). Of the 17
  not scored, 6 are questions that paraphrase or misspell the node label they
  quote (`"Is Home Orthodox"` for a node reading `Is the Home Orthodox?`;
  `"memorizaton"` for `memorization`) and 11 don't quote two labels at all — both
  dataset noise rather than parser gaps.

  `edge_count` and `max_indegree` are now the weakest subtypes, and neither
  involves label matching. On their failures gold is usually *lower* than our
  count (`gold - ours` is −1 or −2 in most cases), i.e. the parser finds edges the
  gold answer doesn't count. That's an unexplored over-generation issue and the
  next thing to look at. Both conventions were checked against gold and the
  deduplicating one is right: edge_count 97.1% deduped vs 96.4% raw,
  max_indegree 95.1% vs 94.9% counting parallel edges separately.

  > Earlier revisions of this table reported **98.8%** over 4826 questions. That
  > figure silently excluded every degree question: the harness tested for
  > `"in-degree"`, but FlowVQA writes it as one word (`"indegree"`), so 449
  > indegree questions fell through unscored — and `max_outdegree` (101 more)
  > had no graph-tool function at all. Both are fixed; the number moved to 98.5%
  > because indegree, at 95.1%, is the worst-performing subtype.
  >
  > A later revision reported **98.5%** over 5376 questions, with 129 skipped as
  > label-unresolved. Those 129 were a Reader bug, not ambiguity: `_clean_label`
  > stripped apostrophes from both ends of a label, truncating the 569 labels that
  > legitimately end in one, so they no longer matched the label quoted in the
  > question. Fixing that plus existential matching for duplicate labels brought
  > coverage to 99.7%.

- **M2 in progress / M5 first result** — Examiner built, and the §8 ablation run on
  a 60-item stratified sample (Qwen3-8B, 4-bit, greedy). The headline is that
  decomposition pays off exactly where a deterministic tool can replace the LLM,
  and is neutral where the LLM is still doing the work:

  | question type | single-pass baseline | full pipeline | delta |
  |---|---|---|---|
  | topological (43% of the benchmark) | 46.7% | **100.0%** | **+53.3** |
  | content (57%) | 82.2% | 80.0% | −2.2 |

  Topological is exact match on both arms, no judge involved. Content is judged by
  `flowmind/judge.py`; reproduce with `tools/run_ablation.py` then
  `tools/score_run.py`.

  Three things that shaped those numbers and belong in any write-up:

  1. **The content deficit was mostly a formatting choice, not the architecture.**
     Feeding the Examiner the serialized node/edge listing scored −8.9 against the
     baseline; feeding it the same raw Mermaid the baseline sees recovers 3 of the 4
     lost answers, leaving −2.2 (one answer in 45, i.e. noise). Run both with
     `--representation graph|mermaid`. With representations matched, the two arms
     frequently return byte-identical answers — greedy decoding, same prompt, and a
     revision loop that fires 2–3 times in 60 items.
  2. **The revision loop contributes nothing measurable yet.** It triggers on
     graph-grounding failures, which this model rarely produces. For the Examiner to
     earn its place the self-checks have to fire more often *and* catch real errors.
  3. **The judge was chosen by measurement, and it mattered.** Phi-4-mini scores
     100% on the negative controls in `tools/validate_judge.py`; Mistral-7B scores
     82.7% and inflated the same 90 answers from 77.8% to 97.8%. Agreement with
     exact match on topological questions does *not* validate a judge — both models
     scored 30/30 there while differing by 20 points on content.

  **Do not compare the text-path content numbers to FlowVQA's leaderboard.** The
  published baselines (GPT-4V 68.42%, TextFlow ~82.7%) read the flowchart as an
  *image*; this pipeline reads the ground-truth Mermaid source, which is a strictly
  easier task. Only the VLM Reader results are comparable in kind.

- **Vision Reader (stretch)** — zero-shot Qwen3-VL-2B, image → Mermaid → graph:
  **0.860 edge F1, 0.971 node-label recall**, cycle recall 6/9. Node shapes come out
  at the majority-class baseline (61.4% ≈ the 60.2% from always guessing `process`),
  so shape recognition is a perception gap rather than a formatting one. Resolution
  is *not* the bottleneck — edge F1 rose with downscaling (0.788 full-res → 0.932 at
  ≤0.6×), so a larger GPU would not help. A one-shot prompt ablation is a measured
  negative result: it induced shape syntax but degraded structure and leaked its own
  example into three answers (`FLOWMIND_VLM_PROMPT=v1|v2`).

- **M3 not started** — Planner and `metrics.behavioral_equivalence` are stubs.

  Question-type classifier (no-LLM router fallback): **99.5%** on `test_full.json`
  (`python tools/train_router.py`).

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
