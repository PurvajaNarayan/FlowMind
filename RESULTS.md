# FlowMind — Results

All figures below were produced by the commands listed in each section and are
reproducible from `main`. Everything is measured on `data/train_full.json` unless
stated otherwise; the only result touching `test_full.json` is the router classifier.

---

## 0. Summary

The project asks whether decomposing flowchart question-answering into a routed
multi-agent pipeline buys accuracy over a single LLM pass. The answer we can defend
is more specific than a yes or a no:

> **Decomposition pays off exactly where a deterministic tool can replace the LLM,
> and is neutral where the LLM is still doing the work.**

| question type | share of benchmark | single-pass baseline | full pipeline | delta |
|---|---|---|---|---|
| topological | 43% | 46.7% | **100.0%** | **+53.3** |
| content (3 types) | 57% | 82.2% | 80.0% | −2.2 |

The topological figure is exact match on both arms with no LLM judge involved. The
content figure is judged by a model whose discrimination we measured first.

Supporting results:

- **Deterministic topological lane: 98.7%** over 99.7% of the 5,516 topological
  questions in train.
- **Intent router: 99.54%** on the held-out test split, and **100.0%** on
  topological questions specifically.
- **Vision Reader (stretch goal): 0.860 edge F1, 0.971 node-label recall**
  zero-shot from flowchart images.
- Four measured negative results, described in §7.

---

## 1. Experimental setup

**Dataset.** FlowVQA (Singh et al., ACL Findings 2024). `train_full.json` holds
1,319 records — 261 `code` (FloCo-sourced), 407 `instruct` (Instructables), 651
`wiki` (WikiHow) — and 12,938 question–answer pairs. `test_full.json` holds 9,475
further pairs with no key overlap. Each record carries a Mermaid.js script, a
markdown summary, and for the `code` subset the original Python. Images ship
separately as 2,532 PNGs in two layouts (`main` top-down, `bottom_top` for the
directional-bias robustness set); records join to images by key, since the JSON has
no image field.

Question types split 43% topological / 57% natural language. In the test split:
topological 4,076, applied_scenario 1,936, fact_retrieval 1,878,
flow_referential 1,585.

**Models.** All open-weight, all run locally on free hardware.

| role | model | licence | notes |
|---|---|---|---|
| answering LLM | `Qwen/Qwen3-8B` | Apache 2.0 | 4-bit (nf4), fp16 compute |
| judge | `microsoft/Phi-4-mini-instruct` | MIT | 3.8B, chosen by measurement (§6) |
| vision Reader | `Qwen/Qwen3-VL-2B-Instruct` | Apache 2.0 | fp16 |
| router classifier | TF-IDF + logistic regression | — | no LLM |

**Hardware.** Google Colab T4 (15 GB, Turing). Turing has no bf16, so fp16 compute
throughout; Phi-4-mini additionally needs `attn_implementation="sdpa"` because it
defaults to flash-attention, which Turing does not support.

**Decoding.** Greedy (`do_sample=False`) everywhere. Re-running a sweep reproduces
bit-identically — verified by repeating a 20-image VLM sweep and getting byte-equal
output. Consequently no result below carries sampling variance, and reported deltas
need no multiple-seed control.

---

## 2. Deterministic topological lane

### Result

**98.7% (5,429 / 5,499)**, covering 5,499 of the 5,516 topological questions in
train — **99.7% coverage**.

| subtype | accuracy | scored |
|---|---|---|
| node_count | 100.0% | 1207/1207 |
| max_outdegree | 100.0% | 101/101 |
| shortest_path | 99.7% | 861/864 |
| direct_predecessor | 99.6% | 807/810 |
| direct_successor | 99.2% | 854/861 |
| edge_count | 97.1% | 1172/1207 |
| max_indegree | 95.1% | 427/449 |

```bash
python tools/parser_coverage.py data/train_full.json
```

### Method

The Mermaid script is parsed into a `FlowGraph` (nodes with label and shape, edges
with optional Yes/No labels), converted to a networkx `DiGraph`, and each question
subtype dispatched to a pure function: `node_count`, `edge_count`,
`shortest_path_edges`, `is_direct_predecessor`, `is_direct_successor`,
`max_indegree`, `max_outdegree`. No model, no randomness, no API. The whole sweep
over 1,319 charts runs in about half a second.

### Three defects the harness exposed

Each of these inflated or hid a number, and each is worth reporting because the
lesson generalises.

**Silent exclusion by phrasing.** The harness matched `"in-degree"` and
`"in degree"`. FlowVQA writes it as one word, `"indegree"`, so the branch never
fired and **449 questions fell through unscored**. The headline covered only 87.5%
of the topological set while appearing to cover all of it.

**A missing implementation.** A further **101 questions** ask for maximum
*outdegree*, for which no graph-tool function existed. Spec §7.2 lists only
`max_indegree`. Added; it scores 101/101.

**A parser bug corrupting shared state.** `_clean_label` was
`.strip('"').strip("'")`, which removes quote characters from each end
independently. That truncated every label ending in an apostrophe — **569 labels
across 182 charts**. `"Iterate over the list 'nums'"` in a question could never
match `Iterate over the list 'nums` in the graph, which accounted for all **129**
questions the harness had to skip as unresolvable. Since `FlowGraph` is the contract
every downstream component consumes, this was wrong data flowing to the Examiner
too, not merely a harness problem.

Stripping *balanced* apostrophes is not a safe fix either: 9 labels genuinely start
and end with one, e.g. `'sum' = 'sum' + 'i'`. An apostrophe pair is therefore
treated as a wrapper only when nothing between it is an apostrophe.

**Duplicate labels.** Charts routinely contain two nodes labelled "End". Resolution
returned the first arbitrarily, and ambiguous adjacency questions scored 50% and
33% against 99.5%+ for unambiguous ones. Resolution now returns all candidates and
each question applies its own semantics — "is X a direct predecessor of End" is
true if X precedes *any* node labelled End.

### How the number moved

| revision | accuracy | coverage |
|---|---|---|
| initial | 98.8% | 4826/5516 (87.5%) |
| degree questions scored | 98.5% | 5376/5516 (97.5%) |
| label resolution fixed | **98.7%** | **5499/5516 (99.7%)** |

The dip to 98.5% is the point: `max_indegree` at 95.1% is the weakest subtype and
had been invisible. A figure covering 99.7% of the set is more defensible than a
higher one that quietly omits its worst category.

### Counting conventions, checked rather than assumed

FlowVQA's gold could count a decision's parallel Yes/No edges into one target as
one edge or two. We tested both against gold:

| metric | deduplicated | raw |
|---|---|---|
| edge_count | **97.1%** | 96.4% |
| max_indegree | **95.1%** | 94.9% |

Deduplication is correct on both.

### Residual

17 questions unscored: 6 where the question paraphrases or misspells the label it
quotes (`"Is Home Orthodox"` for a node reading `Is the Home Orthodox?`;
`"memorizaton"` for `memorization`), and 11 that never quote two labels. Both are
dataset noise, not parser gaps.

`edge_count` and `max_indegree` are now the weakest subtypes and neither involves
label matching. On their failures gold is usually *below* our count (`gold − ours`
is −1 or −2 in most cases), so the parser finds edges the gold answer does not
count. Diagnosed but unfixed; the likely ceiling is one or two points.

---

## 3. Intent router

### Result

**99.54%** on `test_full.json` (9,475 questions), trained on train's 12,938.

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| applied_scenario | 0.9974 | 0.9959 | 0.9966 | 1936 |
| fact_retrieval | 0.9894 | 0.9931 | 0.9912 | 1878 |
| flow_referential | 0.9880 | 0.9855 | 0.9867 | 1585 |
| **topological** | **1.0000** | **1.0000** | **1.0000** | **4076** |

```bash
python tools/train_router.py
```

### Why the topological figure is the one that matters

TF-IDF (word 1–2 grams + char 3–5 grams) into logistic regression. No LLM.

The perfect topological score is the important number, not the 99.54%. Routing a
counting question to the LLM instead of the graph tool costs roughly 58 points on
that question (§4), so a routing error there is expensive. Errors the classifier
*does* make are all confusions among the three natural-language types, which route
to the same branch and therefore cost nothing.

Three-tier dispatch, best signal first: the dataset's own `type` label when
evaluating on FlowVQA (free and correct), the trained classifier for unlabelled
free-form questions, and a keyword heuristic if the model file is absent.

---

## 4. Single-pass baseline

### Result

**40.0%** on topological questions, against 98.7% for the graph tool on the same
question type.

```bash
python tools/run_baseline.py --n 120 --save runs/baseline.jsonl
```

Replicated across two independent samples:

| sample | topological | adjacency | counting |
|---|---|---|---|
| 100 items, `code` only | 10/25 = 40.0% | 8/9 = 88.9% | 2/16 = 12.5% |
| 120 items, all subsets | 12/30 = 40.0% | 8/9 = 88.9% | 4/21 = 19.0% |

### The mechanism: it is not counting at all

The aggregate hides the interesting part. Adjacency questions — *"is X a direct
predecessor of Y"* — are a local lookup in the text, and the model handles them at
**88.9%**. Counting questions require aggregating over the whole chart, and it
manages **19.0%**.

The failure is not arithmetic error. Across 21 counting questions the model used
only **11 distinct answers**, and 16 of 21 were values it also gave for a
*different* flowchart:

```
'20' ×3 (0 right)   '5' ×3 (0 right)   '12' ×3 (0 right)
'10' ×3 (1 right)   '18' ×2 (0 right)  '14' ×2 (1 right)
```

It answered `20` for `wiki00023`, `wiki00123` and `wiki00673` — three unrelated
charts. Ten of 21 answers are multiples of five. This is a model emitting a
plausible magnitude, not enumerating and slipping.

Consistent with that, accuracy degrades with chart size, since larger charts make a
guessed magnitude less likely to land:

| subset | topological | counting only |
|---|---|---|
| code | 50% | 3/8 |
| instruct | 40% | 1/7 |
| wiki | 30% | 0/6 |

Zero correct counts on `wiki`, the subset with the longest charts.

### Sampling

The first baseline was drawn from `code` only, because `train_full.json` is ordered
code-first and the sampler stratified on question type alone. Sampling is now a full
(subset × question-type) grid with a cap of 2 questions per flowchart and a seeded
shuffle inside each cell. At n=120 that gives 40 per subset, 30 per type, spread
over 113 charts rather than 20.

The per-chart cap alone made things worse before the shuffle was added: with
selection following question position, and a chart's topological questions ordered
`node_count, edge_count, shortest_path, predecessor…`, the cap returned counting
questions only and dropped adjacency to zero — discarding the split that carries
the finding.

---

## 5. The ablation

### Result

n=60, stratified over 12 (subset × question-type) cells.

| question type | single-pass | full pipeline | delta |
|---|---|---|---|
| topological (n=15) | 46.7% | **100.0%** | **+53.3** |
| content (n=45/arm) | 82.2% | 80.0% | −2.2 |

```bash
python tools/run_ablation.py --n 60 --representation mermaid --save runs/ablation.jsonl
python tools/score_run.py runs/ablation.jsonl --save runs/scored.jsonl
```

Content, by question type (both arms pooled, Phi-4-mini judge):

| type | accuracy |
|---|---|
| fact_retrieval | 28/30 = 93.3% |
| applied_scenario | 26/30 = 86.7% |
| flow_referential | 19/30 = 63.3% |

`flow_referential` is the hardest type by a wide margin, and — see §5.2 — its 63.3%
is unmoved by input representation.

### 5.1 Design decisions that make the comparison honest

**The baseline receives raw Mermaid, not the parsed graph.** Passing the graph would
hand the baseline the Reader's output, and the Reader's contribution is part of what
the ablation is meant to isolate.

**The Examiner's self-check never sees the gold answers.** An earlier version
triggered revision on `content_match(reply, item.answers)` and interpolated
`item.answers` into the retry prompt, so on a retry the model was shown the answer
key. It also set `correct = (verdict == "accept")`, where `accept` came from that
same gold comparison — so the pipeline was scored correct exactly when it decided it
was correct, using the references, while the baseline got one blind attempt. The
delta would have been guaranteed positive and would have measured leakage rather
than architecture. Revision now triggers only on checks computable from the graph:
an empty answer, a quoted phrase absent from the chart, or no lexical overlap with
the chart at all.

**Belief and correctness are recorded separately.** `verdict` means "passed its own
self-checks" and is not a correctness claim. Correctness is decided afterwards by
the judge, once the pipeline has committed.

**Both arms are scored identically** — exact match for topological, the judge for
content — and content is deliberately not scored inline, so the placeholder
`content_match` never enters the headline.

### 5.2 Three-quarters of the content deficit was a formatting choice

With the Examiner reading the serialized node/edge listing, content came in at
**−8.9**. Feeding it the same raw Mermaid the baseline sees recovers 3 of the 4 lost
answers:

| Examiner reads | examiner | single_pass | delta |
|---|---|---|---|
| node/edge listing | 33/45 (73.3%) | 37/45 (82.2%) | −8.9 |
| raw Mermaid | 36/45 (80.0%) | 37/45 (82.2%) | −2.2 |

`single_pass` is identical on both rows, as it must be — the baseline never sees the
representation and decoding is greedy — which confirms the comparison is clean. The
system prompt for the Mermaid mode is pinned byte-identical to the baseline's by a
test, so the only remaining difference between the arms is the revision loop.

The residual −2.2 is one answer in 45, i.e. noise. And with representations matched
the two arms frequently return **byte-identical answers** (`'Return the hash of x'`,
`'Continue to the next node.'`, `'Baking Dish'`). The pipeline is converging on being
the baseline.

### 5.3 Why: the revision loop cannot fire on errors

It fired 2–3 times in 60 items. Replaying every self-check over the judged runs
gives the reason:

| run | catch rate | false-alarm rate |
|---|---|---|
| Mermaid representation | 1/9 = 11.1% | 1/36 = 2.8% |
| node/edge listing | 0/12 = 0.0% | 1/33 = 3.0% |

```bash
python tools/eval_selfcheck.py runs/scored.jsonl --branch examiner
```

*Catch rate* is the ceiling on how much the loop can ever help: of answers the judge
called wrong, how many the check flags. *False-alarm rate* is the cost: correct
answers sent back for revision.

The checks ask "is this answer about this chart?" Every real error **is** about the
chart — the model names genuine steps and picks the wrong one. One observed failure
was answering with the correct answer to a *different question about the same
flowchart*: perfectly grounded, completely wrong.

Question-aware checks were added (adjacency against the graph, and step-count
verification), but the addressable population is small: only **367 of 7,422 content
questions (4.9%)** both name a resolvable node and ask about ordering. Everything
else is prose reasoning with nothing in the graph to check it against.

**So the loop is structurally unable to affect results in this task, which is why
the content delta is ≈0.** That is a measured limitation of gold-free
self-verification, not a tuning failure.

---

## 6. Evaluation methodology

Spec §9 left the content judge open between an LLM judge, embedding similarity, or
both. FlowVQA settles it: the benchmark's own protocol (paper §3.2) gives three
evaluators — GPT-3.5, Llama-2 70B, Mixtral 8×7B — the response, the question and all
three gold answers; each writes a chain-of-thought rationale then a binary label,
scored by majority vote. The authors call it "length-invariant paraphrase detection"
and state explicitly that it **surpasses similarity metrics and rule-based
matching**.

Embedding similarity would additionally fail on the distinction this task turns on:
"returns the index i" and "does not return the index i" are near-identical as
vectors and opposite in meaning.

### Two deviations from the published protocol

**One judge, not three.** Llama-2 70B and Mixtral 8×7B do not fit free hardware. The
ensemble can be reinstated by running with several judge models and majority-voting.

**The judge is not the model under test.** Qwen3-8B grading its own answers is
self-preference bias. What matters is *lineage*: note that DeepSeek's small distills
are trained onto Qwen and Llama bases, so the Qwen variants would reintroduce the
problem despite the different name.

### Judge selection by measurement

Four controls over 25 content questions: a reference answer verbatim (expect
correct); a reference answer from a *different* question — fluent, confident and
definitely wrong, with no surface cue (expect incorrect); the reference with its
claim negated; the reference with its digits changed.

| control | Phi-4-mini (3.8B) | Mistral-7B |
|---|---|---|
| gold — true positive | 25/25 (100%) | 25/25 (100%) |
| swapped — true negative | 25/25 (100%) | 23/25 (92%) |
| negated — true negative | 25/25 (100%) | 19/25 (76%) |
| renumbered — true negative | 2/2 (100%) | 1/2 (50%) |
| **balanced accuracy** | **100.0%** | 91.3% |

```bash
python tools/validate_judge.py --n 25
```

Phi-4-mini is perfect at 3.8B while Mistral accepts roughly one wrong answer in
six, and is weakest precisely where it matters — negation, at 76%, which is what
`flow_referential` questions hinge on. The smaller model wins because judging is
constrained binary classification, not generation.

The choice changed the numbers materially: the same 90 content answers scored
**77.8% under Phi and 97.8% under Mistral**. Mistral marked both `'7 steps.'` and
`'6 steps.'` correct for the same question — they cannot both be right.

### A validation method that does not work

We initially validated the judge by agreement with exact match on the *topological*
questions, where truth is unambiguous. Both judges scored **30/30** there while
differing by 20 points on content. Topological answers are `9` and `Yes`; agreeing
on them certifies nothing about the hard case. Reported here because it is an easy
mistake to make and it looks like rigour.

**Caveat that survives a perfect control score.** Every control is *definitely*
wrong by construction. 100% rules out rubber-stamping; it does not certify
calibration on partially-correct near-misses, which is the class real model errors
fall into.

---

## 7. Vision Reader (stretch goal)

### Result

Zero-shot Qwen3-VL-2B, image → Mermaid → `FlowGraph`, scored against the graph
parsed from the ground-truth Mermaid.

| metric | value |
|---|---|
| edge F1 | **0.860** |
| edge precision / recall | 0.881 / 0.841 |
| edge F1 including Yes/No labels | 0.855 |
| node-label recall | **0.971** |
| cycle recall | 6/9 charts with loops (66.7%) |
| shape accuracy | 218/355 = 61.4% |

```bash
python tools/eval_vlm.py --n 20 --layout main --save runs/vlm.jsonl
python tools/analyze_vlm_run.py runs/vlm.jsonl --per-sample
```

### The design that makes this cheap to supervise

The VLM emits *Mermaid text*, not a graph. That reuses the already-tested parser,
gives free ground-truth supervision from the dataset's own `mermaid` field, and
makes the VLM a drop-in Reader backend — nothing downstream can tell which Reader
produced a graph.

### Text is read well; shapes are not perceived at all

Node-label recall of 0.971 means the model transcribes the words in a flowchart
almost perfectly. Shapes are a different story:

```
gold shape mix   process 219, decision 56, io 51, terminal 38
pred shape mix   process 365, decision 3
```

No terminals, no IO, three decisions against 56. The 61.4% shape accuracy is
indistinguishable from the **60.2%** you get by labelling every node `process` — the
majority-class baseline. Shape information is not being produced at all. Spec §7.1
requires all four shapes.

### Resolution is not the bottleneck

| | edge F1 | label recall |
|---|---|---|
| full resolution (n=7) | 0.788 | 0.956 |
| mildly downscaled (n=7) | 0.880 | 0.976 |
| heavily downscaled ≤0.6× (n=5) | **0.932** | 0.987 |

Accuracy *rose* as images shrank, so a larger GPU would not help. The apparent
effect is confounded with subset — full-resolution samples are `code`-heavy, and
`code` is the hardest subset (0.799 vs instruct's 0.918) because its charts contain
loops and symbolic expressions.

### Why long charts crashed, and the fix

Nine of the first 15 samples died with CUDA OOM, requesting up to 15.97 GiB from a
14.56 GiB card. The cause is chart height, not subset: renders are a fixed 1568 px
wide and grow downward, and Qwen3-VL compresses by 32, so visual tokens ≈
pixels/1024.

| | pixels | visual tokens |
|---|---|---|
| succeeded | 2.47 – 4.93 MP | ~2,400 – 4,800 |
| OOM | 7.58 – 18.88 MP | ~7,400 – **18,433** |

The tallest chart, `wiki00031`, is 1568 × 12038. Capping total pixels just above the
largest observed success, plus freeing CUDA memory between samples, takes the sweep
to 20/20. Memory was also fragmenting across the loop, so later samples failed on
allocations earlier ones had survived.

### The count metric is misleading and should not be the headline

`eval_vlm` reports node/edge **count** match. It is both too strict and too shallow:

```
count-match:  node 10/20 (50%), edge 2/20 (10%), both 1/20 (5%)
edge F1:      0.860
```

Too strict, because it is exact equality — mean absolute node-count error was 0.53
with 18/19 samples within ±1, yet exact match was 10/20. Too shallow, because
matching totals says nothing about structure: one sample was "off by one" on both
counts while every shape was wrong and both loops were missing.

---

## 8. Negative results

Four, all measured rather than assumed.

**One-shot prompting made the VLM worse.** Adding a worked example to the prompt to
teach shapes:

| | v1 (no example) | v2 (one-shot) |
|---|---|---|
| label recall | **0.971** | 0.836 |
| edge F1 | **0.860** | 0.729 |
| cycle recall | 6/9 | 5/10 |
| shape accuracy | 61.4% | 55.3% |

It *did* induce the syntax — the predicted mix moved from `{process 365, decision 3}`
to `{process 238, io 101, terminal 9, decision 20}` — but applies shapes wrongly
(101 `io` against 56 gold), so accuracy fell *below* the majority baseline. It also
**leaked its own example**: three charts came back as exactly 7 nodes with label
recall ≈0.1, matching the 7-node demo in the prompt rather than the image. Reverted;
kept selectable via `FLOWMIND_VLM_PROMPT=v1|v2`.

Since neither prompt perceives shapes, shape recognition is a perception gap, and
prompting cannot close it. That is the evidence-based argument for a LoRA fine-tune.

**The revision loop cannot help** — §5.3. Only 4.9% of content questions are
graph-verifiable.

**Most of the content deficit was formatting, not architecture** — §5.2. −8.9
became −2.2 on changing the Examiner's input.

**Agreement with exact match does not validate a judge** — §6. Both judges scored
30/30 while differing 20 points on content.

---

## 9. Limitations and threats to validity

**Sample sizes.** The ablation is n=60: 15 topological and 45 content per arm. The
−2.2 content delta is a single answer. Per-subset content splits are 30 items each.
The +53.3 topological delta is robust at n=15 only because the effect is very large.
VLM figures are n=20, and the per-subset breakdown there is n≈7.

**Train only.** Every result except the router classifier is measured on
`train_full.json`. Nothing has been run against `test_full.json`.

**Not comparable to FlowVQA's leaderboard.** The published baselines — GPT-4V at
68.42% majority vote, TextFlow up to ~82.7% — read the flowchart as an **image**.
Our text pipeline reads the ground-truth **Mermaid source**, which is strictly more
information and a substantially easier task. Only the Vision Reader results (§7) are
comparable in kind. Absolute content numbers should not be set against those
figures.

**Judge deviations.** One judge rather than the specified three, and a 3.8B open
model rather than GPT-3.5 / Llama-2 70B / Mixtral 8×7B. Controls are all
definitely-wrong by construction, so they do not certify calibration on
partially-correct answers.

**Single answering model.** All LLM results use Qwen3-8B at 4-bit. The counting
failure in §4 may be specific to this model or this quantisation; we have not tested
whether a larger model counts.

**Unexplained parser behaviour.** On `edge_count` and `max_indegree` failures gold is
usually below our count, so the parser finds edges gold does not. Worth one or two
points on the deterministic lane.

---

## 10. Not done

- **Planner (M3)** and `metrics.behavioral_equivalence` are stubs. The `code` subset
  (261 train records) carries the original Python, so behavioral equivalence —
  running generated and original functions on random inputs and comparing outputs —
  would give a second fully deterministic metric with no judge required. This is the
  largest remaining gap and the cheapest remaining rigour.
- Nothing evaluated on `test_full.json` beyond the router.
- LoRA fine-tuning of the Vision Reader, which §7 and §8 jointly motivate.
- The `bottom_top` directional-bias ablation, which the dataset provides and which
  would test whether the VLM reads structure or layout position.

---

## 11. Reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                        # 72 tests

# deterministic lane (CPU, ~0.5s)
python tools/parser_coverage.py data/train_full.json

# router classifier (CPU)
python tools/train_router.py

# GPU from here: pip install -r requirements-vlm.txt && pip install bitsandbytes
python tools/run_baseline.py  --n 120 --save runs/baseline.jsonl
python tools/run_ablation.py  --n 60 --representation mermaid --save runs/ablation.jsonl
python tools/score_run.py     runs/ablation.jsonl --save runs/scored.jsonl
python tools/validate_judge.py --n 25
python tools/eval_selfcheck.py runs/scored.jsonl --branch examiner
python tools/eval_vlm.py      --n 20 --layout main --save runs/vlm.jsonl
python tools/analyze_vlm_run.py runs/vlm.jsonl --per-sample
```

`notebooks/ablation_run.ipynb` runs the GPU sections end to end on Colab.
