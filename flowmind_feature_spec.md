# FlowMind: Multi-Agent Flowchart Understanding, QA & Code Synthesis

**Status:** draft v0.2 — edit freely, this is meant to be a starting point, not a locked spec.
*(v0.2: replaced the fixed sequential pipeline with intent-based routing — see §5)*
**Dataset:** [FlowVQA](https://github.com/flowvqa/flowvqa) (Singh et al., ACL Findings 2024)
**Course context:** NLP class project

---

## 1. Problem statement

FlowVQA pairs flowchart images with a Mermaid.js structural script and four types of
questions (fact retrieval, applied scenario, flow referential, topological). Most existing
approaches (TextFlow, SRA) answer these questions with a single LLM pass over a text
representation of the flowchart. This project instead builds a **three-agent pipeline** that
reads, verifies, and acts on the flowchart, and measures whether that decomposition actually
buys more accuracy than a single-pass baseline.

## 2. Goals

- Build a Reader → Examiner → Planner pipeline that answers FlowVQA questions and, for the
  code-sourced subset, regenerates and validates executable Python from the flowchart.
- Get at least one fully deterministic (non-LLM) evaluation lane working (topological
  questions via graph algorithms) so the project has a hard, arguable accuracy number and
  not just LLM-judged fuzziness everywhere.
- Produce a clean ablation: single-pass baseline vs. full pipeline, on the same sample set.

## 3. Non-goals

- Not attempting to beat published SOTA (TextFlow's ~82.7%) — the point is the agent
  architecture and evaluation rigor, not chasing a leaderboard number.
- Not covering the full 22,413 QA pairs — working set is a few hundred examples,
  stratified across question types (see §7).
- Not building a production UI. CLI / notebook + a written report is the deliverable.
- Vision-based flowchart reading is a stretch goal, not the default path (see §9 open
  question on images).

## 4. Users

Primarily: whoever is grading this. Secondarily: imagine a learner who's been handed a
flowchart of an algorithm and wants (a) a plain-English answer to a question about it, and
(b) working code that implements it — that's the scenario the code-subset milestone is
modeling.

## 5. System overview

**Not a fixed sequential pipeline.** Earlier drafts of this spec assumed every input runs
through Reader → Examiner → Planner in sequence. That's wrong: the flowchart is a constant,
but what happens after reading it depends entirely on the *intent* of the question. The
Reader always runs first (it has no idea yet what will be asked), then the system routes to
exactly one of three paths:

```
Flowchart + question
        |
        v
   Reader agent  (always runs, parses flowchart -> graph)
        |
        |-- intent: topological  --> Graph tool        (deterministic, no LLM)
        |-- intent: content      --> Examiner agent     (answers + self-checks)
        |-- intent: code_request --> Planner agent      (generates + tests code)
        |
        v
   Response to user
```

Only one branch runs per question. Intent comes from the dataset's own `type` field when
evaluating on FlowVQA itself (free and already correct — see `flowvqa_router.py`), or from a
classifier when someone types a free-form question that isn't pre-labeled. The revision loop
from earlier drafts still exists, but it's now scoped *inside* the Examiner's branch only
(it can send a "this doesn't check out" signal back to the Reader) — it doesn't span all
three paths anymore. The Planner's branch gets its own correctness check instead: behavioral
equivalence against the original `code` field.

Shared resource: a **graph tool** (networkx-based, built and tested — `flowvqa_graph_utils.py`)
that answers every topological question type directly, and a **router**
(`flowvqa_router.py`, also tested) that does the intent dispatch above.

## 6. Confirmed data schema

Pulled directly from `train_full.json` (the `code` / FloCo-sourced entries):

```json
{
  "code00453": {
    "key": "code00453",
    "code": "def find_fixed_point(arr, n): ...",   // present for `code` entries only
    "summary": "markdown pseudocode, START/INPUT/PROCESS/DECISION/OUTPUT/END style",
    "mermaid": "flowchart TD\n    A([\"Start\"]) --> B[...] ...",
    "qa": {
      "1": { "Q": "...", "A1": "...", "A2": "...", "A3": "...", "type": "fact_retrieval" },
      "4": { "Q": "How many nodes exist in the given flowchart?", "A1": "9", "type": "topological" }
    }
  }
}
```

Notes:
- `A1`/`A2`/`A3` are paraphrases of the same gold answer — use all three as references when
  scoring, not exact string match against `A1` alone.
- `type` is one of `fact_retrieval`, `applied_scenario`, `flow_referential`, `topological`.
- Topological questions include: node count, edge count, shortest-path edge count between
  two named nodes, direct-predecessor check, direct-successor check, max in-degree.
- No image field appears in this subset's records — **open question, see §9**.
- `wiki`/`instruct` prefixed keys almost certainly lack the `code` field; assumed to have
  the same `mermaid`/`qa` structure — verify this before building the wiki/instruct milestone.

## 7. Functional requirements

### 7.0 Router / intent classifier
| | |
|---|---|
| Input | question text, plus `qa_type` when evaluating on FlowVQA itself |
| Output | one of `topological` / `content` / `code_request`, plus the dispatched result |
| Requirement | when `qa_type` is available, route off it directly — don't waste a classification step on a label you already have |
| Fallback | a keyword-based heuristic classifier for free-form questions with no label (implemented in `flowvqa_router.py`); swap for an LLM call once this stops being the bottleneck |
| Already done | `route()`, `route_from_dataset_type()`, `classify_intent_heuristic()` in `flowvqa_router.py`, tested against real samples |

### 7.1 Reader agent
| | |
|---|---|
| Input | `mermaid` script (text-only v1) |
| Output | a graph object: nodes with label text, edges with optional Yes/No labels |
| Requirement | must parse all four Mermaid node shapes (rounded, parallelogram, rect, diamond) and both edge styles (`-->`, `-->\|label\|`) |
| Already done | `mermaid_to_graph()` in `flowvqa_graph_utils.py`, tested against 3 real samples |
| Stretch | vision-based reading directly from the flowchart image, if images turn out to be available |

### 7.2 Graph tool (shared utility, not conversational)
| Function | Answers |
|---|---|
| `node_count` / `edge_count` | node/edge count questions |
| `shortest_path_edges(a, b)` | shortest-path edge-count questions |
| `is_direct_predecessor(a, b)` / `is_direct_successor(a, b)` | adjacency questions |
| `max_indegree` | max in-degree questions |

All five already implemented and unit-verified.

### 7.3 Examiner agent
| | |
|---|---|
| Input | question text + type, the Reader's graph (topological questions never reach this agent — they're handled entirely by the router + graph tool in §7.0) |
| Behavior | generate an answer from the graph + node text, then score it against `A1`/`A2`/`A3` |
| Output | answer, verdict (`accept` / `revise`), and — if `revise` — a flag sent back to the Reader with what looked wrong (e.g. "referenced node not found in graph") |
| Requirement | cap revision loops at 2 attempts per question to avoid runaway cost; this loop is local to the Examiner's branch, it does not involve the Planner |

### 7.4 Planner agent
| | |
|---|---|
| Input | verified graph, `summary` field (optional reference, not shown to the agent by default), source type |
| Output (`code` subset) | (a) an executable Python function reconstructed from the graph, (b) a markdown plan doc |
| Output (`wiki`/`instruct` subset) | markdown plan doc only, with web search invoked only when a step references something the agent can't resolve from the flowchart text alone |
| Requirement | generated Python must be runnable via `exec`/import, not just plausible-looking text |

## 8. Evaluation & acceptance criteria

| Metric | How | Target / comparison point |
|---|---|---|
| Graph extraction accuracy | node/edge counts vs. ground truth from `mermaid` | should be ~100% for text-only v1 (it's the same source); this metric matters once/if vision is added |
| Topological QA accuracy | exact match via graph tool | should approach 100% — if it doesn't, the bug is in the parser, not "understanding" |
| Fact/Applied/Flow-referential accuracy | best-of-3 match against `A1`/`A2`/`A3` (embedding similarity or LLM judge) | compare against published baselines: GPT-4 few-shot 68.42% majority vote, TextFlow up to ~82.7% |
| Code-behavioral correctness (`code` subset only) | run generated Python + original `code` on N random inputs, compare outputs | report % of functions that are behaviorally equivalent — no LLM judge needed |
| Ablation | single-pass baseline vs. full Reader+Examiner+Planner pipeline, same sample set | report accuracy delta and where the revision loop actually changed an answer |

## 9. Open questions

- **Images**: do pre-rendered flowchart images actually ship in the Data repo, or are they
  generated on demand from the Mermaid script? Check the repo file tree before deciding
  whether vision is even feasible as a stretch goal.
- **wiki/instruct schema**: confirm field names match the `code` subset before building
  milestone 2.
- **Judge choice** for the three non-topological question types: single LLM judge vs.
  embedding similarity vs. both — pick one and justify it in the report, since this is the
  fuzziest part of the whole pipeline.

## 10. Milestones

1. **M0 — done**: data loader + Mermaid parser, verified against real samples.
2. **M1**: Reader + graph tool + topological QA, fully deterministic, on the `code` subset
   (~575 flowcharts).
3. **M2**: Examiner agent for the other three question types + multi-reference scoring.
4. **M3**: Planner agent — Python regeneration + behavioral-equivalence testing, markdown
   plan output.
5. **M4**: extend to `wiki`/`instruct` subset, add web search to the Planner.
6. **M5**: ablation study, error analysis, final report.

## 11. Related work to cite

- Singh et al., *FlowVQA* (ACL Findings 2024) — the dataset itself.
- Ye et al., *TextFlow* (NAACL 2025) — closest prior art to the Reader stage.
- Suri et al., *FlowPathAgent / FlowExplainBench* (EMNLP 2025) — attribution/verification
  framing relevant to the Examiner.
- He et al., *Flow2Code* (ACL Findings 2025) — precedent for flowchart-to-code as a
  sub-task, relevant to the Planner.
- The SRA paper (arXiv 2602.13771) — precedent for a shallow/deep reasoning switch,
  relevant to when the Examiner should bother looping back to the Reader.

## 12. Non-functional notes

- Cap API spend: sample size and revision-loop limits above exist specifically to keep this
  affordable on a student budget.
- Log every agent's input/output for each sample — you'll want these traces for the error
  analysis section of the report, and it's much harder to reconstruct after the fact.
- Check the dataset repo's license/citation terms before including flowchart images in any
  public write-up or demo.
