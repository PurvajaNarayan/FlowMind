# FlowMind — Results

This document records everything we have measured, what each measurement means, and
what it can and cannot be used to claim. Every number was produced by the commands
shown alongside it and can be reproduced from the `main` branch.

Unless a section says otherwise, all measurements use the **training split** of the
dataset (`data/train_full.json`). Only one result — the question router in §4 — has
been measured on the held-out test split.

---

## 1. How to read this document

I use a handful of terms repeatedly. Here is what each one means in this project, so
nothing later depends on guessing.

**Flowchart / chart.** One diagram from the dataset. Each one comes with a Mermaid.js
script (a text description of the boxes and arrows), a rendered PNG image, and a set
of questions about it.

**Mermaid script.** The text source of a flowchart. For example
`A(["Start"]) --> B["Check the value"]` means "a rounded box labelled Start has an
arrow pointing to a rectangular box labelled Check the value". The dataset gives us
these scripts, which is why reading the flowchart as *text* is much easier than
reading it as a *picture*.

**Graph.** Once a Mermaid script is parsed, we hold it as a graph: a list of boxes
(called *nodes*) and a list of arrows between them (called *edges*). Everything in
the system operates on this graph.

**The three subsets.** The dataset's flowcharts come from three sources, and we
report results separately for each because they behave differently:

- `code` — flowcharts of Python algorithms. Short, full of loops and symbols.
- `instruct` — flowcharts from Instructables craft projects. Long, plain-English.
- `wiki` — flowcharts from WikiHow articles. Longest of the three.

**The four question types.** Every question in the dataset is labelled as one of:

- *topological* — questions about the structure of the diagram, answerable by
  counting or by checking connections. "How many nodes exist?" "Is box X directly
  before box Y?" These make up **43%** of the benchmark.

- *fact_retrieval* — "What does the flowchart output when a fixed point is found?"
- *applied_scenario* — "John has done step 3; what should he do next?"
- *flow_referential* — "What are the two conditions that lead to the loop ending?"

The last three all require reading and understanding English, and together they make
up the remaining **57%**.

**The two systems we compare.** The whole project rests on one comparison:

- **The single-pass baseline** — we hand the entire Mermaid script and the question
  to one LLM in a single request, and take whatever it says. No parsing, no routing,
  no retries. This is roughly what earlier published work does.

- **The full pipeline** — we parse the chart into a graph, decide what kind of
  question it is, and send it to a specialist: ordinary code for structural
  questions, an LLM for language questions.

Earlier drafts of this document called these two "arms", which is standard
experiment-speak but unhelpful. They are just *two systems answering the same
questions*, and the interesting quantity is the difference between their scores.

**Exact match.** A way of scoring where the answer must equal the correct answer
precisely. If the right answer is `9` and the system says `10`, that scores zero —
there is no partial credit. We use exact match only for structural questions, where
answers are single numbers or Yes/No, so there is no ambiguity about whether a
response is right.

**Judged / grader.** Language questions cannot be scored by exact match, because
there are many correct ways to phrase an answer. For those we use a separate LLM
whose only job is to read the question, the reference answers, and our system's
answer, and say whether they mean the same thing. §7 explains how we chose that
grader and how we checked it was trustworthy.

**Coverage.** The fraction of questions that our measurement actually managed to
score. This matters more than it sounds: a scoring script can silently skip
questions it does not recognise, and then a high accuracy figure is high only on the
subset it happened to look at. §3 describes a case where this went wrong.

**n=60, and "stratified".** `n` is simply how many questions were used in a
measurement. *Stratified* means the questions were deliberately spread evenly across
categories rather than picked at random — see §6.3, which explains why that turned
out to matter a great deal.

**Greedy decoding.** LLMs can generate text randomly, producing a different answer
each time you ask. We turned that off, so the model always picks its single
most-likely next word. The consequence is that re-running any measurement gives
*byte-for-byte identical* results. We verified this by repeating a sweep and
comparing outputs. It means no number here has random variation in it, and no
difference we report could be explained by luck of the draw.

**4-bit / quantised.** Large models normally store each number in 16 bits of memory.
Compressing them to 4 bits makes the model roughly four times smaller, so an 8-billion
parameter model fits in about 5 GB instead of 16 GB. We do this because the free GPU
we use has only 15 GB. There is a small accuracy cost, which we have not measured.

---

## 2. The question this project is trying to answer

The dataset (FlowVQA) pairs flowcharts with questions. Previously published
approaches answer those questions by giving a text version of the flowchart to one
large language model and asking it to respond.

Our claim is that this is wasteful, because the four question types are not alike.
Some of them — "how many boxes are there?" — are not really language problems at
all. They are graph problems, and a graph problem can be solved by ordinary code
exactly and instantly. Others genuinely need language understanding.

So the system we built looks at each question, decides which kind it is, and sends it
to whichever component suits it. The question this document answers is whether that
decision-making actually improves results, and by how much.

**The short answer**, which the rest of the document supports:

> Splitting the work up helps enormously for the questions where ordinary code can
> replace the language model, and makes essentially no difference for the questions
> where the language model is still doing the work.

| question type | share of dataset | one LLM call | our pipeline | difference |
|---|---|---|---|---|
| structural (topological) | 43% | 46.7% | **100.0%** | **+53.3 points** |
| language (other three) | 57% | 82.2% | 80.0% | −2.2 points |

Read that as: on structural questions, a single LLM call gets 46.7% right, while our
pipeline gets everything right — an improvement of 53.3 percentage points. On
language questions the two systems perform the same to within one answer.

---

## 3. Result: answering structural questions with code instead of a model

### What we did

Structural questions ask about the shape of the diagram. There are seven kinds in the
dataset, and each one can be answered by a short piece of ordinary code operating on
the parsed graph:

| question | how the code answers it |
|---|---|
| "How many nodes exist?" | counts the list of boxes |
| "How many edges exist?" | counts the distinct arrows |
| "How many edges on the shortest path from X to Y?" | breadth-first search |
| "Is X a direct predecessor of Y?" | checks whether an arrow X→Y is in the list |
| "Is X a direct successor of Y?" | checks whether an arrow Y→X is in the list |
| "What is the maximum indegree?" | counts arrows *into* each box, takes the largest |
| "What is the maximum outdegree?" | counts arrows *out of* each box, takes the largest |

No model is involved, nothing is random, and the whole run over all 1,319 flowcharts
takes about half a second.

### The result

**98.7% correct, on 99.7% of all the structural questions in the training split.**

In plain terms: there are 5,516 structural questions in the training data. Our
scoring script was able to attempt 5,499 of them, and got 5,429 right.

| question kind | accuracy | number right |
|---|---|---|
| node count | 100.0% | 1207 / 1207 |
| maximum outdegree | 100.0% | 101 / 101 |
| shortest path length | 99.7% | 861 / 864 |
| is X directly before Y | 99.6% | 807 / 810 |
| is X directly after Y | 99.2% | 854 / 861 |
| edge count | 97.1% | 1172 / 1207 |
| maximum indegree | 95.1% | 427 / 449 |

```bash
python tools/parser_coverage.py data/train_full.json
```

The 17 questions we could not attempt are cases where the question misspells or
rewords the box label it refers to — one asks about `"Is Home Orthodox"` when the box
actually reads `Is the Home Orthodox?`, another says `"memorizaton"` where the box
says `memorization`. These are errors in the dataset, not in our parser.

### Three bugs this measurement exposed, and why they are worth reporting

Building the scoring script found three real defects. Each of them made a reported
number wrong in a way that was invisible from the number itself, which is the reason
they are written up here rather than quietly fixed.

**Bug 1: a whole category of question was being skipped without saying so.**

Our scoring script looked for questions containing the phrase `"in-degree"`. The
dataset writes it as one word, `"indegree"`. Because the two do not match, the script
never recognised those questions and skipped all **449** of them.

Worse, it skipped them silently. The reported accuracy was 98.8%, which looked
excellent, but it was 98.8% *of the questions the script happened to recognise* —
only 87.5% of the questions that actually existed. Nothing in the output hinted at
the missing 12.5%.

**Bug 2: one kind of question had no implementation at all.**

A further **101** questions ask for the maximum *outdegree* — the largest number of
arrows leaving any single box. We had written the function for indegree (arrows
coming in) but not outdegree (arrows going out). Those questions were unanswerable.
Adding the function took three lines and it now answers all 101 correctly.

**Bug 3: the parser was corrupting box labels, which affected everything downstream.**

The code that cleaned up box labels was stripping quote characters from both ends of
the text. That is fine for a label like `"Start"`, which becomes `Start`. But a label
like `"Iterate over the list 'nums'"` genuinely *ends* with an apostrophe, and the
cleanup removed it, producing `Iterate over the list 'nums` instead.

**569 labels across 182 flowcharts** end with an apostrophe, so all of them were
being mangled. When a question then asked about `"Iterate over the list 'nums'"`, we
could not find a matching box, and the question became unanswerable — this accounted
for all **129** questions the script had been skipping as unresolvable.

This bug mattered more than the other two, because the parsed graph is what *every*
part of the system reads. It was not a scoring problem; it was wrong data being
handed to the Examiner as well.

The obvious fix — strip apostrophes only when they appear at both ends — is also
wrong: 9 labels genuinely begin and end with one, such as `'sum' = 'sum' + 'i'`. So
an apostrophe pair is now treated as a wrapper only when there is no other
apostrophe between them.

**Bug 4 (smaller): boxes with identical labels were resolved arbitrarily.**

Flowcharts often contain two boxes both labelled "End". When a question asked about
"End", we picked whichever one came first. Questions affected by this scored **50%
and 33%**, against 99.5% or better for questions where the label was unique. We now
gather all the matching boxes and answer accordingly: "is X directly before End" is
true if X is directly before *any* box labelled End.

### How the headline number changed as these were fixed

| stage | accuracy | how many questions were scored |
|---|---|---|
| before any fixes | 98.8% | 4826 of 5516 (87.5%) |
| after scoring degree questions | 98.5% | 5376 of 5516 (97.5%) |
| after fixing label handling | **98.7%** | **5499 of 5516 (99.7%)** |

Notice the accuracy briefly went *down*, from 98.8% to 98.5%. That is the point.
Maximum-indegree questions are the hardest of the seven kinds at 95.1%, and they had
been excluded entirely. Including them pulled the average down while making the
figure honest. A 98.7% that covers 99.7% of the questions is far more defensible
than a 98.8% that quietly omits its own worst category.

### One design decision we checked rather than assumed

When a decision box has two arrows — a "Yes" branch and a "No" branch — that both
lead to the same destination box, is that one arrow or two? The dataset's own correct
answers could have taken either view, and we had no way to know which without
checking.

So we computed both and compared each against the dataset's answers:

| question kind | counting them as one | counting them as two |
|---|---|---|
| edge count | **97.1%** | 96.4% |
| maximum indegree | **95.1%** | 94.9% |

Counting them as one arrow is correct on both. This is recorded because it is the
sort of assumption that is easy to make silently and hard to notice later.

### What is still wrong

Edge count (97.1%) and maximum indegree (95.1%) are now the weakest of the seven
kinds, and neither involves matching labels, so the label fixes above do not touch
them. Looking at the cases we get wrong, the dataset's answer is usually *smaller*
than ours — meaning our parser is finding arrows that the dataset's answer does not
count. We have diagnosed this far but not fixed it. It is probably worth one or two
percentage points.

---

## 4. Result: deciding which component should answer each question

### What we did

Something has to look at a question and decide whether it goes to the code or to the
language model. We trained a small classifier for this: it turns the question text
into numerical features (which words and letter-sequences appear) and feeds them to a
logistic regression. There is no LLM involved and it runs in milliseconds.

### The result

**99.54% correct** on the held-out test split — 9,475 questions the classifier had
never seen, having been trained on the 12,938 in the training split.

| question type | how often it was right when it predicted this type | how often it found this type | number of questions |
|---|---|---|---|
| applied_scenario | 99.74% | 99.59% | 1,936 |
| fact_retrieval | 98.94% | 99.31% | 1,878 |
| flow_referential | 98.80% | 98.55% | 1,585 |
| **topological (structural)** | **100.00%** | **100.00%** | **4,076** |

```bash
python tools/train_router.py
```

### Why the last row is the number that matters

The 99.54% overall figure is less important than the perfect score on structural
questions, for a specific reason.

Sending a structural question to the language model instead of to the code costs
roughly 53 percentage points on that question (§5 and §6). So a mistake *there* is
expensive. The classifier makes no such mistakes: 4,076 out of 4,076.

Its actual mistakes are all confusions between the three language types — thinking a
`fact_retrieval` question is a `flow_referential` one, for instance. But all three of
those go to the same place anyway, so those mistakes cost nothing at all.

---

## 5. Result: what happens when one LLM answers structural questions

### What we did

We gave the whole Mermaid script and the question to Qwen3-8B in a single request and
recorded its answer. This is the comparison point for the whole project: it is what
you get without any of the pipeline.

### The result

**40.0%** on structural questions, against 98.7% for the code.

```bash
python tools/run_baseline.py --n 120 --save runs/baseline.jsonl
```

We ran this twice on different samples of questions and got the same figure:

| sample | overall | connection questions | counting questions |
|---|---|---|---|
| 100 questions, `code` charts only | 40.0% (10/25) | 88.9% (8/9) | 12.5% (2/16) |
| 120 questions, all three subsets | 40.0% (12/30) | 88.9% (8/9) | 19.0% (4/21) |

### The interesting part: it is not counting at all

The 40% average hides two very different behaviours.

**Connection questions** — "is box X directly before box Y?" — only require looking
at one line of the Mermaid script to see whether that arrow is written there. The
model handles these at **88.9%**.

**Counting questions** — "how many boxes are there?" — require going through the
entire chart and keeping a running total. The model manages **19.0%**.

And the way it fails is revealing. Across 21 counting questions it produced only
**11 different answers**, and 16 of the 21 were numbers it had also given for a
completely different flowchart:

```
answered '20' three times   (0 correct)
answered  '5' three times   (0 correct)
answered '12' three times   (0 correct)
answered '10' three times   (1 correct)
answered '18' twice         (0 correct)
answered '14' twice         (1 correct)
```

It answered `20` for three unrelated flowcharts. Ten of the 21 answers were multiples
of five.

That pattern is not what arithmetic mistakes look like. Arithmetic mistakes cluster
near the right answer. This looks like a model producing a number of plausible
*magnitude* without enumerating anything — the way a person might glance at a
diagram and say "about twenty" without counting.

The subset breakdown supports that reading. Larger charts make a guessed magnitude
less likely to be right, and accuracy falls accordingly:

| subset | chart size | structural accuracy | counting only |
|---|---|---|---|
| `code` | smallest | 50% | 3/8 |
| `instruct` | medium | 40% | 1/7 |
| `wiki` | largest | 30% | 0/6 |

Zero correct counts on the subset with the longest flowcharts.

### A sampling mistake worth recording

Our first run of this measurement drew all 100 of its questions from the `code`
subset. Neither `wiki` (651 flowcharts) nor `instruct` (407) appeared at all.

The cause: the dataset file happens to list `code` records first, and our sampler
spread questions evenly across the four *question types* but not across the three
*subsets*. Taking the first questions of each type therefore took them all from the
front of the file.

This was not a harmless imperfection. The subsets behave measurably differently — the
table above shows 50% versus 30% — so a `code`-only sample cannot speak for the
dataset.

The sampler now spreads questions across all twelve combinations of subset and
question type, and caps how many questions come from any single flowchart at two, so
that answers stay reasonably independent of one another.

Adding that cap on its own made things briefly *worse*, in a way worth knowing about.
Without shuffling, the sampler took questions in the order they appear in the file,
and a flowchart's structural questions are always listed as node count, edge count,
shortest path, predecessor, and so on. Capping at two questions per chart therefore
returned *only* counting questions and no connection questions at all — destroying
precisely the 88.9%-versus-19.0% split that is the most informative thing in this
section. Shuffling within each group fixed it.

---

## 6. Result: the head-to-head comparison

### What we did

We took 60 questions, ran both systems on exactly the same questions, and compared.

The 60 questions were spread evenly across twelve groups — every combination of three
subsets and four question types, five questions in each. They came from 57 different
flowcharts, with no more than two questions taken from any one chart.

```bash
python tools/run_ablation.py --n 60 --representation mermaid --save runs/ablation.jsonl
python tools/score_run.py runs/ablation.jsonl --save runs/scored.jsonl
```

### The result

| question type | one LLM call | our pipeline | difference |
|---|---|---|---|
| structural (15 questions) | 46.7% | **100.0%** | **+53.3 points** |
| language (45 questions each) | 82.2% | 80.0% | −2.2 points |

Language questions broken down by type:

| type | accuracy | number right |
|---|---|---|
| fact_retrieval | 93.3% | 28 / 30 |
| applied_scenario | 86.7% | 26 / 30 |
| flow_referential | 63.3% | 19 / 30 |

`flow_referential` is markedly the hardest. These are questions about ordering and
consequence — "what are the two conditions that could have led here?" — and as §6.3
shows, its 63.3% does not budge no matter how we present the chart to the model.

### 6.1 Four decisions that keep this comparison honest

A comparison like this is easy to rig accidentally. Four choices matter.

**The single-LLM system gets the raw Mermaid script, not our parsed graph.** If we
gave it our parsed graph, we would be handing it the work our Reader does, and the
Reader's contribution is part of what we are trying to measure.

**Our pipeline is never shown the correct answers while it is working.** This one was
a genuine bug we had to fix. An earlier version of the Examiner decided whether to
retry by comparing its answer against the dataset's correct answers — and then put
those correct answers *into the retry prompt*. So on a second attempt the model was
literally shown the answer key and asked again.

It was worse than that. The pipeline was also *scored* by whether it had decided to
accept its own answer, and that decision used the correct answers. So the pipeline
was marked right exactly when it had decided it was right, using information the
single-LLM system never saw. The comparison would have been guaranteed to favour the
pipeline, and it would have been measuring cheating rather than architecture.

Retrying is now triggered only by checks that use the flowchart alone: is the answer
empty, does it quote a step that does not exist in the chart, does it share no
vocabulary with the chart at all.

**"The pipeline thinks it is right" and "the pipeline is right" are recorded
separately.** The Examiner's own verdict on its answer is stored, but it is not used
as the score. The score comes afterwards from the independent grader.

**Both systems are scored the same way** — exact match for structural questions, the
grader for language questions — and language questions are deliberately not scored
during the run, so that a placeholder scoring function cannot leak into the headline
figure.

### 6.2 Where the small language-question deficit came from

Our pipeline scored 2.2 points *below* the single-LLM system on language questions.
Before investigating, we assumed that said something about the architecture. It did
not.

The two systems differed in one unremarked way: the single-LLM system reads the raw
Mermaid script, while our Examiner was reading a reformatted list —

```
Nodes:

- A (terminal): Start
- B (process): Check the value
Edges:

- A -> B
```

That reformatting is lossy in a subtle way: it discards the visual grouping and
ordering that the original script has, and models have seen a very large amount of
Mermaid during training but very little of our bespoke listing format.

So we ran the Examiner again, giving it the identical raw Mermaid the other system
gets, with a system prompt pinned to be byte-identical:

| what the Examiner reads | Examiner | one LLM call | difference |
|---|---|---|---|
| our reformatted node/edge list | 33/45 (73.3%) | 37/45 (82.2%) | −8.9 points |
| raw Mermaid script | 36/45 (80.0%) | 37/45 (82.2%) | −2.2 points |

Changing only the format recovered three of the four lost answers. **About
three-quarters of what looked like an architectural weakness was a formatting
choice.**

Two things confirm the comparison is clean. The single-LLM column is identical on
both rows, exactly as it must be — that system never sees the reformatting, and
greedy decoding removes randomness, so any movement there would have meant something
else had changed. And with formats matched, the two systems frequently return
*byte-identical* answers: `'Return the hash of x'`, `'Continue to the next node.'`,
`'Baking Dish'`.

That last observation is the important one. The remaining −2.2 points is a single
answer out of 45, which is well inside the noise of a sample this size. Our pipeline,
on language questions, has converged on *being* the single-LLM system.

### 6.3 Why: the retry mechanism almost never activates

Our pipeline is supposed to differ from a single LLM call by checking its own answer
and retrying when the check fails. Across 60 questions, it retried **two or three
times**. So for practical purposes it made one call, like the baseline, which is why
the two scores are the same.

We measured why, by replaying every check over the answers the grader had already
judged:

| run | how many wrong answers got flagged | how many right answers got wrongly flagged |
|---|---|---|
| raw Mermaid format | 1 of 9 (11.1%) | 1 of 36 (2.8%) |
| reformatted list | 0 of 12 (0.0%) | 1 of 33 (3.0%) |

```bash
python tools/eval_selfcheck.py runs/scored.jsonl --branch examiner
```

The first column is the ceiling on how much retrying can ever help: if a check never
fires on a wrong answer, no number of retries improves anything. The second column is
the cost: every one of those is a *correct* answer being sent back and possibly
replaced with a worse one.

The reason the checks miss is structural rather than a matter of tuning. They ask
*"is this answer about this flowchart?"* — and every real error **is** about the
flowchart. The model names genuine steps from the chart and picks the wrong one. In
one case it answered with the correct answer to a *different question about the same
flowchart*: perfectly grounded in the chart, and completely wrong.

We then added checks that consider the question, not just the chart — verifying that
a claimed "next step" really is adjacent in the graph, and that a claimed step count
matches the actual path length. But the population these can examine is small. Only
**367 of 7,422** language questions (**4.9%**) both name a box explicitly and ask
about ordering. The other 95% are prose reasoning with nothing in the graph to check
them against.

**So the retry mechanism is structurally unable to affect the outcome in this task.**
That is a measured limitation of self-checking without access to the answers, not a
bug we failed to fix, and it fully explains why the language-question difference is
approximately zero.

---

## 7. How we chose and checked the grader

### Why a grader is needed at all

Language questions cannot be scored by string comparison. The dataset gives three
correct paraphrases for each question, and a system's answer may be correct while
matching none of them word-for-word. For example, the reference answers might be
`It outputs the index i`, `The index i is returned`, and
`It returns i, the fixed point index`, while our system says
`The index 'i' as fixed point` — correct, and identical to none of them.

So we use a separate language model as a grader: it reads the question, the three
reference answers, and our answer, and says whether ours means the same thing as any
of them.

### Why not something simpler

The obvious cheaper option is *embedding similarity* — converting both texts to
numerical vectors and measuring the angle between them. We rejected it for two
reasons.

First, the benchmark's own authors rejected it. FlowVQA's paper specifies a grading
protocol: three evaluator models each read the response, the question, and all three
reference answers, write out their reasoning, then give a yes/no verdict, and the
final score is a majority vote. The authors describe this as "length-invariant
paraphrase detection" and state explicitly that it outperforms similarity metrics and
rule-based matching. Using a different method would produce a number comparable to
nothing in the literature.

Second, similarity cannot handle negation, which these questions turn on constantly.
"Returns the index i" and "does *not* return the index i" are nearly identical as
vectors and opposite in meaning.

### Two ways we deviate from the published protocol

**We use one grader, not three.** Two of the three models the paper uses (Llama-2 70B
and Mixtral 8×7B) do not fit on free hardware. The three-model version can be
reconstructed by running our tool with several grader models and taking a majority
vote; we use one for cost.

**The grader is never the model being graded.** If Qwen3-8B graded its own answers,
it would tend to approve its own style of response — and it is the first objection
any reader would raise. What matters is that the grader comes from a *different model
family*, and this is less obvious than it sounds: DeepSeek's small models are built
on top of Qwen and Llama base models, so the Qwen-based ones would quietly reintroduce
the problem despite the different name.

### How we checked the grader was actually trustworthy

We did not want to assume. So we built four tests, using 25 real questions, and
checked whether the grader gets each one right:

1. **Give it a correct answer, verbatim from the dataset.** It should say correct. If
   it does not, the grader is too harsh and will understate every system it grades.
2. **Give it a correct answer belonging to a *different question*.** Fluent,
   confident, well-formed, and definitely wrong — with no clue in the wording to give
   it away. It should say incorrect. This is the hardest test.
3. **Give it a correct answer with its meaning reversed** ("does not" inserted). It
   should say incorrect.
4. **Give it a correct answer with the numbers changed.** It should say incorrect.

| test | Phi-4-mini (3.8B) | Mistral-7B |
|---|---|---|
| approves a genuinely correct answer | 25/25 (100%) | 25/25 (100%) |
| rejects a different question's answer | 25/25 (100%) | 23/25 (92%) |
| rejects a reversed-meaning answer | 25/25 (100%) | 19/25 (76%) |
| rejects a wrong-number answer | 2/2 (100%) | 1/2 (50%) |
| **average of both directions** | **100.0%** | 91.3% |

```bash
python tools/validate_judge.py --n 25
```

Phi-4-mini is perfect on all four at 3.8 billion parameters, while Mistral-7B — the
larger model — accepts roughly one definitely-wrong answer in six. Mistral's weakest
test is reversed meaning at 76%, which is the worst possible weakness here, because
that is exactly what `flow_referential` questions depend on.

The smaller model wins because grading is a narrow yes/no classification, not
open-ended generation. Size helps with the latter far more than the former.

**This choice changed the results materially.** The same 90 answers scored **77.8%
under Phi-4-mini and 97.8% under Mistral-7B**. Mistral marked *both* `'7 steps.'` and
`'6 steps.'` as correct for the same question — they cannot both be right. All figures
in this document use Phi-4-mini.

### A checking method that looked rigorous and was not

Our first attempt at validating the grader was to run it on the *structural*
questions, where we already know the right answer with certainty, and measure how
often it agreed. Both graders scored **30 out of 30**.

That number is worthless. Structural answers are `9` and `Yes` — deciding whether `9`
matches `9` is trivial, and agreeing on it says nothing about the hard case. The two
graders scored identically on that test while differing by 20 percentage points on
the questions that actually matter.

This is recorded because it is an easy mistake to make and it has the outward
appearance of diligence.

**One caveat that survives even a perfect score.** All four tests use answers that are
*definitely* wrong by construction. Passing them rules out a grader that approves
everything. It does not prove the grader handles *partially* correct answers well —
answers that get most of the way there and miss a detail — and that is the category
real model mistakes usually fall into.

---

## 8. Result: reading flowcharts from images instead of text

### Why this exists

Everything above reads the Mermaid script, which the dataset provides. That is
convenient and slightly artificial: in real life you would have a picture of a
flowchart, not its source code. If our system requires the source code, it only works
on flowcharts that already came with source code.

So we also built a Reader that takes the PNG image, uses a vision-language model to
write out the Mermaid script, and then feeds that into the same parser everything
else uses. Nothing downstream can tell which Reader produced the graph.

This design has a practical advantage: because the dataset also contains the *true*
Mermaid script, we can compare what the model wrote against what it should have
written, at no labelling cost.

### The result

| measurement | value | what it means |
|---|---|---|
| node-label recall | **0.971** | of the box labels that should be there, 97.1% were written out |
| edge F1 | **0.860** | combined measure of arrows found and arrows invented; 1.0 is perfect |
| edge precision | 0.881 | of the arrows it wrote, 88.1% are real |
| edge recall | 0.841 | of the real arrows, 84.1% were written |
| loop detection | 6 of 9 (66.7%) | of charts containing a loop, how many it reproduced with a loop |
| box-shape accuracy | 61.4% | see below — this number is misleading |

```bash
python tools/eval_vlm.py --n 20 --layout main --save runs/vlm.jsonl
python tools/analyze_vlm_run.py runs/vlm.jsonl --per-sample
```

### It reads the words very well and does not see the shapes at all

97.1% label recall means the model transcribes the *text* in a flowchart almost
perfectly.

Box shapes are another matter. Mermaid distinguishes rounded boxes (start/end),
parallelograms (input/output), rectangles (a step), and diamonds (a decision), and
the specification requires the Reader to identify all four. What the model produced:

```
what the charts actually contain:  219 rectangles, 56 diamonds, 51 parallelograms, 38 rounded
what the model wrote:              365 rectangles,  3 diamonds,  0 parallelograms,  0 rounded
```

No rounded boxes, no parallelograms, and three diamonds against 56.

This is why the 61.4% shape accuracy is misleading. Rectangles are 60.2% of all boxes,
so simply labelling *everything* a rectangle scores 60.2% — and 61.4% is
indistinguishable from that. The model is not partially recognising shapes; it is not
producing shape information at all, and scoring near the "always guess the commonest
answer" baseline.

### A larger GPU would not help

We expected image resolution to be the limiting factor, and it is not:

| image size | edge F1 | label recall |
|---|---|---|
| full resolution (7 charts) | 0.788 | 0.956 |
| moderately shrunk (7 charts) | 0.880 | 0.976 |
| heavily shrunk to ≤60% (5 charts) | **0.932** | 0.987 |

Accuracy went *up* as images got smaller. This is confounded, and we say so: the
full-resolution samples are mostly `code` charts, which are the hardest subset (0.799
against `instruct`'s 0.918) because they contain loops and symbolic expressions. But
whichever way the confound runs, there is no evidence that more pixels would help, so
there is no case for spending money on a bigger GPU.

### Why long flowcharts crashed, and how we fixed it

Nine of the first fifteen images failed with an out-of-memory error, one of them
requesting 15.97 GB from a card with 14.56 GB available.

The cause is chart *height*. Every rendered flowchart is 1,568 pixels wide, and grows
downward as it gets longer. The vision model converts the image into "visual tokens"
at roughly one token per 32×32 pixel block, and then has to process all of them at
once:

| | image size | visual tokens |
|---|---|---|
| succeeded | 2.5 – 4.9 megapixels | about 2,400 – 4,800 |
| ran out of memory | 7.6 – 18.9 megapixels | about 7,400 – **18,433** |

The tallest chart, `wiki00031`, is 1,568 × 12,038 pixels — 18,433 visual tokens for
one image.

The fix caps total pixels just above the largest size we observed working, shrinking
anything above it, and frees GPU memory between images. Memory had also been
fragmenting across the run, so later images failed on allocations that earlier ones
had survived. All 20 images now complete.

### The metric we were originally using was the wrong one

Our first vision measurement compared *counts*: does the model produce the same
number of boxes and arrows as the real chart?

```
count comparison:  boxes right 10/20 (50%), arrows right 2/20 (10%), both right 1/20 (5%)
edge F1:           0.860
```

Those describe the same 20 charts. The count comparison is simultaneously too harsh
and too shallow.

Too harsh, because it demands exact equality. The average error was 0.53 boxes, with
18 of 19 charts within one box of correct — and yet exact agreement was only 10 of 20.
Being off by one scores the same as being off by thirty.

Too shallow, because matching totals says nothing about structure. One chart had
counts off by exactly one in each direction, which reads as "nearly right", while
every box shape was wrong and both loops were missing.

---

## 9. Things that did not work

Four attempts failed, all measured rather than guessed at. They are reported because a
negative result with a mechanism behind it is more useful than an unexplained one.

### Giving the vision model a worked example made it worse

To try to fix the shape problem, we added a small worked example to the prompt showing
all four box shapes in use.

| | without example | with example |
|---|---|---|
| node-label recall | **0.971** | 0.836 |
| edge F1 | **0.860** | 0.729 |
| loop detection | 6/9 | 5/10 |
| box-shape accuracy | 61.4% | 55.3% |

Every structural measure got worse. And it did two specific harmful things.

It made the model *attempt* shapes without perceiving them. The output moved from
`365 rectangles, 3 diamonds` to `238 rectangles, 101 parallelograms, 9 rounded, 20
diamonds` — but wrongly, with 101 parallelograms against 56 in the real charts. So it
went from "always guess rectangle" (scoring the 60.2% baseline) to "guess shapes
badly" (55.3%), which is worse than the baseline.

And it copied its own example. Three flowcharts came back as *exactly seven boxes*
with label recall around 0.1 — matching the seven-box example in the prompt rather
than the image in front of it.

Reverted. Both prompts remain available so the comparison can be reproduced.

Since neither prompt version perceives shapes, shape recognition is a limitation of
the model's perception rather than of how we phrase the request, and prompting cannot
close it. That is the evidence-based argument for fine-tuning the model, if that is
pursued.

### The retry mechanism cannot help

Covered in §6.3. Only 4.9% of language questions can be checked against the graph at
all.

### Most of the pipeline's language-question deficit was formatting

Covered in §6.2. −8.9 points became −2.2 by changing what the Examiner reads.

### Checking the grader on easy questions proved nothing

Covered in §7. Both graders scored 30/30 on structural questions while differing by
20 points on language questions.

---

## 10. What these numbers can and cannot be used to claim

### Sample sizes are small

The head-to-head comparison uses 60 questions: 15 structural, and 45 language
questions per system. The −2.2 point language difference is **one answer**. The
per-subset language breakdowns are 30 questions each. The +53.3 point structural
difference is trustworthy at 15 questions only because the effect is enormous — 7/15
against 15/15 — not because 15 is a comfortable sample.

Vision measurements use 20 charts, and the per-subset breakdown there is about 7
charts each. Treat those splits as suggestive, not established.

### Everything except the router is measured on training data

We have not run the pipeline against `test_full.json`. The router classifier is the
only component with a held-out result. This is the most straightforward gap to close.

### These numbers must not be compared to FlowVQA's published leaderboard

This is the most important caveat in the document.

The published results — GPT-4V at 68.42%, TextFlow at up to about 82.7% — were
obtained by models reading the flowchart **as an image**. Our text pipeline reads the
flowchart's **Mermaid source code**, which the dataset happens to provide. That is
strictly more information and a substantially easier task.

So our 82.2% for a single LLM call is not "comparable to TextFlow", and it is not
evidence of anything about our system relative to theirs. Only the vision results in
§8 are comparable in kind, because only those start from an image.

We made this comparison ourselves early in the project and it was never valid.

### The grader deviates from the specified protocol

One grader instead of three, and a 3.8-billion-parameter open model instead of
GPT-3.5, Llama-2 70B and Mixtral 8×7B. Our validation tests all use definitely-wrong
answers, so they do not establish how the grader handles partially-correct ones.

### Only one answering model was tested

All language-model results use Qwen3-8B compressed to 4 bits. The counting failure in
§5 may be particular to this model or to the compression; we have not tested whether
a larger model counts properly.

### One parser behaviour remains unexplained

On the edge-count and maximum-indegree questions we get wrong, the dataset's answer is
usually smaller than ours, meaning our parser finds arrows the dataset does not count.
Worth one or two points on §3's figure.

---

## 11. What has not been built

**The Planner and behavioural-equivalence testing.** This is the largest remaining
gap. The `code` subset — 261 flowcharts in the training split — comes with the
original Python function the flowchart describes. The plan was to generate Python
*from* the flowchart, then run both the generated and original functions on random
inputs and compare their outputs. That is a completely deterministic measure of
correctness with no grader and no ambiguity, of the same kind as §3's 98.7%. It is
also the cheapest remaining source of rigour, and it currently has no data at all.

**Any evaluation on the test split** beyond the router.

**Fine-tuning the vision model**, which §8 and §9 jointly motivate as the only
plausible route to shape recognition.

**The directional-bias check.** The dataset ships every flowchart rendered a second
time bottom-to-top. Comparing the vision Reader across the two layouts would show
whether it reads the diagram's structure or merely its top-to-bottom position. The
tooling supports it; we have not run it.

---

## 12. Reproducing everything here

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                        # 72 tests

# structural questions, and the router: CPU only, seconds
python tools/parser_coverage.py data/train_full.json
python tools/train_router.py

# everything below needs a GPU:
#   pip install -r requirements-vlm.txt && pip install bitsandbytes accelerate
python tools/run_baseline.py   --n 120 --save runs/baseline.jsonl
python tools/run_ablation.py   --n 60 --representation mermaid --save runs/ablation.jsonl
python tools/score_run.py      runs/ablation.jsonl --save runs/scored.jsonl
python tools/validate_judge.py --n 25
python tools/eval_selfcheck.py runs/scored.jsonl --branch examiner
python tools/eval_vlm.py       --n 20 --layout main --save runs/vlm.jsonl
python tools/analyze_vlm_run.py runs/vlm.jsonl --per-sample
```

`notebooks/ablation_run.ipynb` runs the GPU sections end to end on Google Colab,
including the setup.
