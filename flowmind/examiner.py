"""Examiner agent (spec §7.3). [Owner: B]

Answers content questions (fact_retrieval / applied_scenario / flow_referential)
from the graph + node text, then self-checks the answer. Topological questions
never reach here — the router sends them to graph_tool.

The revision loop is LOCAL to this branch. Re-running the Reader on `revise`
would be a no-op for the deterministic text Reader (same mermaid in, same graph
out), so instead this re-prompts the same LLM with its previous answer and why it
was rejected, asking it to reconsider. Capped at MAX_REVISIONS attempts.

THE SELF-CHECK MUST NOT SEE THE GOLD ANSWERS
--------------------------------------------
An earlier version triggered revision on `content_match(reply, item.answers)` and
interpolated `item.answers` into the retry prompt, so on a retry the model was
shown the reference answers and asked again. That invalidates the §8 ablation: the
pipeline arm would get three attempts with the answer key visible while the
baseline arm gets one blind attempt, so the delta is guaranteed positive and
measures leakage rather than architecture.

Revision is therefore triggered only by checks computable from the GRAPH alone --
which is also what §7.3's own example describes ("referenced node not found in
graph"):

  * an empty answer
  * an answer that quotes a phrase appearing nowhere in the flowchart
  * an answer with no lexical overlap with the flowchart at all

`verdict` now means "passed its own self-checks", NOT "is correct". Correctness is
a separate, after-the-fact judgement made by flowmind.judge via
tools/score_run.py, once the pipeline has committed to an answer. Keeping the two
apart is what makes the ablation meaningful -- and the gap between them is itself
a result, since it says whether the Examiner's self-assessment is calibrated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from flowmind.data import QAItem
from flowmind.llm import LLMClient, get_client
from flowmind.schema import FlowGraph

MAX_REVISIONS = 2  # spec §7.3

SYSTEM = (
    "You answer questions about flowcharts. You are given the flowchart as a "
    "list of nodes and edges. Answer only from the graph. Be concise: a phrase "
    "or a single sentence. Do not restate the question and do not explain your "
    "reasoning."
)

# Wording deliberately identical to eval.ablation.SYSTEM. With
# representation="mermaid" the Examiner then differs from the single-pass baseline
# in exactly one respect -- the revision loop -- which is what makes the
# representation experiment below a controlled comparison rather than a confound.
SYSTEM_MERMAID = (
    "You answer questions about flowcharts. You are given the flowchart as a "
    "Mermaid.js script. Answer only from the flowchart. Be concise: a phrase or a "
    "single sentence. Do not restate the question and do not explain your "
    "reasoning."
)

REPRESENTATIONS = ("graph", "mermaid")

# WHY REPRESENTATION IS SELECTABLE
# --------------------------------
# On the first 60-item ablation the pipeline scored BELOW the single-pass baseline
# on content questions (-4.4 points under one judge, -8.9 under another). The loop
# is not the explanation: it fired twice in 60 items, so the content arm was
# essentially a single call.
#
# What did differ was the input. The Examiner reads the serialized node/edge
# listing; the baseline reads the raw Mermaid. Mermaid preserves ordering and
# nesting and is a format models have seen a great deal of, so it may simply be
# the better representation -- in which case the deficit says nothing about the
# agent architecture and everything about a formatting choice.
#
# Running the Examiner over both representations separates those two explanations.
# Grounding checks still run against the graph either way, since they need node
# and edge identity rather than surface text.


@dataclass
class ExaminerResult:
    answer: str
    # "accept" = passed the graph-only self-checks. NOT a correctness claim;
    # see the module docstring.
    verdict: str
    revise_reason: str | None = None  # sent back to Reader when verdict == "revise"
    revisions: int = 0
    attempts: list[dict] = field(default_factory=list)  # per-attempt prompt/answer/reason
    representation: str = "graph"   # which view of the chart the model was shown


# Words too common to count as evidence that an answer is grounded in a chart.
_STOPWORDS = frozenset("""
a an the is are was were be been being do does did done to of in on at for from with by as
it its this that these those and or but not no if then else when what which who whom how
many much you your yours we our they their there here i me my he she his her
step steps node nodes edge edges flowchart chart diagram box boxes answer question
""".split())

# Quoted spans, in any of the quote styles a model reaches for. Three characters
# minimum so single letters and stray apostrophes are not treated as citations.
_QUOTED_RE = re.compile(
    r'"([^"]{3,})"' r"|'([^']{3,})'" r"|`([^`]{3,})`" r"|“([^”]{3,})”"
)


def _graph_to_text(graph: FlowGraph) -> str:
    """Serialize a FlowGraph into the node/edge listing the LLM sees.

    The Examiner is handed a parsed FlowGraph, not raw mermaid, so this has no
    equivalent elsewhere in the repo (the Reader only goes mermaid -> graph).
    """
    lines = ["Nodes:"]
    for n in graph.nodes:
        lines.append(f"- {n.id} ({n.shape.value}): {n.label}")
    lines.append("Edges:")
    for e in graph.edges:
        suffix = f" [{e.label}]" if e.label else ""
        lines.append(f"- {e.source} -> {e.target}{suffix}")
    return "\n".join(lines)


def build_content_prompt(graph: FlowGraph, item: QAItem,
                         representation: str = "graph") -> str:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}, "
                         f"got {representation!r}")
    body = (item.mermaid.strip() if representation == "mermaid"
            else _graph_to_text(graph))
    header = "Flowchart (Mermaid):" if representation == "mermaid" else "Flowchart:"
    return "\n".join([header, body, "", f"Question: {item.question.strip()}"])


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", (text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def _quoted_spans(text: str) -> list[str]:
    return [next(g for g in m.groups() if g) for m in _QUOTED_RE.finditer(text or "")]


def check_grounding(reply: str, graph: FlowGraph) -> str | None:
    """Why this answer should be revised, judged against the graph alone.

    Returns a reason, or None if the answer passes. Never consults gold answers —
    see the module docstring for why that distinction decides whether the §8
    ablation means anything.
    """
    text = (reply or "").strip()
    if not text:
        return "the answer was empty"

    chart_text = " ".join(
        [n.label for n in graph.nodes] + [e.label or "" for e in graph.edges]
    ).lower()

    # A quoted phrase is the model citing the chart. If the citation is not in the
    # chart, it invented a step -- exactly §7.3's "referenced node not found".
    for span in _quoted_spans(text):
        if span.strip().lower() not in chart_text:
            return (f"it cites {span!r}, which does not appear anywhere in the "
                    f"flowchart")

    # No overlap at all suggests the answer is not about this chart. Only applied
    # to answers long enough that overlap should be expected: terse-but-correct
    # replies ("Yes", "0", "Byte") legitimately share nothing with node labels and
    # must not be punished for brevity.
    words = _content_words(text)
    if len(words) >= 3 and not (words & _content_words(chart_text)):
        return "it does not reference anything in the flowchart"

    return None


# --- question-aware checks -------------------------------------------------------
#
# check_grounding only asks "is this answer about this chart?". Measured against
# seven answers the judge marked wrong, it caught zero of them, because every real
# error is *about* the chart: the model names real steps and picks the wrong one.
# One observed failure was answering with the correct answer to a DIFFERENT question
# about the same flowchart -- perfectly grounded, completely wrong.
#
# These checks bring the question into it. Both are computable from question +
# graph, never from gold, and both are deliberately conservative: they fire only
# when the graph settles the matter, because a false alarm sends a correct answer
# back for revision and can make things worse.

_ORDER_AFTER = ("next", "after", "following", "follows", "subsequent", "then")
_ORDER_BEFORE = ("before", "previous", "preceding", "precedes", "prior")
_COUNT_ASK = ("how many", "number of")


# Node references in a question, in every quoting style FlowVQA uses.
#
# Topological questions use doubled double-quotes: ""like this."" Content questions
# use SINGLE quotes inside prose -- "Upon reaching the 'calculate median' step".
# An earlier version handled only the double-quoted form, lifted from
# parser_coverage which deals with topological questions, so on content questions it
# always returned [] and every check downstream of it silently never ran. Measured:
# 0 of 7422 content questions contain a double-quoted span, while 1769 contain a
# single-quoted one and 923 of those resolve to a real node label.
_Q_SPANS = re.compile(
    r'""(.*?)""'          # ""doubled"" -- topological style
    r'|"([^"]{3,})"'      # "plain double"
    r"|'([^']{3,})'"      # 'single' -- content style, the one that was missing
)


def _labels_quoted_in_question(question: str) -> list[str]:
    """Every quoted span in the question that might name a node.

    Three-character minimum so stray apostrophes and single letters are not treated
    as node references.
    """
    out = []
    for m in _Q_SPANS.finditer(question or ""):
        span = next((g for g in m.groups() if g), "").strip()
        if span:
            out.append(span)
    return out


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().rstrip(".").lower()


# A quoted span matching this many nodes by substring is too vague to anchor on:
# "count" hits "Set count to 0", "count < n + 1?" and "Increment count", and taking
# all three unions their neighbourhoods until almost any answer looks adjacent.
_MAX_AMBIGUOUS_ANCHORS = 2


def _resolve(graph: FlowGraph, label: str) -> list[str]:
    """Node ids a quoted span could name. Exact label match wins; substring is a
    fallback, and is dropped entirely when it is too ambiguous to be informative."""
    want = _norm_label(label)
    if not want:
        return []
    exact = [n.id for n in graph.nodes if _norm_label(n.label) == want]
    if exact:
        return exact
    loose = [n.id for n in graph.nodes if want in _norm_label(n.label)]
    return loose if len(loose) <= _MAX_AMBIGUOUS_ANCHORS else []


def _mentions(reply: str, label: str) -> bool:
    """Does the answer clearly refer to this node? Substring on the normalised
    label, which is strict enough to avoid matching on one shared word."""
    lab = _norm_label(label)
    return bool(lab) and len(lab) > 3 and lab in _norm_label(reply)


def _check_adjacency(reply: str, graph: FlowGraph, question: str) -> str | None:
    """For "what comes next after X" style questions, is the named step adjacent?

    Fires only when the answer names some other node in the chart *and* names none
    of the ones actually adjacent to X. Answers that paraphrase rather than quote a
    label mention nothing exactly, so they pass -- deliberately.
    """
    q = (question or "").lower()
    after = any(w in q for w in _ORDER_AFTER)
    before = any(w in q for w in _ORDER_BEFORE)
    if not (after or before):
        return None

    anchors = [i for lab in _labels_quoted_in_question(question) for i in _resolve(graph, lab)]
    if not anchors:
        return None

    if after:
        adj = {e.target for e in graph.edges if e.source in anchors}
    else:
        adj = {e.source for e in graph.edges if e.target in anchors}
    if not adj:
        return None

    labels = {n.id: n.label for n in graph.nodes}
    if any(_mentions(reply, labels.get(i, "")) for i in adj):
        return None                                   # named a genuine neighbour

    named = [labels[i] for i in labels
             if i not in adj and i not in anchors and _mentions(reply, labels[i])]
    if named:
        direction = "follows" if after else "precedes"
        return (f"it names {named[0][:40]!r}, which is a step in the flowchart but "
                f"not one that {direction} the step the question asks about")
    return None


def _check_count(reply: str, graph: FlowGraph, question: str) -> str | None:
    """For "how many steps between X and Y" questions, does the number check out?

    Only fires when the question quotes two resolvable labels, so the graph gives a
    definite answer. Both the edge count and the node count along the path are
    accepted, since "steps" is used for either.
    """
    q = (question or "").lower()
    if not any(w in q for w in _COUNT_ASK):
        return None
    nums = [int(m) for m in re.findall(r"\b(\d{1,3})\b", reply or "")]
    if not nums:
        return None

    labs = _labels_quoted_in_question(question)
    if len(labs) < 2:
        return None
    A, B = _resolve(graph, labs[0]), _resolve(graph, labs[1])
    if not (A and B):
        return None

    import networkx as nx
    g = graph.to_networkx()
    lengths = []
    for a in A:
        for b in B:
            try:
                lengths.append(nx.shortest_path_length(g, a, b))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
    if not lengths:
        return None
    d = min(lengths)
    if any(n in (d, d + 1) for n in nums):            # edges, or nodes on the path
        return None
    return (f"it says {nums[0]}, but the flowchart gives {d} edges "
            f"({d + 1} steps) between the two named nodes")


def check_answers_question(reply: str, graph: FlowGraph,
                           question: str) -> str | None:
    """Question-aware self-checks. Gold is never consulted."""
    for check in (_check_adjacency, _check_count):
        reason = check(reply, graph, question)
        if reason:
            return reason
    return None


def check_answer(reply: str, graph: FlowGraph, question: str) -> str | None:
    """Every self-check, in order of cost. Grounding first since it is cheapest."""
    return (check_grounding(reply, graph)
            or check_answers_question(reply, graph, question))


def _revise_prompt(base_prompt: str, prev_answer: str, reason: str) -> str:
    return (
        f"{base_prompt}\n\n"
        f"Your previous answer was: {prev_answer!r}\n"
        f"It was rejected because {reason}.\n"
        "Answer again using only the flowchart above."
    )


def answer(graph: FlowGraph, item: QAItem, client: LLMClient | None = None,
           max_revisions: int = MAX_REVISIONS, max_new_tokens: int = 128,
           representation: str = "graph") -> ExaminerResult:
    """Generate a content answer and self-check it against the graph.

    On a failed check, re-prompts the same LLM with its previous answer and the
    reason it was rejected, up to `max_revisions` retries. If it still fails,
    returns verdict="revise" as a final unresolved state rather than presenting an
    ungrounded answer as though it had passed.

    `item` is used for the question text only. The gold answers in `item.answers`
    are deliberately never read here — correctness is judged afterwards, by
    flowmind.judge, once the pipeline has committed.
    """
    client = client or get_client()
    base_prompt = build_content_prompt(graph, item, representation=representation)
    system = SYSTEM_MERMAID if representation == "mermaid" else SYSTEM
    prompt = base_prompt
    attempts: list[dict] = []

    revisions = 0
    while True:
        reply = client.complete(prompt, system=system,
                                max_new_tokens=max_new_tokens).strip()
        reason = check_answer(reply, graph, item.question)
        attempts.append({"prompt": prompt, "answer": reply, "reason": reason})

        if reason is None:
            return ExaminerResult(answer=reply, verdict="accept",
                                  revisions=revisions, attempts=attempts,
                                  representation=representation)

        if revisions >= max_revisions:
            return ExaminerResult(answer=reply, verdict="revise", revise_reason=reason,
                                  revisions=revisions, attempts=attempts,
                                  representation=representation)

        prompt = _revise_prompt(base_prompt, reply, reason)
        revisions += 1
