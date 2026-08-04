"""FlowMind chat frontend.

A small Flask UI over the real pipeline (Reader -> router -> {graph_tool |
graph_tool_llm | Examiner}), separate from the FlowVQA eval scripts in
tools/. One composer field accepts pasted Mermaid, an attached image, and the
question together; /analyze runs them through the same components the
notebooks and tools/run_*.py use and returns the final answer plus the
diagram to render.

Not part of the eval pipeline -- doesn't read data/, doesn't write runs/.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

# The default LLM (llm.DEFAULT_MODEL_ID, Qwen3-8B) isn't cached locally and
# would trigger a ~16GB download on first use. Qwen3-4B already is (see
# llm.py's own fp16-fallback note) -- set it here rather than in llm.py so
# eval reproducibility elsewhere is untouched. Override with FLOWMIND_LLM_MODEL.
os.environ.setdefault("FLOWMIND_LLM_MODEL", "Qwen/Qwen3-4B")

from flask import Flask, jsonify, render_template, request

from flowmind import graph_tool as gt
from flowmind.data import QAItem
from flowmind.examiner import answer as examiner_answer
from flowmind.graph_tool_llm import answer_topological_llm
from flowmind.llm import get_client
from flowmind.planner import _ordered_steps  # reuse the DFS step-walk, not the code-gen contract
from flowmind.reader.mermaid_reader import mermaid_to_graph
from flowmind.router import route
from flowmind.schema import FlowGraph

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB, generous for a flowchart photo

# --- splitting one composer message into (mermaid, question) -------------------

_FENCE_RE = re.compile(r"```(?:mermaid)?\s*\n(.*?)```", re.S)
_DIRECTIVE_RE = re.compile(r"^\s*(flowchart|graph)\s+(TD|LR|RL|BT|TB)\b", re.I)
# A line still "looks like Mermaid" if it has an edge arrow, a bare/shaped node
# reference, or one of the metadata keywords the parser itself skips.
_MERMAID_LINE_RE = re.compile(
    r'[-.=]{2,}>'
    r'|^\s*[A-Za-z0-9_]+\s*[\[{(]'
    r'|^\s*(subgraph|end|classDef|class |style |linkStyle|click |direction)\b',
    re.I,
)


def split_composer_text(text: str) -> tuple[str, str]:
    """Pull a Mermaid diagram out of one freeform composer message.

    Preferred form is a fenced ```mermaid block (unambiguous). Falls back to
    raw-pasted Mermaid followed by the question on its own line: everything
    from a `flowchart`/`graph` directive is kept as Mermaid while lines keep
    looking like Mermaid syntax, and the first line that doesn't -- typically
    the question -- ends the block.
    """
    text = text.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        mermaid = fence.group(1).strip()
        question = (text[:fence.start()] + " " + text[fence.end():]).strip()
        return mermaid, question

    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if _DIRECTIVE_RE.match(l)), None)
    if start is None:
        return "", text

    end = start + 1
    while end < len(lines) and (not lines[end].strip() or _MERMAID_LINE_RE.search(lines[end])):
        end += 1
    mermaid = "\n".join(lines[start:end]).strip()
    question = "\n".join(lines[:start] + lines[end:]).strip()
    return mermaid, question


# --- code_request lane ----------------------------------------------------------
#
# flowmind.planner.plan() needs the *original* code to extract a matching
# function signature (see its module docstring) -- there is no such thing for
# a flowchart a user just typed in, so it doesn't fit here. This lane reuses
# only the graph -> ordered-steps walk and asks the LLM directly, with no
# behavioral-equivalence contract to satisfy.

CODE_SYSTEM = (
    "You write a concise markdown plan and a single self-contained Python "
    "function that implement the algorithm described by a flowchart, and "
    "that answer the user's specific question about it. Output a short plan "
    "as bullet points, then the function inside one fenced ```python block. "
    "No explanation outside the plan and the code block."
)


def _freeform_code_answer(graph: FlowGraph, question: str) -> str:
    steps = _ordered_steps(graph)
    prompt = f"Flowchart steps, in order:\n{steps}\n\nQuestion: {question}"
    return get_client().complete(prompt, system=CODE_SYSTEM, max_new_tokens=512).strip()


# --- routes -----------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    message = (request.form.get("message") or "").strip()
    image = request.files.get("image")
    has_image = bool(image and image.filename)

    if not message and not has_image:
        return jsonify(error="Type a question (and paste a diagram or attach an image)."), 400

    mermaid_text, question = split_composer_text(message)
    source = "pasted Mermaid"

    if not mermaid_text:
        if not has_image:
            return jsonify(error=(
                "I couldn't find a Mermaid diagram in your message and no image was "
                "attached. Paste the diagram (a ```mermaid fenced block is most "
                "reliable) or attach a picture of the flowchart."
            )), 400
        suffix = Path(image.filename).suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            image.save(tmp.name)
            try:
                from flowmind.reader.vlm_reader import image_to_mermaid
                mermaid_text = image_to_mermaid(tmp.name)
            except Exception as exc:
                return jsonify(error=(
                    "Couldn't read the flowchart image with the VLM reader "
                    f"({exc}). Install requirements-vlm.txt, or paste the "
                    "Mermaid text instead."
                )), 500
        source = "image, read via Qwen3-VL"

    if not question:
        return jsonify(error="What's your question about this flowchart?"), 400

    try:
        graph = mermaid_to_graph(mermaid_text)
    except Exception as exc:
        return jsonify(error=f"Couldn't parse that Mermaid text: {exc}"), 400
    if not graph.nodes:
        return jsonify(error="Couldn't find any flowchart nodes in that Mermaid text."), 400

    intent = route(question)

    try:
        if intent == "topological":
            kind, pred, unresolved = gt.answer_topological(graph, question)
            branch = "graph tool (deterministic)"
            if kind is None:
                kind, pred, unresolved = answer_topological_llm(graph, question, get_client())
                branch = "graph tool (LLM picked the function)"

            if kind is None:
                answer_text = ("That doesn't match a topological question the graph "
                               "tool understands (node/edge counts, degree, shortest "
                               "path, direct predecessor/successor).")
            elif unresolved or pred is None:
                answer_text = ("The flowchart doesn't resolve that — check the node "
                               "names in your question against the diagram.")
            else:
                answer_text = str(pred)
            meta = {"branch": branch}

        elif intent == "code_request":
            answer_text = _freeform_code_answer(graph, question)
            meta = {"branch": "code generator (LLM, from the flowchart's step order)"}

        else:  # "content"
            item = QAItem(sample_key="webapp", question_id="1", question=question,
                         answers=[], qa_type=intent, mermaid=mermaid_text)
            result = examiner_answer(graph, item, representation="mermaid")
            answer_text = result.answer
            meta = {"branch": "examiner (LLM + graph self-check)",
                    "verdict": result.verdict, "revisions": result.revisions}
    except Exception as exc:
        return jsonify(error=f"The model backend hit an error: {exc}"), 500

    return jsonify(ok=True, mermaid=mermaid_text, question=question,
                   source=source, intent=intent, answer=answer_text, **meta)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
