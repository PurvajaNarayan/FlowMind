"""Scoring (spec §8). [Owner: C]

Four lanes:
  - graph_extraction: node/edge counts vs ground-truth mermaid (matters for VLM).
  - topological: exact match via graph_tool (should approach 100%).
  - content: best-of-3 vs A1/A2/A3 (embedding sim or LLM judge — spec §9).
  - behavioral_equivalence: run generated vs original Python on random inputs.
"""

from __future__ import annotations

import ast
import builtins as _builtins_module
import math
import multiprocessing
import random

from flowmind.reader.mermaid_reader import mermaid_to_graph
from flowmind.schema import FlowGraph


def graph_extraction_accuracy(pred: FlowGraph, gold_mermaid: str) -> dict:
    """Compare a (possibly VLM-produced) graph against the graph parsed from the
    ground-truth mermaid. Returns node/edge count match + exactness."""
    gold = mermaid_to_graph(gold_mermaid)
    return {
        "node_count_match": len(pred.nodes) == len(gold.nodes),
        "edge_count_match": len(pred.edges) == len(gold.edges),
        "pred_nodes": len(pred.nodes),
        "gold_nodes": len(gold.nodes),
        "pred_edges": len(pred.edges),
        "gold_edges": len(gold.edges),
    }


def topological_exact_match(prediction, gold: str) -> bool:
    """Exact match for graph-tool answers. gold comes from A1."""
    return str(prediction).strip() == str(gold).strip()


def content_match(prediction: str, references: list[str]) -> bool:
    """Best-of-3 match against A1/A2/A3.

    TODO(C): pick embedding similarity vs LLM judge and justify in the report
    (spec §9). Placeholder below is a naive containment check for wiring only.
    """
    p = prediction.strip().lower()
    return any(p and (p in r.lower() or r.lower() in p) for r in references)


# --- behavioral equivalence (spec §8 code-behavioral row) --------------------
#
# SANDBOXING: each call runs in its own subprocess with a curated builtins
# whitelist (no open/__import__/os/sys -- import statements fail naturally
# since __import__ is absent) and a wall-clock timeout that kills a hung
# process. This bounds runaway/infinite loops and isolates crashes from the
# caller. It is NOT adversarial-attacker-proof; it is sized for grading
# LLM-generated algorithmic code, not for running hostile input.

_SAFE_BUILTIN_NAMES = (
    "range", "len", "abs", "min", "max", "sum", "sorted", "reversed",
    "list", "dict", "tuple", "set", "frozenset", "str", "int", "float",
    "bool", "bytes", "enumerate", "zip", "map", "filter", "isinstance",
    "issubclass", "round", "divmod", "pow", "all", "any", "next", "iter",
    "Exception", "ValueError", "TypeError", "IndexError", "KeyError",
    "AttributeError", "StopIteration", "ZeroDivisionError",
    "ArithmeticError", "OverflowError", "RecursionError",
    "NotImplementedError", "RuntimeError", "NameError", "UnboundLocalError",
)
_SAFE_BUILTINS = {name: getattr(_builtins_module, name) for name in _SAFE_BUILTIN_NAMES
                  if hasattr(_builtins_module, name)}


def _run_in_subprocess(code: str, func_name: str, args: tuple, queue) -> None:
    try:
        g = {"__builtins__": dict(_SAFE_BUILTINS)}
        exec(code, g)
        fn = g.get(func_name)
        if fn is None or not callable(fn):
            queue.put(("error", f"{func_name!r} not defined"))
            return
        result = fn(*args)
        queue.put(("ok", result))
    except Exception as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _sandboxed_call(code: str, func_name: str, args: tuple,
                    timeout: float = 2.0) -> tuple[str, object]:
    """Run `func_name(*args)` from `code` in a separate process.

    Returns ("ok", value), ("error", message), or ("error", "timeout") if the
    process doesn't finish within `timeout` seconds (it is terminated).
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_in_subprocess, args=(code, func_name, args, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return ("error", "timeout")
    if not queue.empty():
        return queue.get()
    return ("error", f"process exited with code {proc.exitcode} and no result")


def _values_match(a: object, b: object) -> bool:
    """Tolerant equality: float-vs-float (or int-vs-float) via math.isclose,
    lists/tuples compared elementwise, everything else by ==."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
        except (TypeError, ValueError):
            return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_values_match(x, y) for x, y in zip(a, b))
    return a == b


def behavioral_equivalence(generated_code: str, original_code: str,
                           func_name: str, inputs: list[tuple]) -> float:
    """Run both functions on the given inputs, return fraction of matching
    outputs (spec §8 code-behavioral row).

    Exceptions are handled symmetrically: both sides raising counts as
    agreement regardless of exception type -- the generated function does not
    have to fail the exact same way, only fail exactly when the original
    does. One side raising and the other not is a disagreement.
    """
    if not inputs:
        raise ValueError("inputs must be non-empty")
    agreements = 0
    for args in inputs:
        gen_status, gen_val = _sandboxed_call(generated_code, func_name, args)
        orig_status, orig_val = _sandboxed_call(original_code, func_name, args)
        if gen_status == "error" and orig_status == "error":
            agreements += 1
        elif gen_status == "ok" and orig_status == "ok":
            agreements += _values_match(gen_val, orig_val)
    return agreements / len(inputs)


# --- input generation ---------------------------------------------------
#
# generate_inputs infers each parameter's ROLE from how original_code's AST
# actually USES it -- not from its name. Name-guessing was tried first and
# produced real false positives: a param named `A` (not matching any "looks
# like a list" name hint) got a random int instead of a list, so both the
# original and a genuinely buggy generated function raised the same
# "int is not subscriptable" TypeError on every trial -- scored as 100%
# "agreement" without the algorithm ever running. Usage analysis catches this
# because `A` is subscripted (`A[i]`) regardless of what it's called.
#
# Only ast.parse is used here, never exec -- so a module-level default-
# argument expression that references an undefined name (seen in the dataset:
# a sentinel like `arg=UNSET`) can't blow up introspection either.

_LIST_METHODS = {"append", "sort", "pop", "extend", "insert", "remove", "reverse"}
_STRING_METHODS = {"isalpha", "isdigit", "isalnum", "upper", "lower", "strip",
                   "split", "join", "isspace", "startswith", "endswith", "replace"}
# Last-resort tier when usage analysis finds no evidence at all for a param.
_LIST_HINTS = ("arr", "array", "lst", "nums", "values", "items", "elements", "seq", "list")
_STR_HINTS = ("str", "text", "word", "name", "line", "string")


def _range_bound_name(call: ast.Call) -> str | None:
    """For range(x) or range(a, x), return x's name if it's a bare Name."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "range"):
        return None
    args = call.args
    if len(args) == 1 and isinstance(args[0], ast.Name):
        return args[0].id
    if len(args) >= 2 and isinstance(args[1], ast.Name):
        return args[1].id
    return None


def _analyze_param_usage(func_node: ast.AST, param_names: list[str]) -> dict[str, dict]:
    """Infer a role for each parameter from how func_node's body uses it.

    Returns {name: {"role": "list"|"string"|"size"|"index"|"generic",
                    "domain": set[int]|None,     # role == "list"
                    "sized_list": str|None}}     # role in ("size", "index")
    """
    param_set = set(param_names)
    subscript_bases: dict[str, set[str]] = {}
    bound_of: dict[str, str] = {}
    list_method_target: set[str] = set()
    string_method_target: set[str] = set()
    bare_iterated: set[str] = set()
    bare_iter_source: dict[str, str] = {}
    equality_literals: dict[str, set[int]] = {}

    def names_in(node: ast.AST) -> set[str]:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    for node in ast.walk(func_node):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            base = node.value.id
            subscript_bases.setdefault(base, set()).update(names_in(node.slice))

        elif isinstance(node, ast.Compare):
            left = node.left
            for op, right in zip(node.ops, node.comparators):
                if isinstance(left, ast.Name) and isinstance(right, ast.Name):
                    if isinstance(op, (ast.Lt, ast.LtE)):
                        bound_of[left.id] = right.id
                    elif isinstance(op, (ast.Gt, ast.GtE)):
                        bound_of[right.id] = left.id
                if isinstance(op, ast.Eq):
                    sub = lit = None
                    if isinstance(left, ast.Subscript) and isinstance(right, ast.Constant) \
                            and isinstance(right.value, int):
                        sub, lit = left, right.value
                    elif isinstance(right, ast.Subscript) and isinstance(left, ast.Constant) \
                            and isinstance(left.value, int):
                        sub, lit = right, left.value
                    if sub is not None and isinstance(sub.value, ast.Name):
                        equality_literals.setdefault(sub.value.id, set()).add(lit)
                left = right

        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            var, it = node.target.id, node.iter
            if isinstance(it, ast.Name):
                bare_iterated.add(it.id)
                bare_iter_source[var] = it.id
            elif isinstance(it, ast.Call):
                bound_name = _range_bound_name(it)
                if bound_name:
                    bound_of[var] = bound_name

        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.attr in _LIST_METHODS:
                list_method_target.add(node.value.id)
            elif node.attr in _STRING_METHODS:
                string_method_target.add(node.value.id)

    # A bare-iterated name is string-like if the loop variable it produces is
    # used with a string method (`for ch in text: ch.isalpha()` -> text).
    for loop_var, source in bare_iter_source.items():
        if loop_var in string_method_target:
            string_method_target.add(source)

    string_params = {p for p in param_names if p in string_method_target}
    list_params = {p for p in param_names
                   if p not in string_params
                   and (p in subscript_bases or p in list_method_target or p in bare_iterated)}

    def sizes_which_list(p: str) -> str | None:
        for lst in list_params:
            for idx_name in subscript_bases.get(lst, ()):
                if idx_name == p:
                    return lst
                seen, cur = set(), idx_name
                while cur in bound_of and cur not in seen:
                    seen.add(cur)
                    cur = bound_of[cur]
                    if cur == p:
                        return lst
        return None

    roles: dict[str, dict] = {}
    for p in param_names:
        if p in string_params:
            roles[p] = {"role": "string"}
        elif p in list_params:
            roles[p] = {"role": "list", "domain": equality_literals.get(p)}
        else:
            sized = sizes_which_list(p)
            if sized:
                roles[p] = {"role": "size", "sized_list": sized}
            else:
                indexed = next((lst for lst in list_params
                               if p in subscript_bases.get(lst, ())), None)
                if indexed:
                    roles[p] = {"role": "index", "sized_list": indexed}
                else:
                    roles[p] = {"role": "generic"}
    return roles


def _generate_from_roles(roles: dict[str, dict], param_names: list[str],
                         n: int, seed: int) -> list[tuple]:
    rng = random.Random(seed)
    trials: list[tuple] = []
    for _ in range(n):
        lists: dict[str, list[int]] = {}
        for p in param_names:
            if roles[p]["role"] == "list":
                length = rng.randint(1, 8)
                domain = roles[p].get("domain")
                if domain:
                    pool = sorted(domain)
                    lists[p] = [rng.choice(pool) if rng.random() < 0.8
                               else rng.randint(-20, 20) for _ in range(length)]
                else:
                    lists[p] = [rng.randint(-20, 20) for _ in range(length)]

        args = []
        for p in param_names:
            info = roles[p]
            role = info["role"]
            if role == "list":
                args.append(list(lists[p]))
            elif role == "size":
                paired = lists.get(info["sized_list"])
                args.append(len(paired) if paired is not None else rng.randint(1, 8))
            elif role == "index":
                paired = lists.get(info["sized_list"]) or []
                args.append(rng.randint(0, max(len(paired) - 1, 0)))
            elif role == "string":
                args.append(rng.choice(["hello", "world", "abc", "test", ""]))
            else:  # generic: a name hint is still free signal if usage found nothing
                name = p.lower()
                if any(h in name for h in _LIST_HINTS):
                    args.append([rng.randint(-20, 20) for _ in range(rng.randint(1, 8))])
                elif any(h in name for h in _STR_HINTS):
                    args.append(rng.choice(["hello", "world", "abc", "test", ""]))
                else:
                    args.append(rng.randint(-50, 50))
        trials.append(tuple(args))
    return trials


def generate_inputs(original_code: str, func_name: str, n: int = 20,
                    seed: int = 0) -> list[tuple]:
    """Guess type-appropriate random inputs by analyzing how original_code's
    AST actually uses each parameter (spec §8 TODO).

    Only parses original_code, never execs it -- introspection can't be
    broken by a module-level default-argument expression referencing an
    undefined name.

    Raises ValueError if the first parameter is `self`: some `code` entries in
    the dataset are methods extracted out of class context (they reference
    undefined instance attributes), not standalone functions, and cannot be
    meaningfully tested with synthesized inputs. Callers should treat this as
    "not applicable", not a failure.
    """
    tree = ast.parse(original_code)
    func_node = next(
        (node for node in tree.body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name),
        None,
    )
    if func_node is None:
        raise ValueError(f"{func_name!r} not defined in original_code")

    param_names = [a.arg for a in func_node.args.args]
    if param_names and param_names[0] == "self":
        raise ValueError(
            f"{func_name!r} looks like a method, not a standalone function "
            "(first parameter is 'self') -- cannot synthesize a receiver"
        )

    roles = _analyze_param_usage(func_node, param_names)
    return _generate_from_roles(roles, param_names, n, seed)
