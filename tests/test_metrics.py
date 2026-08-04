"""behavioral_equivalence / generate_inputs, no LLM, no GPU.

Uses hand-written code strings so these tests are fast and deterministic; the
sample.json fixture's find_fixed_point(arr, n) is used for generate_inputs
since it matches the project's real dataset shape.
"""

import pytest

from flowmind.eval.metrics import (
    _sandboxed_call,
    behavioral_equivalence,
    generate_inputs,
)

ORIGINAL = "def add(a, b):\n    return a + b\n"
IDENTICAL = "def add(a, b):\n    return a + b\n"
WRONG = "def add(a, b):\n    return a - b\n"
RAISES_ON_ZERO = "def add(a, b):\n    if b == 0:\n        raise ValueError('no')\n    return a + b\n"
RAISES_DIFFERENTLY_ON_ZERO = (
    "def add(a, b):\n    if b == 0:\n        return 1 / 0\n    return a + b\n"
)


def test_identical_code_scores_perfectly():
    inputs = [(1, 2), (3, 4), (-1, 5)]
    assert behavioral_equivalence(IDENTICAL, ORIGINAL, "add", inputs) == 1.0


def test_wrong_code_scores_zero():
    inputs = [(1, 2), (3, 4)]
    assert behavioral_equivalence(WRONG, ORIGINAL, "add", inputs) == 0.0


def test_symmetric_exceptions_count_as_agreement():
    """Both raise (different exception types) on b=0 -- still agreement."""
    inputs = [(1, 0), (2, 3)]
    score = behavioral_equivalence(RAISES_DIFFERENTLY_ON_ZERO, RAISES_ON_ZERO,
                                   "add", inputs)
    assert score == 1.0


def test_one_sided_exception_is_a_disagreement():
    inputs = [(1, 0)]  # original raises, generated (plain add) does not
    assert behavioral_equivalence(ORIGINAL, RAISES_ON_ZERO, "add", inputs) == 0.0


def test_float_tolerance():
    gen = "def half(x):\n    return x / 2 + 1e-10\n"
    orig = "def half(x):\n    return x / 2\n"
    assert behavioral_equivalence(gen, orig, "half", [(10,)]) == 1.0


def test_empty_inputs_raises():
    with pytest.raises(ValueError):
        behavioral_equivalence(IDENTICAL, ORIGINAL, "add", [])


def test_sandboxed_call_kills_infinite_loop():
    status, val = _sandboxed_call("def f():\n    while True:\n        pass\n",
                                  "f", (), timeout=0.3)
    assert status == "error"
    assert val == "timeout"


def test_sandboxed_call_blocks_import():
    status, val = _sandboxed_call("import os\ndef f():\n    return os.getcwd()\n",
                                  "f", ())
    assert status == "error"


# --- generate_inputs ------------------------------------------------------

FIND_FIXED_POINT = "def find_fixed_point(arr, n):\n    for i in range(n):\n        if arr[i] == i:\n            return i\n    return -1\n"


def test_generate_inputs_ties_size_param_to_list_length():
    trials = generate_inputs(FIND_FIXED_POINT, "find_fixed_point", n=15, seed=0)
    assert len(trials) == 15
    for arr, n in trials:
        assert isinstance(arr, list)
        assert n == len(arr)


def test_generate_inputs_never_causes_index_error_on_original():
    trials = generate_inputs(FIND_FIXED_POINT, "find_fixed_point", n=15, seed=0)
    g = {}
    exec(FIND_FIXED_POINT, g)
    fn = g["find_fixed_point"]
    for args in trials:
        fn(*args)  # must not raise IndexError


def test_generate_inputs_rejects_self_first_param():
    code = "def close(self):\n    return self.value\n"
    with pytest.raises(ValueError):
        generate_inputs(code, "close")


def test_generate_inputs_raises_for_missing_function():
    with pytest.raises(ValueError):
        generate_inputs(FIND_FIXED_POINT, "not_a_real_function")


# --- usage-based role inference: the real dataset bugs that motivated it ---
#
# Manual review of a real run caught false positives from the old name-only
# heuristic: single-letter array params (A, B, C) and non-hinted size params
# (p, q, r) fell back to random ints, so the original crashed on every trial
# for a reason unrelated to the algorithm, and a genuinely buggy generated
# function scored 1.0 by "agreeing" on that crash. These tests use the real
# dataset code verbatim.

FIND_CLOSEST = (
    "import sys\n"
    "def find_closet(A, B, C, p, q, r):\n"
    "    diff = sys.maxsize\n"
    "    res_i = 0\n"
    "    res_j = 0\n"
    "    res_k = 0\n"
    "    i = 0\n"
    "    j = 0\n"
    "    k = 0\n"
    "    while i < p and j < q and k < r:\n"
    "        minimum = min(A[i], min(B[j], C[k]))\n"
    "        maximum = max(A[i], max(B[j], C[k]))\n"
    "        if maximum - minimum < diff:\n"
    "            res_i = i\n"
    "            res_j = j\n"
    "            res_k = k\n"
    "            diff = maximum - minimum\n"
    "        if diff == 0:\n"
    "            break\n"
    "        if A[i] == minimum:\n"
    "            i = i + 1\n"
    "        elif B[j] == minimum:\n"
    "            j = j + 1\n"
    "        else:\n"
    "            k = k + 1\n"
    "    return A[res_i], B[res_j], C[res_k]\n"
)

FIND_MIN_SWAPS = (
    "def find_Min_Swaps(arr, n):\n"
    "    noOfZeroes = [0] * n\n"
    "    count = 0\n"
    "    noOfZeroes[n - 1] = 1 - arr[n - 1]\n"
    "    for i in range(n - 2, -1, -1):\n"
    "        noOfZeroes[i] = noOfZeroes[i + 1]\n"
    "        if arr[i] == 0:\n"
    "            noOfZeroes[i] = noOfZeroes[i] + 1\n"
    "    for i in range(0, n):\n"
    "        if arr[i] == 1:\n"
    "            count = count + noOfZeroes[i]\n"
    "    return count\n"
)

RETURN_LETTERS = (
    "def return_letters_from_string(text):\n"
    "    out = ''\n"
    "    for letter in text:\n"
    "        if letter.isalpha():\n"
    "            out += letter\n"
    "    return out\n"
)


def test_single_letter_array_params_are_recognized_as_lists():
    """A/B/C don't match any name hint but are subscripted -- usage analysis
    must still type them as lists, and p/q/r as each one's own size."""
    trials = generate_inputs(FIND_CLOSEST, "find_closet", n=15, seed=0)
    g: dict = {}
    exec(FIND_CLOSEST, g)
    fn = g["find_closet"]
    for A, B, C, p, q, r in trials:
        assert isinstance(A, list) and isinstance(B, list) and isinstance(C, list)
        assert p == len(A) and q == len(B) and r == len(C)
        fn(A, B, C, p, q, r)  # must not raise -- the bug this fix targets


def test_equality_checks_narrow_the_value_domain():
    """arr[i] == 0 and arr[i] == 1 should bias generated values toward {0, 1},
    not the wide generic range -- otherwise a binary-array algorithm never
    gets exercised."""
    trials = generate_inputs(FIND_MIN_SWAPS, "find_Min_Swaps", n=30, seed=0)
    all_values = [v for arr, _n in trials for v in arr]
    assert all_values, "expected at least one generated element"
    in_domain = sum(1 for v in all_values if v in (0, 1))
    assert in_domain / len(all_values) > 0.5


def test_bare_char_iteration_with_string_method_infers_string_role():
    trials = generate_inputs(RETURN_LETTERS, "return_letters_from_string", n=10, seed=0)
    for (text,) in trials:
        assert isinstance(text, str)


def test_generic_param_with_no_usage_evidence_falls_back_to_int():
    code = "def f(x):\n    return x\n"
    trials = generate_inputs(code, "f", n=5, seed=0)
    for (x,) in trials:
        assert isinstance(x, int)
