from flowmind.graph_tool_llm import (
    answer_topological_llm,
    answer_topological_with_fallback,
    build_prompt,
)
from flowmind.llm import ScriptedClient


def _reply(function, labels):
    import json
    return json.dumps({"function": function, "labels": labels})


def test_llm_picks_zero_arg_function(graph):
    # free-form phrasing the keyword parser would miss; LLM picks node_count,
    # deterministic compute returns the real count (7 nodes in the fixture).
    client = ScriptedClient(_reply("node_count", []))
    assert answer_topological_llm(graph, "tally up the boxes", client) == ("node_count", 7, False)


def test_llm_picks_two_arg_function(graph):
    # Start -> ... -> End is 5 edges in the fixture.
    client = ScriptedClient(_reply("shortest_path", ["Start", "End"]))
    assert answer_topological_llm(graph, "how far apart are the ends", client) == (
        "shortest_path", 5, False,
    )


def test_llm_reply_tolerates_code_fences(graph):
    fenced = "```json\n" + _reply("edge_count", []) + "\n```"
    client = ScriptedClient(fenced)
    assert answer_topological_llm(graph, "count the arrows", client) == ("edge_count", 7, False)


def test_unresolvable_label_flags_unresolved(graph):
    client = ScriptedClient(_reply("direct_predecessor", ["Start", "Nonexistent"]))
    kind, pred, unresolved = answer_topological_llm(graph, "does start precede foo", client)
    assert (kind, pred, unresolved) == ("direct_predecessor", None, True)


def test_bad_json_reply_is_unhandled(graph):
    client = ScriptedClient("I think it's about 7 nodes.")
    assert answer_topological_llm(graph, "whatever", client) == (None, None, False)


def test_unknown_function_is_rejected(graph):
    client = ScriptedClient(_reply("delete_everything", []))
    assert answer_topological_llm(graph, "wreck it", client) == (None, None, False)


def test_wrong_label_count_is_rejected(graph):
    # shortest_path needs 2 labels; one is invalid.
    client = ScriptedClient(_reply("shortest_path", ["Start"]))
    assert answer_topological_llm(graph, "path length", client) == (None, None, False)


def test_fallback_skips_llm_when_parser_handles_it(graph):
    client = ScriptedClient(_reply("node_count", []))
    result = answer_topological_with_fallback(graph, "How many nodes exist?", client)
    assert result == ("node_count", 7, False)
    assert client.prompts == []  # deterministic parser answered; LLM never called


def test_fallback_invokes_llm_on_parser_miss(graph):
    client = ScriptedClient(_reply("edge_count", []))
    result = answer_topological_with_fallback(graph, "count the arrows please", client)
    assert result == ("edge_count", 7, False)
    assert len(client.prompts) == 1  # parser missed, LLM was consulted


def test_prompt_lists_node_labels(graph):
    prompt = build_prompt(graph, "anything")
    assert "Start" in prompt and "End" in prompt
    assert "node_count" in prompt  # tool doc is present
