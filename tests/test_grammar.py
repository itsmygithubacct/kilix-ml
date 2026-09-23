import json
from pathlib import Path

import pytest

from kilix_ml.grammar import compile_tools, TokenMask

TOOLS = json.loads((Path(__file__).parents[1] / "domains/kilix_panes/tools.json").read_text())


def walk(grammar, raw):
    state = grammar.initial_state()
    for byte in raw:
        state = grammar.advance(state, bytes([byte]))
        assert state is not None
    assert grammar.is_complete(state)


def test_real_tool_schemas_compile_and_walk():
    g = compile_tools(TOOLS, "open vim in a pane on the right", max_calls=1)
    walk(g, b'[{"name":"open_pane","arguments":{"side":"right","program":"vim"}}]')
    walk(g, b'[]')


def test_source_constrained_and_bounds():
    g = compile_tools(TOOLS, "close the vim pane", max_calls=1)
    state = g.initial_state()
    assert g.advance(state, b'[{"name":"close_pane","arguments":{"pane":"invented"}}]') is None
    g2 = compile_tools(TOOLS, "resize pane 80", max_calls=1)
    assert g2.advance(g2.initial_state(), b'[{"name":"resize_pane","arguments":{"direction":"wider","amount":51}}]') is None


def test_unknown_property_rejected_and_mask():
    g = compile_tools(TOOLS, "rename tab Alpha", max_calls=1)
    assert g.advance(g.initial_state(), b'[{"name":"rename_tab","arguments":{"name":"Alpha","extra":1}}]') is None
    raw = b'[]'
    mask = TokenMask(g, [b'[', b']', b'x'])
    assert mask.allowed(g.initial_state()) == [0]
    walk(g, raw)


def test_unsupported_schema_fails_loudly():
    with pytest.raises(ValueError, match="unsupported schema"):
        compile_tools([{"name":"bad", "parameters":{"type":"object", "properties":{"x":{"type":"number"}}, "required":[], "additionalProperties":False}}], "")


def test_long_request_and_independent_source_fields():
    request = "Please open vim and call the new pane editor; then leave all other panes exactly as they are today."
    assert len(request) >= 80
    g = compile_tools(TOOLS, request, max_calls=1)
    walk(g, b'[{"name":"open_pane","arguments":{"program":"vim","name":"editor"}}]')
    assert g.advance(g.initial_state(), b'[{"name":"open_pane","arguments":{"program":"nano","name":"editor"}}]') is None
    assert g.advance(g.initial_state(), b'[{"name":"open_pane","arguments":{"program":"vim","name":"editorx"}}]') is None


def test_input_or_canonical_tab_numbers_and_multi_call_array():
    g = compile_tools(TOOLS, "close tab 3", max_calls=2)
    walk(g, b'[{"name":"close_tab","arguments":{"tab":"3"}},{"name":"go_to_tab","arguments":{"tab":"previous"}}]')
    assert g.advance(g.initial_state(), b'[{"name":"close_tab","arguments":{"tab":"12"}}]') is None


def test_mutations_incomplete_sources_required_fields_duplicates_and_call_limit():
    g = compile_tools(TOOLS, "open vim in a pane", max_calls=1)
    assert g.advance(g.initial_state(), b'[{"name":"open_pane","arguments":{"pane":"current"}}]') is None
    assert g.advance(g.initial_state(), b'[{"name":"close_pane","arguments":{}}]') is None
    assert g.advance(g.initial_state(), b'[{"name":"resize_pane","arguments":{"direction":"wider","direction":"shorter"}}]') is None
    assert g.advance(g.initial_state(), b'[{"name":"open_pane","arguments":{"program":""}}]') is None
    assert g.advance(g.initial_state(), b'[{"name":"close_pane","arguments":{"pane":"vim","command":"vim"}}]') is None
    first = b'{"name":"close_pane","arguments":{"pane":"current"}}'
    state = g.advance(g.initial_state(), b"[" + first)
    assert state is not None and not g.is_complete(state)
    assert ord(",") not in g.allowed_bytes(state)
    assert g.advance(state, b",") is None
    quote_prefix = g.advance(g.initial_state(), b'[{"name":"open_pane","arguments":{"program":"')
    assert quote_prefix is not None and not g.is_complete(quote_prefix)


def test_unconstrained_strings_are_valid_json():
    tool = [{"name": "echo", "parameters": {"type": "object", "properties": {
        "text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}}]
    g = compile_tools(tool, "", max_calls=1)
    walk(g, b'[{"name":"echo","arguments":{"text":""}}]')
    walk(g, '[{"name":"echo","arguments":{"text":"quote: \\\" snowman: ☃"}}]'.encode())
    assert g.advance(g.initial_state(), b'[{"name":"echo","arguments":{"text":"bad\nline"}}]') is None
    assert g.advance(g.initial_state(), b'[{"name":"echo","arguments":{"text":"\\u12xz"}}]') is None
    assert g.advance(g.initial_state(), b'[{"name":"echo","arguments":{"text":"\\uD800"}}]') is None


def test_const_enum_sibling_constraints_are_not_ignored():
    def tool(value):
        return [{"name": "t", "parameters": {"type": "object", "properties": {
            "x": value}, "required": ["x"], "additionalProperties": False}}]
    with pytest.raises(ValueError, match="unsupported constraints"):
        compile_tools(tool({"type": "integer", "enum": [1], "minimum": 2}), "")
    with pytest.raises(ValueError, match="unsupported constraints"):
        compile_tools(tool({"type": "string", "const": "x", "x-source": "input"}), "x")


def test_10000_random_completions_parse_and_validate_independently():
    import random
    jsonschema = pytest.importorskip("jsonschema")

    request = "Please open vim and call the new pane editor; close tab 3 if asked; keep the rest unchanged."
    g = compile_tools(TOOLS, request, max_calls=2)
    validators = {tool["name"]: jsonschema.Draft202012Validator(tool["parameters"]) for tool in TOOLS}
    by_name = {tool["name"]: tool for tool in TOOLS}
    rng = random.Random(92173)

    def unique_object(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise AssertionError(f"duplicate JSON object key: {key}")
            out[key] = value
        return out

    for _ in range(10_000):
        state = g.initial_state()
        output = bytearray()
        while not g.is_complete(state):
            options = tuple(g.allowed_bytes(state))
            assert options
            byte = rng.choice(options)
            output.append(byte)
            state = g.advance(state, bytes((byte,)))
            assert state is not None
        decoded = json.loads(output, object_pairs_hook=unique_object)
        assert isinstance(decoded, list) and len(decoded) <= 2
        for call in decoded:
            assert set(call) == {"name", "arguments"}
            tool = by_name[call["name"]]
            validators[call["name"]].validate(call["arguments"])
            for key, prop in tool["parameters"]["properties"].items():
                if key not in call["arguments"]:
                    continue
                source = prop.get("x-source")
                if source == "input":
                    assert call["arguments"][key] and call["arguments"][key] in request
                elif source == "input-or-canonical":
                    value = call["arguments"][key]
                    assert value in request or value in {
                        "current", "left", "right", "above", "below", "next", "previous",
                        "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    }
