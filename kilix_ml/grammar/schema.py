"""Compile the supported JSON schema subset into a byte-level NFA."""
from __future__ import annotations

import json

from .mask import _NFA, ByteGrammar


def _quoted(s):
    return json.dumps(s, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _literal(nfa, raw):
    return nfa.literal(raw)


def _json_string(nfa):
    # JSON string: opening quote, zero or more valid code units, closing quote.
    branches = []
    for lo, hi in ((0x20, 0x21), (0x23, 0x5b), (0x5d, 0x7e)):
        a, b = nfa.node(), nfa.node()
        nfa.charset(a, b, range(lo, hi + 1))
        branches.append((a, b))
    for esc in (b'\\"', b'\\\\', b'\\/', b'\\b', b'\\f', b'\\n', b'\\r', b'\\t'):
        branches.append(nfa.literal(esc))
    hexes = b"0123456789abcdefABCDEF"
    hex_digit = lambda: _char(nfa, hexes)
    non_surrogate = nfa.alt([
        nfa.concat([_char(nfa, b"0123456789aAbBcC"), hex_digit(), hex_digit(), hex_digit()]),
        nfa.concat([_char(nfa, b"dD"), _char(nfa, b"01234567"), hex_digit(), hex_digit()]),
        nfa.concat([_char(nfa, b"eEfF"), hex_digit(), hex_digit(), hex_digit()]),
    ])
    high_surrogate = nfa.concat([_char(nfa, b"dD"), _char(nfa, b"89aAbB"), hex_digit(), hex_digit()])
    low_surrogate = nfa.concat([_char(nfa, b"dD"), _char(nfa, b"cCdDeEfF"), hex_digit(), hex_digit()])
    escaped_unicode = nfa.alt([
        nfa.concat([nfa.literal(b"\\u"), non_surrogate]),
        nfa.concat([nfa.literal(b"\\u"), high_surrogate, nfa.literal(b"\\u"), low_surrogate]),
    ])
    branches.append(escaped_unicode)
    branches.extend(_utf8_fragments(nfa))
    repeated = nfa.star(nfa.alt(branches))
    return nfa.concat([nfa.literal(b'"'), repeated, nfa.literal(b'"')])


def _char(nfa, allowed):
    a, b = nfa.node(), nfa.node()
    nfa.charset(a, b, allowed)
    return a, b


def _utf8_fragments(nfa):
    c = range(0x80, 0xc0)
    out = []
    out.append(nfa.concat([_char(nfa, range(0xc2, 0xe0)), _char(nfa, c)]))
    out.append(nfa.concat([_char(nfa, [0xe0]), _char(nfa, range(0xa0, 0xc0)), _char(nfa, c)]))
    out.append(nfa.concat([_char(nfa, range(0xe1, 0xed)), _char(nfa, c), _char(nfa, c)]))
    out.append(nfa.concat([_char(nfa, [0xed]), _char(nfa, range(0x80, 0xa0)), _char(nfa, c)]))
    out.append(nfa.concat([_char(nfa, range(0xee, 0xf0)), _char(nfa, c), _char(nfa, c)]))
    out.append(nfa.concat([_char(nfa, [0xf0]), _char(nfa, range(0x90, 0xc0)), _char(nfa, c), _char(nfa, c)]))
    out.append(nfa.concat([_char(nfa, range(0xf1, 0xf4)), _char(nfa, c), _char(nfa, c), _char(nfa, c)]))
    out.append(nfa.concat([_char(nfa, [0xf4]), _char(nfa, range(0x80, 0x90)), _char(nfa, c), _char(nfa, c)]))
    return out


def _source_fragment(nfa, request):
    try:
        request_size = len(request.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("request contains an invalid Unicode surrogate") from exc
    if request_size > 4096:
        raise ValueError("request exceeds 4096 UTF-8 bytes supported by source grounding")
    # A linear DAG recognizes every nonempty contiguous request span. The opening
    # quote branches through the first byte of any character, while each boundary
    # can either close the string or continue with the next request character.
    chars = [_quoted(ch)[1:-1] for ch in request]
    opening, closing = nfa.literal(b'"'), nfa.literal(b'"')
    finish = nfa.node()
    boundaries = [nfa.node() for _ in range(len(chars) + 1)]
    for i, encoded in enumerate(chars):
        target = boundaries[i + 1]
        if len(encoded) == 1:
            nfa.edge(opening[1], target, encoded[0])
            nfa.edge(boundaries[i], target, encoded[0])
        else:
            continuation = nfa.node()
            nfa.edge(opening[1], continuation, encoded[0])
            nfa.edge(boundaries[i], continuation, encoded[0])
            rest = nfa.literal(encoded[1:])
            nfa.epsilon(continuation, rest[0])
            nfa.epsilon(rest[1], target)
    for i in range(1, len(boundaries)):
        nfa.epsilon(boundaries[i], closing[0])
    nfa.epsilon(closing[1], finish)
    return opening[0], finish


def _source_schema_fragment(nfa, schema, request):
    source = schema.get("x-source")
    if source not in ("input", "input-or-canonical"):
        raise ValueError(f"unsupported x-source: {source}")
    input_fragment = _source_fragment(nfa, request)
    if source == "input":
        return input_fragment
    canonical = ["current", "left", "right", "above", "below", "next", "previous"] + [str(i) for i in range(1, 10)]
    return nfa.alt([input_fragment] + [nfa.literal(_quoted(v)) for v in canonical])


def _schema_fragment(nfa, schema, request):
    allowed_keys = {"type", "enum", "const", "anyOf", "minimum", "maximum", "x-source", "description"}
    unknown = set(schema) - allowed_keys
    if unknown:
        raise ValueError(f"unsupported schema keywords: {sorted(unknown)}")
    if "anyOf" in schema:
        if set(schema) != {"anyOf"}:
            raise ValueError("anyOf cannot be combined with sibling constraints")
        return nfa.alt([_schema_fragment(nfa, option, request) for option in schema["anyOf"]])
    if "const" in schema or "enum" in schema:
        keyword = "const" if "const" in schema else "enum"
        allowed_for_value = {keyword, "description", "type"}
        if set(schema) - allowed_for_value:
            raise ValueError(f"unsupported constraints combined with {keyword}")
        values = [schema[keyword]] if keyword == "const" else schema[keyword]
        if not isinstance(values, list) and keyword == "enum":
            raise ValueError("enum must be an array")
        if "type" in schema:
            expected = schema["type"]
            for value in values:
                matches = ((expected == "string" and isinstance(value, str)) or
                           (expected == "integer" and type(value) is int) or
                           (expected == "boolean" and type(value) is bool) or
                           (expected == "object" and isinstance(value, dict)) or
                           (expected == "array" and isinstance(value, list)) or
                           (expected == "null" and value is None))
                if not matches:
                    raise ValueError(f"{keyword} value conflicts with its declared type")
        fragments = [nfa.literal(json.dumps(v, ensure_ascii=False, separators=(",", ":")).encode()) for v in values]
        return fragments[0] if keyword == "const" else nfa.alt(fragments)
    typ = schema.get("type")
    if typ == "string":
        if set(schema) - {"type", "x-source", "description"}:
            raise ValueError("unsupported constraints on string schema")
        if "x-source" in schema:
            return _source_schema_fragment(nfa, schema, request)
        return _json_string(nfa)
    if typ == "integer":
        if set(schema) - {"type", "minimum", "maximum", "description"}:
            raise ValueError("unsupported constraints on integer schema")
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if lo is None or hi is None or type(lo) is not int or type(hi) is not int or lo > hi:
            raise ValueError("integer schema requires valid minimum and maximum")
        if hi - lo > 10000:
            raise ValueError("integer schema range exceeds the supported 10001 values")
        return nfa.alt([nfa.literal(str(i).encode()) for i in range(lo, hi + 1)])
    if typ == "boolean":
        if set(schema) - {"type", "description"}:
            raise ValueError("unsupported constraints on boolean schema")
        return nfa.alt([nfa.literal(b"true"), nfa.literal(b"false")])
    raise ValueError(f"unsupported schema type: {typ!r}")


def _object_fragment(nfa, schema, request):
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("tool parameters must be a closed object schema")
    extra = set(schema) - {"type", "properties", "required", "additionalProperties"}
    if extra:
        raise ValueError(f"unsupported object schema keywords: {sorted(extra)}")
    props = schema.get("properties", {})
    required_list = schema.get("required", [])
    if not isinstance(props, dict) or not isinstance(required_list, list) or any(not isinstance(k, str) for k in required_list):
        raise ValueError("object properties must be an object and required must be a string array")
    if len(required_list) != len(set(required_list)):
        raise ValueError("required property names must be unique")
    if any(not isinstance(k, str) for k in props):
        raise ValueError("property names must be strings")
    required = set(required_list)
    if not required <= set(props):
        raise ValueError("required property missing from properties")
    names = list(props)
    if len(names) > 24:
        raise ValueError("object schema has too many properties")
    close = nfa.node()
    # Two paths per property index track whether a previous field was emitted.
    before = [(nfa.node(), nfa.node()) for _ in range(len(names) + 1)]
    for i, key in enumerate(names):
        value = _schema_fragment(nfa, props[key], request)
        for seen in (0, 1):
            src = before[i][seen]
            if key not in required:
                nfa.epsilon(src, before[i + 1][seen])
            field = nfa.concat([_literal(nfa, (b"," if seen else b"") + _quoted(key) + b":"), value])
            nfa.epsilon(src, field[0])
            nfa.epsilon(field[1], before[i + 1][1])
    nfa.epsilon(before[-1][0], close)
    nfa.epsilon(before[-1][1], close)
    return nfa.concat([_literal(nfa, b"{"), (before[0][0], close), _literal(nfa, b"}")])


def _tool_call(nfa, tool, request):
    prefix = _literal(nfa, b'{"name":' + _quoted(tool["name"]) + b',"arguments":')
    obj = _object_fragment(nfa, tool["parameters"], request)
    return nfa.concat([prefix, obj, _literal(nfa, b"}")])


def compile_tools(tools, request: str, max_calls: int = 32):
    if type(max_calls) is not int or max_calls < 0:
        raise ValueError("max_calls must be a nonnegative integer")
    if not isinstance(request, str):
        raise TypeError("request must be text")
    nfa = _NFA()
    empty = _literal(nfa, b"[]")
    if max_calls == 0 or not tools:
        nfa.start, nfa.accept = empty
        return ByteGrammar(nfa, max_calls=0)
    calls = [_tool_call(nfa, t, request) for t in tools]
    call_start, _ = nfa.alt(calls)
    joined_end = nfa.node()
    for _, end in calls:
        nfa.epsilon(end, joined_end)
    open_array = _literal(nfa, b"[")
    close_array = _literal(nfa, b"]")
    comma = _literal(nfa, b",")
    array_end = nfa.node()
    nfa.epsilon(open_array[1], call_start)
    nfa.epsilon(joined_end, close_array[0])
    nfa.epsilon(close_array[1], array_end)
    nfa.epsilon(joined_end, comma[0])
    nfa.epsilon(comma[1], call_start)
    empty = _literal(nfa, b"[]")
    start, accept = nfa.alt([empty, (open_array[0], array_end)])
    nfa.start, nfa.accept = start, accept
    return ByteGrammar(nfa, call_ends=(joined_end,), call_commas=(comma[0],), max_calls=max_calls)
