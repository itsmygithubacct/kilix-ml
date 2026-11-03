"""Supported const/enum values must share the renderer's byte contract."""
import copy
import json

import numpy as np
import pytest

from kilix_ml.export import export_bundle, expected_shapes, load_bundle
from kilix_ml.format import canonical_calls, compact, example, prefix
from kilix_ml.grammar import compile_tools
from kilix_ml.model.config import Config
from kilix_ml.runtime.complete import Engine
from kilix_ml.tokenizer import SPECIALS, Tokenizer, byte_alphabet


def byte_tokenizer():
    vocab = {s: i for i, s in enumerate(SPECIALS)}
    vocab.update({char: i + len(SPECIALS) for i, char in byte_alphabet().items()})
    return Tokenizer({'format': 'kilix-byte-bpe/v1', 'specials': SPECIALS,
                      'vocab': vocab, 'merges': []})


def tool(schema):
    return [{'name': 'pair', 'parameters': {'type': 'object', 'additionalProperties': False,
             'properties': {'payload': schema}, 'required': ['payload']}}]


def call(value):
    return [{'name': 'pair', 'arguments': {'payload': value}}]


def accepts(grammar, raw):
    state = grammar.advance(grammar.initial_state(), raw)
    return state is not None and grammar.is_complete(state)


def stored(value):
    # Independent corpus representation: JSON object keys sorted, arrays intact.
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


OBJECT = {'z': 'Z', 'a': 'A'}
ARRAY = [{'z': 'Z', 'a': 'A'}, 'second', [{'z': 2, 'a': 1}, False, None]]
NESTED = {'z': ARRAY, 'a': {'z': 'last', 'a': 'first'}}
LITERALS = [
    pytest.param({'type': 'object', 'const': OBJECT}, [OBJECT], id='object-const'),
    pytest.param({'type': 'object', 'enum': [OBJECT, {'z': 'Y', 'a': 'B'}]},
                 [OBJECT, {'z': 'Y', 'a': 'B'}], id='object-enum'),
    pytest.param({'type': 'array', 'const': ARRAY}, [ARRAY], id='array-const'),
    pytest.param({'type': 'array', 'enum': [ARRAY, [NESTED, 'other']]},
                 [ARRAY, [NESTED, 'other']], id='array-enum'),
    pytest.param({'const': NESTED}, [NESTED], id='nested-object-const'),
]


@pytest.mark.parametrize('schema,values', LITERALS)
def test_literal_targets_and_grammars_ignore_object_insertion_order(schema, values):
    schemas = tool(copy.deepcopy(schema))
    corpus_schemas = stored(schemas)
    original_schema_bytes = compact(schemas)
    tokenizer = byte_tokenizer()
    assert prefix(tokenizer, schemas) == prefix(tokenizer, corpus_schemas)
    for value in values:
        caller = call(copy.deepcopy(value))
        corpus = stored(caller)
        original_call_bytes = compact(caller)
        # Keep model framing name-before-arguments; sort only argument values.
        expected = compact(call(stored(value))).encode()
        for definition in (schemas, corpus_schemas):
            grammar = compile_tools(definition, 'pair', max_calls=1)
            for supplied in (caller, corpus):
                target = compact(canonical_calls(supplied, definition)).encode()
                assert target == expected
                assert accepts(grammar, target)
                assert example(tokenizer, 'pair', supplied, definition) == example(
                    tokenizer, 'pair', caller, corpus_schemas)
        assert compact(caller) == original_call_bytes
    assert compact(schemas) == original_schema_bytes


@pytest.mark.parametrize('schema,values', LITERALS)
def test_literal_grammar_requires_the_same_canonical_bytes(schema, values):
    expected = compact(call(stored(values[0]))).encode()
    for definition in (tool(schema), tool(stored(schema))):
        grammar = compile_tools(definition, 'pair', max_calls=1)
        assert accepts(grammar, expected)
        # Noncanonical bytes must not depend on which schema dict was supplied.
        assert not accepts(grammar, compact(call(values[0])).encode())
        assert not accepts(grammar, compact(call({'a': 'not a declared value'})).encode())
        assert accepts(grammar, b'[]')


@pytest.mark.parametrize('keyword', ['const', 'enum'])
def test_data_array_order_and_unequal_values_are_not_normalized_away(keyword):
    schema = {'type': 'array', keyword: ARRAY if keyword == 'const' else [ARRAY]}
    grammar = compile_tools(tool(schema), 'pair', max_calls=1)
    target = compact(canonical_calls(call(copy.deepcopy(ARRAY)), tool(schema))).encode()
    assert accepts(grammar, target)
    reversed_array = list(reversed(stored(ARRAY)))
    assert not accepts(grammar, compact(call(reversed_array)).encode())
    changed = stored(ARRAY)
    changed[0]['a'] = 'different'
    assert not accepts(grammar, compact(call(changed)).encode())
    assert json.loads(target)[0]['arguments']['payload'] == ARRAY


def test_enum_alternatives_and_tool_list_order_are_preserved():
    tokenizer = byte_tokenizer()
    schemas = tool({'enum': [ARRAY, [NESTED, 'other']]})
    swapped = copy.deepcopy(schemas)
    swapped[0]['parameters']['properties']['payload']['enum'].reverse()
    assert prefix(tokenizer, schemas) != prefix(tokenizer, swapped)
    second = copy.deepcopy(schemas[0])
    second['name'] = 'second'
    assert prefix(tokenizer, schemas + [second]) != prefix(tokenizer, [second] + schemas)
    calls = call(ARRAY) + [{'name': 'second', 'arguments': {'payload': ARRAY}}]
    assert [c['name'] for c in canonical_calls(calls, schemas + [second])] == ['pair', 'second']


def test_exact_literal_bytes_types_unicode_escapes_and_outer_framing():
    value = {'a': [True, False, None, 7, -2, 1.25, -0.0],
             'z': [{'a': 'quote " slash \\ newline\n tab\t', 'z': 'café ☃ é'}]}
    schemas = tool({'const': value})
    result = canonical_calls(call(value), schemas)
    expected = ('[{"name":"pair","arguments":{"payload":{"a":[true,false,null,7,-2,1.25,-0.0],'
                '"z":[{"a":"quote \\" slash \\\\ newline\\n tab\\t","z":"café ☃ é"}]}}}]').encode()
    assert compact(result).encode() == expected
    assert accepts(compile_tools(schemas, 'pair'), expected)
    assert [type(v) for v in result[0]['arguments']['payload']['a']] == [
        bool, bool, type(None), int, int, float, float]
    result[0]['arguments']['payload']['z'][0]['a'] = 'changed copy'
    assert value['z'][0]['a'] == 'quote " slash \\ newline\n tab\t'


@pytest.mark.parametrize('schema', [
    {'type': 'object'},
    {'type': 'object', 'properties': {'x': {'type': 'string'}}, 'additionalProperties': False},
    {'type': 'array', 'items': {'type': 'string'}},
    {'type': 'number'},
])
def test_general_nested_schema_support_is_not_added(schema):
    with pytest.raises(ValueError, match='unsupported schema'):
        compile_tools(tool(schema), 'pair')


@pytest.mark.parametrize('schema,message', [
    ({'enum': 'not an array'}, 'enum must be an array'),
    ({'type': 'integer', 'const': True}, 'conflicts with its declared type'),
    ({'const': OBJECT, 'enum': [OBJECT]}, 'unsupported constraints'),
    ({'type': 'string', 'const': 'x', 'x-source': 'input'}, 'unsupported constraints'),
])
def test_existing_literal_schema_errors_are_preserved(schema, message):
    with pytest.raises(ValueError, match=message):
        compile_tools(tool(schema), 'pair')


def test_serialization_errors_and_shared_containers():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match='Circular reference detected'):
        compact(canonical_calls(call(cyclic), tool({'const': []})))
    with pytest.raises(ValueError, match='Circular reference detected'):
        prefix(byte_tokenizer(), tool({'const': cyclic}))
    with pytest.raises(ValueError):
        prefix(byte_tokenizer(), tool({'const': float('nan')}))
    shared = {'z': 2, 'a': 1}
    result = canonical_calls(call([shared, shared]), tool({'const': [shared, shared]}))
    assert compact(result) == '[{"name":"pair","arguments":{"payload":[{"a":1,"z":2},{"a":1,"z":2}]}}]'
    with pytest.raises(ValueError, match='undeclared arguments'):
        canonical_calls([{'name': 'pair', 'arguments': {'unknown': 1}}], tool({'const': []}))


@pytest.mark.parametrize('keyword', ['const', 'enum'])
def test_literal_bundle_reload_and_runtime_prefix_cache(tmp_path, monkeypatch, keyword):
    tokenizer = byte_tokenizer()
    config = Config(vocab_size=tokenizer.vocab_size, width=16, heads=2, kv_heads=1,
                    layers=1, context=2048)
    weights = {name: np.zeros(shape, dtype=np.float32) for name, shape in expected_shapes(config).items()}
    export_bundle(tmp_path / 'bundle', config, weights, tokenizer, {'stage': 'fixture'})
    _, reloaded, _ = load_bundle(tmp_path / 'bundle')
    schemas = tool({'type': 'array', keyword: ARRAY if keyword == 'const' else [ARRAY]})
    corpus_schemas = stored(schemas)
    assert example(reloaded, 'pair', call(ARRAY), schemas) == example(
        tokenizer, 'pair', call(stored(ARRAY)), corpus_schemas)
    assert prefix(reloaded, schemas) == prefix(tokenizer, corpus_schemas)
    expected = compact(call(stored(ARRAY)))
    engine = Engine(tmp_path / 'bundle', max_new_tokens=192)
    original_forward = engine.model.forward
    prompt_length = len(prefix(reloaded, schemas)) + len(reloaded.encode('pair')) + 4

    def choose_one_call(ids, cache=None, *, last_only=False):
        logits, cache = original_forward(ids, cache, last_only=last_only)
        # Fixture choices enter exactly one call and then close the call array;
        # the real grammar determines every literal byte, including object order.
        logits[:] = 0
        chosen = '{' if cache.length == prompt_length + 1 else ']'
        logits[-1, reloaded.encode(chosen)[0]] = 1
        return logits, cache

    monkeypatch.setattr(engine.model, 'forward', choose_one_call)
    first = engine.complete('pair', schemas)
    second = engine.complete('pair', corpus_schemas)
    assert first.metrics['finished'] and second.metrics['finished']
    assert compact(first.calls) == compact(second.calls) == expected
    assert not first.metrics['prefix_cache_hit'] and second.metrics['prefix_cache_hit']
