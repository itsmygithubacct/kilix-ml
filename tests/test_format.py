import json
from pathlib import Path
import pytest

from kilix_ml.format import canonical_calls, compact, example, prefix
from kilix_ml.grammar import compile_tools


class ByteTokenizer:
    special = {name: 256 + i for i, name in enumerate(
        ['<bos>', '<eos>', '<|tools|>', '<|end|>', '<|user|>', '<|assistant|>', '<|calls|>'])}

    def encode(self, text):
        return list(text.encode())


def test_training_and_decoding_ignore_schema_object_insertion_order():
    tools = json.loads((Path(__file__).parents[1] / 'domains/kilix_panes/tools.json').read_text())
    # Model training reads corpus files written with sort_keys=True, while
    # application schemas retain their original insertion order.
    stored = json.loads(json.dumps(tools, sort_keys=True))
    tokenizer = ByteTokenizer()
    expected_prefix = ([tokenizer.special['<bos>'], tokenizer.special['<|tools|>']]
                       + tokenizer.encode(compact(stored)) + [tokenizer.special['<|end|>']])
    assert prefix(tokenizer, tools) == prefix(tokenizer, stored) == expected_prefix
    request = 'open vim on the right and name it editor'
    calls = [{'name': 'open_pane', 'arguments': {'side': 'right', 'program': 'vim', 'name': 'editor'}}]
    assert example(tokenizer, request, calls, tools) == example(tokenizer, request, calls, stored)
    answer = compact(canonical_calls(calls, tools)).encode()
    assert answer == b'[{"name":"open_pane","arguments":{"name":"editor","program":"vim","side":"right"}}]'
    for schemas in (tools, stored):
        grammar = compile_tools(schemas, request)
        state = grammar.advance(grammar.initial_state(), answer)
        assert state is not None and grammar.is_complete(state)


def test_tool_list_order_is_preserved():
    tools = [{'name': 'z', 'parameters': {}}, {'name': 'a', 'parameters': {}}]
    assert prefix(ByteTokenizer(), tools) != prefix(ByteTokenizer(), list(reversed(tools)))


@pytest.mark.parametrize('name,arguments,utterance', [
    ('open_pane', {'side': 'right', 'program': 'vim', 'name': 'editor'}, 'open vim on the right named editor'),
    ('open_tab', {'program': 'vim', 'name': 'editor'}, 'open a tab with vim named editor'),
    ('resize_pane', {'direction': 'wider', 'amount': 3}, 'make this pane wider by 3'),
    ('run_in_pane', {'pane': 'left', 'command': 'git diff'}, 'run git diff in the left pane'),
    ('rename_pane', {'pane': 'left', 'name': 'editor'}, 'name the left pane editor'),
])
def test_grammar_accepts_sorted_corpus_targets(name, arguments, utterance):
    tools = json.loads((Path(__file__).parents[1] / 'domains/kilix_panes/tools.json').read_text())
    # This is the stored corpus representation, independent of the renderer.
    answer = compact([{'name': name, 'arguments': dict(sorted(arguments.items()))}]).encode()
    grammar = compile_tools(tools, utterance)
    state = grammar.advance(grammar.initial_state(), answer)
    assert state is not None and grammar.is_complete(state)


def test_enum_array_order_is_preserved():
    tools = [{'name': 'arrange', 'parameters': {'type': 'object', 'additionalProperties': False,
              'properties': {'layout': {'type': 'string', 'enum': ['grid', 'stack']}}, 'required': ['layout']}}]
    reverse = json.loads(json.dumps(tools))
    reverse[0]['parameters']['properties']['layout']['enum'].reverse()
    assert prefix(ByteTokenizer(), tools) != prefix(ByteTokenizer(), reverse)
