import numpy as np
import pytest
import torch
from safetensors.numpy import load_file
from kilix_ml.export import export_bundle, load_bundle
from kilix_ml.model.config import Config
from kilix_ml.model.san import SAN
from kilix_ml.tokenizer import train
from kilix_ml.format import canonical_calls, compact, example, prefix
from kilix_ml.grammar import compile_tools
from kilix_ml.runtime.complete import Engine
import json


def test_safe_bundle_and_corruption(tmp_path):
    tok, _ = train(['abc abc'] * 3, tmp_path / 'tok.json', 280)
    model = SAN(Config(vocab_size=tok.vocab_size, width=16, heads=2, kv_heads=1, layers=1))
    weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    export_bundle(tmp_path / 'bundle', model.config, weights, tok, {'stage': 'test'})
    reference = load_file(tmp_path / 'bundle/model.safetensors')
    loaded, tokenizer, _ = load_bundle(tmp_path / 'bundle')
    for name, value in reference.items():
        np.testing.assert_array_equal(value, loaded.weights[name])
    path = tmp_path / 'bundle/tokenizer.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='digest mismatch'):
        load_bundle(tmp_path / 'bundle')


def test_serialization_contract_survives_bundle_reload(tmp_path):
    tools = [{'name': 'split', 'parameters': {'type': 'object', 'additionalProperties': False,
             'properties': {'side': {'type': 'string', 'enum': ['right', 'left']},
                            'name': {'type': 'string', 'x-source': 'input'}},
             'required': ['side', 'name']}}]
    stored = json.loads(json.dumps(tools, sort_keys=True))
    request = 'split right and call it editor'
    tok, _ = train([compact(stored), request] * 3, tmp_path / 'tok.json', 280)
    torch.manual_seed(12)
    model = SAN(Config(vocab_size=tok.vocab_size, width=16, heads=2, kv_heads=1, layers=1, context=1024))
    export_bundle(tmp_path / 'bundle', model.config,
                  {k: v.detach().numpy() for k, v in model.state_dict().items()}, tok, {'stage': 'test'})
    _, reloaded, _ = load_bundle(tmp_path / 'bundle')
    calls = [{'name': 'split', 'arguments': {'side': 'right', 'name': 'editor'}}]
    ids, labels = example(reloaded, request, calls, tools)
    assert (ids, labels) == example(tok, request, calls, stored)
    assert prefix(reloaded, tools) == prefix(tok, stored)
    answer = compact(canonical_calls(calls, stored)).encode()
    grammar = compile_tools(tools, request)
    state = grammar.advance(grammar.initial_state(), answer)
    assert state is not None and grammar.is_complete(state)
    engine = Engine(tmp_path / 'bundle', max_new_tokens=96)
    first = engine.complete(request, tools)
    second = engine.complete(request, stored)
    assert first.calls == second.calls
    assert first.metrics['finished'] == second.metrics['finished']
    assert second.metrics['prefix_cache_hit']
