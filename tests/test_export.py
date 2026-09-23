import numpy as np
import pytest
import torch
from safetensors.numpy import load_file
from kilix_ml.export import export_bundle, load_bundle
from kilix_ml.model.config import Config
from kilix_ml.model.san import SAN
from kilix_ml.tokenizer import train


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
