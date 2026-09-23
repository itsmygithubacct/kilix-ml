import json
import os
import subprocess
import sys
import torch
from kilix_ml.export import export_bundle
from kilix_ml.model.config import Config
from kilix_ml.model.san import SAN
from kilix_ml.tokenizer import train


def test_runtime_is_numpy_only_and_reuses_prefix(tmp_path):
    tok, _ = train(['hello world'] * 20, tmp_path / 'tokenizer.json', 280)
    torch.manual_seed(5)
    model = SAN(Config(vocab_size=tok.vocab_size, width=16, heads=2, kv_heads=1, layers=1, context=512))
    export_bundle(tmp_path / 'bundle', model.config, {k: v.detach().numpy() for k, v in model.state_dict().items()}, tok, {'test': True})
    code = '''
import json, sys
from kilix_ml.runtime.complete import Engine
e = Engine(sys.argv[1], max_new_tokens=40)
tools = [{'name':'ping','parameters':{'type':'object','properties':{},'required':[],'additionalProperties':False}}]
results = [e.complete('hello',tools) for _ in range(3)]
assert len({json.dumps(r.calls,sort_keys=True) for r in results})==1
assert results[1].metrics['prefix_cache_hit']
assert 'torch' not in sys.modules and 'tokenizers' not in sys.modules and 'safetensors' not in sys.modules
print(json.dumps(results[0].calls))
'''
    outputs = [subprocess.check_output([sys.executable, '-c', code, str(tmp_path / 'bundle')], text=True) for _ in range(2)]
    assert outputs[0] == outputs[1]
