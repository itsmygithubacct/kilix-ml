import json
import numpy as np
import torch
from types import SimpleNamespace
from kilix_ml.model.san import SAN
from kilix_ml.export import sha256
from kilix_ml.model.config import Config
from kilix_ml.tokenizer import train
from kilix_ml.train.run import arguments, prepare_sft_validation, run, run_sft_validation
from kilix_ml.train.data import PretrainBatches, SFTBatches


def test_batch_resume_and_token_mix(tmp_path):
    for name in ['a', 'b']:
        np.arange(100, dtype='<u2').tofile(tmp_path / name)
    batches = PretrainBatches(tmp_path / 'a', tmp_path / 'b', 10, 8)
    x, _ = batches.next()
    assert batches.offsets == [56, 24]
    state = json.loads(json.dumps(batches.state_dict()))
    expected, _ = batches.next()
    restored = PretrainBatches(tmp_path / 'a', tmp_path / 'b', 10, 8)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(restored.next()[0], expected)


def test_training_reproducibility_and_checkpoint_load(tmp_path):
    tok, _ = train(['one two three'] * 10, tmp_path / 'tokenizer.json', 280)
    config = Config(vocab_size=tok.vocab_size, width=16, heads=2, kv_heads=1, layers=1, context=32)
    (tmp_path / 'config.json').write_text(json.dumps(config.to_dict()))
    streams = {}
    for name in ['fineweb', 'synth']:
        path = tmp_path / (name + '.u16')
        np.asarray(tok.encode('one two three') * 100, dtype='<u2').tofile(path)
        streams[name] = {'sha256': sha256(path)}
    (tmp_path / 'data.json').write_text(json.dumps({'streams': streams}))
    base = ['--stage', 'pretrain', '--tokenizer', str(tmp_path / 'tokenizer.json'),
            '--config', str(tmp_path / 'config.json'), '--fineweb', str(tmp_path / 'fineweb.u16'),
            '--synth', str(tmp_path / 'synth.u16'), '--data-manifest', str(tmp_path / 'data.json'),
            '--device', 'cpu', '--steps', '3', '--batch', '2', '--accumulate', '1', '--sequence', '16']
    digests = []
    for name in ['run1', 'run2']:
        args = arguments(base + ['--out', str(tmp_path / name)])
        result = run(args)
        assert result['tokens'] == 96
        bundle = tmp_path / name / 'export-00000003/model.safetensors'
        digests.append(sha256(bundle))
        state = torch.load(tmp_path / name / 'checkpoint.pt', weights_only=True)
        assert state['step'] == 3
    assert digests[0] == digests[1]
    resumed = run(arguments(base + ['--out', str(tmp_path / 'run1'), '--resume']))
    assert resumed['step'] == 3


def test_sft_validation_includes_all_swap_rows_and_restores_mode(tmp_path):
    tok, _ = train(['swap panes left right above below'] * 10, tmp_path / 'tok.json', 280)
    swap_tool = {'name': 'swap_panes', 'parameters': {
        'type': 'object', 'properties': {'side': {'type': 'string', 'enum': ['left', 'right', 'above', 'below']}},
        'required': ['side'], 'additionalProperties': False}}
    other_tool = {'name': 'rename_tab', 'parameters': {
        'type': 'object', 'properties': {'name': {'type': 'string'}},
        'required': ['name'], 'additionalProperties': False}}

    class FixtureCorpus:
        root = tmp_path
        def resolve_tools(self, _row):
            return [swap_tool, other_tool]

    rows = []
    for i in range(137):
        rows.append({'request': f'rename tab {i}', 'toolset': 'kilix',
                     'calls': [{'name': 'rename_tab', 'arguments': {'name': f'tab{i}'}}]})
    sides = ['left', 'right', 'above', 'below']
    for i in range(30):
        rows.append({'request': f'swap pane {sides[i % 4]} #{i}', 'toolset': 'kilix',
                     'calls': [{'name': 'swap_panes', 'arguments': {'side': sides[i % 4]}}]})
    part = tmp_path / 'kilix'
    part.mkdir()
    (part / 'val.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    sets, swap_rows, support, sources = prepare_sft_validation(
        FixtureCorpus(), tok, 'kilix', context=512, seed=91)
    assert len(sets['kilix_val']) == 128
    assert len(swap_rows) == 30
    assert support['rows'] == 30 and support['calls'] == 30
    assert support['side_support'] == {'left': 8, 'right': 8, 'above': 7, 'below': 7}
    assert sources['swap_panes_val']['sampled_rows'] == 30

    model = SAN(Config(vocab_size=tok.vocab_size, width=16, heads=2, kv_heads=1, layers=1, context=512))
    model.train()
    results = run_sft_validation(model, {'swap_panes_val': swap_rows}, 8, 'cpu',
                                 pad_id=tok.special['<pad>'], step=17, final=True)
    metric = results['losses']['swap_panes_val']
    assert metric['rows'] == 30 and metric['target_tokens'] > 0
    assert np.isfinite(metric['validation_loss'])
    assert model.training is True
    assert all(param.grad is None for param in model.parameters())

    model.eval()
    run_sft_validation(model, {'kilix_val': sets['kilix_val'][:4]}, 2, 'cpu',
                        pad_id=tok.special['<pad>'], step=17)
    assert model.training is False
