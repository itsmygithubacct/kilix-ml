import importlib.util
import json
from pathlib import Path

PACK = Path(__file__).resolve().parents[1] / 'domains' / 'kilix_apps'
spec = importlib.util.spec_from_file_location('kilix_apps_generate', PACK / 'generate.py')
generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate)
TOOLS = {tool['name']: tool for tool in json.loads((PACK / 'tools.json').read_text())}


def test_generation_is_deterministic_and_uses_only_the_five_tools():
    first, _ = generate.generate(0, 3)
    second, _ = generate.generate(0, 3)
    assert first == second and len(first) > 1000
    for row in first:
        for tool, args in row['actions']:
            assert tool in TOOLS
            assert set(args) <= set(TOOLS[tool]['parameters']['properties'])


def test_every_bound_value_is_canonical():
    vocab = generate.load_vocab()
    for row in generate.generate(0, 2)[0]:
        for tool, args in row['actions']:
            if tool == 'launch':
                assert any(args['app'] in vocab[k] for k in ('app', 'game', 'tool'))
            if tool == 'show':
                assert args['item'] in vocab['item'] and isinstance(args['on'], bool)
            if tool == 'game':
                assert args['game'] in vocab['game'] and isinstance(args['available'], bool)


def test_excluded_requests_are_dropped():
    rows, _ = generate.generate(0, 1)
    rows2, dropped = generate.generate(0, 1, {generate._fold(rows[0]['query'])})
    assert dropped == 1 and rows[0]['query'] not in [r['query'] for r in rows2]
