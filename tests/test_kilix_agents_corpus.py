import importlib.util
import json
from pathlib import Path

PACK = Path(__file__).resolve().parents[1] / 'domains' / 'kilix_agents'
spec = importlib.util.spec_from_file_location('kilix_agents_generate', PACK / 'generate.py')
generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate)
TOOLS = {tool['name']: tool for tool in json.loads((PACK / 'tools.json').read_text())}


def test_generation_is_deterministic_and_uses_only_the_three_tools():
    first, _ = generate.generate(0, 3)
    second, _ = generate.generate(0, 3)
    assert first == second and len(first) > 1000
    assert set(TOOLS) == {'agent', 'wait', 'tell'}
    for row in first:
        for tool, args in row['actions']:
            assert tool in TOOLS
            assert set(args) <= set(TOOLS[tool]['parameters']['properties'])


def test_canonical_values_and_payloads_as_said():
    vocab = generate.load_vocab()
    for row in generate.generate(0, 2)[0]:
        for tool, args in row['actions']:
            if tool == 'agent':
                assert args['agent'] in vocab['agent']
                assert args.get('place', 'tab') in TOOLS['agent']['parameters']['properties'][
                    'place']['enum']
            if tool == 'wait':
                assert args['for'] in ('idle', 'waiting')
            for key in ('prompt', 'text', 'model', 'dir'):
                if key in args:
                    assert args[key] in row['query'], (key, row['query'])


def test_excluded_requests_are_dropped():
    rows, _ = generate.generate(0, 1)
    rows2, dropped = generate.generate(0, 1, {generate._fold(rows[0]['query'])})
    assert dropped == 1 and rows[0]['query'] not in [r['query'] for r in rows2]
