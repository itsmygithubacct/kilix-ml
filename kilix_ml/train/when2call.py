"""Convert only supported no-call SFT rows from NVIDIA When2Call.

The source's clarification/inability answers become [] in our no-call format;
we do not claim to preserve its natural-language answering benchmark.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from ..export import sha256

REPO = 'nvidia/When2Call'
REVISION = '0582f7749df63a96fdc3070932e83e72396ace53'
FILE = 'train/when2call_train_sft.jsonl'


def schema(source, *, root=False):
    aliases = {'str': 'string', 'dict': 'object', 'int': 'integer', 'bool': 'boolean'}
    kind = aliases.get(source.get('type'), source.get('type'))
    out = {'type': kind}
    if kind not in ('object', 'string', 'integer', 'boolean'):
        raise ValueError('unsupported_type')
    if kind == 'object' and not root:
        raise ValueError('nested_object')
    if 'description' in source:
        out['description'] = source['description']
    if kind == 'object':
        out.update(properties={k: schema(v) for k, v in source.get('properties', {}).items()},
                   required=source.get('required', []), additionalProperties=False)
    if 'enum' in source:
        out['enum'] = source['enum']
    if kind == 'integer':
        if 'minimum' not in source or 'maximum' not in source:
            raise ValueError('unbounded_integer')
        out.update(minimum=source['minimum'], maximum=source['maximum'])
    # Defaults do not make an unprovided required parameter safe to invent.
    for key in source:
        if key not in ('type', 'description', 'properties', 'required', 'additionalProperties',
                       'enum', 'minimum', 'maximum', 'default'):
            raise ValueError('unsupported_keyword')
    return out


def convert(source, destination, exclusions=()):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    excluded = set()
    fold = lambda s: ' '.join(re.sub(r'[^\w\s]', ' ', s.casefold()).split())
    for path in exclusions:
        for line in Path(path).read_text().splitlines():
            excluded.add(fold(json.loads(line)['request']))
    rows = {'train': [], 'val': []}
    counts, seen = Counter(), set()
    for line in Path(source).read_text().splitlines():
        raw = json.loads(line)
        counts['input'] += 1
        messages = raw.get('messages', [])
        if len(messages) != 2 or [m['role'] for m in messages] != ['user', 'assistant']:
            counts['unsupported_conversation'] += 1
            continue
        request, answer = messages[0]['content'], messages[1]['content']
        # Conservative marker filter; other answers are not assumed to be no-calls.
        if not any(x in answer.lower() for x in ('unable', 'could you', 'please provide', 'please specify',
                                                  'i need', "i'll need", 'can you', 'cannot', "can't")):
            counts['unclassified_answer'] += 1
            continue
        key = fold(request)
        if key in excluded or key in seen:
            counts['excluded_or_duplicate'] += 1
            continue
        try:
            tools = []
            for item in raw['tools']:
                tool = json.loads(item) if isinstance(item, str) else item
                parameters = dict(tool['parameters'])
                if 'required' in tool:
                    parameters['required'] = tool['required']
                tools.append({'name': tool['name'], 'description': tool.get('description', ''),
                              'parameters': schema(parameters, root=True)})
            from ..grammar import compile_tools
            compile_tools(tools, request)  # Conversion must fit the actual inference grammar.
        except (ValueError, KeyError, TypeError) as exc:
            counts[str(exc)] += 1
            continue
        seen.add(key)
        split = 'val' if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'
        rows[split].append({'request': request, 'calls': [], 'tools': tools,
                            'source': REPO, 'tag': 'when2call_no_call'})
    files = {}
    for split, values in rows.items():
        p = destination / (split + '.jsonl')
        p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values))
        files[p.name] = {'rows': len(values), 'sha256': sha256(p)}
    manifest = {'source': REPO, 'revision': REVISION, 'file': FILE, 'source_sha256': sha256(source),
                'converter_sha256': sha256(__file__),
                'license': 'CC-BY-4.0', 'author': 'NVIDIA Corporation', 'counts': dict(counts),
                'transformation': 'Supported schemas; explicit clarification/inability answers -> []; no source test data.',
                'files': files}
    (destination / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--exclude', action='append', type=Path, default=[])
    a = p.parse_args()
    if a.source is None:
        from huggingface_hub import hf_hub_download
        a.source = hf_hub_download(REPO, FILE, repo_type='dataset', revision=REVISION, token=False)
    print(json.dumps(convert(a.source, a.out, a.exclude), indent=2))
