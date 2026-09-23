"""Explicit online acquisition, separate from offline training/inference.

Only public, commit-pinned datasets are accessed. No access token is needed.
"""
import argparse
import hashlib
import json
import itertools
from pathlib import Path
import numpy as np
from ..export import sha256
from ..tokenizer import Tokenizer

SOURCES = {
    'fineweb': {'repo': 'HuggingFaceFW/fineweb-edu', 'revision': '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9',
                'config': 'sample-100BT', 'license': 'ODC-By-1.0', 'weight': 0.7},
    'synth': {'repo': 'PleIAs/SYNTH', 'revision': '0d6813a2966662c39f22f0b9af28a0c1c9f7a437',
              'config': None, 'license': 'CC-BY-4.0', 'weight': 0.3},
}


def documents(source):
    from datasets import load_dataset
    spec = SOURCES[source]
    dataset = load_dataset(spec['repo'], name=spec['config'], revision=spec['revision'],
                           split='train', streaming=True, token=False)
    for row in dataset:
        if source == 'fineweb':
            text = row['text']
        else:
            if row.get('language') not in ('en', 'English', 'english'):
                continue
            # Reasoning traces are not needed for a direct tool-call model.
            text = str(row.get('query') or '') + '\n\n' + str(row.get('synthetic_answer') or '')
        if text.strip():
            yield text


def acquire(destination, *, bytes_per_source=None, token_counts=None, tokenizer=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / 'acquisition.json'
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = {'format': 'kilix-pretrain-data/v1', 'sources': SOURCES, 'streams': {},
                'synth_fields': ['query', 'synthetic_answer'], 'synth_language': 'English',
                'mix_unit': 'tokens', 'mix': {'fineweb': 0.7, 'synth': 0.3}}
    native = tokenizer.native_encoder() if tokenizer is not None else None
    for name in SOURCES:
        suffix = '.jsonl' if tokenizer is None else '.u16'
        path = destination / (name + suffix)
        if path.exists():
            raise FileExistsError(path)
        limit = bytes_per_source[name] if tokenizer is None else token_counts[name]
        consumed, rows = 0, 0
        temp = path.with_suffix(path.suffix + '.partial')
        if native is None:
            inputs = ((text, None) for text in documents(name))
        else:
            def encoded():
                stream = iter(documents(name))
                while batch := list(itertools.islice(stream, 128)):
                    for item in native.encode_batch(batch, add_special_tokens=False):
                        yield None, item.ids
            inputs = encoded()
        with temp.open('wb') as f:
            for text, encoded_ids in inputs:
                if tokenizer is None:
                    data = (json.dumps({'text': text}, ensure_ascii=False) + '\n').encode()
                    consumed += len(text.encode())
                    f.write(data)
                else:
                    ids = encoded_ids + [tokenizer.special['<eos>']]
                    ids = ids[:max(0, limit - consumed)]
                    f.write(np.asarray(ids, dtype='<u2').tobytes())
                    consumed += len(ids)
                rows += 1
                if rows % 1000 == 0:
                    print(json.dumps({'source': name, 'rows': rows, 'units': consumed, 'target': limit}), flush=True)
                if consumed >= limit:
                    break
        if consumed < limit:
            raise ValueError(f'{name} source exhausted at {consumed}, requested {limit}')
        temp.rename(path)
        manifest['streams'][name] = {'file': path.name, 'sha256': sha256(path), 'rows': rows,
                                     'units': consumed, 'unit': 'text_bytes' if tokenizer is None else 'tokens'}
        (destination / 'progress.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if tokenizer is not None:
        manifest['tokenizer_sha256'] = hashlib.sha256(json.dumps(tokenizer.data, sort_keys=True).encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path)
    p.add_argument('--tokens', type=int, default=30_000_000_000)
    p.add_argument('--sample-bytes', type=int, default=50_000_000)
    a = p.parse_args()
    if a.tokenizer:
        acquire(a.out, token_counts={'fineweb': a.tokens * 7 // 10, 'synth': a.tokens - a.tokens * 7 // 10},
                tokenizer=Tokenizer.load(a.tokenizer))
    else:
        acquire(a.out, bytes_per_source={'fineweb': a.sample_bytes * 7 // 10, 'synth': a.sample_bytes - a.sample_bytes * 7 // 10})
