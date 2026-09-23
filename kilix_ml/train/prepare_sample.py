"""Tokenize acquired prose locally for the throughput probe, without networking."""
import argparse
import json
from pathlib import Path
import numpy as np
from ..export import sha256
from ..tokenizer import Tokenizer


def prepare(sample, tokenizer_path, out):
    sample, out = Path(sample), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    tok = Tokenizer.load(tokenizer_path)
    native = tok.native_encoder()
    source = json.loads((sample / 'acquisition.json').read_text())
    manifest = {'format': 'kilix-pretrain-data/v1', 'purpose': 'throughput probe sample; repeats explicitly counted',
                'sources': source['sources'], 'tokenizer_sha256': sha256(tokenizer_path), 'streams': {}}
    for name in ('fineweb', 'synth'):
        input_path = sample / (name + '.jsonl')
        if sha256(input_path) != source['streams'][name]['sha256']:
            raise ValueError('source sample checksum mismatch')
        path = out / (name + '.u16')
        rows, count = 0, 0
        with path.open('wb') as f, input_path.open() as source_file:
            for line in source_file:
                text = json.loads(line)['text']
                ids = native.encode(text).ids + [tok.special['<eos>']]
                np.asarray(ids, dtype='<u2').tofile(f)
                count += len(ids)
                rows += 1
        manifest['streams'][name] = {'sha256': sha256(path), 'tokens': count, 'rows': rows}
    (out / 'acquisition.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sample', 'tokenizer', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.sample, a.tokenizer, a.out), indent=2))
