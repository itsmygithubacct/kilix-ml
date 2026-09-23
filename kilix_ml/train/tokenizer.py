"""Train the shared tokenizer from acquired prose and training-only tool text."""
import argparse
import json
from pathlib import Path
from ..export import sha256
from ..tokenizer import train


def build(sample, corpus, output, when2call=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = [Path(sample) / (s + '.jsonl') for s in ('fineweb', 'synth')]
    tool_text = Path(corpus) / 'tokenizer_text.txt'
    extra = Path(when2call) / 'train.jsonl' if when2call else None

    def texts():
        for path in paths:
            with path.open() as f:
                for line in f:
                    yield json.loads(line)['text']
        # Repeated tool text gives identifiers enough counts without consulting eval.
        for _ in range(3):
            yield from tool_text.read_text().splitlines()
        if extra:
            for line in extra.read_text().splitlines():
                row = json.loads(line)
                yield row['request']
                yield json.dumps(row['tools'], ensure_ascii=False)
    tokenizer, native = train(texts(), output / 'tokenizer.json', 8192)
    if tokenizer.vocab_size != 8192:
        raise ValueError('not enough text to fill 8192-token vocabulary')
    checks = ['Hello world', 'run echo "Hello" in the pane', '\x00 café 🙂 你好',
              '1234567890', '<|calls|>', '{"name":"move_tab","arguments":{"position":3}}']
    fast = tokenizer.native_encoder()
    for text in checks:
        assert tokenizer.encode(text) == fast.encode(text).ids
        assert tokenizer.decode(tokenizer.encode(text)) == text
    manifest = {'format': 'kilix-tokenizer-training/v1', 'vocab_size': tokenizer.vocab_size,
                'sample_manifest_sha256': sha256(Path(sample) / 'acquisition.json'),
                'corpus_manifest_sha256': sha256(Path(corpus) / 'MANIFEST.json'),
                'inputs': {p.name: sha256(p) for p in paths + [tool_text] + ([extra] if extra else [])},
                'tokenizer_sha256': sha256(output / 'tokenizer.json'),
                'algorithm': 'byte BPE; ASCII words and identifiers merge, digits/punctuation isolated'}
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sample', 'corpus', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--when2call', type=Path)
    a = p.parse_args()
    print(json.dumps(build(a.sample, a.corpus, a.out, a.when2call), indent=2))
