"""Deterministic local data readers; training itself does not use the network."""
import json
from pathlib import Path
import numpy as np
from ..export import sha256
from ..format import example


def jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class Corpus:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'MANIFEST.json').read_text())
        for name, metadata in self.manifest['files'].items():
            if sha256(self.root / name) != metadata['sha256']:
                raise ValueError(f'corpus digest mismatch: {name}')
        self.tools = {name: json.loads((self.root / 'tools' / (name + '.json')).read_text())
                      for name in ('kilix_panes', 'synthetic_home', 'generic_registry')}

    def resolve_tools(self, row):
        toolset = row['toolset']
        return self.tools[toolset] if isinstance(toolset, str) else [self.tools['generic_registry'][q] for q in toolset]

    def encode(self, tokenizer, part, split='train', context=3072):
        rows, lengths = [], []
        for row in jsonl(self.root / part / (split + '.jsonl')):
            ids, labels = example(tokenizer, row['request'], row['calls'], self.resolve_tools(row))
            if len(ids) > context:
                raise ValueError(f'{part} example has {len(ids)} tokens, exceeds context={context}')
            rows.append((np.array(ids, dtype=np.int64), np.array(labels, dtype=np.int64)))
            lengths.append(len(ids))
        if not rows:
            raise ValueError(f'empty corpus partition {part}/{split}')
        return rows, {'rows': len(rows), 'min': min(lengths), 'max': max(lengths), 'mean': float(np.mean(lengths))}


class SFTBatches:
    def __init__(self, groups, probabilities, batch_size, pad_id, seed=0):
        self.groups, self.probabilities = groups, np.array(probabilities) / sum(probabilities)
        self.batch_size, self.pad_id = batch_size, pad_id
        self.rng = np.random.default_rng(seed)

    def next(self):
        rows = []
        for _ in range(self.batch_size):
            group = self.groups[self.rng.choice(len(self.groups), p=self.probabilities)]
            rows.append(group[self.rng.integers(len(group))])
        length = max(len(ids) for ids, _ in rows)
        ids = np.full((len(rows), length), self.pad_id, dtype=np.int64)
        labels = np.full_like(ids, -100)
        for i, (tokens, targets) in enumerate(rows):
            ids[i, :len(tokens)], labels[i, :len(tokens)] = tokens, targets
        return ids, labels

    def state_dict(self):
        return {'rng': self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state['rng']


class PretrainBatches:
    """Fixed 7:3 token-block schedule; each stream is separately memory mapped.

    Acquisition manifests record exact source revisions. Epoch wrap is counted;
    tokens processed and unique tokens are distinct in reports.
    """
    def __init__(self, fineweb, synth, batch_size, sequence, seed=0):
        self.paths = [str(fineweb), str(synth)]
        self.streams = [np.memmap(p, dtype='<u2', mode='r') for p in self.paths]
        self.batch_size, self.sequence = batch_size, sequence
        if any(len(a) < sequence + 1 for a in self.streams):
            raise ValueError('token streams too short')
        self.offsets, self.blocks, self.wraps = [0, 0], 0, [0, 0]

    def next(self):
        rows = []
        for _ in range(self.batch_size):
            source = 0 if self.blocks % 10 < 7 else 1
            self.blocks += 1
            a = self.streams[source]
            offset = self.offsets[source]
            if offset + self.sequence + 1 > len(a):
                offset = 0
                self.wraps[source] += 1
            rows.append(np.asarray(a[offset:offset + self.sequence + 1], dtype=np.int64))
            self.offsets[source] = offset + self.sequence
        ids = np.stack(rows)
        return ids, ids.copy()

    def state_dict(self):
        return {'offsets': self.offsets, 'blocks': self.blocks, 'wraps': self.wraps}

    def load_state_dict(self, state):
        self.offsets, self.blocks, self.wraps = state['offsets'], state['blocks'], state['wraps']
