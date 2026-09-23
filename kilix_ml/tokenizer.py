"""Byte BPE: optional Rust trainer, independent standard-library encoder.

ASCII words/identifiers can merge; each digit, punctuation and whitespace is
isolated. All 256 bytes exist, so unseen Unicode always round-trips.
"""
from functools import lru_cache
import json
from pathlib import Path
import re

SPECIALS = ['<bos>', '<eos>', '<pad>', '<|system|>', '<|tools|>', '<|user|>',
            '<|assistant|>', '<|calls|>', '<|text|>', '<|result|>', '<|end|>']
PATTERN = r'[A-Za-z_]+|[0-9]|[^A-Za-z_0-9]'


def byte_alphabet():
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {b: chr(b) for b in visible}
    for b in range(256):
        if b not in mapping:
            mapping[b] = chr(256 + len(mapping) - len(visible))
    return mapping


class Tokenizer:
    def __init__(self, data):
        if data.get('format') != 'kilix-byte-bpe/v1' or data.get('specials') != SPECIALS:
            raise ValueError('unsupported tokenizer format')
        self.data = data
        self.vocab = data['vocab']
        self.ranks = {tuple(pair): i for i, pair in enumerate(data['merges'])}
        self.special = {s: self.vocab[s] for s in SPECIALS}
        self.alphabet = byte_alphabet()
        inverse = {v: k for k, v in self.alphabet.items()}
        self.token_bytes = [b''] * len(self.vocab)
        if sorted(self.vocab.values()) != list(range(len(self.vocab))):
            raise ValueError('noncontiguous vocabulary')
        for word, i in self.vocab.items():
            if word not in self.special:
                self.token_bytes[i] = bytes(inverse[c] for c in word)
        self.vocab_size = len(self.vocab)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(self.data, ensure_ascii=False, sort_keys=True) + '\n')

    @lru_cache(maxsize=32768)
    def _piece(self, piece):
        parts = [self.alphabet[b] for b in piece.encode('utf-8')]
        while len(parts) > 1:
            candidates = [(self.ranks.get((a, b), float('inf')), i)
                          for i, (a, b) in enumerate(zip(parts, parts[1:]))]
            rank, i = min(candidates)
            if rank == float('inf'):
                break
            parts[i:i+2] = [parts[i] + parts[i+1]]
        return tuple(self.vocab[p] for p in parts)

    def encode(self, text):
        # Untrusted text cannot introduce control tokens.
        return [i for match in re.finditer(PATTERN, text) for i in self._piece(match.group())]

    def decode(self, ids, *, show_special=False):
        inverse = {v: k for k, v in self.special.items()}
        return b''.join(inverse[i].encode() if show_special and i in inverse else self.token_bytes[i]
                        for i in ids).decode('utf-8', errors='replace')

    def native_encoder(self):
        """Optional acquisition accelerator; never used or imported by inference."""
        from tokenizers import Tokenizer as Native, Regex, models, pre_tokenizers
        native = Native(models.BPE(vocab=self.vocab, merges=[tuple(p) for p in self.data['merges']]))
        native.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(PATTERN), behavior='isolated'),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        return native


def train(texts, output, vocab_size=8192):
    from tokenizers import Tokenizer as Native, Regex, models, pre_tokenizers, trainers
    native = Native(models.BPE())
    native.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                 special_tokens=SPECIALS, initial_alphabet=list(byte_alphabet().values()),
                                 show_progress=False)
    native.train_from_iterator(texts, trainer=trainer)
    model = json.loads(native.to_str())['model']
    merges = [p.split(' ') if isinstance(p, str) else p for p in model['merges']]
    tokenizer = Tokenizer({'format': 'kilix-byte-bpe/v1', 'specials': SPECIALS,
                           'vocab': model['vocab'], 'merges': merges})
    tokenizer.save(output)
    return tokenizer, native
