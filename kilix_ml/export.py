"""Non-executable fp32 tensor export with verified, atomic bundle publication."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import numpy as np
from .format import FORMAT
from .model.config import Config


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_tensors(path, weights):
    header, buffers, offset = {}, [], 0
    for name, value in sorted(weights.items()):
        value = np.ascontiguousarray(value, dtype='<f4')
        raw = value.tobytes()
        header[name] = {'dtype': 'F32', 'shape': list(value.shape), 'data_offsets': [offset, offset + len(raw)]}
        buffers.append(raw)
        offset += len(raw)
    raw_header = json.dumps(header, separators=(',', ':')).encode()
    raw_header += b' ' * (-len(raw_header) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(raw_header)))
        f.write(raw_header)
        for raw in buffers:
            f.write(raw)


def load_tensors(path):
    size = Path(path).stat().st_size
    with open(path, 'rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated tensor header')
        length, = struct.unpack('<Q', prefix)
        if length > min(size - 8, 8 * 1024 * 1024):
            raise ValueError('invalid tensor header size')
        header = json.loads(f.read(length))
    weights, end = {}, 0
    for name, entry in sorted(header.items(), key=lambda item: item[1].get('data_offsets', [0])[0]):
        if name == '__metadata__':
            continue
        lo, hi = entry['data_offsets']
        shape = entry['shape']
        if (entry['dtype'] != 'F32' or lo != end or hi < lo or hi > size - 8 - length
                or any(type(n) is not int or n < 0 for n in shape)
                or int(np.prod(shape, dtype=object)) * 4 != hi - lo):
            raise ValueError(f'invalid tensor entry: {name}')
        weights[name] = np.memmap(path, dtype='<f4', mode='r', offset=8 + length + lo, shape=tuple(shape))
        end = hi
    if end != size - 8 - length:
        raise ValueError('tensor payload has unclaimed bytes')
    return weights


def expected_shapes(c):
    shapes = {'embedding.weight': (c.vocab_size, c.width), 'norm.weight': (c.width,)}
    for i in range(c.layers):
        p = f'blocks.{i}.'
        for name in ('q', 'o', 'gate'):
            shapes[p + name + '.weight'] = (c.width, c.width)
        for name in ('k', 'v'):
            shapes[p + name + '.weight'] = (c.kv_heads * c.head_dim, c.width)
        for name in ('norm', 'post_norm'):
            shapes[p + name + '.weight'] = (c.width,)
        for name in ('q_norm', 'k_norm'):
            shapes[p + name + '.weight'] = (c.head_dim,)
    return shapes


def export_bundle(destination, config, weights, tokenizer, training_manifest):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError('tokenizer/model vocabulary mismatch')
    if {k: tuple(v.shape) for k, v in weights.items()} != expected_shapes(config):
        raise ValueError('model tensor names/shapes mismatch')
    temporary = Path(tempfile.mkdtemp(prefix='.export-', dir=destination.parent))
    try:
        save_tensors(temporary / 'model.safetensors', weights)
        tokenizer.save(temporary / 'tokenizer.json')
        (temporary / 'training.json').write_text(json.dumps(training_manifest, indent=2, sort_keys=True) + '\n')
        manifest = {'format': FORMAT, 'config': config.to_dict(), 'dtype': 'float32',
                    'parameters': sum(v.size for v in weights.values()),
                    'files': {name: sha256(temporary / name) for name in
                              ('model.safetensors', 'tokenizer.json', 'training.json')}}
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def load_bundle(directory):
    from .tokenizer import Tokenizer
    from .runtime.numpy_model import NumpyModel
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('format') != FORMAT or set(manifest.get('files', {})) != {'model.safetensors', 'tokenizer.json', 'training.json'}:
        raise ValueError('unsupported or incomplete bundle')
    for name, digest in manifest['files'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'digest mismatch: {name}')
    config = Config(**manifest['config'])
    tokenizer = Tokenizer.load(directory / 'tokenizer.json')
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError('vocabulary mismatch')
    weights = load_tensors(directory / 'model.safetensors')
    if {k: tuple(v.shape) for k, v in weights.items()} != expected_shapes(config):
        raise ValueError('tensor names/shapes mismatch')
    return NumpyModel(config, weights), tokenizer, manifest

