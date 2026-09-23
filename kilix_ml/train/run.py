"""Offline, resumable pretraining and full-parameter tool SFT.

Run ``python -m kilix_ml.train.run --help``. A probe is pretraining with a
short --seconds limit; it retains its checkpoint for continuation.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from ..export import export_bundle, load_bundle, sha256
from ..model.config import Config
from ..model.san import SAN
from ..tokenizer import Tokenizer
from .data import Corpus, PretrainBatches, SFTBatches, jsonl
from ..format import example

VALIDATION_FORMAT_VERSION = 1
VALIDATION_LIMITS = {'generic': 128, 'kilix': 128, 'synthetic_home': 32}


def prepare_sft_validation(corpus, tokenizer, stage, *, context, seed=341):
    """Create fixed validation rows without exposing any of them to SFTBatches."""
    split_name = 'dev' if stage == 'synthetic_home' else 'val'
    source_part = stage if stage != 'generic' else 'generic'
    source_path = corpus.root / source_part / f'{split_name}.jsonl'
    limit = VALIDATION_LIMITS[stage]
    rng = random.Random(seed)
    sampled, swap_rows = [], []
    seen_non_swap = 0
    side_counts = {'left': 0, 'right': 0, 'above': 0, 'below': 0}
    swap_calls = 0
    for row in jsonl(source_path):
        tools = corpus.resolve_tools(row)
        ids, labels = example(tokenizer, row['request'], row['calls'], tools)
        if len(ids) > context:
            raise ValueError(f'{source_path} example has {len(ids)} tokens, exceeds context={context}')
        encoded = (np.asarray(ids, dtype=np.int64), np.asarray(labels, dtype=np.int64))
        is_swap = stage == 'kilix' and any(call['name'] == 'swap_panes' for call in row['calls'])
        if is_swap:
            swap_rows.append(encoded)
            for call in row['calls']:
                if call['name'] == 'swap_panes':
                    swap_calls += 1
                    side = call.get('arguments', {}).get('side')
                    if side in side_counts:
                        side_counts[side] += 1
            continue
        seen_non_swap += 1
        if len(sampled) < limit:
            sampled.append(encoded)
        else:
            slot = rng.randrange(seen_non_swap)
            if slot < limit:
                sampled[slot] = encoded
    if not sampled:
        raise ValueError(f'empty SFT validation partition: {source_path}')
    validation = {f'{source_part}_{split_name}': sampled}
    sources = {f'{source_part}_{split_name}': {'path': str(source_path),
                 'source_rows': seen_non_swap + len(swap_rows), 'sampled_rows': len(sampled),
                 'sample_limit': limit, 'sampling': 'deterministic reservoir; swap rows excluded'} }
    swap_support = {'source_path': str(source_path), 'rows': len(swap_rows), 'calls': swap_calls,
                    'side_support': side_counts, 'expected_rows': 30}
    if stage == 'kilix':
        if not swap_rows:
            raise ValueError('Kilix SFT validation has no swap_panes rows')
        sources['swap_panes_val'] = {'path': str(source_path), 'source_rows': len(swap_rows),
                                     'sampled_rows': len(swap_rows), 'sample_limit': None,
                                     'sampling': 'all swap_panes validation rows'}
    return validation, swap_rows, swap_support, sources


def run_sft_validation(model, validation_sets, batch_size, device, *, pad_id, step, final=False):
    """Report token-weighted validation loss; do not alter weights, optimizer, or RNG."""
    was_training = model.training
    model.eval()
    out = {'step': step, 'final': bool(final), 'losses': {}}
    try:
        with torch.inference_mode():
            for name, rows in validation_sets.items():
                weighted_loss = 0.0
                target_tokens = 0
                for start in range(0, len(rows), batch_size):
                    batch = rows[start:start + batch_size]
                    length = max(len(ids) for ids, _ in batch)
                    x = np.full((len(batch), length), pad_id, dtype=np.int64)
                    y = np.full_like(x, -100)
                    for i, (ids, labels) in enumerate(batch):
                        x[i, :len(ids)] = ids
                        y[i, :len(labels)] = labels
                    x_tensor = torch.from_numpy(x).to(device)
                    y_tensor = torch.from_numpy(y).to(device)
                    count = int(np.count_nonzero(y[:, 1:] != -100))
                    amp = torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else contextlib.nullcontext()
                    with amp:
                        loss = model(x_tensor, y_tensor)
                    weighted_loss += float(loss) * count
                    target_tokens += count
                if target_tokens <= 0:
                    raise ValueError(f'validation split {name} has no target tokens')
                out['losses'][name] = {'validation_loss': weighted_loss / target_tokens,
                                       'rows': len(rows), 'target_tokens': target_tokens}
        return out
    finally:
        model.train(was_training)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['pretrain', 'generic', 'kilix', 'synthetic_home'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--preset', choices=['7.2m', '25m'], default='7.2m')
    p.add_argument('--config', type=Path)
    p.add_argument('--corpus', type=Path)
    p.add_argument('--fineweb', type=Path)
    p.add_argument('--synth', type=Path)
    p.add_argument('--data-manifest', type=Path)
    p.add_argument('--when2call', type=Path, help='converted, verified When2Call directory for generic SFT')
    p.add_argument('--initialize', type=Path, help='verified exported bundle from preceding stage')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--tokens', type=int, default=30_000_000_000)
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--seconds', type=float, default=0)
    p.add_argument('--deadline', type=float, default=0, help='Unix timestamp, hard stop between steps')
    p.add_argument('--sequence', type=int, default=1024)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--accumulate', type=int, default=32)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--validation-every', type=int, default=250,
                   help='SFT validation interval in optimizer steps; final validation always runs')
    p.add_argument('--seed', type=int, default=341)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--compile', action='store_true')
    return p.parse_args(argv)


def run(a):
    if min(a.batch, a.accumulate, a.sequence, a.checkpoint_every, a.validation_every) <= 0:
        raise ValueError('batch/accumulation/sequence/checkpoint/validation interval must be positive')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    if a.device == 'cpu':
        torch.use_deterministic_algorithms(True)
    tokenizer = Tokenizer.load(a.tokenizer)
    if a.initialize:
        loaded, _, initial_manifest = load_bundle(a.initialize)
        config = loaded.config
    else:
        config = Config(**json.loads(a.config.read_text())) if a.config else Config.preset(a.preset)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(f'vocabulary {tokenizer.vocab_size} differs from model {config.vocab_size}')
    if a.stage != 'pretrain' and not a.initialize and not a.resume:
        raise ValueError('SFT requires a pretrained --initialize bundle or --resume')
    model = SAN(config).to(a.device)
    if a.initialize:
        model.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in loaded.weights.items()})
        del loaded
    lr = a.lr if a.lr is not None else (0.006 if config.width == 256 else 0.003) if a.stage == 'pretrain' else 0.0003
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.ndim >= 2], 'weight_decay': 0.1},
        {'params': [p for p in model.parameters() if p.ndim < 2], 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.95), fused=a.device.startswith('cuda'))
    data_info = {}
    validation_sets = {}
    swap_validation_rows = []
    swap_validation_support = None
    if a.stage == 'pretrain':
        if not a.fineweb or not a.synth or not a.data_manifest:
            raise ValueError('pretraining needs both local token streams and --data-manifest')
        provenance = json.loads(a.data_manifest.read_text())
        for name, path in [('fineweb', a.fineweb), ('synth', a.synth)]:
            if sha256(path) != provenance['streams'][name]['sha256']:
                raise ValueError(f'{name} stream digest mismatch')
        data = PretrainBatches(a.fineweb, a.synth, a.batch, a.sequence, a.seed)
        if a.sequence + 1 > config.context:
            raise ValueError('pretraining sequence + target exceeds context')
        data_info = provenance
    else:
        corpus = Corpus(a.corpus)
        parts, probabilities = {
            'generic': (['generic', 'cross'], [85, 15]),
            'kilix': (['kilix', 'generic', 'cross'], [65, 20, 15]),
            'synthetic_home': (['synthetic_home'], [100]),
        }[a.stage]
        groups = []
        for part in parts:
            rows, stats = corpus.encode(tokenizer, part, context=config.context)
            groups.append(rows)
            data_info[part] = stats
        validation_sets, swap_validation_rows, swap_validation_support, validation_sources = prepare_sft_validation(
            corpus, tokenizer, a.stage, context=config.context, seed=a.seed)
        if a.when2call and a.stage == 'generic':
            from .data import jsonl
            from ..format import example
            manifest = json.loads((a.when2call / 'MANIFEST.json').read_text())
            if sha256(a.when2call / 'train.jsonl') != manifest['files']['train.jsonl']['sha256']:
                raise ValueError('When2Call digest mismatch')
            extra = []
            skipped = 0
            for row in jsonl(a.when2call / 'train.jsonl'):
                ids, labels = example(tokenizer, row['request'], row['calls'], row['tools'])
                if len(ids) > config.context:
                    skipped += 1
                    continue
                extra.append((np.array(ids, dtype=np.int64), np.array(labels, dtype=np.int64)))
            if not extra:
                raise ValueError('no supported When2Call examples within context')
            groups.append(extra)
            probabilities = [75, 15, 10]
            data_info['when2call'] = {'manifest_sha256': sha256(a.when2call / 'MANIFEST.json'),
                                      'rows': len(extra), 'over_context_dropped': skipped}
        data_info['corpus_manifest_sha256'] = sha256(a.corpus / 'MANIFEST.json')
        data = SFTBatches(groups, probabilities, a.batch, tokenizer.special['<pad>'], a.seed)
    code = hashlib.sha256()
    code_root = Path(__file__).resolve().parents[1]
    for path in sorted(code_root.rglob('*.py')):
        code.update(str(path.relative_to(code_root)).encode() + b'\0' + path.read_bytes() + b'\0')
    signature = {'stage': a.stage, 'config': config.to_dict(), 'tokenizer_sha256': sha256(a.tokenizer),
                 'code_sha256': code.hexdigest(),
                 'validation_format_version': VALIDATION_FORMAT_VERSION,
                 'validation_every': a.validation_every,
                 'initialized_from_manifest_sha256': sha256(a.initialize / 'manifest.json') if a.initialize else None,
                 'data': data_info, 'seed': a.seed, 'batch': a.batch, 'accumulate': a.accumulate,
                 'sequence': a.sequence, 'lr': lr, 'warmup': a.warmup, 'tokens_target': a.tokens, 'steps_target': a.steps}
    checkpoint = a.out / 'checkpoint.pt'
    if a.out.exists() and not a.resume and any(a.out.iterdir()):
        raise FileExistsError(f'{a.out} already contains a run; use --resume explicitly')
    a.out.mkdir(parents=True, exist_ok=True)
    step, tokens, cumulative_seconds = 0, 0, 0.0
    if a.resume:
        state = torch.load(checkpoint, map_location=a.device, weights_only=True)
        if state['signature'] != signature:
            raise ValueError('resume configuration/data mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        data.load_state_dict(state['data'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if a.device.startswith('cuda') and state['cuda_rng']:
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        step, tokens, cumulative_seconds = state['step'], state['tokens'], state['seconds']
    forward = torch.compile(model) if a.compile else model
    stopping = [False]
    previous = {sig: signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    started, starting_tokens = time.monotonic(), tokens
    last_loss = None
    reason = 'target'
    validation_history = []

    def save():
        state = {'signature': signature, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'data': data.state_dict(), 'step': step, 'tokens': tokens,
                 'seconds': cumulative_seconds + time.monotonic() - started,
                 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else []}
        temporary = checkpoint.with_suffix('.tmp')
        torch.save(state, temporary)
        os.replace(temporary, checkpoint)
        return state

    try:
        while tokens < a.tokens and (not a.steps or step < a.steps):
            if stopping[0] or (a.seconds and time.monotonic() - started >= a.seconds) or (a.deadline and time.time() >= a.deadline):
                reason = 'signal' if stopping[0] else 'time_limit'
                break
            progress = step / max(1, a.steps) if a.steps else tokens / a.tokens
            factor = min(1.0, (step + 1) / max(a.warmup, 1))
            if progress > 0.8:
                factor *= 0.1 + 0.9 * (1 + math.cos(math.pi * min(1, (progress - 0.8) / 0.2))) / 2
            for group in optimizer.param_groups:
                group['lr'] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=a.device)
            step_tokens = 0
            data_before_step = copy.deepcopy(data.state_dict())
            interrupted_step = False
            for micro in range(a.accumulate):
                if stopping[0] or (a.deadline and time.time() >= a.deadline):
                    interrupted_step = True
                    break
                ids, labels = data.next()
                x = torch.from_numpy(ids).to(a.device)
                y = torch.from_numpy(labels).to(a.device)
                amp = torch.autocast('cuda', dtype=torch.bfloat16) if a.device.startswith('cuda') else contextlib.nullcontext()
                with amp:
                    loss = forward(x, y)
                (loss / a.accumulate).backward()
                total_loss += loss.detach() / a.accumulate
                step_tokens += int(np.count_nonzero(labels[:, 1:] != -100))
            if interrupted_step:
                data.load_state_dict(data_before_step)
                optimizer.zero_grad(set_to_none=True)
                reason = 'signal' if stopping[0] else 'time_limit'
                break
            total_loss = float(total_loss)
            if not math.isfinite(total_loss):
                raise FloatingPointError('nonfinite loss')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            tokens += step_tokens
            last_loss = total_loss
            elapsed = time.monotonic() - started
            metrics = {'step': step, 'tokens': tokens, 'loss': total_loss, 'lr': lr * factor,
                       'elapsed_seconds': elapsed, 'tokens_per_second': (tokens - starting_tokens) / elapsed}
            if step % 10 == 0 or step == 1:
                print(json.dumps(metrics), flush=True)
            with open(a.out / 'metrics.jsonl', 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            if a.stage != 'pretrain' and step % a.validation_every == 0:
                validation = run_sft_validation(model, validation_sets, a.batch, a.device,
                                                pad_id=tokenizer.special['<pad>'], step=step)
                if swap_validation_rows:
                    swap_loss = run_sft_validation(
                        model, {'swap_panes_val': swap_validation_rows}, a.batch, a.device,
                        pad_id=tokenizer.special['<pad>'], step=step)['losses']['swap_panes_val']
                    validation['losses']['swap_panes_val'] = swap_loss | swap_validation_support
                validation['sources'] = validation_sources
                validation_history.append(validation)
                with open(a.out / 'validation.jsonl', 'a') as f:
                    f.write(json.dumps(validation, sort_keys=True) + '\n')
                print(json.dumps({'validation_loss': validation}, sort_keys=True), flush=True)
            progress_path = os.environ.get('KILIX_PROGRESS_FILE')
            if progress_path:
                progress_file = Path(progress_path)
                progress_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = progress_file.with_suffix('.tmp')
                temporary.write_text(str(tokens) + '\n')
                os.replace(temporary, progress_file)
            if step % a.checkpoint_every == 0:
                save()
        state = save()
        final_validation = None
        if a.stage != 'pretrain':
            final_validation = run_sft_validation(model, validation_sets, a.batch, a.device,
                                                  pad_id=tokenizer.special['<pad>'], step=step, final=True)
            if swap_validation_rows:
                swap_loss = run_sft_validation(
                    model, {'swap_panes_val': swap_validation_rows}, a.batch, a.device,
                    pad_id=tokenizer.special['<pad>'], step=step, final=True)['losses']['swap_panes_val']
                final_validation['losses']['swap_panes_val'] = swap_loss | swap_validation_support
            final_validation['sources'] = validation_sources
            validation_history.append(final_validation)
            with open(a.out / 'validation.jsonl', 'a') as f:
                f.write(json.dumps(final_validation, sort_keys=True) + '\n')
        summary = signature | {'step': step, 'tokens': tokens, 'seconds': state['seconds'],
                               'loss': last_loss, 'stop_reason': reason, 'parameters': model.parameter_count(),
                               'data_state': data.state_dict(), 'torch': torch.__version__, 'device': a.device,
                               'validation_loss': final_validation,
                               'validation_history_rows': len(validation_history)}
        destination = a.out / f'export-{step:08d}'
        if not destination.exists():
            export_bundle(destination, config, {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}, tokenizer, summary)
        (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'export': str(destination), 'summary': summary}), flush=True)
        return summary
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    run(arguments())
