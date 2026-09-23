#!/usr/bin/env python3
"""Run one bounded phase on the already selected rental. Contains no credentials."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--stage', choices=['probe', 'acquire', 'pretrain', 'generic', 'kilix', 'synthetic_home'], required=True)
    a = p.parse_args()
    inputs, runs = a.job / 'inputs', a.job / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    seconds = int(os.environ['KILIX_STAGE_SECONDS'])
    deadline = time.time() + max(1, seconds - 90)
    tokenizer = inputs / 'tokenizer' / 'tokenizer.json'
    if a.stage == 'acquire':
        cmd = [sys.executable, '-m', 'kilix_ml.train.acquire', '--out', str(a.job / 'pretraining'),
               '--tokenizer', str(tokenizer), '--tokens', '30000000000']
        subprocess.run(cmd, check=True, timeout=max(1, seconds - 30))
        return
    source = inputs / 'probe' if a.stage == 'probe' else a.job / 'pretraining'
    base = [sys.executable, '-m', 'kilix_ml.train.run', '--stage', 'pretrain' if a.stage == 'probe' else a.stage,
            '--out', str(runs / a.stage), '--tokenizer', str(tokenizer), '--preset', '7.2m',
            '--device', 'cuda', '--deadline', str(deadline)]
    if a.stage in ('probe', 'pretrain'):
        base += ['--fineweb', str(source / 'fineweb.u16'), '--synth', str(source / 'synth.u16'),
                 '--data-manifest', str(source / 'acquisition.json'), '--tokens', '30000000000',
                 '--batch', '8', '--accumulate', '32', '--sequence', '1024', '--compile']
    else:
        preceding = {'generic': 'pretrain', 'kilix': 'generic', 'synthetic_home': 'generic'}[a.stage]
        summary = json.loads((runs / preceding / 'summary.json').read_text())
        if summary['step'] < 1:
            raise ValueError('preceding stage has no trained steps')
        bundle = runs / preceding / f"export-{summary['step']:08d}"
        base += ['--initialize', str(bundle), '--corpus', str(inputs / 'corpus'), '--steps',
                 '1000' if a.stage == 'synthetic_home' else '2000', '--batch', '4', '--accumulate', '8',
                 '--lr', '0.0003', '--warmup', '100']
        if a.stage == 'generic':
            base += ['--when2call', str(inputs / 'when2call')]
    subprocess.run(base, check=True)
    summary = json.loads((runs / a.stage / 'summary.json').read_text())
    if summary['step'] < 1:
        raise RuntimeError('phase completed without a training step')
    if a.stage == 'probe':
        rows = [json.loads(line) for line in (runs / 'probe' / 'metrics.jsonl').read_text().splitlines()]
        steady = rows[len(rows) // 2:]
        if len(steady) >= 2:
            throughput = (steady[-1]['tokens'] - steady[0]['tokens']) / (steady[-1]['elapsed_seconds'] - steady[0]['elapsed_seconds'])
        else:
            throughput = rows[-1]['tokens_per_second']
        report = {'target_probe_seconds': 1800, 'actual_training_seconds': summary['seconds'],
                  'steady_tokens_per_second': throughput, 'hours_for_30B_at_measured_rate': 30e9 / throughput / 3600,
                  'probe_weights_used_for_main_training': False,
                  'note': 'Probe repeats the small public sample; main pretraining starts from scratch on the full pinned streams.'}
        (runs / 'probe' / 'throughput.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
