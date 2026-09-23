#!/usr/bin/env python3
"""Exercise the exact GPU architecture before the paid throughput probe."""
import argparse
import json
import numpy as np
import torch
from kilix_ml.model.config import Config
from kilix_ml.model.san import SAN
from kilix_ml.runtime.numpy_model import NumpyModel

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--probe-tokens', required=True)
a = p.parse_args()
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError('CUDA with BF16 support is required')
torch.manual_seed(341)
torch.set_num_threads(1)
model = SAN(Config.preset('7.2m')).cuda().eval()
ids = torch.tensor([[1, 31, 22, 7811, 4, 9]], device='cuda')
with torch.no_grad():
    actual = model(ids)[0].float().cpu().numpy()
weights = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
expected, _ = NumpyModel(model.config, weights).forward(ids[0].cpu().numpy())
error = float(np.max(np.abs(expected - actual)))
if error > 1e-4:
    raise AssertionError(f'GPU/NumPy logit parity failed: {error}')
model.train()
tokens = np.memmap(a.probe_tokens, dtype='<u2', mode='r')
x = torch.tensor(np.asarray(tokens[:8 * 1025], dtype=np.int64).reshape(8, 1025), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss = model(x, x)
loss.backward()
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
print(json.dumps({'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
                  'cuda': torch.version.cuda, 'parameters': model.parameter_count(),
                  'max_abs_fp32_logit_error': error, 'bf16_loss': float(loss),
                  'gradient_norm': float(norm), 'allocated_peak_mb': torch.cuda.max_memory_allocated() / 2**20,
                  'status': 'smoke_only_weights_discarded'}))
