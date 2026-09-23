# kilix-ml

Small, independently implemented tool models: PyTorch training and an offline
NumPy-only inference runtime. MIT, by itsmygithubacct.

## Status

Training and inference code are implemented. No model is promoted for use in
Kilix until it passes the independent safety, accuracy, latency, memory, and
second-domain gates. Untrained smoke-test exports are not usable model releases.

The nominal **7.2M** configuration has **7,351,808** parameters: 20 attention-only
layers, width 256, 8 query/4 KV heads, RoPE, Q/K RMS normalization, post-attention
normalization, output gates and tied embeddings. The 25M configuration is
25,189,376 parameters.

The complete 14-tool prefix needs about 2,291 tokens. Context is therefore 3,072,
while base-pretraining sequences remain 1,024. Prompts are never silently cut.

## Install and check

Python 3.11+; dependency lock generated with uv 0.12.3. Runtime needs only NumPy.

```sh
uv sync --locked
uv sync --locked --extra train --extra test
uv run python -m pytest -q
unshare -Urn uv run --no-sync python -m pytest -q
```

Tests cover Torch/NumPy logits, incremental decoding, BF16 backward, non-pickle
model export, corruption rejection, repeated CPU training digests, isolated-process
runtime imports, and 10,000 independently schema-checked grammar walks.

## Inference

```python
from kilix_ml.runtime.complete import Engine
from kilix_ml.domain import Domain

domain = Domain("domains/kilix_panes")
engine = Engine("/path/to/verified/model-bundle")
result = engine.complete("open a pane on the right", tools=domain.tools)
checked = domain.validate_many([{"request": "open a pane on the right", "calls": result.calls}])
```

`complete(messages, tools=None)` also supports text completion. Structured results
contain `text`, `calls`, `confidence` and `metrics`; confidence remains `None` until
calibrated. Calls are proposals. This library never executes them. The Kilix pack
uses the pinned kilix-needle bridge for semantic checks; execution and confirmation
remain at that application boundary.

Persistent JSONL process, or add `--request` for a single request:

```sh
python -m kilix_ml --bundle /path/to/bundle --domain domains/kilix_panes --validate
```

Inputs use `{"request":"…"}` or `{"messages":[…]}`. The runtime verifies every
bundle file digest, caches the system/tool prefix, batches forced JSON punctuation,
and reuses KV buffers. It defaults BLAS threads to one before importing NumPy;
explicit caller environment settings win. Set those variables before importing
NumPy in a larger host process too. Engine calls are serialized with a lock.

## Training

`models/7.2m.json` records a 30B-token pretraining target followed by generic,
Kilix, and synthetic-home supervised fine-tuning. A short throughput probe can
measure the training setup on a repeating public sample; its weights are
excluded from the main training run.

Acquisition is an explicit online step (`kilix_ml.train.acquire`). Training reads
only verified local token streams and corpus files. FineWeb-Edu/SYNTH are mixed
7:3 by fixed-size token blocks. Stream wraps are counted rather than represented
as new data. See [NOTICE.md](NOTICE.md) for sources, licenses and transformations.

```sh
python -m kilix_ml.train.run --help
python -m kilix_ml.train.when2call --help
python -m kilix_ml.evaluate --help
```

The same renderer defines training labels and inference prompts. Pretraining uses
all next-token targets; SFT masks the prompt. Stages save optimizer/RNG/data state
for explicit resume and export fp32 safetensors plus verified JSON manifests.
Stage ancestry, source-code digest, data digests, parameter counts and token counts
are recorded. Optimizer checkpoints are training-only; runtime bundles contain no
pickle. The second pack is `domains/synthetic_home`.

Independent held-out cases, training data, and generated weights live outside
this source directory. Training artifacts must not contain credentials or live
Kilix user data.
