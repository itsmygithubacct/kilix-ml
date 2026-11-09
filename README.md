# kilix-ml

Small, independently implemented tool models: PyTorch training and an offline
NumPy-only inference runtime. MIT, by itsmygithubacct.

**Status:** reaches users inside [kilix-needle](https://github.com/itsmygithubacct/kilix-needle#readme),
which carries this repository as its `third_party/kilix-ml` submodule; the
Kilix content catalog installs Needle (`kilix install kilix-needle`) with its
submodules. Plebian-OS and Kilix do not install kilix-ml on its own.

## What it does

- Defines small attention-only tool-calling models (`models/7.2m.json`) and
  trains them with PyTorch (`kilix_ml.train`).
- Runs exported bundles on the CPU with NumPy only, under grammar-constrained
  decoding against a tool schema (`kilix_ml.runtime`).
- Holds the domain packs: corpora, tool lists and validators for Kilix panes,
  apps and agents, plus a synthetic-home pack. The `kilix_panes`, `kilix_apps`
  and `kilix_agents` packs are the tuning library that `kilix-needle tune`
  reads.
- Evaluates checkpoints and exports fp32 safetensors with verified JSON
  manifests and no pickle.

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

Schema object keys and generated argument keys use lexicographic order in both
training and inference, matching the sorted corpus JSON. This applies recursively
to object values in supported `const` and `enum` literals, including objects in
constant arrays. Tool lists, enum alternatives and data arrays retain their
supplied order; caller-owned values are not mutated. General nested object and
array schemas remain unsupported. Older evaluations that used unsorted schemas
must be rerun; custom checkpoints trained with a different ordering need
revalidation.

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

## How it works

```
 domains/<pack>/  corpus templates, tools.json, validator
        |  generate.py fills templates into requests and expected calls
        v
 kilix_ml.train   tokenizer, pretraining, SFT  ->  safetensors + manifest
        |  kilix_ml.export: digests every file, no pickle
        v
 kilix_ml.runtime NumPy forward pass, cached tool prefix, grammar mask
        |  calls are proposals only
        v
 kilix-needle     semantic checks, confirmation, execution
```

`kilix_ml.format` renders the same prompt for training labels and inference.
`kilix_ml.grammar` turns each tool's JSON schema into a mask over the next
token, so a decoded call always parses against the schema. The runtime never
executes a call; the Kilix packs check proposals through kilix-needle, which
owns confirmation and execution.

## Documentation

- [NOTICE.md](NOTICE.md) — training data sources, licences and transformations
- [domains/kilix_panes/README.md](domains/kilix_panes/README.md) — the panes contract, its 14 tools and the batch validator
- [domains/kilix_panes/LICENCES-TRAINING.md](domains/kilix_panes/LICENCES-TRAINING.md) — licences of the panes training dependencies
- [domains/kilix_apps/README.md](domains/kilix_apps/README.md) — the apps job's training library
- [domains/kilix_agents/TEMPLATE-SPEC.md](domains/kilix_agents/TEMPLATE-SPEC.md) — template rules for the agents pack
- [domains/synthetic_home/README.md](domains/synthetic_home/README.md) — the second-domain schema pack

## Related repositories

- [kilix-needle](https://github.com/itsmygithubacct/kilix-needle#readme) — the application that pins this repository as a submodule, tunes from its packs and checks its proposals
- [kilix-content](https://github.com/itsmygithubacct/kilix-content#readme) — the catalog that installs kilix-needle

## Part of Plebian-OS

kilix-ml is one of the repositories behind
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
It is installed on demand as a submodule of kilix-needle. The map of every
repository and how it ships is on the
[components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## Licence

MIT. See [LICENSE](LICENSE).
