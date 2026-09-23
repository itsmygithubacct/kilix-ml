from dataclasses import dataclass, field
import json
import time
import threading
import numpy as np
from ..export import load_bundle
from ..format import conversation, prefix
from ..grammar import compile_tools, TokenMask
from ..grammar.mask import TokenTrie


@dataclass
class Result:
    text: str = ''
    calls: list = field(default_factory=list)
    confidence: float | None = None
    metrics: dict = field(default_factory=dict)

    @property
    def function_calls(self):
        return self.calls


class Engine:
    """Greedy local completion. Calls are proposals; this class never executes them.

    confidence is deliberately None until independently calibrated. Keep the
    domain's semantic validator and confirmation policy at the execution seam.
    """
    def __init__(self, bundle, *, max_new_tokens=192):
        self.model, self.tokenizer, self.manifest = load_bundle(bundle)
        self.max_new_tokens = max_new_tokens
        self._prefix_key = None
        self._prefix_cache = None
        self._lock = threading.Lock()
        self._token_trie = TokenTrie(self.tokenizer.token_bytes)

    def _prefill(self, ids, cache=None):
        for start in range(0, len(ids), 64):
            logits, cache = self.model.forward(ids[start:start + 64], cache, last_only=True)
        return logits, cache

    def complete(self, messages, tools=None):
        with self._lock:
            return self._complete(messages, tools)

    def _complete(self, messages, tools=None):
        started = time.perf_counter()
        tok = self.tokenizer
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        if not messages or messages[-1]['role'] != 'user':
            raise ValueError('completion requires a final user message')
        request = messages[-1]['content']
        if len(request.encode()) > 4096:
            raise ValueError('request exceeds 4096 bytes')
        system = messages[0]['content'] if messages[0]['role'] == 'system' else None
        pinned = prefix(tok, tools, system)
        mode = '<|calls|>' if tools is not None else '<|text|>'
        prompt = conversation(tok, messages, tools) + [tok.special['<|assistant|>'], tok.special[mode]]
        if len(prompt) + self.max_new_tokens > self.model.config.context:
            raise ValueError('prompt and generation budget exceed context; no silent truncation')
        key = tuple(pinned)
        hit = key == self._prefix_key
        if not hit:
            self._prefix_cache = None
            _, self._prefix_cache = self._prefill(pinned)
            self._prefix_key = key
        self._prefix_cache.length = len(pinned)
        logits, cache = self._prefill(prompt[len(pinned):], self._prefix_cache)
        prefilled = time.perf_counter()
        grammar = compile_tools(tools, request) if tools is not None else None
        mask = TokenMask(grammar, trie=self._token_trie) if grammar is not None else None
        state = grammar.initial_state() if grammar is not None else None
        generated, finished = [], False
        pending = []
        forced_tokens = 0
        mask_seconds = 0.0
        for _ in range(self.max_new_tokens):
            if grammar is not None and grammar.is_complete(state):
                finished = True
                break
            mask_start = time.perf_counter()
            if mask is not None:
                allowed = mask.allowed(state)
                if not allowed:
                    break
            else:
                allowed = [i for i, raw in enumerate(tok.token_bytes) if raw] + [tok.special['<|end|>'], tok.special['<eos>']]
            # Sorted IDs give deterministic tie-breaking across runs.
            allowed = np.asarray(sorted(allowed), dtype=np.int64)
            mask_seconds += time.perf_counter() - mask_start
            if len(allowed) == 1:
                chosen = int(allowed[0])
                forced_tokens += 1
            else:
                # Fixed JSON punctuation does not require a model decision.
                # Prefill its accumulated tokens together at the next branch.
                if pending:
                    logits, cache = self._prefill(pending, cache)
                    pending = []
                chosen = int(allowed[np.argmax(logits[-1, allowed])])
            if mask is None and chosen in (tok.special['<|end|>'], tok.special['<eos>']):
                finished = True
                break
            generated.append(chosen)
            if mask is not None:
                state = mask.advance(state, chosen)
            pending.append(chosen)
        if grammar is not None:
            finished = grammar.is_complete(state)
        decoded = tok.decode(generated)
        calls = json.loads(decoded) if tools is not None and finished else []
        return Result(text=decoded if tools is None else '', calls=calls, metrics={
            'prompt_tokens': len(prompt), 'generated_tokens': len(generated), 'prefix_cache_hit': hit,
            'forced_tokens': forced_tokens,
            'prefill_ms': (prefilled - started) * 1000, 'mask_ms': mask_seconds * 1000,
            'total_ms': (time.perf_counter() - started) * 1000,
            'finished': finished, 'status': 'complete' if finished else 'generation_limit',
        })
