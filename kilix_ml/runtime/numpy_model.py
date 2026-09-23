"""Reference fp32 implementation with an incremental, caller-owned KV cache."""
import numpy as np
from ..model.config import Config


class KVCache:
    """One reusable allocation. Resetting length restores a pinned prefix in place."""
    def __init__(self, config):
        shape = (config.kv_heads, config.context, config.head_dim)
        self.storage = [(np.empty(shape, dtype=np.float32), np.empty(shape, dtype=np.float32))
                        for _ in range(config.layers)]
        self.length = 0

    def __getitem__(self, i):
        k, v = self.storage[i]
        return k[:, :self.length], v[:, :self.length]


def rms(x, weight, eps):
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * weight


def rope(x, positions, theta):
    d = x.shape[-1]
    angles = positions.astype(np.float32)[:, None] * np.float32(theta) ** (-np.arange(0, d, 2, dtype=np.float32) / d)
    cos, sin = np.cos(angles), np.sin(angles)
    y = np.empty_like(x)
    y[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    y[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    return y


class NumpyModel:
    def __init__(self, config: Config, weights):
        self.config, self.weights = config, weights

    def forward(self, ids, cache=None, *, last_only=False):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or not len(ids):
            raise ValueError("expected a nonempty token sequence")
        c, w = self.config, self.weights
        if cache is None:
            cache = KVCache(c)
        offset = cache.length
        if offset + len(ids) > c.context:
            raise ValueError("context limit exceeded")
        if np.any(ids < 0) or np.any(ids >= c.vocab_size):
            raise ValueError("invalid token ID")
        x = w['embedding.weight'][ids]
        positions = np.arange(offset, offset + len(ids))
        for i in range(c.layers):
            p = f'blocks.{i}.'
            n = rms(x, w[p+'norm.weight'], c.norm_eps)
            q = (n @ w[p+'q.weight'].T).reshape(-1, c.heads, c.head_dim).transpose(1, 0, 2)
            k = (n @ w[p+'k.weight'].T).reshape(-1, c.kv_heads, c.head_dim).transpose(1, 0, 2)
            v = (n @ w[p+'v.weight'].T).reshape(-1, c.kv_heads, c.head_dim).transpose(1, 0, 2)
            q = rope(rms(q, w[p+'q_norm.weight'], c.norm_eps), positions, c.rope_theta)
            k = rope(rms(k, w[p+'k_norm.weight'], c.norm_eps), positions, c.rope_theta)
            key_buffer, value_buffer = cache.storage[i]
            key_buffer[:, offset:offset + len(ids)] = k
            value_buffer[:, offset:offset + len(ids)] = v
            k, v = key_buffer[:, :offset + len(ids)], value_buffer[:, :offset + len(ids)]
            # Group query heads against shared K/V directly; never copy a long
            # pinned prefix just to broadcast its heads.
            grouped_q = q.reshape(c.kv_heads, -1, c.head_dim)
            scores = (grouped_q @ k.transpose(0, 2, 1)).reshape(c.heads, len(ids), -1) / np.float32(np.sqrt(c.head_dim))
            scores = np.where(np.arange(offset + len(ids))[None, :] <= positions[:, None], scores, -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            prob = np.exp(scores)
            prob /= prob.sum(axis=-1, keepdims=True)
            a = (prob.reshape(c.kv_heads, -1, offset + len(ids)) @ v).reshape(c.heads, len(ids), c.head_dim)
            a = a.transpose(1, 0, 2).reshape(-1, c.width)
            gate = 1 / (1 + np.exp(-np.clip(n @ w[p+'gate.weight'].T, -80, 80)))
            x = x + rms((a * gate) @ w[p+'o.weight'].T, w[p+'post_norm.weight'], c.norm_eps)
        if last_only:
            x = x[-1:]
        logits = rms(x, w['norm.weight'], c.norm_eps) @ w['embedding.weight'].T
        cache.length = offset + len(ids)
        return logits, cache
