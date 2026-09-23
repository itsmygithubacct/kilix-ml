"""Attention-only decoder, implemented independently for kilix-ml."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight.to(x.dtype)


def rope(x, positions, theta):
    d = x.shape[-1]
    angles = positions.float()[:, None] * theta ** (-torch.arange(0, d, 2, device=x.device).float() / d)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = self.config = config
        d, kd = c.width, c.kv_heads * c.head_dim
        self.norm = RMSNorm(d, c.norm_eps)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, kd, bias=False)
        self.v = nn.Linear(d, kd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.norm_eps)
        self.post_norm = RMSNorm(d, c.norm_eps)

    def forward(self, x, positions):
        c = self.config
        b, t, _ = x.shape
        n = self.norm(x)
        q = self.q_norm(self.q(n).view(b, t, c.heads, c.head_dim).transpose(1, 2))
        k = self.k_norm(self.k(n).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2))
        v = self.v(n).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        a = a.transpose(1, 2).reshape(b, t, c.width) * torch.sigmoid(self.gate(n))
        return x + self.post_norm(self.o(a))


class SAN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width, config.norm_eps)
        self.apply(self._init)
        # Scale residual projections with depth.
        for block in self.blocks:
            nn.init.normal_(block.o.weight, std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids, labels=None):
        if ids.ndim != 2 or ids.shape[1] > self.config.context:
            raise ValueError("expected batch by sequence within context limit")
        x = self.embedding(ids)
        positions = torch.arange(ids.shape[1], device=ids.device)
        for block in self.blocks:
            x = block(x, positions)
        logits = F.linear(self.norm(x), self.embedding.weight)
        if labels is None:
            return logits
        return F.cross_entropy(logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                               labels[:, 1:].reshape(-1), ignore_index=-100)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
