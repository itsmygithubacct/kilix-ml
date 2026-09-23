from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    vocab_size: int = 8192
    width: int = 256
    layers: int = 20
    heads: int = 8
    kv_heads: int = 4
    context: int = 3072
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    def __post_init__(self):
        if min(self.vocab_size, self.width, self.layers, self.heads, self.kv_heads, self.context) <= 0:
            raise ValueError("dimensions must be positive")
        if self.width % self.heads or self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("GQA dimensions must divide and head dimension must be even")

    @property
    def head_dim(self):
        return self.width // self.heads

    def to_dict(self):
        return asdict(self)

    @classmethod
    def preset(cls, name):
        if name == "7.2m":
            return cls()
        if name == "25m":
            return cls(width=512)
        raise ValueError(f"unknown model preset: {name}")
