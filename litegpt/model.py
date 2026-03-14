"""
Decoder-only transformer (GPT architecture).

Auto-scaling: all hyperparameters derive from a single --depth parameter.
  model_dim = depth * 64
  num_heads  = max(1, model_dim // 128)   → head_dim = 128 always
  mlp_dim    = model_dim * 4 (rounded to multiple of 64)

Modern choices from nanochat:
  - RoPE positional embeddings
  - QK normalization (prevents attention entropy collapse)
  - ReLU² activation (not SwiGLU — simpler, similar performance)
  - Learnable per-layer residual scalars
  - Untied input/output embeddings
  - F.scaled_dot_product_attention (FlashAttention on CUDA, fallback on MPS/CPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    depth: int = 12
    vocab_size: int = 50304   # padded GPT-2 vocab (multiple of 64)
    seq_len: int = 2048

    @property
    def model_dim(self) -> int:
        return self.depth * 64

    @property
    def num_heads(self) -> int:
        # head_dim stays ~128; minimum 1 head for tiny smoke-test models
        return max(1, self.model_dim // 128)

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads

    @property
    def mlp_dim(self) -> int:
        return ((self.model_dim * 4) // 64) * 64


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def build_rope_cache(
    seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) each of shape (seq_len, head_dim)."""
    # Compute in float32 for precision, cast at the end
    half = head_dim // 2
    theta = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)                    # (T, head_dim/2)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)  # (T, head_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, H, T, D) | cos, sin: (1, 1, T, D)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim  = config.head_dim
        self.model_dim = config.model_dim

        self.qkv = nn.Linear(config.model_dim, 3 * config.model_dim, bias=False)
        self.out  = nn.Linear(config.model_dim, config.model_dim, bias=False)

        # QK norm — prevents attention logit explosion in deep models
        self.q_norm = RMSNorm(config.head_dim)
        self.k_norm = RMSNorm(config.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.model_dim, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # B H T D
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE — cos/sin: (1, 1, T, D)
        cos_t = cos.unsqueeze(0).unsqueeze(0)
        sin_t = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rope(q, k, cos_t, sin_t)

        # SDPA: uses FlashAttention kernel on CUDA SM80+, falls back gracefully elsewhere
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class MLP(nn.Module):
    """Two-layer MLP with ReLU² activation (simpler than SwiGLU, similar quality)."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.model_dim, config.mlp_dim, bias=False)
        self.fc2 = nn.Linear(config.mlp_dim, config.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)) ** 2)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1  = RMSNorm(config.model_dim)
        self.attn   = CausalSelfAttention(config)
        self.norm2  = RMSNorm(config.model_dim)
        self.mlp    = MLP(config)

        # Learnable residual scalars — lets each layer calibrate its own contribution
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.alpha1 * self.attn(self.norm1(x), cos, sin)
        x = x + self.alpha2 * self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.embed  = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.norm   = RMSNorm(config.model_dim)
        self.head   = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        # Untied embeddings (no weight sharing between embed and head)

        # RoPE buffers — filled by setup_rope() after moving to device
        self.register_buffer("rope_cos", torch.zeros(config.seq_len, config.head_dim))
        self.register_buffer("rope_sin", torch.zeros(config.seq_len, config.head_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def setup_rope(self):
        """Call once after moving the model to its target device/dtype."""
        cos, sin = build_rope_cache(
            self.config.seq_len,
            self.config.head_dim,
            self.rope_cos.device,
            self.rope_cos.dtype,
        )
        self.rope_cos.copy_(cos)
        self.rope_sin.copy_(sin)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.shape
        x = self.embed(idx)

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        c = self.config
        return (
            f"GPT(depth={c.depth}, dim={c.model_dim}, heads={c.num_heads}, "
            f"head_dim={c.head_dim}, mlp_dim={c.mlp_dim}, "
            f"params={self.num_parameters()/1e6:.1f}M)"
        )
