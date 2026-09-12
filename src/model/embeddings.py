"""Rotary Position Embeddings (RoPE) for the Sparrow MoE architecture.

Computes complex rotary frequency tensors and applies rotation to query and key
representations in Grouped-Query Attention (GQA).
"""

from typing import Tuple
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor.

    Args:
        x: Input tensor of shape (..., head_dim).

    Returns:
        Tensor of identical shape with half-dimensions rotated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precomputes and caches rotary position embeddings (RoPE).

    Supports dynamic sequence length expansion and sequence offset for KV caching.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Calculate inverse frequency: theta_i = base^(-2(i-1)/dim)
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Compute outer product: (seq_len, dim / 2)
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate frequencies to match head_dim: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns cos and sin tensors shaped for broadcasting.

        Args:
            x: Reference tensor for device and dtype matching.
            seq_len: Sequence length of current query/key tokens.
            offset: Position offset (used when querying single tokens during KV cache generation).

        Returns:
            cos: Tensor of shape (1, 1, seq_len, dim)
            sin: Tensor of shape (1, 1, seq_len, dim)
        """
        end_pos = offset + seq_len
        if end_pos > self.max_seq_len_cached or self.cos_cached.device != x.device:
            self._set_cos_sin_cache(max(end_pos, self.max_seq_len), device=x.device, dtype=x.dtype)

        cos = self.cos_cached[offset:end_pos].to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:end_pos].to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to Query and Key tensors.

    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim).
        k: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim).
        cos: Cosine tensor broadcastable to q and k.
        sin: Sine tensor broadcastable to q and k.

    Returns:
        Rotated (q, k) tuple of identical shapes.
    """
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
