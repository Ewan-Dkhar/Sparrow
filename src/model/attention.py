"""Grouped-Query Attention (GQA) with RoPE and FlashAttention-2 integration for Sparrow.

GQA reduces Key/Value cache footprint by sharing key and value heads across groups
of query heads, drastically optimizing inference memory on 8GB VRAM targets.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.embeddings import RotaryEmbedding, apply_rotary_emb


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE).

    Args:
        d_model: Input and output representation dimension.
        n_heads: Number of Query attention heads.
        n_kv_heads: Number of Key and Value heads (n_heads % n_kv_heads must be 0).
        head_dim: Dimensionality of each individual attention head.
        max_seq_len: Maximum supported sequence length for RoPE caching.
        rope_theta: Base frequency for rotary embeddings.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_groups = n_heads // n_kv_heads

        # Projections
        self.w_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # Rotary position embeddings
        self.rope = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for Grouped-Query Attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            kv_cache: Optional tuple of past (keys, values) each of shape
                      (batch, n_kv_heads, past_seq_len, head_dim).
            use_cache: Whether to return updated KV cache for autoregressive generation.

        Returns:
            out: Attention output of shape (batch, seq_len, d_model).
            new_kv_cache: Updated KV cache tuple if use_cache is True, else None.
        """
        b, s, _ = x.shape

        # Linear projections
        q = self.w_q(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Determine sequence position offset for RoPE cache
        offset = kv_cache[0].shape[2] if kv_cache is not None else 0
        cos, sin = self.rope(q, seq_len=s, offset=offset)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Update Key/Value cache if present
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        # Repeat KV heads for Grouped-Query Attention broadcast
        if self.num_groups > 1:
            k_expanded = torch.repeat_interleave(k, repeats=self.num_groups, dim=1)
            v_expanded = torch.repeat_interleave(v, repeats=self.num_groups, dim=1)
        else:
            k_expanded, v_expanded = k, v

        # Compute scaled dot product attention (dispatches to FlashAttention-2 when available)
        # Is causal if we are processing a prompt without a pre-existing KV cache
        is_causal = (s > 1) and (kv_cache is None)
        attn_out = F.scaled_dot_product_attention(
            query=q,
            key=k_expanded,
            value=v_expanded,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        # Concatenate heads and project back to d_model
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, self.n_heads * self.head_dim)
        out = self.w_o(attn_out)

        return out, new_kv_cache
