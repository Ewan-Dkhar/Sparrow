"""Main Transformer block and SparrowMoE decoder model.

Integrates RMSNorm, Grouped-Query Attention (GQA) with RoPE, and Sparse MoE layers
featuring noisy Top-2 routing across 8 SwiGLU experts with auxiliary load balancing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import GroupedQueryAttention
from src.model.router import SparseMoELayer


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Avoids re-centering mean to simplify computation while retaining training stability.

    Args:
        dim: Hidden feature dimension.
        eps: Epsilon value added for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for RMSNorm.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor scaled by learnable gain parameter.
        """
        # Calculate root mean square along last dimension
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + self.eps)
        return x * rsqrt * self.weight


@dataclass
class SparrowConfig:
    """Hyperparameter configuration for the Sparrow MoE architecture."""

    name: str = "sparrow-moe-150m"
    vocab_size: int = 50257
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 64
    d_ff: int = 1024
    n_experts: int = 8
    top_k: int = 2
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    router_noise_mult: float = 1.0
    aux_loss_coef: float = 0.01
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SparrowConfig":
        """Creates configuration from dictionary (e.g., loaded from YAML)."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def count_parameters(self) -> Tuple[int, int]:
        """Calculates total and active parameter counts.

        Returns:
            (total_params, active_params_per_token)
        """
        # Embedding: vocab_size * d_model
        embed_params = self.vocab_size * self.d_model
        # Per layer:
        # 1) Attn: Wq (d_model * n_heads * head_dim) + Wk + Wv + Wo
        attn_params = (
            self.d_model * (self.n_heads * self.head_dim)
            + 2 * self.d_model * (self.n_kv_heads * self.head_dim)
            + (self.n_heads * self.head_dim) * self.d_model
        )
        # 2) Router: w_gate (d_model * n_experts) + w_noise (d_model * n_experts)
        router_params = 2 * (self.d_model * self.n_experts)
        # 3) Experts: n_experts * SwiGLU (3 * d_model * d_ff)
        expert_params = 3 * self.d_model * self.d_ff
        all_experts_params = self.n_experts * expert_params
        active_experts_params = self.top_k * expert_params
        # 4) Norms: 2 * d_model per layer
        norm_params = 2 * self.d_model

        layer_total = attn_params + router_params + all_experts_params + norm_params
        layer_active = attn_params + router_params + active_experts_params + norm_params

        total = embed_params + (self.n_layers * layer_total) + self.d_model
        active = embed_params + (self.n_layers * layer_active) + self.d_model

        if not self.tie_word_embeddings:
            total += embed_params
            active += embed_params

        return total, active


class SparrowBlock(nn.Module):
    """Transformer decoder block with pre-RMSNorm, GQA attention, and sparse MoE FFN.

    Args:
        config: SparrowConfig instance.
    """

    def __init__(self, config: SparrowConfig):
        super().__init__()
        self.norm_1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = GroupedQueryAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
        )
        self.norm_2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.moe = SparseMoELayer(
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_experts=config.n_experts,
            top_k=config.top_k,
            noise_mult=config.router_noise_mult,
            aux_loss_coef=config.aux_loss_coef,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Forward pass through Sparrow Transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            kv_cache: Past key/value states for autoregressive decoding.
            use_cache: Whether to return updated key/value cache.

        Returns:
            x: Block output tensor of shape (batch, seq_len, d_model).
            new_kv_cache: Updated KV cache tuple.
            aux_loss: Auxiliary load-balancing loss from the MoE layer.
        """
        # Attention sub-layer with residual connection
        normed_attn_in = self.norm_1(x)
        attn_out, new_kv_cache = self.attn(normed_attn_in, kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out

        # MoE FFN sub-layer with residual connection
        normed_moe_in = self.norm_2(x)
        moe_out, aux_loss = self.moe(normed_moe_in)
        x = x + moe_out

        return x, new_kv_cache, aux_loss


class SparrowMoE(nn.Module):
    """Sparrow Sparse Mixture of Experts (MoE) Language Model.

    A modern decoder-only architecture featuring:
    - RMSNorm
    - RoPE & Grouped-Query Attention (GQA)
    - SwiGLU Feed-Forward Networks
    - Noisy Top-2 Sparse MoE routing across 8 experts with auxiliary balancing loss

    Args:
        config: Model configuration specifying dimensions, layers, and experts.
    """

    def __init__(self, config: SparrowConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer decoder blocks
        self.layers = nn.ModuleList([
            SparrowBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization layer
        self.norm_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Output projection (lm_head)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie word embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize model weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights using scaled normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass of the Sparrow MoE Language Model.

        Args:
            input_ids: Tensor of token IDs of shape (batch, seq_len).
            targets: Optional ground-truth token targets of shape (batch, seq_len).
            past_kv_caches: Optional list of KV caches, one per layer.
            use_cache: Whether to construct and return new KV caches for generation.

        Returns:
            logits: Output prediction scores of shape (batch, seq_len, vocab_size).
            total_aux_loss: Accumulated load-balancing loss across all MoE layers.
            loss: Optional total cross-entropy loss + aux loss (if targets provided).
            new_kv_caches: Updated KV caches for all layers (if use_cache is True).
        """
        _, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)  # (batch, seq_len, d_model)

        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        new_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = [] if use_cache else None  # type: ignore

        # Iterate through each Sparrow Transformer block
        for i, layer in enumerate(self.layers):
            layer_cache = past_kv_caches[i] if past_kv_caches is not None else None
            x, new_cache, layer_aux_loss = layer(x, kv_cache=layer_cache, use_cache=use_cache)
            total_aux_loss = total_aux_loss + layer_aux_loss
            if use_cache and new_cache is not None:
                new_kv_caches.append(new_cache)

        # Final RMS normalization and lm_head projection
        x = self.norm_f(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        # Calculate cross entropy loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten logits and targets for cross entropy computation
            flat_logits = logits.view(-1, self.config.vocab_size)
            flat_targets = targets.view(-1)
            ce_loss = F.cross_entropy(flat_logits, flat_targets)
            loss = ce_loss + total_aux_loss

        return logits, total_aux_loss, loss, new_kv_caches
