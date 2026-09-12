"""Sparrow Model Package.

Contains the modern sparse Mixture of Experts (MoE) mini-LLM architecture.
"""

from src.model.attention import GroupedQueryAttention
from src.model.embeddings import RotaryEmbedding, apply_rotary_emb
from src.model.expert import SwiGLU
from src.model.router import SparseMoELayer, Top2Router
from src.model.sparrow_moe import RMSNorm, SparrowBlock, SparrowConfig, SparrowMoE

__all__ = [
    "GroupedQueryAttention",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "SwiGLU",
    "Top2Router",
    "SparseMoELayer",
    "RMSNorm",
    "SparrowBlock",
    "SparrowConfig",
    "SparrowMoE",
]
