"""SwiGLU Feed-Forward Network (FFN) expert module for the Sparrow MoE architecture.

SwiGLU combines Swish (SiLU) gating with linear up/down projections:
    SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU expert network block.

    Args:
        d_model: Input and output embedding dimension.
        d_ff: Intermediate hidden dimension of the expert.
        bias: Whether to include bias terms in linear layers (default: False for modern LLMs).
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Gate and up projection layers
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        # Down projection back to model dimension
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SwiGLU expert.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model) or (num_tokens, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model) or (num_tokens, d_model).
        """
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)
