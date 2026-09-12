"""Noisy Top-2 Router and Sparse Mixture of Experts (MoE) layer for Sparrow.

Implements token routing across 8 independent SwiGLU experts with Gaussian noise
injection during training, top-2 gating, and auxiliary load-balancing loss calculation.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.expert import SwiGLU


class Top2Router(nn.Module):
    """Noisy Top-2 Router with auxiliary load-balancing loss.

    Args:
        d_model: Token embedding dimension.
        n_experts: Total number of experts (default: 8).
        noise_mult: Noise magnitude multiplier for training exploration (default: 1.0).
        aux_loss_coef: Weight multiplier for load balancing auxiliary loss (default: 0.01).
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        noise_mult: float = 1.0,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.noise_mult = noise_mult
        self.aux_loss_coef = aux_loss_coef

        # Gate projection for baseline routing logits
        self.w_gate = nn.Linear(d_model, n_experts, bias=False)
        # Learnable noise projection for tunable exploration variance
        self.w_noise = nn.Linear(d_model, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Routes tokens to top-2 experts and computes auxiliary load balancing loss.

        Args:
            x: Input token representations of shape (num_tokens, d_model).

        Returns:
            top2_weights: Normalized routing weights of shape (num_tokens, 2).
            top2_indices: Expert indices of shape (num_tokens, 2).
            aux_loss: Scalar tensor representing auxiliary load-balancing loss.
        """
        clean_logits = self.w_gate(x)  # (N, n_experts)

        if self.training and self.noise_mult > 0.0:
            # Add Gaussian noise scaled by softplus of learned noise projection
            raw_noise = torch.randn_like(clean_logits)
            noise_std = F.softplus(self.w_noise(x))
            noisy_logits = clean_logits + (raw_noise * noise_std * self.noise_mult)
            routing_logits = noisy_logits
        else:
            routing_logits = clean_logits

        # Compute full softmax distribution over all experts
        router_probs = F.softmax(routing_logits, dim=-1)  # (N, n_experts)

        # Select Top-2 experts
        top2_weights, top2_indices = torch.topk(router_probs, k=2, dim=-1)  # (N, 2)

        # Re-normalize Top-2 weights so they sum to 1.0 per token
        top2_weights = top2_weights / (
            top2_weights.sum(dim=-1, keepdim=True) + 1e-9
        )

        # --- Auxiliary Load-Balancing Loss ---
        # P_i: Average probability assigned to expert i across all tokens
        p_i = router_probs.mean(dim=0)  # (n_experts,)

        # f_i: Fraction of tokens assigned to expert i in top-2
        # Mask shape: (N, 2, n_experts) -> reduced over top_k dimension to (N, n_experts)
        mask = F.one_hot(top2_indices, num_classes=self.n_experts).any(dim=1).float()
        f_i = mask.mean(dim=0)  # (n_experts,)

        # Auxiliary loss penalizes correlation between dispatch frequency and probability
        aux_loss = self.aux_loss_coef * self.n_experts * torch.sum(f_i * p_i)

        return top2_weights, top2_indices, aux_loss


class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts layer containing Top-2 Router and 8 SwiGLU experts.

    Args:
        d_model: Input/output dimension.
        d_ff: Expert intermediate hidden dimension.
        n_experts: Total number of experts (default: 8).
        top_k: Number of routed experts per token (default: 2).
        noise_mult: Multiplier for router noise injection.
        aux_loss_coef: Load balancing loss coefficient.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        noise_mult: float = 1.0,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = Top2Router(
            d_model=d_model,
            n_experts=n_experts,
            noise_mult=noise_mult,
            aux_loss_coef=aux_loss_coef,
        )

        self.experts = nn.ModuleList([
            SwiGLU(d_model=d_model, d_ff=d_ff) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass dispatching tokens to top-2 experts and aggregating outputs.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            output: Combined tensor of shape (batch, seq_len, d_model).
            aux_loss: Auxiliary load-balancing loss scalar.
        """
        b, s, d = x.shape
        flat_x = x.reshape(-1, d)  # (N, d)

        # Obtain routing weights, expert assignments, and load balancing loss
        top2_weights, top2_indices, aux_loss = self.router(flat_x)  # (N, 2), (N, 2), scalar

        out = torch.zeros_like(flat_x)

        # Dispatch tokens to each of the 8 experts
        for expert_idx in range(self.n_experts):
            # Check where this expert was selected (either top-1 or top-2)
            # expert_mask has shape (N, 2)
            expert_mask = top2_indices == expert_idx
            token_indices, k_positions = torch.where(expert_mask)

            if token_indices.numel() == 0:
                continue

            # Extract assigned tokens and corresponding gating weights
            selected_tokens = flat_x[token_indices]
            expert_out = self.experts[expert_idx](selected_tokens)
            weights = top2_weights[token_indices, k_positions].unsqueeze(-1)

            # Accumulate weighted expert output
            out.index_add_(0, token_indices, expert_out * weights)

        return out.reshape(b, s, d), aux_loss
