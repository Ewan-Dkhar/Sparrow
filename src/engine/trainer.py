"""Mixed-precision training loop for Sparrow MoE with Rich terminal monitoring.

Optimized for 8GB VRAM (RTX 4060) utilizing bfloat16 / float16 AMP, gradient accumulation,
gradient clipping, and auxiliary load-balancing loss tracking.
"""

import math
import os
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from src.model.sparrow_moe import SparrowConfig, SparrowMoE


class SparrowTrainer:
    """Trainer executing mixed precision training on Sparrow MoE."""

    def __init__(
        self,
        model: SparrowMoE,
        dataloader: DataLoader,
        lr: float = 3.0e-4,
        min_lr: float = 3.0e-5,
        weight_decay: float = 0.1,
        warmup_steps: int = 200,
        max_steps: int = 5000,
        grad_accum_steps: int = 8,
        grad_clip: float = 1.0,
        precision: str = "bfloat16",
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        console: Optional[Console] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.console = console or Console()

        # Mixed precision setup
        self.precision_dtype = (
            torch.bfloat16 if precision == "bfloat16" and torch.cuda.is_bf16_supported()
            else torch.float16 if precision == "float16"
            else torch.float32
        )
        self.use_amp = self.device.type == "cuda" and self.precision_dtype != torch.float32
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.use_amp and self.precision_dtype == torch.float16)
        )

        # Separate weight decay parameters
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)

    def get_lr(self, step: int) -> float:
        """Computes learning rate with linear warmup and cosine decay."""
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        if step > self.max_steps:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.lr - self.min_lr)

    def train(self) -> None:
        """Runs training loop across dataset with Rich progress bar."""
        self.model.train()
        step = 0
        running_loss = 0.0
        running_aux_loss = 0.0
        data_iter = iter(self.dataloader)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("[green]loss: {task.fields[loss]:.4f}"),
            TextColumn("[magenta]aux: {task.fields[aux]:.4f}"),
            TextColumn("[yellow]vram: {task.fields[vram]}"),
            console=self.console,
        )

        with progress:
            task = progress.add_task(
                "Training Sparrow MoE",
                total=self.max_steps,
                loss=0.0,
                aux=0.0,
                vram="0MB",
            )

            start_time = time.time()
            total_tokens = 0

            while step < self.max_steps:
                self.optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                step_aux_loss = 0.0

                for _ in range(self.grad_accum_steps):
                    try:
                        x, y = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.dataloader)
                        x, y = next(data_iter)

                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    total_tokens += x.numel()

                    with torch.amp.autocast(
                        device_type=self.device.type,
                        dtype=self.precision_dtype,
                        enabled=self.use_amp,
                    ):
                        _, aux_loss, loss, _ = self.model(x, targets=y)
                        scaled_loss = loss / self.grad_accum_steps

                    self.scaler.scale(scaled_loss).backward()
                    step_loss += loss.item() / self.grad_accum_steps
                    step_aux_loss += aux_loss.item() / self.grad_accum_steps

                # Unscale and clip gradients
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # Optimizer step and lr adjustment
                curr_lr = self.get_lr(step)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = curr_lr

                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss = 0.9 * running_loss + 0.1 * step_loss if step > 0 else step_loss
                running_aux_loss = 0.9 * running_aux_loss + 0.1 * step_aux_loss if step > 0 else step_aux_loss

                # VRAM tracking
                if self.device.type == "cuda":
                    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    vram_str = f"{vram_mb:.0f}MB"
                else:
                    vram_str = "N/A"

                step += 1
                progress.update(
                    task,
                    advance=1,
                    loss=running_loss,
                    aux=running_aux_loss,
                    vram=vram_str,
                )

                if step % 500 == 0:
                    self.save_checkpoint(step)

        elapsed = time.time() - start_time
        self.console.print(
            f"[bold green]Training finished![/bold green] "
            f"Trained {self.max_steps} steps ({total_tokens / elapsed:.1f} tokens/s)."
        )

    def save_checkpoint(self, step: int) -> None:
        """Saves model weights and configuration."""
        ckpt_path = self.checkpoint_dir / f"sparrow_step_{step}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "config": self.model.config,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            ckpt_path,
        )
