#!/usr/bin/env python3
"""Sparrow: Modern Sparse Mixture of Experts (MoE) Mini-LLM CLI.

Entry point supporting interactive REPL chat, mixed-precision training,
and architecture inspection.
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch

# Ensure Sparrow root directory is on Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.cli.renderer import TerminalRenderer
from src.cli.repl import SparrowREPL
from src.cli.session import ChatSession
from src.data.dataset import SyntheticDemoDataset, TextChunkDataset
from src.data.tokenizer import SparrowTokenizer
from src.engine.generator import SparrowGenerator
from src.engine.trainer import SparrowTrainer
from src.model.sparrow_moe import SparrowConfig, SparrowMoE

# Register custom config class as safe global for PyTorch 2.6+ weights_only loading
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([SparrowConfig])


def load_config(config_path: str | Path) -> SparrowConfig:
    """Loads configuration from YAML file or defaults."""
    path = Path(config_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
            model_cfg = raw_cfg.get("model", {})
            return SparrowConfig.from_dict(model_cfg)
    return SparrowConfig()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparrow MoE Mini-LLM CLI")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["chat", "train", "info"],
        default="chat",
        help="Execution mode: 'chat' for REPL, 'train' for training loop, 'info' for model specs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_8gb.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Text file path for training (if None, synthetic data is used for demo)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override total training steps (e.g., 500)",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    renderer = TerminalRenderer()

    # Load and validate checkpoint if specified
    ckpt = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: '{args.checkpoint}'")

        renderer.print_system_message(f"Loading checkpoint from {args.checkpoint}...")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            # Fallback for legacy checkpoints with custom unallowlisted objects
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "config" in ckpt:
            loaded_cfg = ckpt["config"]
            if isinstance(loaded_cfg, dict):
                config = SparrowConfig.from_dict(loaded_cfg)
            elif isinstance(loaded_cfg, SparrowConfig):
                config = loaded_cfg

    if args.mode == "info":
        renderer.print_banner(config)
        total_p, active_p = config.count_parameters()
        renderer.console.print(
            f"[bold green]✓ Configuration verified![/bold green] "
            f"Total: {total_p / 1e6:.1f}M params | Active: {active_p / 1e6:.1f}M per token."
        )
        return

    # Initialize model with resolved configuration and weights
    model = SparrowMoE(config)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])

    # Target dtype (bfloat16 for RTX 4060)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    model.to(device=device, dtype=dtype)

    if args.mode == "train":
        renderer.print_banner(config)
        tokenizer = SparrowTokenizer()

        # Load training options from config
        raw_cfg = {}
        if Path(args.config).is_file():
            with open(args.config, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
        train_cfg = raw_cfg.get("training", {})
        hw_cfg = raw_cfg.get("hardware", {})

        batch_size = int(train_cfg.get("batch_size", 4))
        grad_accum = int(train_cfg.get("gradient_accumulation_steps", 8))
        lr = float(train_cfg.get("learning_rate", 3.0e-4))
        min_lr = float(train_cfg.get("min_learning_rate", 3.0e-5))
        max_steps = int(args.steps or train_cfg.get("max_steps", 5000))
        warmup_steps = int(train_cfg.get("warmup_steps", 200))
        save_interval = int(train_cfg.get("save_interval", 1000))
        ckpt_dir = train_cfg.get("checkpoint_dir", "checkpoints")
        ckpt_name = train_cfg.get("checkpoint_name", "sparrow_model.pt")
        precision = hw_cfg.get("precision", "bfloat16")

        if args.train_file:
            renderer.print_system_message(f"Ingesting training data from {args.train_file}...")
            dataset = TextChunkDataset(args.train_file, seq_len=config.max_seq_len // 2, tokenizer=tokenizer)
        else:
            renderer.print_system_message("No train file specified; using synthetic dataset for demonstration...")
            dataset = SyntheticDemoDataset(vocab_size=config.vocab_size, seq_len=512, size=200)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        trainer = SparrowTrainer(
            model=model,
            dataloader=dataloader,
            lr=lr,
            min_lr=min_lr,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            grad_accum_steps=grad_accum,
            save_interval=save_interval,
            checkpoint_dir=ckpt_dir,
            checkpoint_name=ckpt_name,
            precision=precision,
            device=str(device),
            console=renderer.console,
        )
        trainer.train()

    elif args.mode == "chat":
        tokenizer = SparrowTokenizer()
        generator = SparrowGenerator(model=model, tokenizer=tokenizer, device=device)
        session = ChatSession(tokenizer=tokenizer)
        repl = SparrowREPL(
            model=model,
            generator=generator,
            session=session,
            renderer=renderer,
            config=config,
        )
        repl.run()


if __name__ == "__main__":
    main()
