# 🐦 Sparrow: Sparse Mixture of Experts (MoE) Mini-LLM

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6%2B%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/target-NVIDIA%20RTX%204060%20(8GB)-76b900.svg)](https://www.nvidia.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

```
   ____                                   
  / __/ ___  ___ _  ____  ____  ___  __    __
 _\ \  / _ \/ _ `/ / __/ / __/ / _ \ | |/|/ /
/___/ / .__/\_,_/ /_/   /_/    \___/ |__,__/ 
     /_/                                      
       Sparse Mixture of Experts (MoE) Mini-LLM
```

**Sparrow** is a modern, sparse Mixture of Experts (MoE) decoder-only mini-LLM written from scratch in pure PyTorch. It features a complete ~133M parameter architecture with **~57M active parameters per token**, designed to comfortably train, fine-tune, and run on consumer hardware (such as an NVIDIA GeForce RTX 4060 with 8GB VRAM) via a rich, interactive terminal interface.

---

## 🚀 Key Architectural Features

* **Sparse Mixture of Experts (MoE)**: Decouples model parameter capacity from compute cost per token. Distributes tokens across 8 independent SwiGLU expert networks using a **Noisy Top-2 Router** with auxiliary load-balancing loss ($\mathcal{L}_{\text{aux}}$) to prevent routing collapse.
* **RMSNorm (Root Mean Square Layer Normalization)**: Replaces traditional LayerNorm with zero mean-centering overhead, delivering up to 50% faster GPU kernel execution and improved stability.
* **Rotary Position Embeddings (RoPE)**: Direct 2D coordinate rotation applied to Query and Key states at every attention layer for relative position invariance.
* **Grouped-Query Attention (GQA)**: Configured with 8 Query heads and 4 Key/Value heads, halving KV cache memory footprint during autoregressive generation.
* **FlashAttention-2 / SDPA Integration**: Employs PyTorch's native `scaled_dot_product_attention` for fast, memory-efficient attention in `bfloat16`.
* **Rich Terminal REPL**: Marries `prompt_toolkit` (input history, keybindings, slash commands) with `rich` for live streaming Markdown generation and GPU status telemetry.
* **Zero-RAM Memory-Mapped Ingestion**: Streaming chunked tokenization pipeline capable of pre-tokenizing 500M+ tokens without RAM spikes, loaded into training via `np.memmap` in 0.0001 seconds.

---

## 📊 Model Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **132.8 Million** | Full network capacity across all 8 experts |
| **Active Parameters** | **57.3 Million** | Parameters evaluated per token (Top-2 experts) |
| **Layers** | 8 | Transformer decoder blocks |
| **Hidden Dimension ($d_{\text{model}}$)** | 512 | Token representation vector width |
| **Attention Heads** | 8 Query heads / 4 KV heads | Grouped-Query Attention (group size = 2) |
| **Head Dimension** | 64 | Dimension per attention head |
| **FFN / Expert Dimension ($d_{\text{ff}}$)** | 1024 | Intermediate dimension of each SwiGLU expert |
| **Experts ($E$)** | 8 | Total independent FFN expert networks per layer |
| **Active Experts ($k$)** | 2 | Top-$k$ routed experts per token |
| **Context Window** | 2048 tokens | Maximum sequence length with RoPE |
| **Vocabulary Size** | 50,257 | Byte-Pair Encoding (BPE via `tiktoken`) |
| **Target Precision** | `bfloat16` / `float16` | Native mixed-precision targeting 8GB VRAM |

---

## 📁 Repository Structure

```
sparrow/
├── pyproject.toml             # uv dependencies and build specification
├── configs/
│   └── default_8gb.yaml       # Hyperparams tuned for RTX 4060 (8GB VRAM)
├── src/
│   ├── cli/
│   │   ├── repl.py            # prompt_toolkit REPL & slash command handling
│   │   ├── renderer.py        # Rich terminal UI, ASCII banner & live Markdown streaming
│   │   └── session.py         # Chat conversation history & context window manager
│   ├── model/
│   │   ├── attention.py       # Grouped-Query Attention (GQA) & FlashAttention-2
│   │   ├── embeddings.py      # Rotary Position Embeddings (RoPE) math & caching
│   │   ├── expert.py          # SwiGLU Feed-Forward Network
│   │   ├── router.py          # Noisy Top-2 Router & auxiliary load-balancing loss
│   │   └── sparrow_moe.py     # RMSNorm, SparrowBlock, and SparrowMoE Model
│   ├── engine/
│   │   ├── trainer.py         # Mixed-precision (BF16) training loop with live Rich telemetry
│   │   └── generator.py       # Autoregressive generation with KV caching & Top-P / Top-K
│   └── data/
│       ├── dataset.py         # TextChunkDataset (supports raw text, .pt, and .bin memmap)
│       ├── tokenizer.py       # Tiktoken GPT-2 BPE wrapper
│       └── prepare_tinystories.py # High-speed streaming downloader & pre-tokenizer
├── tests/
│   └── test_checkpoint.py     # Automated regression tests for safe serialization
├── docs/
│   └── ARCHITECTURE_FROM_SCRATCH.md # Complete educational guide and math derivations
└── run.py                     # Main CLI entry point (chat, train, info modes)
```

---

## 🛠️ Installation & Setup

Sparrow strictly uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

### 1. Prerequisites
* Linux (Arch Linux, Ubuntu, Debian, etc.)
* NVIDIA GPU with CUDA drivers (e.g. RTX 4060, RTX 3060, or better)
* `uv` package manager installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/Ewan-Dkhar/Sparrow.git
cd Sparrow
uv sync
```

Verify that PyTorch recognizes your CUDA GPU:
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), 'BF16 Supported:', torch.cuda.is_bf16_supported())"
```

---

## ⚡ Quick Start

### 1. View Architecture & Parameter Budget
Inspect the model configuration and parameter counts:
```bash
uv run python run.py --mode info
```

### 2. Interactive Terminal Chat
Launch the rich interactive REPL with live streaming Markdown:
```bash
uv run python run.py --mode chat
```

#### Available Slash Commands inside the REPL:
* `/help` — Display command cheat sheet
* `/clear` — Reset conversation history and clear KV cache context
* `/gpu` — Display live VRAM allocation, reserved memory, and peak usage
* `/params` — Print detailed parameter breakdown (total vs. active per token)
* `/temp <value>` — Adjust generation temperature dynamically (e.g., `/temp 0.8`)
* `/exit` — Exit the REPL cleanly

---

## 📚 Training on TinyStories

### 1. Download & Pre-Tokenize TinyStories
We provide an ingestion script that downloads TinyStories directly from Hugging Face and pre-tokenizes the text in memory-efficient streaming chunks:

```bash
# Quick start (~5.5M tokens, downloads and prepares in ~10 seconds)
uv run python src/data/prepare_tinystories.py --split valid

# Full training set (~548M tokens, downloads ~2.1 GB, prepares in ~1.5 minutes)
uv run python src/data/prepare_tinystories.py --split train
```

This creates:
* `data/tinystories_valid.pt` (Pre-tokenized validation tensor)
* `data/tinystories_train.bin` (Binary memory-mapped integer array)

### 2. Start Training
Launch the mixed-precision training loop:

```bash
# Train for 1,000 steps on validation split (~20 minutes)
uv run python run.py --mode train --train-file data/tinystories_valid.pt --steps 1000

# Full 1-Epoch training on 548M token dataset (~15,000 steps, ~5.5 hours on RTX 4060)
uv run python run.py --mode train --train-file data/tinystories_train.bin --steps 15000
```

#### Training Performance Metrics (RTX 4060 8GB):
* **Throughput**: **~25,000 tokens/sec** (~1.3 seconds per step)
* **VRAM Usage**: **~4.4 GB** (rock solid, leaving >3.5 GB headroom)
* **Tokens per Step**: 32,768 ($4 \text{ batch} \times 8 \text{ accum} \times 1024 \text{ context}$)

### 3. Chat with Your Trained Checkpoint
Checkpoints are automatically saved to `checkpoints/` (e.g. `sparrow_step_1000.pt`, `sparrow_step_5000.pt`). Test generation at any time:

```bash
uv run python run.py --mode chat --checkpoint checkpoints/sparrow_step_5000.pt
```

---

## 🧪 Testing

Run the automated test suite to verify checkpoint serialization and memory-mapped loading:

```bash
uv run python tests/test_checkpoint.py
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
