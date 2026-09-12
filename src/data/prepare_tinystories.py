#!/usr/bin/env python3
"""Dataset downloader and pre-tokenizer for TinyStories.

Downloads raw TinyStories subsets from Hugging Face and pre-tokenizes them
into binary tensor files for zero-overhead training ingestion.
"""

import argparse
from pathlib import Path
import sys
import urllib.request
import numpy as np
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.tokenizer import SparrowTokenizer

DATA_URLS = {
    # 21.5 MB - ~5.5M tokens, downloads in ~3s, ideal for rapid local training & fine-tuning
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    # 2.1 GB - ~500M tokens, full training set
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
}


def download_file(url: str, dest_path: Path, console: Console) -> None:
    """Downloads a remote file with a Rich progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            task = progress.add_task(f"Downloading {dest_path.name}", total=total_size)
            with open(dest_path, "wb") as f:
                chunk_size = 64 * 1024  # 64 KB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))


def tokenize_and_save(raw_path: Path, out_path: Path, console: Console, chunk_lines: int = 100_000) -> None:
    """Stream-tokenizes raw text file in memory-efficient chunks directly to a binary file.

    Prevents Out-Of-Memory (OOM) killer crashes on multi-gigabyte datasets by keeping
    memory footprint below 200 MB regardless of file size.
    """
    console.print(f"[bold yellow]Tokenizing[/bold yellow] {raw_path.name} in streaming chunks...")
    tokenizer = SparrowTokenizer()

    total_bytes = raw_path.stat().st_size
    temp_bin = out_path.with_suffix(".bin.tmp")
    total_tokens = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]{task.fields[tokens]} tokens"),
        console=console,
    )

    with open(raw_path, "r", encoding="utf-8") as f_in, open(temp_bin, "wb") as f_out:
        with progress:
            task = progress.add_task("Tokenizing", total=total_bytes, tokens="0")
            lines_buffer = []
            bytes_read = 0

            for line in f_in:
                lines_buffer.append(line)
                bytes_read += len(line.encode("utf-8"))

                if len(lines_buffer) >= chunk_lines:
                    text_chunk = "".join(lines_buffer)
                    tokens = tokenizer.encode(text_chunk)
                    token_arr = np.array(tokens, dtype=np.int32)
                    f_out.write(token_arr.tobytes())
                    total_tokens += len(tokens)

                    lines_buffer.clear()
                    progress.update(task, completed=bytes_read, tokens=f"{total_tokens:,}")

            if lines_buffer:
                text_chunk = "".join(lines_buffer)
                tokens = tokenizer.encode(text_chunk)
                token_arr = np.array(tokens, dtype=np.int32)
                f_out.write(token_arr.tobytes())
                total_tokens += len(tokens)
                lines_buffer.clear()
                progress.update(task, completed=total_bytes, tokens=f"{total_tokens:,}")

    if temp_bin.exists():
        temp_bin.replace(out_path)

    mb = out_path.stat().st_size / (1024 * 1024)
    console.print(
        f"[bold green]✓ Pre-tokenization complete![/bold green] "
        f"Encoded {total_tokens:,} tokens ({mb:.1f} MB saved to {out_path.name})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare TinyStories dataset for Sparrow")
    parser.add_argument(
        "--split",
        type=str,
        choices=["valid", "train"],
        default="valid",
        help="Dataset split: 'valid' (21.5 MB, ~5.5M tokens, recommended to start) or 'train' (2.1 GB, full)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Output directory to save dataset files",
    )

    args = parser.parse_args()
    console = Console()
    out_dir = ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    url = DATA_URLS[args.split]
    raw_file = out_dir / f"tinystories_{args.split}.txt"
    bin_file = out_dir / f"tinystories_{args.split}.bin"
    pt_file = out_dir / f"tinystories_{args.split}.pt"

    console.print(f"[bold magenta]Sparrow TinyStories Ingestion[/bold magenta] [dim](split: {args.split})[/dim]")

    if not raw_file.exists():
        console.print(f"Fetching from: {url}")
        download_file(url, raw_file, console)
    else:
        console.print(f"[green]✓ Raw text already exists at {raw_file}[/green]")

    target_file = bin_file
    if not target_file.exists() and not pt_file.exists():
        tokenize_and_save(raw_file, target_file, console)
    else:
        existing = target_file if target_file.exists() else pt_file
        console.print(f"[green]✓ Pre-tokenized dataset already exists at {existing}[/green]")
        target_file = existing

    console.print()
    console.print("[bold green]Ready to train![/bold green] Run the following command:")
    console.print(f"[cyan]uv run python run.py --mode train --train-file {target_file}[/cyan]")


if __name__ == "__main__":
    main()
