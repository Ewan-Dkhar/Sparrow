#!/usr/bin/env python3
"""Dataset builder and tokenizer for Blended Multi-Domain Continued Pretraining.

Streams and tokenizes:
1. Educational / Encyclopedic text (Simple English Wikipedia) - 40%
2. Python Code (flytech/python-codes-25k & alpaca code instructions) - 40%
3. Narrative Replay (TinyStories from local .bin) - 20%

Interleaves them into homogeneous, memory-mapped binary files (.bin) for
zero-overhead training ingestion while preventing catastrophic forgetting.
"""

import argparse
from pathlib import Path
import random
import sys
from typing import Iterator, Optional
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.tokenizer import SparrowTokenizer


def iter_wiki_tokens(tokenizer: SparrowTokenizer, split: str = "train") -> Iterator[int]:
    """Streams and tokenizes Simple English Wikipedia articles."""
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split=split, streaming=True)
    for sample in ds:
        title = sample.get("title", "").strip()
        text = sample.get("text", "").strip()
        if not text:
            continue
        full_text = f"# {title}\n\n{text}\n<|endoftext|>\n"
        token_ids = tokenizer.encode(full_text)
        yield from token_ids


def iter_code_tokens(tokenizer: SparrowTokenizer) -> Iterator[int]:
    """Streams and tokenizes Python code from multiple coding datasets."""
    from datasets import load_dataset

    # 1. Flytech Python Codes 25K
    try:
        ds_fly = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
        for sample in ds_fly:
            text = sample.get("text", "")
            if not text:
                instruction = sample.get("instruction", "").strip()
                input_ctx = sample.get("input", "").strip()
                output = sample.get("output", "").strip()
                if input_ctx:
                    text = f"# Task: {instruction}\n# Context: {input_ctx}\n\n{output}"
                else:
                    text = f"# Task: {instruction}\n\n{output}"
            full_text = f"{text.strip()}\n<|endoftext|>\n"
            yield from tokenizer.encode(full_text)
    except Exception as e:
        print(f"[Warning] Failed streaming flytech/python-codes-25k: {e}")

    # 2. Alpaca Python Instructions 18K
    try:
        ds_alpaca = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
        for sample in ds_alpaca:
            instruction = sample.get("instruction", "").strip()
            input_ctx = sample.get("input", "").strip()
            output = sample.get("output", "").strip()
            if not output:
                continue
            if input_ctx:
                text = f"# Problem: {instruction}\n# Context:\n{input_ctx}\n\n# Solution:\n{output}"
            else:
                text = f"# Problem: {instruction}\n\n# Solution:\n{output}"
            full_text = f"{text}\n<|endoftext|>\n"
            yield from tokenizer.encode(full_text)
    except Exception as e:
        print(f"[Warning] Failed streaming python_code_instructions_18k_alpaca: {e}")


def iter_stories_tokens(stories_path: Path) -> Iterator[int]:
    """Streams replay tokens directly from local tokenized TinyStories .bin file."""
    if not stories_path.is_file():
        raise FileNotFoundError(
            f"TinyStories replay file '{stories_path}' not found. "
            "Please run src/data/prepare_tinystories.py first."
        )
    memmap = np.memmap(stories_path, dtype=np.int32, mode="r")
    num_tokens = len(memmap)
    idx = 0
    # Yield continuously with looping
    while True:
        chunk_size = min(65536, num_tokens - idx)
        if chunk_size <= 0:
            idx = 0
            continue
        chunk = memmap[idx : idx + chunk_size]
        yield from chunk.tolist()
        idx += chunk_size


def build_blended_dataset(
    out_path: Path,
    total_tokens: int,
    stories_bin_path: Path,
    wiki_ratio: float = 0.40,
    code_ratio: float = 0.40,
    stories_ratio: float = 0.20,
    block_tokens: int = 4096,
    console: Optional[Console] = None,
) -> None:
    """Builds an interleaved multi-domain binary dataset."""
    console = console or Console()
    tokenizer = SparrowTokenizer()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(".bin.tmp")

    console.print(
        f"[bold green]Starting Data Blending[/bold green] -> [cyan]{out_path.name}[/cyan]\n"
        f"Target Tokens: [bold]{total_tokens:,}[/bold] | "
        f"Mix: [blue]Wiki: {wiki_ratio*100:.0f}%[/blue] │ "
        f"[magenta]Code: {code_ratio*100:.0f}%[/magenta] │ "
        f"[yellow]Stories Replay: {stories_ratio*100:.0f}%[/yellow]"
    )

    wiki_gen = iter_wiki_tokens(tokenizer)
    code_gen = iter_code_tokens(tokenizer)
    stories_gen = iter_stories_tokens(stories_bin_path)

    wiki_count = 0
    code_count = 0
    stories_count = 0
    written_count = 0

    # Number of block tokens proportional to ratios
    wiki_block = int(block_tokens * (wiki_ratio / (wiki_ratio + code_ratio + stories_ratio)))
    code_block = int(block_tokens * (code_ratio / (wiki_ratio + code_ratio + stories_ratio)))
    stories_block = block_tokens - wiki_block - code_block

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]{task.fields[tok]} tok"),
        TextColumn("[blue]W:{task.fields[w]}"),
        TextColumn("[magenta]C:{task.fields[c]}"),
        TextColumn("[yellow]S:{task.fields[s]}"),
        console=console,
    )

    write_buffer = []
    flush_size = 65_536  # Flush to disk in 64k token blocks

    with progress:
        task = progress.add_task(
            "Building Blended Dataset",
            total=total_tokens,
            tok="0",
            w="0",
            c="0",
            s="0",
        )

        with open(temp_path, "wb") as f_out:
            while written_count < total_tokens:
                rem = total_tokens - written_count

                # 1. Pull Wikipedia block
                w_target = min(wiki_block, rem)
                w_tokens = []
                for _ in range(w_target):
                    try:
                        w_tokens.append(next(wiki_gen))
                    except StopIteration:
                        wiki_gen = iter_wiki_tokens(tokenizer)
                        w_tokens.append(next(wiki_gen))
                wiki_count += len(w_tokens)
                write_buffer.extend(w_tokens)
                written_count += len(w_tokens)
                rem = total_tokens - written_count

                # 2. Pull Python Code block
                if rem > 0:
                    c_target = min(code_block, rem)
                    c_tokens = []
                    for _ in range(c_target):
                        try:
                            c_tokens.append(next(code_gen))
                        except StopIteration:
                            code_gen = iter_code_tokens(tokenizer)
                            c_tokens.append(next(code_gen))
                    code_count += len(c_tokens)
                    write_buffer.extend(c_tokens)
                    written_count += len(c_tokens)
                    rem = total_tokens - written_count

                # 3. Pull TinyStories replay block
                if rem > 0:
                    s_target = min(stories_block, rem)
                    s_tokens = []
                    for _ in range(s_target):
                        s_tokens.append(next(stories_gen))
                    stories_count += len(s_tokens)
                    write_buffer.extend(s_tokens)
                    written_count += len(s_tokens)

                # Flush to disk if buffer reaches flush_size
                if len(write_buffer) >= flush_size:
                    np.array(write_buffer, dtype=np.int32).tofile(f_out)
                    write_buffer.clear()

                progress.update(
                    task,
                    completed=min(written_count, total_tokens),
                    tok=f"{written_count:,}",
                    w=f"{wiki_count // 1000}k",
                    c=f"{code_count // 1000}k",
                    s=f"{stories_count // 1000}k",
                )

            # Flush any remaining tokens in buffer
            if write_buffer:
                np.array(write_buffer, dtype=np.int32).tofile(f_out)
                write_buffer.clear()

    temp_path.replace(out_path)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    console.print(
        f"[bold green]✓ Dataset successfully generated![/bold green]\n"
        f"File: [cyan]{out_path}[/cyan] ({file_size_mb:.1f} MB, {written_count:,} tokens)\n"
        f"Wiki: {wiki_count:,} | Code: {code_count:,} | Stories Replay: {stories_count:,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and tokenize blended Wikipedia + Python Code + TinyStories dataset for Sparrow."
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default="data/blended_train.bin",
        help="Path for blended training binary file",
    )
    parser.add_argument(
        "--output-val",
        type=str,
        default="data/blended_val.bin",
        help="Path for blended validation binary file",
    )
    parser.add_argument(
        "--train-tokens",
        type=int,
        default=50_000_000,
        help="Number of tokens for blended training dataset (e.g. 50,000,000)",
    )
    parser.add_argument(
        "--val-tokens",
        type=int,
        default=2_000_000,
        help="Number of tokens for blended validation dataset (e.g. 2,000,000)",
    )
    parser.add_argument(
        "--stories-bin",
        type=str,
        default="data/tinystories_train.bin",
        help="Path to pre-tokenized TinyStories binary file for replay buffer",
    )
    parser.add_argument(
        "--wiki-ratio",
        type=float,
        default=0.40,
        help="Proportion of Wikipedia text (default 0.40)",
    )
    parser.add_argument(
        "--code-ratio",
        type=float,
        default=0.40,
        help="Proportion of Python code (default 0.40)",
    )
    parser.add_argument(
        "--stories-ratio",
        type=float,
        default=0.20,
        help="Proportion of TinyStories replay (default 0.20)",
    )

    args = parser.parse_args()
    console = Console()
    stories_path = Path(args.stories_bin)

    if not stories_path.is_file():
        console.print(f"[bold red]Error:[/bold red] TinyStories file not found at '{stories_path}'.")
        sys.exit(1)

    # 1. Build validation set
    if args.val_tokens > 0:
        val_path = Path(args.output_val)
        console.print("[bold yellow]1/2: Generating Validation Split...[/bold yellow]")
        build_blended_dataset(
            out_path=val_path,
            total_tokens=args.val_tokens,
            stories_bin_path=stories_path,
            wiki_ratio=args.wiki_ratio,
            code_ratio=args.code_ratio,
            stories_ratio=args.stories_ratio,
            console=console,
        )

    # 2. Build training set
    train_path = Path(args.output_train)
    console.print("\n[bold yellow]2/2: Generating Training Split...[/bold yellow]")
    build_blended_dataset(
        out_path=train_path,
        total_tokens=args.train_tokens,
        stories_bin_path=stories_path,
        wiki_ratio=args.wiki_ratio,
        code_ratio=args.code_ratio,
        stories_ratio=args.stories_ratio,
        console=console,
    )


if __name__ == "__main__":
    main()
