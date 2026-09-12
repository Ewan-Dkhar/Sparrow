"""PyTorch Dataset for text ingestion (e.g. TinyStories or plain text corpora).

Tokenizes raw text into fixed-length autoregressive chunks (x, y) where y is shifted by one token.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.tokenizer import SparrowTokenizer


class TextChunkDataset(Dataset):
    """Autoregressive text dataset for language model training.

    Splits tokenized sequences into overlapping or non-overlapping blocks of
    size (seq_len + 1) to yield (input_ids, targets).

    Args:
        data: Either a file path string / Path, a raw string, or a pre-tokenized 1D tensor.
        seq_len: Sequence length for each training example.
        tokenizer: SparrowTokenizer instance (if data is text).
    """

    def __init__(
        self,
        data: Union[str, Path, torch.Tensor, List[int]],
        seq_len: int = 1024,
        tokenizer: Optional[SparrowTokenizer] = None,
    ):
        super().__init__()
        self.seq_len = seq_len

        if isinstance(data, torch.Tensor):
            self.tokens = data.long()
        elif isinstance(data, (list, tuple)):
            self.tokens = torch.tensor(data, dtype=torch.long)
        elif isinstance(data, (str, Path)):
            path = Path(data)
            if path.is_file():
                if path.suffix == ".bin":
                    # Zero-overhead memory-mapped binary file for massive datasets
                    self.tokens = np.memmap(path, dtype=np.int32, mode="r")
                elif path.suffix == ".pt":
                    loaded = torch.load(path, weights_only=True)
                    self.tokens = loaded.long() if isinstance(loaded, torch.Tensor) else torch.tensor(loaded, dtype=torch.long)
                else:
                    text = path.read_text(encoding="utf-8")
                    if tokenizer is None:
                        tokenizer = SparrowTokenizer()
                    token_ids = tokenizer.encode(text)
                    self.tokens = torch.tensor(token_ids, dtype=torch.long)
            else:
                text = str(data)
                if tokenizer is None:
                    tokenizer = SparrowTokenizer()
                token_ids = tokenizer.encode(text)
                self.tokens = torch.tensor(token_ids, dtype=torch.long)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Ensure tokens are sufficiently long for at least one chunk
        if len(self.tokens) < self.seq_len + 1:
            # Pad or repeat to satisfy minimum sequence length
            repeat_count = (self.seq_len + 2) // len(self.tokens) + 1
            if isinstance(self.tokens, torch.Tensor):
                self.tokens = self.tokens.repeat(repeat_count)
            else:
                self.tokens = np.tile(self.tokens, repeat_count)

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        chunk = self.tokens[start_idx:end_idx]

        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk.astype(np.int64))
        elif not isinstance(chunk, torch.Tensor):
            chunk = torch.tensor(chunk, dtype=torch.long)
        else:
            chunk = chunk.long()

        input_ids = chunk[:-1]
        targets = chunk[1:]
        return input_ids, targets


class SyntheticDemoDataset(Dataset):
    """Synthetic dataset generating random tokens for benchmarking and pipeline testing."""

    def __init__(self, vocab_size: int = 50257, seq_len: int = 512, size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        return tokens[:-1], tokens[1:]
