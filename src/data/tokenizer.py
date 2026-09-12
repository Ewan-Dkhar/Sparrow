"""Tiktoken BPE tokenizer wrapper for Sparrow.

Provides fast Byte-Pair Encoding (BPE) tokenization, decoding, and special token handling.
"""

from typing import List, Union
import tiktoken
import torch


class SparrowTokenizer:
    """Wrapper around OpenAI's tiktoken for fast BPE tokenization.

    Defaults to the GPT-2 encoding (50,257 tokens) which pairs with the default
    Sparrow vocab size, but can also wrap cl100k_base or custom encodings.

    Args:
        encoding_name: Name of tiktoken encoding (default: "gpt2").
    """

    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

        # Define special tokens
        self.eos_token = "<|endoftext|>"
        self.bos_token = "<|endoftext|>"
        self.pad_token = "<|endoftext|>"

        self.eos_token_id = self.enc.encode(self.eos_token, allowed_special={self.eos_token})[0]
        self.bos_token_id = self.eos_token_id
        self.pad_token_id = self.eos_token_id

    def encode(
        self,
        text: str,
        return_tensors: bool = False,
        device: torch.device | str | None = None,
    ) -> Union[List[int], torch.Tensor]:
        """Encodes string into a sequence of token IDs.

        Args:
            text: Input string to tokenize.
            return_tensors: Whether to return a PyTorch Tensor of shape (1, seq_len).
            device: Device to place tensor on if return_tensors is True.

        Returns:
            List of integer IDs or 2D PyTorch Tensor.
        """
        tokens = self.enc.encode(text, allowed_special="all")
        if return_tensors:
            return torch.tensor([tokens], dtype=torch.long, device=device)
        return tokens

    def decode(self, tokens: Union[List[int], torch.Tensor]) -> str:
        """Decodes token IDs back into string text.

        Args:
            tokens: List of token IDs or 1D/2D PyTorch Tensor.

        Returns:
            Decoded string text.
        """
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim == 2:
                tokens = tokens.squeeze(0)
            tokens = tokens.tolist()
        return self.enc.decode(tokens)
