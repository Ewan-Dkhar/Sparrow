"""Autoregressive text generation engine with KV cache and sampling methods for Sparrow.

Supports greedy decoding, temperature scaling, top-k, nucleus (top-p) sampling,
repetition penalties, and streaming callbacks for rich terminal output.
"""

from typing import Callable, Iterator, List, Optional
import torch
import torch.nn.functional as F

from src.data.tokenizer import SparrowTokenizer
from src.model.sparrow_moe import SparrowMoE


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[List[int]] = None,
) -> int:
    """Samples next token index from model logits.

    Args:
        logits: Unnormalized logits of shape (vocab_size,).
        temperature: Sampling randomness scale (<= 0.0 triggers greedy decoding).
        top_k: Limits sampling pool to top-k highest probability tokens.
        top_p: Nucleus sampling probability threshold (0.0 to 1.0).
        repetition_penalty: Multiplicative discount on previously generated tokens.
        generated_tokens: List of previously generated token IDs to apply penalty to.

    Returns:
        Next token integer ID.
    """
    logits = logits.clone()

    # Apply repetition penalty
    if repetition_penalty != 1.0 and generated_tokens:
        for prev_token in set(generated_tokens):
            if logits[prev_token] < 0:
                logits[prev_token] *= repetition_penalty
            else:
                logits[prev_token] /= repetition_penalty

    # Greedy decoding if temperature is effectively zero
    if temperature <= 0.001:
        return int(torch.argmax(logits).item())

    # Apply temperature
    logits = logits / temperature

    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    # Top-P (Nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right so that first token above threshold is retained
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


class SparrowGenerator:
    """Handles autoregressive generation using SparrowMoE with KV caching."""

    def __init__(
        self,
        model: SparrowMoE,
        tokenizer: SparrowTokenizer,
        device: torch.device | str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
        on_token_callback: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        """Streams generated tokens one by one as they are decoded.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k parameter.
            top_p: Top-p nucleus parameter.
            repetition_penalty: Multiplicative penalty for repeated tokens.
            stop_tokens: List of token IDs that immediately halt generation.
            on_token_callback: Optional callback invoked with each newly decoded token string.

        Yields:
            Decoded string fragment for each step.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors=True, device=self.device)
        curr_tokens = input_ids[0].tolist()
        generated: List[int] = []

        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_token_id]

        # Initial prompt evaluation with KV cache construction
        logits, _, _, past_kv_caches = self.model(
            input_ids,
            use_cache=True,
        )

        last_token_logits = logits[0, -1, :]
        next_tok = sample_next_token(
            logits=last_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=generated,
        )

        for _ in range(max_new_tokens):
            if next_tok in stop_tokens:
                break

            generated.append(next_tok)
            token_text = self.tokenizer.decode([next_tok])
            if on_token_callback:
                on_token_callback(token_text)
            yield token_text

            # Single-token step passing cached past KV states
            tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
            logits, _, _, past_kv_caches = self.model(
                tok_tensor,
                past_kv_caches=past_kv_caches,
                use_cache=True,
            )

            last_token_logits = logits[0, -1, :]
            next_tok = sample_next_token(
                logits=last_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated,
            )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generates full completion for prompt.

        Returns:
            Complete generated text completion string.
        """
        pieces = list(
            self.stream_generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        )
        return "".join(pieces)
