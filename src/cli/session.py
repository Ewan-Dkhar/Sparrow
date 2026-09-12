"""Chat session and context window management for the Sparrow CLI.

Tracks conversation history, applies prompt templates, and trims oldest turns
when context approaches the model's sequence length limit.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from src.data.tokenizer import SparrowTokenizer


@dataclass
class ChatMessage:
    """A single turn in the chat session."""
    role: str  # "system", "user", "assistant"
    content: str


class ChatSession:
    """Manages chat turns, prompt formatting, and context window trimming.

    Args:
        tokenizer: Tokenizer instance for measuring context token length.
        system_prompt: Default system instruction string.
        max_context_tokens: Maximum token length for the chat history (default: 1536).
    """

    def __init__(
        self,
        tokenizer: Optional[SparrowTokenizer] = None,
        system_prompt: str = "You are Sparrow, a sparse Mixture of Experts mini-LLM.",
        max_context_tokens: int = 1536,
    ):
        self.tokenizer = tokenizer or SparrowTokenizer()
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.messages: List[ChatMessage] = []

    def reset(self) -> None:
        """Clears conversation history."""
        self.messages.clear()

    def add_message(self, role: str, content: str) -> None:
        """Appends a message to conversation history."""
        self.messages.append(ChatMessage(role=role, content=content))
        self._trim_context()

    def format_prompt(self) -> str:
        """Formats the entire conversation history into an autoregressive prompt string."""
        formatted = [f"<|system|>\n{self.system_prompt}\n"]
        for msg in self.messages:
            if msg.role == "user":
                formatted.append(f"<|user|>\n{msg.content}\n")
            elif msg.role == "assistant":
                formatted.append(f"<|assistant|>\n{msg.content}\n")
        formatted.append("<|assistant|>\n")
        return "".join(formatted)

    def _trim_context(self) -> None:
        """Trims oldest user/assistant turns if token count exceeds max_context_tokens."""
        prompt = self.format_prompt()
        tokens = self.tokenizer.encode(prompt)
        while len(tokens) > self.max_context_tokens and len(self.messages) > 1:
            # Drop the oldest user-assistant pair or first turn
            self.messages.pop(0)
            prompt = self.format_prompt()
            tokens = self.tokenizer.encode(prompt)

    @property
    def turn_count(self) -> int:
        """Returns number of messages in current session."""
        return len(self.messages)
