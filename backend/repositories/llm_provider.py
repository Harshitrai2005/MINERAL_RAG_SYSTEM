"""
LLM Provider Interface
─────────────────────────────────────────────────────────────────────────────
Contract for any text-generation backend (Gemini, OpenAI, Claude, local
Ollama model, etc). RAGService depends only on this interface.

Swapping LLM providers later (e.g. Gemini -> Claude) means writing one new
class here — no changes to RAGService, routes, or prompts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class LLMResponse:
    answer: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


class LLMProvider(ABC):
    """Abstract interface every LLM backend must implement."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> LLMResponse:
        """Generate a complete response in one call."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> Iterator[str]:
        """Generate a response, yielding tokens/chunks as they arrive."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the LLM API is reachable and the key is valid."""
        raise NotImplementedError
