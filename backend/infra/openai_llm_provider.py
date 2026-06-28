"""
OpenAI LLM Provider (Fallback)
─────────────────────────────────────────────────────────────────────────────
Concrete LLMProvider implementation using OpenAI's chat.completions API.

This is the fallback provider when Groq is unavailable or the user prefers
GPT-4o. The interface is identical to GroqLLMProvider — swapping providers
means changing one env var (LLM_PROVIDER=openai) with zero code changes in
RAGService or routes.

MODELS (ordered by preference):
  gpt-4o-mini   — fast, cheap, excellent quality for RAG ($0.15/1M in tokens)
  gpt-4o        — best quality, higher cost
  gpt-3.5-turbo — legacy fallback, very cheap

REQUIRED ENV VARS:
  OPENAI_API_KEY  — from https://platform.openai.com/api-keys
  LLM_PROVIDER    — set to "openai" to activate this provider

HOW TO SWITCH:
  In .env:
      LLM_PROVIDER=openai
      OPENAI_API_KEY=sk-...

  render.yaml already has OPENAI_API_KEY as a sync: false secret slot.
"""

from __future__ import annotations

import time
from typing import Iterator

from repositories.llm_provider import LLMProvider, LLMResponse
from utils.logger import setup_logger

logger = setup_logger(__name__)

_PREFERRED_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo",
]


class OpenAILLMProvider(LLMProvider):
    """
    OpenAI GPT provider — drop-in replacement for GroqLLMProvider.
    Implements the same LLMProvider interface with identical method signatures.
    """

    def __init__(self, api_key: str):
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY is not set — OpenAI LLM calls will fail. "
                "Get a key at https://platform.openai.com/api-keys"
            )
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key or "placeholder")
            self._model_candidates = self._discover_models()
        except ImportError:
            logger.error(
                "openai package not installed. "
                "Add 'openai>=1.0.0' to requirements.txt and reinstall."
            )
            self._client = None
            self._model_candidates = _PREFERRED_MODELS

    def _discover_models(self) -> list[str]:
        """
        Try to list available models from OpenAI and filter to preferred list.
        Falls back to hard-coded list on any failure.
        """
        try:
            available = {m.id for m in self._client.models.list().data}
            ordered = [m for m in _PREFERRED_MODELS if m in available]
            if not ordered:
                return _PREFERRED_MODELS
            logger.info(f"OpenAI models available: {ordered}")
            return ordered
        except Exception as exc:
            logger.warning(f"OpenAI model discovery failed ({exc}); using defaults.")
            return _PREFERRED_MODELS

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> LLMResponse:
        """Generate a complete response with retry on rate limit."""
        if self._client is None:
            return LLMResponse(
                answer="OpenAI provider not available — openai package missing.",
                model="none",
                error="import_error",
            )

        for model_name in self._model_candidates:
            for attempt in range(3):
                try:
                    from openai import RateLimitError, APIStatusError, APITimeoutError

                    response = self._client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    answer = response.choices[0].message.content or "No response generated."
                    usage = response.usage
                    return LLMResponse(
                        answer=answer,
                        model=model_name,
                        input_tokens=usage.prompt_tokens if usage else 0,
                        output_tokens=usage.completion_tokens if usage else 0,
                    )

                except Exception as exc:
                    exc_name = type(exc).__name__
                    if "RateLimitError" in exc_name:
                        wait = 2 ** attempt
                        logger.warning(
                            f"OpenAI rate limit on '{model_name}' "
                            f"(attempt {attempt + 1}/3) — waiting {wait}s"
                        )
                        time.sleep(wait)
                        continue
                    if "APITimeoutError" in exc_name:
                        logger.warning(f"OpenAI timeout on '{model_name}' (attempt {attempt + 1}/3)")
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                            continue
                        break
                    logger.warning(f"OpenAI error on '{model_name}': {str(exc)[:120]}")
                    break

        return LLMResponse(
            answer=(
                "The OpenAI LLM service is temporarily unavailable. "
                "Please check your OPENAI_API_KEY and try again shortly."
            ),
            model="none",
            error="all_models_exhausted",
        )

    def stream(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> Iterator[str]:
        """Stream response tokens."""
        if self._client is None:
            yield "[OpenAI provider not available — openai package missing]"
            return

        for model_name in self._model_candidates:
            try:
                stream = self._client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                return

            except Exception as exc:
                logger.warning(f"OpenAI streaming error on '{model_name}': {str(exc)[:120]}")
                continue

        yield "[OpenAI LLM temporarily unavailable — please check your OPENAI_API_KEY]"

    def health_check(self) -> bool:
        return self._client is not None and len(self._model_candidates) > 0
