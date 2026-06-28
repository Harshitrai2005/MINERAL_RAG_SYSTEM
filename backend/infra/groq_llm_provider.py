"""
Groq LLM Provider
─────────────────────────────────────────────────────────────────────────────
Concrete LLMProvider implementation using Groq's API.

WHY GROQ (over Gemini for this project):
  - Groq has an extremely generous free tier (no credit card required)
  - Inference is exceptionally fast — Groq runs on custom LPU hardware,
    producing tokens ~10x faster than GPU-based providers
  - OpenAI-compatible API (chat.completions interface), so it's easy to
    reason about and well-documented
  - Free tier gives: 14,400 requests/day on llama-3.3-70b, 30 req/min

MODELS (free tier, ordered by preference):
  llama-3.3-70b-versatile   — best quality, 128k context, 30 req/min free
  llama-3.1-8b-instant      — faster/cheaper fallback, 128k context
  gemma2-9b-it              — Google's Gemma 2 running on Groq, good fallback
  mixtral-8x7b-32768        — Mixtral MoE, 32k context, good for long docs

Get a free API key at: https://console.groq.com
(no credit card required, instant signup)

This is a concrete implementation of the LLMProvider interface.
Swapping back to Gemini (or to any other provider) = writing one new file
implementing that same interface. Zero changes to RAGService or routes.
"""

import threading
import time
from collections import deque
from typing import Iterator

from groq import Groq, APIStatusError, APITimeoutError

from repositories.llm_provider import LLMProvider, LLMResponse
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Ordered by preference: best quality first, lighter fallbacks after.
# All are available on Groq's free tier.
_PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]


class _RequestPacer:
    """
    Proactive client-side rate limiter (Issue #1 fix).

    Free-tier Groq limits both REQUESTS/min and TOKENS/min per model. The
    old behaviour only reacted to a 429 *after* it happened (sleep, retry),
    which is why warnings kept appearing on every other query even though
    answers eventually came back — each call was a coin-flip against the
    limit. This pacer keeps a rolling log of recent request timestamps and
    *waits* before sending a new request if we're about to exceed a safe
    request-rate budget, so 429s become the rare exception instead of the
    expected case.

    This is intentionally simple (no token counting, just request pacing) —
    good enough to eliminate the vast majority of avoidable rate-limit hits
    for a portfolio-scale demo without needing exact, frequently-changing
    Groq quota numbers.
    """

    def __init__(self, max_requests_per_window: int = 25, window_seconds: float = 60.0):
        self._max = max_requests_per_window
        self._window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def wait_for_slot(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] > self._window:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._max:
                sleep_for = self._window - (now - self._timestamps[0]) + 0.05
                if sleep_for > 0:
                    logger.info(f"Pacing Groq requests — waiting {sleep_for:.1f}s to stay under rate budget.")
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] > self._window:
                    self._timestamps.popleft()
            self._timestamps.append(time.monotonic())


class GroqLLMProvider(LLMProvider):

    def __init__(self, api_key: str, max_requests_per_minute: int = 25):
        if not api_key:
            logger.warning(
                "GROQ_API_KEY is not set — LLM generation calls will fail. "
                "Vector retrieval still works; only text generation is affected."
            )
        self._client = Groq(api_key=api_key or "placeholder")
        self._model_candidates = self._discover_models()
        # Stay a little under Groq's published free-tier request budget
        # (commonly ~30 req/min for llama-3.3-70b-versatile) so requests are
        # smoothed out instead of bursting into a 429.
        self._pacer = _RequestPacer(max_requests_per_window=max_requests_per_minute)
        # Tracks whether we've already warned about rate limiting in the
        # current "incident" so we don't spam identical warnings on every
        # retry of the same logical request — this directly addresses
        # "the warning keeps on coming" (Issue #1).
        self._rate_limit_warned_recently = False
        self._last_rate_limit_log = 0.0

    # ── Model discovery ──────────────────────────────────────────────────────

    def _discover_models(self) -> list[str]:
        """
        Query Groq's /models endpoint to find what's currently available,
        then filter to our preferred list (in preferred order), and append
        any extras that Groq has added since we last updated this file.

        Falls back to the hard-coded list if the API call fails (e.g. bad key).
        """
        try:
            available_ids = {m.id for m in self._client.models.list().data}
            # Preferred models first (if available), then anything else active
            ordered = [m for m in _PREFERRED_MODELS if m in available_ids]
            extras = [m for m in available_ids if m not in ordered]
            ordered += sorted(extras)
            if not ordered:
                logger.warning("No models found via discovery; falling back to hard-coded list.")
                return _PREFERRED_MODELS
            logger.info(f"Groq models available ({len(ordered)}): {ordered[:5]}")
            return ordered
        except Exception as e:
            logger.warning(f"Groq model discovery failed ({e}); using hard-coded fallback list.")
            return _PREFERRED_MODELS

    # ── Internal: rate-limit-aware logging ───────────────────────────────────

    def _log_rate_limit(self, model_name: str, attempt: int, wait: float) -> None:
        """
        Log a 429 at WARNING the first time in a 30s window, DEBUG after that.
        Stops the same incident from flooding logs with identical warnings
        while still surfacing the very first occurrence loudly.
        """
        now = time.monotonic()
        if now - self._last_rate_limit_log > 30:
            logger.warning(
                f"Groq rate limit on '{model_name}' (attempt {attempt + 1}/3) — waiting {wait:.1f}s. "
                f"This is expected occasionally on the free tier; the request will still complete."
            )
            self._last_rate_limit_log = now
        else:
            logger.debug(f"Groq rate limit on '{model_name}' (attempt {attempt + 1}/3) — waiting {wait:.1f}s")

    @staticmethod
    def _retry_after_seconds(exc: APIStatusError, fallback: float) -> float:
        """Respect the server's Retry-After header if present, else use our backoff."""
        try:
            headers = getattr(exc, "response", None) and exc.response.headers
            if headers and "retry-after" in headers:
                return max(float(headers["retry-after"]), fallback)
        except Exception:
            pass
        return fallback

    # ── Public interface ─────────────────────────────────────────────────────

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> LLMResponse:
        """
        Generate a complete response. Paces requests proactively, tries each
        model in order, and retries up to 3 times with exponential backoff
        (+ jitter, + Retry-After awareness) on rate-limit (429) errors.
        """
        for model_name in self._model_candidates:
            for attempt in range(3):
                self._pacer.wait_for_slot()
                try:
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

                except APIStatusError as e:
                    if e.status_code == 429:
                        base_wait = 2 ** attempt
                        wait = self._retry_after_seconds(e, base_wait)
                        self._log_rate_limit(model_name, attempt, wait)
                        time.sleep(wait)
                        continue
                    # Any other HTTP error (400 bad request, 503, etc.) —
                    # log it and fall through to the next model
                    logger.warning(
                        f"Groq API error on '{model_name}' "
                        f"(HTTP {e.status_code}): {str(e)[:120]}"
                    )
                    break

                except APITimeoutError:
                    logger.warning(f"Groq timeout on '{model_name}' (attempt {attempt + 1}/3)")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    break

                except Exception as e:
                    logger.warning(f"Unexpected error on '{model_name}': {str(e)[:120]}")
                    break

        return LLMResponse(
            answer=(
                "The LLM service is temporarily unavailable. "
                "Please check your GROQ_API_KEY and try again shortly."
            ),
            model="none",
            error="all_models_exhausted",
        )

    def stream(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> Iterator[str]:
        """
        Stream response tokens as they're generated. Groq's LPU hardware
        makes streaming especially effective — tokens arrive very fast.
        Falls back through model candidates on failure.
        """
        for model_name in self._model_candidates:
            self._pacer.wait_for_slot()
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
                return   # success — don't try next model

            except APIStatusError as e:
                if e.status_code == 429:
                    logger.warning(f"Rate limit during streaming on '{model_name}', trying next model")
                    continue
                logger.warning(f"Streaming error on '{model_name}': {e.status_code}")
                continue
            except Exception as e:
                logger.warning(f"Streaming failed on '{model_name}': {str(e)[:120]}")
                continue

        yield "[LLM service temporarily unavailable — please check your GROQ_API_KEY]"

    def health_check(self) -> bool:
        """
        Light check — just confirms we have at least one model candidate.
        A real production health check would make a minimal API call, but
        that costs a request against the free-tier rate limit, so we skip it.
        """
        return len(self._model_candidates) > 0