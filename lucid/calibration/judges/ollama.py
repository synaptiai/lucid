"""Ollama-hosted judge using the same Spiral-Bench v1 rubric.

Calls ``ollama.AsyncClient().chat(...)`` with structured-output JSON
schema = ``SpiralBenchScore.model_json_schema()``. Respects the chunk
size used by Module A (default 10-turn windows). Falls back gracefully:
if the Ollama daemon is unreachable or the model isn't available, the
judge returns an empty list of LabeledTurns and logs a warning — the
calibration run continues without this judge rather than aborting.

**Why Ollama matters for Phase 6B:** it gives us 3 additional raters
(Kimi K2.6, Gemma 4 31B, GLM 5.1 via cloud models) at near-zero API
cost, raising the total to 7 raters. With more raters, Krippendorff α
and Gwet AC1 become better-powered; disagreement audit becomes more
informative (cells where 6 raters agree and one dissents are clearer
signal than 3-vs-1).

**Concurrency + retry:** per-window `asyncio.Semaphore(3)` by default
(local Ollama usually can't sustain more than a few concurrent requests
on consumer hardware). Tenacity retries on schema-validation failures
and transient errors up to 3 times per chunk before marking that chunk
as "judge saw nothing" (empty LabeledTurn).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from lucid.calibration.data import LabeledTurn
from lucid.modules.base import ModuleCorpus
from lucid.modules.module_a_spiralbench import (
    CHUNK_SIZE,
    PROMPT_VERSION,
    SpiralBenchScore,
    _iter_windows,
    _render_window,
    _score_to_findings,
)
from lucid.prompts import load_prompt

if TYPE_CHECKING:
    from ollama import AsyncClient

__all__ = ["OllamaJudge", "sanitize_model_name"]


log = logging.getLogger(__name__)


_INVALID_RATER_CHARS = re.compile(r"[^a-zA-Z0-9]+")


def sanitize_model_name(model: str) -> str:
    """Map an Ollama model id to a short rater-name suffix.

    Examples::

        kimi-k2.6:cloud     → kimi_k2_6
        gemma4:31b-cloud    → gemma4_31b
        glm-5.1:cloud       → glm_5_1
    """
    # strip ``:cloud`` / ``:latest`` etc. suffix
    base = model.split(":", 1)[0]
    cleaned = _INVALID_RATER_CHARS.sub("_", base).strip("_")
    return cleaned.lower() or "unknown"


class OllamaJudge:
    """Ollama-backed Judge using the Spiral-Bench v1 rubric."""

    def __init__(
        self,
        model: str,
        *,
        rater_name: str | None = None,
        chunk_size: int = CHUNK_SIZE,
        host: str | None = None,
        max_concurrency: int = 3,
        client: AsyncClient | None = None,
        max_attempts: int = 3,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        # Lazy import so the test suite doesn't need ollama installed to
        # collect (tests patch the client anyway).
        from ollama import AsyncClient as _AsyncClient

        self._client: AsyncClient = client or _AsyncClient(host=host)
        self._model = model
        self._chunk_size = chunk_size
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_attempts = max_attempts
        self._prompt = load_prompt("a", PROMPT_VERSION)
        self.rater_name = rater_name or f"ollama_{sanitize_model_name(model)}"

    async def run(self, corpus: ModuleCorpus) -> list[LabeledTurn]:
        """Score the corpus; return one LabeledTurn per assistant turn.

        If **every** window fails (daemon unreachable, model missing),
        returns ``[]`` instead of all-empty LabeledTurns — otherwise the
        judge would look identical to "this rater saw no behaviors on
        any turn" and pollute downstream IAA. An empty list is the
        signal the CLI uses to drop this judge from the rater pool.
        """
        # Lazy to avoid circular package import at top-level init time.
        from lucid.calibration.judges import findings_to_labeled_turns

        windows = [
            w
            for conv_id in corpus.conversations
            for w in _iter_windows(
                conv_id,
                corpus.turns_by_conversation.get(conv_id, ()),
                chunk=self._chunk_size,
            )
        ]

        from datetime import UTC, datetime

        from lucid.schemas import Finding

        detected_at = datetime.now(UTC)
        findings: list[Finding] = []
        successful_calls = 0
        failed_calls = 0

        async def _process(window: object) -> None:
            nonlocal successful_calls, failed_calls
            async with self._semaphore:
                score = await self._score_window_safe(window)
            if score is None:
                failed_calls += 1
                return
            successful_calls += 1
            try:
                batch = _score_to_findings(
                    score,
                    window,  # type: ignore[arg-type]
                    audit_run_id=corpus.audit_run_id,
                    prompt_hash=self._prompt.body_hash,
                    detected_at=detected_at,
                )
                findings.extend(batch)
            except Exception as err:
                log.warning(
                    "ollama[%s]: findings-build failed for window: %s",
                    self.rater_name,
                    err,
                )

        if not windows:
            return findings_to_labeled_turns(
                findings, corpus, rater_name=self.rater_name
            )

        await asyncio.gather(*(_process(w) for w in windows))

        if successful_calls == 0 and failed_calls > 0:
            log.error(
                "ollama[%s]: every window failed (%d attempts total); "
                "returning no LabeledTurns so IAA drops this rater.",
                self.rater_name,
                failed_calls,
            )
            return []

        return findings_to_labeled_turns(
            findings, corpus, rater_name=self.rater_name
        )

    async def _score_window_safe(self, window: Any) -> SpiralBenchScore | None:
        """Call Ollama with retry on validation failures. Returns None on
        persistent failure so the caller can emit "judge saw nothing"."""
        last_err: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self._call_once(window)
            except ValidationError as err:
                last_err = err
                log.warning(
                    "ollama[%s] attempt %d/%d: schema mismatch: %s",
                    self.rater_name,
                    attempt,
                    self._max_attempts,
                    str(err)[:200],
                )
            except Exception as err:
                last_err = err
                log.warning(
                    "ollama[%s] attempt %d/%d: %s: %s",
                    self.rater_name,
                    attempt,
                    self._max_attempts,
                    type(err).__name__,
                    str(err)[:200],
                )
                # Non-validation errors likely mean the daemon is down
                # or the model is missing — no point retrying.
                break
        log.error(
            "ollama[%s]: giving up on window after %d attempts: %s",
            self.rater_name,
            self._max_attempts,
            last_err,
        )
        return None

    async def _call_once(self, window: Any) -> SpiralBenchScore:
        response = await self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": self._prompt.body},
                {"role": "user", "content": _render_window(window)},
            ],
            format=SpiralBenchScore.model_json_schema(),
            options={"temperature": 0.0},
            stream=False,
        )
        # Ollama's response may expose the content in different places
        # depending on version; be defensive.
        content = _extract_content(response)
        if not content:
            raise ValueError("ollama response had no message content")
        return SpiralBenchScore.model_validate_json(content)


def _extract_content(response: Any) -> str:
    """Pull the assistant's text content off an Ollama ChatResponse."""
    # Newer ollama-py: response.message.content
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content:
            return str(content)
    # Fallback: dict-style
    if (
        isinstance(response, dict)
        and "message" in response
        and isinstance(response["message"], dict)
    ):
        msg_content = response["message"].get("content")
        if msg_content:
            return str(msg_content)
    return ""
