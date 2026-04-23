"""Module F — Influence Tactics Protocol user-prompt analyzer.

Three-stage filter, described from cheapest to most expensive:

1. **Stage 1 (heuristic)** — pure-Python regex filter in
   :mod:`lucid.modules._f_heuristic_v1`. Scans each user prompt for
   pattern hits from 6 of the 9 ITP categories (the other 3 require
   semantic judgment and cannot be cheaply regex-detected). A prompt
   that matches at least one pattern is a **candidate**; non-matches
   are dropped before any LLM work.

2. **Stage 2 (triage, Sonnet 4.6)** — one Sonnet call per candidate.
   Sonnet decides `proceed`, `drop`, or `unsure`. ``drop`` ends the
   pipeline for that prompt. ``proceed`` and ``unsure`` both advance
   to Stage 3.

3. **Stage 3 (classify, Opus 4.7)** — one Opus call per surviving
   prompt. Opus identifies which ITP categories are present, their
   intensity 1–3, and the triggering phrase.

**Cost framing.** Plan §Phase 7 estimates the full-pipeline cost at
$1.20 per audit vs $10 for single-stage Opus classification on every
prompt. The 3-stage filter is what makes Module F affordable at audit
scale.

**Findings.** One :class:`Finding` per detected tactic. Each finding's
``conversation_id`` and ``turn_ids`` point at the user turn that carried
the tactic. ``behavior`` is the ITP category id (e.g.
``emotional-triggers``). ``intensity`` is 1–3. Prompts with no detected
tactic (Stage 2 ``drop`` or Stage 3 empty array) do not produce
findings; Module F is a positive-finding-only module.

**Error isolation.** Heuristic failures are impossible (pure Python,
small surface). Stage 2 and Stage 3 LLM failures become
:class:`~lucid.modules.base.ModuleError` per prompt and do not abort
the module pass.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from lucid.modules._f_heuristic_v1 import (
    HEURISTIC_VERSION,
    ITP_CATEGORIES,
    heuristic_match,
)
from lucid.modules.base import (
    ModuleCorpus,
    ModuleError,
    ModuleResult,
    extract_result_json,
)
from lucid.prompts import load_prompt
from lucid.schemas import Finding, ModuleName, Role, Turn

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

__all__ = [
    "CITATION_ITP",
    "CLASSIFY_PROMPT_VERSION",
    "HEURISTIC_VERSION",
    "ITP_CATEGORIES",
    "MODEL_OPUS",
    "MODEL_SONNET",
    "MODULE_NAME",
    "TRIAGE_PROMPT_VERSION",
    "ClassifyResult",
    "ITPCategory",
    "ITPTactic",
    "ModuleFITP",
    "TriageDecision",
    "TriageResult",
]


CITATION_ITP = "Influence Tactics Protocol, https://github.com/synaptiai/influence-tactics-protocol"
MODULE_NAME = ModuleName.F_ITP
TRIAGE_PROMPT_VERSION = "triage_v1"
CLASSIFY_PROMPT_VERSION = "classify_v1"
MODEL_OPUS = "claude-opus-4-7"
MODEL_SONNET = "claude-sonnet-4-6"
MAX_TRIAGE_OUTPUT_TOKENS = 300
MAX_CLASSIFY_OUTPUT_TOKENS = 1200
MAX_CONCURRENCY_DEFAULT = 10


ITPCategory = Literal[
    "emotional-triggers",
    "urgent-action-demands",
    "emotional-repetition",
    "false-dilemmas",
    "context-omission",
    "authority-overload",
    "cherry-picked-data",
    "logical-fallacies",
    "framing-techniques",
]

TriageDecision = Literal["proceed", "drop", "unsure"]


# ──────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────


class TriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    decision: TriageDecision
    rationale_category: str  # category id, "multiple", or "none"


class ITPTactic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: ITPCategory
    intensity: Literal[1, 2, 3]
    phrase: str = Field(min_length=1, max_length=200)
    explanation: str = Field(max_length=300)


class ClassifyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    tactics: list[ITPTactic] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────────
# Internal dataclasses
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A user turn flagged by the Stage 1 heuristic."""

    conversation_id: str
    user_turn: Turn
    prior_assistant_turn: Turn | None
    matched_categories: tuple[str, ...]
    matched_snippets: tuple[str, ...]


# ──────────────────────────────────────────────────────────────────────────
# Rendering + injection resistance
# ──────────────────────────────────────────────────────────────────────────


_USER_PROMPT_OPEN = "<USER_PROMPT>"
_USER_PROMPT_CLOSE = "</USER_PROMPT>"
_PRIOR_OPEN = "<PRIOR_ASSISTANT_TURN>"
_PRIOR_CLOSE = "</PRIOR_ASSISTANT_TURN>"
_HEURISTIC_OPEN = "<HEURISTIC_MATCHES>"
_HEURISTIC_CLOSE = "</HEURISTIC_MATCHES>"
_STAGE2_OPEN = "<STAGE_2_DECISION>"
_STAGE2_CLOSE = "</STAGE_2_DECISION>"

_ALL_DELIMITERS: tuple[str, ...] = (
    _USER_PROMPT_OPEN,
    _USER_PROMPT_CLOSE,
    _PRIOR_OPEN,
    _PRIOR_CLOSE,
    _HEURISTIC_OPEN,
    _HEURISTIC_CLOSE,
    _STAGE2_OPEN,
    _STAGE2_CLOSE,
)


def _escape_delimiters(content: str) -> str:
    out = content
    for delim in _ALL_DELIMITERS:
        out = out.replace(delim, delim[0] + " " + delim[1:])
    return out


def _render_triage_request(candidate: _Candidate) -> str:
    prior_text = (
        _escape_delimiters(candidate.prior_assistant_turn.content)
        if candidate.prior_assistant_turn is not None
        else ""
    )
    heuristic_lines = [
        f"matched_categories: {', '.join(candidate.matched_categories)}",
        "matched_snippets: "
        + ", ".join(f'"{_escape_delimiters(s)}"' for s in candidate.matched_snippets),
    ]
    return (
        f"{_USER_PROMPT_OPEN}\n"
        f"{_escape_delimiters(candidate.user_turn.content)}\n"
        f"{_USER_PROMPT_CLOSE}\n\n"
        f"{_PRIOR_OPEN}\n{prior_text}\n{_PRIOR_CLOSE}\n\n"
        f"{_HEURISTIC_OPEN}\n{chr(10).join(heuristic_lines)}\n{_HEURISTIC_CLOSE}\n"
    )


def _render_classify_request(
    candidate: _Candidate,
    triage: TriageResult,
) -> str:
    base = _render_triage_request(candidate)
    stage2_lines = [
        f"decision: {triage.decision}",
        f"rationale_category: {triage.rationale_category}",
    ]
    return base + "\n" + f"{_STAGE2_OPEN}\n{chr(10).join(stage2_lines)}\n{_STAGE2_CLOSE}\n"


# ──────────────────────────────────────────────────────────────────────────
# Candidate detection — Stage 1
# ──────────────────────────────────────────────────────────────────────────


def _prior_assistant_turn(
    turns: Sequence[Turn],
    user_turn_index: int,
) -> Turn | None:
    """Walk backward from the user turn to find the nearest assistant turn."""
    for j in range(user_turn_index - 1, -1, -1):
        if turns[j].role is Role.ASSISTANT:
            return turns[j]
    return None


def _detect_candidates(
    corpus: ModuleCorpus,
) -> list[_Candidate]:
    """Scan every user turn with the Stage 1 heuristic and collect candidates."""
    candidates: list[_Candidate] = []
    for conv_id in corpus.conversations:
        turns = corpus.turns_by_conversation.get(conv_id, ())
        for i, turn in enumerate(turns):
            if turn.role is not Role.USER:
                continue
            if not turn.content.strip():
                continue
            match = heuristic_match(turn.content)
            if not match.is_candidate:
                continue
            candidates.append(
                _Candidate(
                    conversation_id=conv_id,
                    user_turn=turn,
                    prior_assistant_turn=_prior_assistant_turn(turns, i),
                    matched_categories=match.matched_categories,
                    matched_snippets=match.matched_snippets,
                )
            )
    return candidates


# ──────────────────────────────────────────────────────────────────────────
# Finding construction
# ──────────────────────────────────────────────────────────────────────────


def _turn_ids_hash(turn_ids: Sequence[str]) -> str:
    return hashlib.sha256(",".join(sorted(turn_ids)).encode()).hexdigest()


def _finding_id(
    audit_run_id: str,
    conversation_id: str,
    turn_id: str,
    category: str,
) -> str:
    raw = (f"{audit_run_id}:{conversation_id}:{MODULE_NAME.value}:{turn_id}:{category}").encode()
    return hashlib.sha256(raw).hexdigest()


def _tactic_to_finding(
    tactic: ITPTactic,
    *,
    candidate: _Candidate,
    overall_confidence: float,
    reasoning: str,
    audit_run_id: str,
    prompt_hash: str,
    detected_at: datetime,
) -> Finding:
    turn_ids = [candidate.user_turn.id]
    return Finding(
        id=_finding_id(
            audit_run_id,
            candidate.conversation_id,
            candidate.user_turn.id,
            tactic.category,
        ),
        audit_run_id=audit_run_id,
        conversation_id=candidate.conversation_id,
        turn_ids=turn_ids,
        turn_ids_hash=_turn_ids_hash(turn_ids),
        module=MODULE_NAME,
        behavior=tactic.category,
        intensity=tactic.intensity,
        confidence=overall_confidence,
        quote_user=tactic.phrase,
        quote_assistant=None,
        evidence_quotes=[],
        explanation=tactic.explanation,
        citation=CITATION_ITP,
        detected_by=[MODEL_OPUS],
        detected_at=detected_at,
        prompt_version=CLASSIFY_PROMPT_VERSION,
        prompt_hash=prompt_hash,
        metadata={
            "heuristic_version": HEURISTIC_VERSION,
            "heuristic_categories": list(candidate.matched_categories),
            "heuristic_snippets": list(candidate.matched_snippets),
            "reasoning": reasoning,
        },
    )


# ──────────────────────────────────────────────────────────────────────────
# Module class
# ──────────────────────────────────────────────────────────────────────────


class ModuleFITP:
    """``CorpusModule`` implementation of the 3-stage ITP detector.

    Takes an Opus client and a Sonnet client. They may be the same
    ``AsyncAnthropic`` instance; ``model=`` chooses per call. The separate
    params keep the dependency explicit at the call site.
    """

    module_name: ModuleName = MODULE_NAME
    prompt_version: str = CLASSIFY_PROMPT_VERSION

    def __init__(
        self,
        *,
        opus_client: AsyncAnthropic,
        sonnet_client: AsyncAnthropic | None = None,
        max_concurrency: int = MAX_CONCURRENCY_DEFAULT,
        prompt_root: str | None = None,
    ) -> None:
        self._opus = opus_client
        self._sonnet = sonnet_client if sonnet_client is not None else opus_client
        self._semaphore = asyncio.Semaphore(max_concurrency)

        kwargs: dict[str, object] = {}
        if prompt_root is not None:
            from pathlib import Path

            kwargs["root"] = Path(prompt_root)
        self._triage_prompt = load_prompt("f", TRIAGE_PROMPT_VERSION, **kwargs)  # type: ignore[arg-type]
        self._classify_prompt = load_prompt("f", CLASSIFY_PROMPT_VERSION, **kwargs)  # type: ignore[arg-type]

    async def run(self, corpus: ModuleCorpus) -> list[ModuleResult]:
        detected_at = datetime.now(UTC)
        candidates = _detect_candidates(corpus)
        if not candidates:
            return []

        results: list[ModuleResult] = []

        async def _pipeline_one(candidate: _Candidate) -> list[ModuleResult]:
            async with self._semaphore:
                # Stage 2 — Sonnet triage
                try:
                    triage = await self._call_triage(candidate)
                except Exception as exc:
                    return [
                        ModuleError(
                            module=MODULE_NAME,
                            conversation_id=candidate.conversation_id,
                            error_type=f"triage:{type(exc).__name__}",
                            message=str(exc)[:500],
                        )
                    ]
                if triage.decision == "drop":
                    return []

                # Stage 3 — Opus classify
                try:
                    classify = await self._call_classify(candidate, triage)
                except Exception as exc:
                    return [
                        ModuleError(
                            module=MODULE_NAME,
                            conversation_id=candidate.conversation_id,
                            error_type=f"classify:{type(exc).__name__}",
                            message=str(exc)[:500],
                        )
                    ]
                if not classify.tactics:
                    return []

                out: list[ModuleResult] = []
                for tactic in classify.tactics:
                    try:
                        out.append(
                            _tactic_to_finding(
                                tactic,
                                candidate=candidate,
                                overall_confidence=classify.overall_confidence,
                                reasoning=classify.reasoning,
                                audit_run_id=corpus.audit_run_id,
                                prompt_hash=self._classify_prompt.body_hash,
                                detected_at=detected_at,
                            )
                        )
                    except Exception as exc:
                        out.append(
                            ModuleError(
                                module=MODULE_NAME,
                                conversation_id=candidate.conversation_id,
                                error_type=f"classify_build:{type(exc).__name__}",
                                message=str(exc)[:500],
                            )
                        )
                return out

        pipelined = await asyncio.gather(*(_pipeline_one(c) for c in candidates))
        for batch in pipelined:
            results.extend(batch)
        return results

    # ------ LLM call wrappers -------------------------------------------

    async def _call_triage(self, candidate: _Candidate) -> TriageResult:
        content = await self._call_with_retry(
            self._sonnet,
            model=MODEL_SONNET,
            system_text=self._triage_prompt.padded_body,
            user_text=_render_triage_request(candidate),
            max_tokens=MAX_TRIAGE_OUTPUT_TOKENS,
        )
        return TriageResult.model_validate_json(extract_result_json(content))

    async def _call_classify(
        self,
        candidate: _Candidate,
        triage: TriageResult,
    ) -> ClassifyResult:
        content = await self._call_with_retry(
            self._opus,
            model=MODEL_OPUS,
            system_text=self._classify_prompt.padded_body,
            user_text=_render_classify_request(candidate, triage),
            max_tokens=MAX_CLASSIFY_OUTPUT_TOKENS,
        )
        return ClassifyResult.model_validate_json(extract_result_json(content))

    async def _call_with_retry(
        self,
        client: AsyncAnthropic,
        *,
        model: str,
        system_text: str,
        user_text: str,
        max_tokens: int,
    ) -> str:
        transient_exceptions: tuple[type[BaseException], ...] = (
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
        )
        try:
            from anthropic import APIConnectionError, APITimeoutError, RateLimitError

            transient_exceptions = (
                *transient_exceptions,
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
            )
        except Exception:
            pass

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(4),
            wait=wait_random_exponential(multiplier=1, max=60),
            retry=retry_if_exception_type(transient_exceptions),
            reraise=True,
        ):
            with attempt:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": system_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_text}],
                )
        content = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                content = getattr(block, "text", "") or ""
                if content:
                    break
        if not content:
            raise RuntimeError("messages.create returned no text content")
        return content
