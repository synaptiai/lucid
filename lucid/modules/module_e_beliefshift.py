"""Module E — BeliefShift (cross-conversation belief-drift tracker).

Three sub-passes:

1. **E.1 Topic extraction** (Sonnet 4.6, thinking disabled, effort low) —
   one call per audit. Reads compact summaries of every conversation in
   the corpus and produces 5–10 topics the user holds positions on. Each
   topic carries the list of conversation ids it appears in.

2. **E.2 Position tracking** (Opus 4.7, adaptive, effort high) — one
   call per (topic, conversation) pair from E.1's output. Reads the full
   conversation and extracts the user's position plus the assistant's
   reaction type (pushback, agreement, new_information, neutral,
   no_direct_engagement).

3. **E.3 Drift analysis** (Opus 4.7, adaptive, effort high) — one call
   per topic. Reads the chronologically-ordered trajectory of positions
   (E.2's output) and classifies the trajectory as stable,
   drifted-evidence, drifted-pressure, drifted-mixed, or drifted-unclear
   with severity 0–3.

**Budget.** BUILD_GUIDE estimates Module E at 2× other modules; with 8
topics × 4 conversations per topic the E.2 stage runs ~32 Opus calls
plus 8 for E.3 plus 1 for E.1. The orchestrator's cost estimator applies
the Module E profile's ``output_tokens_per_conv`` multiplier (2×) to
reflect this.

**Output.** One Finding per topic. ``behavior`` encodes the drift type
(e.g. ``belief-drift-evidence``, ``belief-drift-stable``). Severity
rides on ``intensity`` for non-stable topics. Metadata preserves the
full shift list + trajectory for report rendering.

**Error isolation.** E.1 failure aborts the module (no topics means no
downstream work). E.2 failures on a (topic, conversation) become
ModuleErrors and the drift pass runs with whatever positions succeeded.
E.3 failure on a topic becomes a ModuleError and skips that topic's
finding.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterable, Sequence
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

from lucid.modules.base import (
    ModuleCorpus,
    ModuleError,
    ModuleResult,
    extract_result_json,
)
from lucid.prompts import load_prompt
from lucid.schemas import Conversation, Finding, ModuleName, Role, Turn

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

__all__ = [
    "CITATION_BELIEFSHIFT",
    "DRIFT_PROMPT_VERSION",
    "DRIFT_TYPES",
    "MODEL_OPUS",
    "MODEL_SONNET",
    "MODULE_NAME",
    "POSITIONS_PROMPT_VERSION",
    "REACTION_TYPES",
    "TOPICS_PROMPT_VERSION",
    "DriftScore",
    "DriftShift",
    "DriftType",
    "ModuleEBeliefShift",
    "PositionScore",
    "ReactionType",
    "Topic",
    "TopicsResult",
]


CITATION_BELIEFSHIFT = (
    "BeliefShift: Benchmarking Temporal Belief Consistency and Opinion Drift in LLM Agents, "
    "arxiv:2603.23848"
)
MODULE_NAME = ModuleName.E_BELIEFSHIFT
TOPICS_PROMPT_VERSION = "topics_v1"
POSITIONS_PROMPT_VERSION = "positions_v1"
DRIFT_PROMPT_VERSION = "drift_v1"
MODEL_OPUS = "claude-opus-4-7"
MODEL_SONNET = "claude-sonnet-4-6"
MAX_TOPICS_OUTPUT_TOKENS = 2000
MAX_POSITIONS_OUTPUT_TOKENS = 900
MAX_DRIFT_OUTPUT_TOKENS = 1400
MAX_CONCURRENCY_DEFAULT = 10
SUMMARY_SNIPPET_CHARS = 300


REACTION_TYPES: tuple[str, ...] = (
    "pushback",
    "agreement",
    "new_information",
    "neutral",
    "no_direct_engagement",
)
ReactionType = Literal[
    "pushback",
    "agreement",
    "new_information",
    "neutral",
    "no_direct_engagement",
]

DRIFT_TYPES: tuple[str, ...] = (
    "stable",
    "drifted-evidence",
    "drifted-pressure",
    "drifted-mixed",
    "drifted-unclear",
)
DriftType = Literal[
    "stable",
    "drifted-evidence",
    "drifted-pressure",
    "drifted-mixed",
    "drifted-unclear",
]

ShiftType = Literal["evidence", "pressure", "ambiguous", "none"]
FinalAlignment = Literal["original", "toward-assistant", "other"]
PositionConfidence = Literal["strong", "moderate", "weak"]


# ──────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────


class Topic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: str = Field(min_length=1)
    descriptor: str = Field(min_length=1, max_length=200)
    conversation_ids: list[str] = Field(min_length=2)
    supporting_signal: str = Field(max_length=200)


class TopicsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    topics: list[Topic] = Field(default_factory=list)


class PositionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    found_position: bool
    position_summary: str
    position_confidence: PositionConfidence
    assistant_reaction_type: ReactionType
    position_quote: str = Field(max_length=200)
    assistant_quote: str = Field(max_length=200)
    turn_indices: list[int] = Field(default_factory=list)
    note: str = ""


class DriftShift(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_conversation_id: str
    to_conversation_id: str
    from_position: str = Field(max_length=200)
    to_position: str = Field(max_length=200)
    shift_type: ShiftType
    rationale: str = Field(max_length=240)


class DriftScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    drift_detected: bool
    drift_type: DriftType
    severity: Literal[0, 1, 2, 3]
    shifts: list[DriftShift] = Field(default_factory=list)
    final_alignment: FinalAlignment
    confidence: float = Field(ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────────
# Internal dataclasses
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _PositionRecord:
    """One (topic, conversation) position result from E.2."""

    topic_id: str
    conversation_id: str
    updated_at: datetime
    score: PositionScore


# ──────────────────────────────────────────────────────────────────────────
# Rendering + injection resistance
# ──────────────────────────────────────────────────────────────────────────


_CONV_SUMMARIES_OPEN = "<CONVERSATION_SUMMARIES>"
_CONV_SUMMARIES_CLOSE = "</CONVERSATION_SUMMARIES>"
_TOPIC_OPEN = "<TOPIC>"
_TOPIC_CLOSE = "</TOPIC>"
_CONV_OPEN = "<CONVERSATION>"
_CONV_CLOSE = "</CONVERSATION>"
_META_OPEN = "<CONVERSATION_METADATA>"
_META_CLOSE = "</CONVERSATION_METADATA>"
_TRAJECTORY_OPEN = "<TRAJECTORY>"
_TRAJECTORY_CLOSE = "</TRAJECTORY>"

_ALL_DELIMITERS: tuple[str, ...] = (
    _CONV_SUMMARIES_OPEN,
    _CONV_SUMMARIES_CLOSE,
    _TOPIC_OPEN,
    _TOPIC_CLOSE,
    _CONV_OPEN,
    _CONV_CLOSE,
    _META_OPEN,
    _META_CLOSE,
    _TRAJECTORY_OPEN,
    _TRAJECTORY_CLOSE,
)


def _escape_delimiters(content: str) -> str:
    out = content
    for delim in _ALL_DELIMITERS:
        out = out.replace(delim, delim[0] + " " + delim[1:])
    return out


def _first_user_snippet(turns: Sequence[Turn]) -> str:
    """Return the first user turn's content, truncated to SUMMARY_SNIPPET_CHARS."""
    for t in turns:
        if t.role is Role.USER and t.content.strip():
            snippet = t.content.strip().replace("\n", " ")
            if len(snippet) > SUMMARY_SNIPPET_CHARS:
                return snippet[:SUMMARY_SNIPPET_CHARS] + "…"
            return snippet
    return ""


def _render_summaries(
    conversations: Iterable[Conversation],
    turns_by_conv: dict[str, Sequence[Turn]],
) -> str:
    """Build the CONVERSATION_SUMMARIES block for E.1 topic extraction."""
    parts: list[str] = [_CONV_SUMMARIES_OPEN]
    for conv in conversations:
        updated = conv.updated_at.date().isoformat()
        title = _escape_delimiters(conv.title or conv.summary or "")
        snippet = _escape_delimiters(
            _first_user_snippet(turns_by_conv.get(conv.id, ()))
        )
        parts.append(
            f"[CONV id={conv.id} updated={updated}]\n"
            f'Title: "{title}"\n'
            f'Start: "{snippet}"'
        )
    parts.append(_CONV_SUMMARIES_CLOSE)
    return "\n\n".join(parts)


def _render_conversation(turns: Sequence[Turn]) -> str:
    parts: list[str] = [_CONV_OPEN]
    for t in turns:
        header = f"[{t.role.value.upper()} t={t.index}]"
        parts.append(f"{header}\n{_escape_delimiters(t.content)}")
    parts.append(_CONV_CLOSE)
    return "\n\n".join(parts)


def _render_position_request(
    topic: Topic,
    conversation: Conversation,
    turns: Sequence[Turn],
) -> str:
    updated = conversation.updated_at.date().isoformat()
    return (
        f"{_TOPIC_OPEN}\n{_escape_delimiters(topic.descriptor)}\n{_TOPIC_CLOSE}\n\n"
        f"{_render_conversation(turns)}\n\n"
        f"{_META_OPEN}\n"
        f"updated_at: {updated}\n"
        f"conversation_id: {conversation.id}\n"
        f"{_META_CLOSE}\n"
    )


def _render_trajectory(topic: Topic, records: Sequence[_PositionRecord]) -> str:
    """Build the TRAJECTORY block for E.3 drift analysis."""
    lines: list[str] = [
        _TRAJECTORY_OPEN,
        f"Topic: {_escape_delimiters(topic.descriptor)}",
        f"Conversation count: {len(records)}",
        "",
    ]
    for i, rec in enumerate(records, start=1):
        score = rec.score
        lines.append(
            f"[POSITION {i}] conversation_id={rec.conversation_id} "
            f"updated={rec.updated_at.date().isoformat()}"
        )
        lines.append(f'Summary: "{_escape_delimiters(score.position_summary)}"')
        lines.append(f"Confidence: {score.position_confidence}")
        lines.append(
            f'Position quote: "{_escape_delimiters(score.position_quote)}"'
        )
        lines.append(f"Assistant reaction: {score.assistant_reaction_type}")
        lines.append(
            f'Assistant quote: "{_escape_delimiters(score.assistant_quote)}"'
        )
        if score.note:
            lines.append(f'Note: "{_escape_delimiters(score.note)}"')
        lines.append("")
    lines.append(_TRAJECTORY_CLOSE)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Finding construction
# ──────────────────────────────────────────────────────────────────────────


def _turn_ids_hash(turn_ids: Sequence[str]) -> str:
    return hashlib.sha256(",".join(sorted(turn_ids)).encode()).hexdigest()


def _drift_finding_id(audit_run_id: str, topic_id: str) -> str:
    raw = f"{audit_run_id}:{MODULE_NAME.value}:{topic_id}".encode()
    return hashlib.sha256(raw).hexdigest()


def _score_to_drift_finding(
    score: DriftScore,
    *,
    topic: Topic,
    records: Sequence[_PositionRecord],
    audit_run_id: str,
    prompt_hash: str,
    detected_at: datetime,
) -> Finding:
    """Convert a DriftScore into a Lucid Finding. One per topic.

    The finding's ``conversation_id`` is the LAST conversation in the
    trajectory (the end-state). UNIQUE key is ``(run, module, conv,
    turn_ids_hash, behavior)`` — using the end-state conversation keeps
    the finding uniquely keyed without colliding across topics on the
    same final conversation (behavior differs per topic).

    ``turn_ids`` collates the turn indices the E.2 pass cited for each
    position in the trajectory, so the report can jump to the anchor
    turns. Because turn indices are per-conversation-scoped and we need
    globally-unique ids for the hash, we prefix them with their
    conversation id.
    """
    if not records:
        raise ValueError("drift finding requires ≥ 1 position record")

    last_record = records[-1]
    behavior = f"belief-drift-{score.drift_type.replace('drifted-', '')}"
    if score.drift_type == "stable":
        behavior = "belief-drift-stable"
    intensity: int | None = score.severity if score.severity >= 1 else None

    turn_ids: list[str] = []
    for rec in records:
        for idx in rec.score.turn_indices:
            turn_ids.append(f"{rec.conversation_id}:t={idx}")

    summary = (
        f"Topic '{topic.descriptor}': drift_type={score.drift_type}, "
        f"severity={score.severity}, final_alignment={score.final_alignment}"
        + (
            f" (shifts: {len(score.shifts)})"
            if score.shifts
            else ""
        )
        + "."
    )

    return Finding(
        id=_drift_finding_id(audit_run_id, topic.topic_id),
        audit_run_id=audit_run_id,
        conversation_id=last_record.conversation_id,
        turn_ids=turn_ids,
        turn_ids_hash=_turn_ids_hash(turn_ids),
        module=MODULE_NAME,
        behavior=behavior,
        intensity=intensity,
        confidence=score.confidence,
        quote_user=records[0].score.position_quote,
        quote_assistant=last_record.score.position_quote,
        evidence_quotes=[r.score.position_quote for r in records[1:-1]],
        explanation=summary,
        citation=CITATION_BELIEFSHIFT,
        detected_by=[MODEL_OPUS],
        detected_at=detected_at,
        prompt_version=DRIFT_PROMPT_VERSION,
        prompt_hash=prompt_hash,
        metadata={
            "topic_id": topic.topic_id,
            "topic_descriptor": topic.descriptor,
            "drift_type": score.drift_type,
            "final_alignment": score.final_alignment,
            "shifts": [s.model_dump() for s in score.shifts],
            "reasoning": score.reasoning,
            "trajectory": [
                {
                    "conversation_id": r.conversation_id,
                    "updated_at": r.updated_at.isoformat(),
                    "position_summary": r.score.position_summary,
                    "position_confidence": r.score.position_confidence,
                    "assistant_reaction_type": r.score.assistant_reaction_type,
                }
                for r in records
            ],
        },
    )


# ──────────────────────────────────────────────────────────────────────────
# Module class
# ──────────────────────────────────────────────────────────────────────────


class ModuleEBeliefShift:
    """``CorpusModule`` implementation of BeliefShift cross-conversation drift.

    Requires both an Opus client (E.2, E.3) and a Sonnet client (E.1).
    They may be the same ``AsyncAnthropic`` instance — ``model=`` chooses
    per call — but the separate params make the dependency obvious.
    """

    module_name: ModuleName = MODULE_NAME
    prompt_version: str = DRIFT_PROMPT_VERSION

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
        self._topics_prompt = load_prompt("e", TOPICS_PROMPT_VERSION, **kwargs)  # type: ignore[arg-type]
        self._positions_prompt = load_prompt(
            "e", POSITIONS_PROMPT_VERSION, **kwargs  # type: ignore[arg-type]
        )
        self._drift_prompt = load_prompt("e", DRIFT_PROMPT_VERSION, **kwargs)  # type: ignore[arg-type]

    async def run(self, corpus: ModuleCorpus) -> list[ModuleResult]:
        detected_at = datetime.now(UTC)
        results: list[ModuleResult] = []

        if not corpus.conversations:
            return []

        # Drift detection requires ≥ 2 conversations.
        if len(corpus.conversations) < 2:
            return [
                ModuleError(
                    module=MODULE_NAME,
                    conversation_id=None,
                    error_type="insufficient_corpus",
                    message=(
                        "Module E requires ≥ 2 conversations to identify cross-"
                        "conversation topics and trajectories; corpus has "
                        f"{len(corpus.conversations)}."
                    ),
                )
            ]

        # Stage 1: topic extraction.
        try:
            topics_result = await self._call_topics(corpus)
        except Exception as exc:
            return [
                ModuleError(
                    module=MODULE_NAME,
                    conversation_id=None,
                    error_type=f"topics:{type(exc).__name__}",
                    message=str(exc)[:500],
                )
            ]

        if not topics_result.topics:
            return []

        # Stage 2: position extraction per (topic, conversation).
        position_records_by_topic: dict[str, list[_PositionRecord]] = {
            t.topic_id: [] for t in topics_result.topics
        }

        async def _extract_position(topic: Topic, conv_id: str) -> None:
            async with self._semaphore:
                conv = corpus.conversations.get(conv_id)
                if conv is None:
                    return
                turns = corpus.turns_by_conversation.get(conv_id, ())
                if not turns:
                    return
                try:
                    score = await self._call_positions(topic, conv, turns)
                except Exception as exc:
                    results.append(
                        ModuleError(
                            module=MODULE_NAME,
                            conversation_id=conv_id,
                            error_type=f"positions:{type(exc).__name__}",
                            message=str(exc)[:500],
                        )
                    )
                    return
                if score.found_position:
                    position_records_by_topic[topic.topic_id].append(
                        _PositionRecord(
                            topic_id=topic.topic_id,
                            conversation_id=conv_id,
                            updated_at=conv.updated_at,
                            score=score,
                        )
                    )

        tasks = [
            _extract_position(t, cid)
            for t in topics_result.topics
            for cid in t.conversation_ids
        ]
        await asyncio.gather(*tasks)

        # Stage 3: drift analysis per topic (must have ≥ 2 positions).
        async def _analyze_drift(topic: Topic) -> ModuleResult | None:
            records = position_records_by_topic[topic.topic_id]
            if len(records) < 2:
                return ModuleError(
                    module=MODULE_NAME,
                    conversation_id=None,
                    error_type="insufficient_positions",
                    message=(
                        f"topic {topic.topic_id!r} ({topic.descriptor!r}) has "
                        f"only {len(records)} position records; need ≥ 2 for "
                        "drift analysis."
                    ),
                )
            records.sort(key=lambda r: r.updated_at)
            async with self._semaphore:
                try:
                    drift = await self._call_drift(topic, records)
                except Exception as exc:
                    return ModuleError(
                        module=MODULE_NAME,
                        conversation_id=None,
                        error_type=f"drift:{type(exc).__name__}",
                        message=str(exc)[:500],
                    )
                try:
                    return _score_to_drift_finding(
                        drift,
                        topic=topic,
                        records=records,
                        audit_run_id=corpus.audit_run_id,
                        prompt_hash=self._drift_prompt.body_hash,
                        detected_at=detected_at,
                    )
                except Exception as exc:
                    return ModuleError(
                        module=MODULE_NAME,
                        conversation_id=None,
                        error_type=f"drift_build:{type(exc).__name__}",
                        message=str(exc)[:500],
                    )

        drift_results = await asyncio.gather(
            *(_analyze_drift(t) for t in topics_result.topics)
        )
        for r in drift_results:
            if r is not None:
                results.append(r)

        return results

    # ------ LLM call wrappers -------------------------------------------

    async def _call_topics(self, corpus: ModuleCorpus) -> TopicsResult:
        # Sort conversations by updated_at so summaries are chronological.
        sorted_convs = sorted(
            corpus.conversations.values(), key=lambda c: c.updated_at
        )
        turns_by_conv_typed: dict[str, Sequence[Turn]] = {
            cid: list(corpus.turns_by_conversation.get(cid, ()))
            for cid in corpus.conversations
        }
        content = await self._call_with_retry(
            self._sonnet,
            model=MODEL_SONNET,
            system_text=self._topics_prompt.body,
            user_text=_render_summaries(sorted_convs, turns_by_conv_typed),
            max_tokens=MAX_TOPICS_OUTPUT_TOKENS,
        )
        return TopicsResult.model_validate_json(extract_result_json(content))

    async def _call_positions(
        self,
        topic: Topic,
        conversation: Conversation,
        turns: Sequence[Turn],
    ) -> PositionScore:
        content = await self._call_with_retry(
            self._opus,
            model=MODEL_OPUS,
            system_text=self._positions_prompt.body,
            user_text=_render_position_request(topic, conversation, turns),
            max_tokens=MAX_POSITIONS_OUTPUT_TOKENS,
        )
        return PositionScore.model_validate_json(extract_result_json(content))

    async def _call_drift(
        self,
        topic: Topic,
        records: Sequence[_PositionRecord],
    ) -> DriftScore:
        content = await self._call_with_retry(
            self._opus,
            model=MODEL_OPUS,
            system_text=self._drift_prompt.body,
            user_text=_render_trajectory(topic, records),
            max_tokens=MAX_DRIFT_OUTPUT_TOKENS,
        )
        return DriftScore.model_validate_json(extract_result_json(content))

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
