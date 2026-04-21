"""Module G — deterministic time/model attribution.

For every conversation in the corpus, emit one :class:`Finding` with
``module=G`` and ``behavior='model=<api-id>'``. Confidence encodes how the
label was obtained:

- ``1.0``  — Claude Code corpus, ``Conversation.model`` already populated.
- ``0.6``  — Claude.ai corpus, inferred from ``updated_at`` via the
  default-model timeline below.
- ``0.3``  — ``updated_at`` predates the earliest timeline entry. Behavior
  is ``'model=unknown'`` so the report can surface the gap rather than
  quietly picking a default.

No LLM calls. The Finding's ``detected_by`` carries the literal string
``"deterministic-attribution"``. ``metadata`` records ``year_month`` for
time bucketing and ``source`` (``"declared"`` vs ``"inferred"``) for
provenance. Report rendering joins by ``conversation_id`` so behavior
findings from Modules A-F/H can be sliced by time and model.

Timeline source: ``docs/methodology.md §5`` (verified 2026-04-21 against
https://platform.claude.com/docs/en/release-notes/api). The timeline is
authoritative here; ``docs/BUILD_GUIDE.md §5`` has stale entries and is
pending update.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime

from lucid.modules.base import ModuleCorpus, ModuleResult
from lucid.schemas import Finding, ModuleName

__all__ = [
    "ANTHROPIC_DEFAULT_MODEL_TIMELINE",
    "CITATION_ATTRIBUTION",
    "DETECTED_BY",
    "MODULE_NAME",
    "PROMPT_VERSION",
    "ModuleGAttribution",
]


CITATION_ATTRIBUTION = (
    "Lucid Module G: deterministic attribution against Anthropic default-model timeline; "
    "source docs/methodology.md §5"
)
MODULE_NAME = ModuleName.G_ATTRIBUTION
PROMPT_VERSION = "v1"
DETECTED_BY = "deterministic-attribution"


# Ordered by release date; each entry is the Anthropic API id that became
# the *consumer-default* Sonnet at that date. Pro-tier users choose Opus
# manually; at attribution time we can only guess the default, so we bias
# toward Sonnet (the free/consumer default on claude.ai). Confidence 0.6
# reflects this bias.
ANTHROPIC_DEFAULT_MODEL_TIMELINE: tuple[tuple[date, str], ...] = (
    (date(2024, 6, 20), "claude-3-5-sonnet-20240620"),
    (date(2024, 10, 22), "claude-3-5-sonnet-20241022"),
    (date(2025, 2, 24), "claude-3-7-sonnet-20250219"),
    (date(2025, 5, 22), "claude-sonnet-4-20250514"),
    (date(2025, 9, 29), "claude-sonnet-4-5"),
    (date(2026, 2, 17), "claude-sonnet-4-6"),
)


def _infer_model(updated_at: datetime) -> tuple[str, float, str]:
    """Return ``(model_api_id, confidence, source)`` for a Claude.ai conversation.

    Binary-search-esque linear scan: timeline is short (≤ 20 entries for the
    hackathon window), no numpy dependency justified.
    """
    target = updated_at.date()
    if target < ANTHROPIC_DEFAULT_MODEL_TIMELINE[0][0]:
        return "unknown", 0.3, "inferred"

    chosen = ANTHROPIC_DEFAULT_MODEL_TIMELINE[0][1]
    for release_date, model_id in ANTHROPIC_DEFAULT_MODEL_TIMELINE:
        if target >= release_date:
            chosen = model_id
        else:
            break
    return chosen, 0.6, "inferred"


def _finding_id(audit_run_id: str, conversation_id: str) -> str:
    """Deterministic per-(run, conversation) finding id.

    Collapsing to the same id on re-run is required by the UNIQUE index in
    ``findings`` (``audit_run_id, module, conversation_id, turn_ids_hash,
    behavior``). Computing it client-side keeps idempotency self-contained
    — the DB still enforces the same invariant at the CHECK layer.
    """
    raw = f"{audit_run_id}:{conversation_id}:{MODULE_NAME.value}".encode()
    return hashlib.sha256(raw).hexdigest()


def _prompt_hash() -> str:
    """sha256 over a literal marker so the provenance field is non-empty.

    Module G has no prompt file, but the Finding contract requires a
    ``prompt_hash``. Using a stable sentinel keeps the column populated and
    immediately recognisable as "deterministic" in downstream joins.
    """
    return hashlib.sha256(b"lucid-module-g-deterministic-v1").hexdigest()


class ModuleGAttribution:
    """Deterministic attribution pass. Implements the CorpusModule protocol."""

    module_name: ModuleName = MODULE_NAME
    prompt_version: str = PROMPT_VERSION

    async def run(self, corpus: ModuleCorpus) -> list[ModuleResult]:
        detected_at = datetime.now(UTC)
        prompt_hash = _prompt_hash()
        results: list[ModuleResult] = []

        for conv_id, conv in corpus.conversations.items():
            if conv.model is not None:
                model_id = conv.model
                confidence = 1.0
                source = "declared"
            else:
                model_id, confidence, source = _infer_model(conv.updated_at)

            year_month = conv.updated_at.strftime("%Y-%m")
            behavior = f"model={model_id}"
            explanation = (
                f"Attribution: {source} model {model_id} for conversation "
                f"updated_at={conv.updated_at.isoformat()}."
            )

            results.append(
                Finding(
                    id=_finding_id(corpus.audit_run_id, conv_id),
                    audit_run_id=corpus.audit_run_id,
                    conversation_id=conv_id,
                    turn_ids=[],
                    turn_ids_hash=hashlib.sha256(b"").hexdigest(),
                    module=MODULE_NAME,
                    behavior=behavior,
                    intensity=None,
                    confidence=confidence,
                    explanation=explanation,
                    citation=CITATION_ATTRIBUTION,
                    detected_by=[DETECTED_BY],
                    detected_at=detected_at,
                    prompt_version=PROMPT_VERSION,
                    prompt_hash=prompt_hash,
                    metadata={
                        "source": source,
                        "year_month": year_month,
                        "model_id": model_id,
                    },
                )
            )

        return results
