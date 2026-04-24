"""Cross-module data types.

Pydantic v2, Python 3.13 idioms: `X | None`, `list[T]`, `dict[K, V]`. No
`Optional[...]` anywhere, no bare `dict` annotations.

Content blocks are modelled as a discriminated union (`Annotated[...,
Field(discriminator="type")]`) so a `tool_use` block carries `tool_name` +
`tool_input` without the `text` field bleeding in, and vice versa.

Schemas intentionally carry every provenance field a Finding must populate
(citation, detected_by, prompt_version, prompt_hash, confidence, timestamps)
as first-class required columns; the SQLite CHECK constraints in
`store/schema.sql` reject inserts missing any of them.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ──────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────


class Source(StrEnum):
    CLAUDE_CODE = "claude-code"
    CLAUDE_AI = "claude-ai"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ModuleName(StrEnum):
    A_SPIRALBENCH = "A"
    B_SHARMA = "B"
    C_SYCEVAL = "C"
    D_PERSPECTIVE = "D"
    E_BELIEFSHIFT = "E"
    F_ITP = "F"
    G_ATTRIBUTION = "G"
    H_MEMORY = "H"


class MemorySupport(StrEnum):
    WELL_SUPPORTED = "well-supported"
    WEAKLY_SUPPORTED = "weakly-supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_DATA = "insufficient-data"
    # Emitted when a memory claim is scoped to a project (or account)
    # whose conversations are not represented in the audit sample. Not
    # a verdict on the claim's truth — a statement that this audit
    # cannot verify it. See module_h_memory.py's source-aware retrieval.
    OUT_OF_SCOPE = "out-of-scope"


AuditStatus = Literal["running", "completed", "failed", "partial", "aborted_pre_spend"]


# Status enum for ``module_progress.status``. Mirrors the SQL CHECK constraint
# in ``store/schema.sql``; bumping either side requires bumping the other.
ModuleProgressStatus = Literal["pending", "running", "completed", "failed", "skipped"]


# ──────────────────────────────────────────────────────────────────────────
# Content blocks — discriminated union
# ──────────────────────────────────────────────────────────────────────────


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextBlock(_Block):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(_Block):
    type: Literal["thinking"] = "thinking"
    thinking: str
    # `signature` length is used as a depth proxy (Laurenzo methodology).
    signature: str | None = None


class ToolUseBlock(_Block):
    type: Literal["tool_use"] = "tool_use"
    tool_name: str
    tool_input: dict[str, object] = Field(default_factory=dict)
    tool_use_id: str | None = None
    # MCP-integration metadata (only present on Claude.ai export).
    mcp_integration: str | None = None
    mcp_server_url: str | None = None
    is_mcp_app: bool | None = None


class ToolResultBlock(_Block):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


# ──────────────────────────────────────────────────────────────────────────
# Corpus objects
# ──────────────────────────────────────────────────────────────────────────


class Conversation(BaseModel):
    """A single conversation from either source.

    `model` is None for Claude.ai conversations until Module G (attribution)
    fills it in from `updated_at` against the release timeline.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    source: Source
    source_path: str
    created_at: datetime
    updated_at: datetime
    model: str | None = None
    title: str | None = None
    summary: str | None = None
    turn_count: int = Field(ge=0)
    project_slug: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    conversation_id: str
    index: int = Field(ge=0)
    role: Role
    content: str
    blocks: list[ContentBlock] = Field(default_factory=list)
    timestamp: datetime | None = None
    parent_message_uuid: str | None = None
    token_count: int | None = Field(default=None, ge=0)


class MemoryClaim(BaseModel):
    """Atomic claim extracted from memories.json for Module H."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: str  # "conversations_memory" | f"project_memories.{uuid}"
    claim_text: str
    category: str | None = None


class MemoryFile(BaseModel):
    """Parsed memories.json contents."""

    model_config = ConfigDict(extra="forbid")

    account_uuid: str
    conversations_memory: str | None = None
    project_memories: dict[str, str] = Field(default_factory=dict)
    extracted_claims: list[MemoryClaim] = Field(default_factory=list)


class Project(BaseModel):
    """Parsed projects.json entry. Context only, not audited directly."""

    model_config = ConfigDict(extra="forbid")

    uuid: str
    name: str
    description: str | None = None
    prompt_template: str | None = None
    created_at: datetime
    updated_at: datetime
    doc_count: int = Field(ge=0)
    doc_char_total: int = Field(ge=0)


# ──────────────────────────────────────────────────────────────────────────
# Audit run / findings
# ──────────────────────────────────────────────────────────────────────────


class CorpusStats(BaseModel):
    """Discovered / sampled counts and coverage used in the report header."""

    model_config = ConfigDict(extra="forbid")

    discovered_conversations: int = Field(ge=0)
    sampled_conversations: int = Field(ge=0)
    discovered_turns: int = Field(ge=0)
    sampled_turns: int = Field(ge=0)
    date_range_start: datetime | None = None
    date_range_end: datetime | None = None
    sources: list[Source] = Field(default_factory=list)


class ModuleTokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0, default=0)
    cache_creation_input_tokens: int = Field(ge=0, default=0)
    usd_cost: float = Field(ge=0.0)


class TokenUsage(BaseModel):
    """Per-module token + cost breakdown, keyed by ModuleName."""

    model_config = ConfigDict(extra="forbid")

    by_module: dict[ModuleName, ModuleTokenUsage] = Field(default_factory=dict)
    orchestrator: ModuleTokenUsage | None = None


class SamplingConfigRecord(BaseModel):
    """Exactly what Sampling was asked to do, persisted for reproducibility."""

    model_config = ConfigDict(extra="forbid")

    n: int = Field(ge=0)
    seed: int
    min_turns: int = Field(ge=0)
    recency_weight: float = Field(ge=0.0, le=1.0)
    recency_window_days: int = Field(ge=0)
    stratify_by_project: bool
    top_n_projects: int = Field(ge=0)


class Finding(BaseModel):
    """A single detection event with full provenance.

    Every field below is required except where annotated `| None`. The
    SQLite schema enforces these with CHECK constraints at the DB level so
    no badly-formed Finding can survive a round-trip. `intensity` is
    `int | None` because Module H (memory support) doesn't use the 1-3
    scale used by behaviour modules.

    `confidence_alpha` / `confidence_beta` are optional Beta-distribution
    posteriors used by the report to render CI whiskers.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    audit_run_id: str
    conversation_id: str | None = None  # None for cross-corpus Module H findings
    turn_ids: list[str] = Field(default_factory=list)
    turn_ids_hash: str  # sha256 of sorted turn_ids; part of the idempotency key
    module: ModuleName
    behavior: str  # a specific label or MemorySupport value for Module H
    intensity: int | None = Field(default=None, ge=1, le=3)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_alpha: float | None = Field(default=None, gt=0.0)
    confidence_beta: float | None = Field(default=None, gt=0.0)
    quote_user: str | None = None
    quote_assistant: str | None = None
    evidence_quotes: list[str] = Field(default_factory=list)
    explanation: str
    citation: str  # the paper / framework that motivated this detection
    detected_by: list[str] = Field(min_length=1)  # must name at least one model ID
    detected_at: datetime
    prompt_version: str  # e.g. "v1"; first-class column, never inferred
    prompt_hash: str  # sha256 of the prompt body used for this finding
    metadata: dict[str, object] = Field(default_factory=dict)


class ModuleProgress(BaseModel):
    """One row in ``module_progress`` — the per-module lifecycle log.

    Lifecycle: ``running`` is set when ``invoke_module`` (or the
    post-session safety net) starts a module; one of ``completed``,
    ``failed``, ``skipped`` is set when the run terminates.

    For modules that short-circuit (Module D without
    ``--include-module-d``, Module H without an embedding provider,
    LLM-backed modules without an Anthropic client) ``started_at`` and
    ``completed_at`` carry the same skip-decision timestamp; the
    duration of a skipped module is undefined-but-reported-as-zero.
    """

    model_config = ConfigDict(extra="forbid")

    audit_run_id: str
    module: ModuleName
    status: ModuleProgressStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None


class AuditRun(BaseModel):
    """One end-to-end audit invocation.

    Checkpointed continuously: `corpus_fingerprint` plus `prompt_versions`
    plus `schema_version` are the resume contract — a resumed run with any
    of these changed is rejected with a clear error rather than silently
    producing a frankenstein set of findings.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    sources: list[Source]
    source_paths: dict[Source, str]
    started_at: datetime
    completed_at: datetime | None = None
    corpus_stats: CorpusStats
    token_usage: TokenUsage
    sampling_config: SamplingConfigRecord
    status: AuditStatus
    corpus_fingerprint: str  # sha256 over sorted '{conv_id}:{content_hash}' list
    prompt_versions: dict[ModuleName, str] = Field(default_factory=dict)
    schema_version: int = Field(ge=1)
    skipped_modules: list[ModuleName] = Field(default_factory=list)


class ReportSection(BaseModel):
    """One agent-written section of the HTML report.

    Populated by the synthesis phase; read by the report renderer.
    ``markdown`` contains inline ``[F:finding_id]`` / ``[T:turn_id]``
    citation tokens that the renderer resolves to hover-links. Every
    id in ``cited_finding_ids`` / ``cited_turn_ids`` MUST exist in the
    run's findings/turns tables — the synthesis validator enforces this
    before the row lands here.

    When ``insufficient_evidence`` is True, the agent declined the
    section. ``markdown`` is empty, citation lists are empty, and
    ``decline_reason`` carries a human-readable explanation. The template
    renders "Section skipped: <decline_reason>" instead of prose.
    """

    model_config = ConfigDict(extra="forbid")

    audit_run_id: str = Field(min_length=1)
    section_id: str = Field(min_length=1, max_length=64)
    markdown: str = Field(default="")
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_turn_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False
    decline_reason: str | None = None
    created_at: datetime

    @model_validator(mode="after")
    def _check_insufficient_evidence_invariant(self) -> ReportSection:
        """Mirror the SQL CHECK in ``store/schema.sql`` at the Pydantic layer.

        Enforces the same two-direction invariant the DB enforces so
        in-memory construction fails loudly on inconsistent state rather
        than waiting for the INSERT to trip the CHECK constraint.
        """
        if self.insufficient_evidence:
            if self.markdown != "":
                raise ValueError(
                    "insufficient_evidence=True requires markdown == ''; "
                    f"got length {len(self.markdown)}"
                )
            if self.cited_finding_ids != []:
                raise ValueError(
                    "insufficient_evidence=True requires cited_finding_ids == []; "
                    f"got {len(self.cited_finding_ids)} entries"
                )
            if self.cited_turn_ids != []:
                raise ValueError(
                    "insufficient_evidence=True requires cited_turn_ids == []; "
                    f"got {len(self.cited_turn_ids)} entries"
                )
            if self.decline_reason is None or self.decline_reason.strip() == "":
                raise ValueError(
                    "insufficient_evidence=True requires a non-empty decline_reason"
                )
        else:
            if self.markdown == "":
                raise ValueError(
                    "insufficient_evidence=False requires non-empty markdown"
                )
            if self.decline_reason is not None:
                raise ValueError(
                    "insufficient_evidence=False requires decline_reason is None; "
                    f"got {self.decline_reason!r}"
                )
        return self
