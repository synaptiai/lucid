"""Module H — Memory-corpus consistency.

Audits the AI-synthesized memories (Claude.ai's `memories.json`) against
the user's actual conversation corpus. For each claim the memory makes
about the user, Module H retrieves the most similar excerpts from the
corpus and asks Opus 4.7 whether the claim is well-supported, weakly-
supported, unsupported, contradicted, or insufficient-data.

**Pipeline:**

1. **Chunk + embed the corpus.** Chunks are turn-pairs (one user turn
   + the adjacent assistant turn). The chunk id is the sha256 of the
   chunk text; persisted embeddings short-circuit re-embed on resume.
2. **Extract claims.** One Opus call per memory chunk against the
   ``extract_v1`` prompt produces atomic per-user claims categorized
   as work/personal/preference/history/belief/skill.
3. **Retrieve.** Each claim is embedded as a query; numpy cosine
   similarity against the corpus matrix surfaces the top-k excerpts.
4. **Short-circuit insufficient.** If the top-1 similarity is below
   :data:`INSUFFICIENT_THRESHOLD`, Module H emits the ``insufficient-
   data`` Finding without a classifier call (saves Opus budget).
5. **Classify.** Opus 4.7 ``classify_v1`` labels the claim vs
   excerpts as one of the five :class:`~lucid.schemas.MemorySupport`
   values.
6. **Two-stage verification.** When the classifier returns
   ``insufficient-data`` AND top-similarity ≥ :data:`INSUFFICIENT_THRESHOLD`
   (the "related content was there but didn't anchor" case), Sonnet
   decomposes the claim into sub-claims via ``refine_v1`` and each
   sub-claim is retrieved + classified. The parent Finding's label is
   the worst label across sub-claims (contradicted > unsupported >
   weakly > well > insufficient).

**Error isolation.** Failures at any stage become
:class:`~lucid.modules.base.ModuleError` entries rather than aborting
the run. Pre-indexing is the one exception: if the embedding provider
is down, the whole module fails. Tests cover each failure point with a
mock that injects the failure at that stage.

**Memories load.** Module H reads :class:`~lucid.schemas.MemoryFile`
rows from the :class:`~lucid.store.sqlite.CorpusStore` at run time; the
ingest layer persisted them earlier. A corpus with no memory files
produces an empty result set (not an error) — not every user exports
memories, and the orchestrator logs this case and continues.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
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
from lucid.modules.embeddings import (
    EmbeddingProvider,
    blob_to_vector,
    chunk_id,
    cosine_similarity_matrix,
    top_k_indices,
    vector_to_blob,
)
from lucid.prompts import load_prompt
from lucid.schemas import (
    Finding,
    MemoryFile,
    MemorySupport,
    ModuleName,
    Project,
    Role,
    Turn,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from lucid.store.sqlite import CorpusStore

__all__ = [
    "CITATION_MODULE_H",
    "CLAIM_CATEGORIES",
    "CLASSIFY_PROMPT_VERSION",
    "EXTRACT_PROMPT_VERSION",
    "INSUFFICIENT_THRESHOLD",
    "MEMORY_CHUNK_BUDGET_CHARS",
    "MODEL_OPUS",
    "MODEL_SONNET",
    "MODULE_NAME",
    "REFINE_PROMPT_VERSION",
    "TOP_K_DEFAULT",
    "ClaimCategory",
    "ClassifyScore",
    "Excerpt",
    "ExtractResult",
    "ExtractedClaim",
    "ModuleHMemory",
    "RefineResult",
    "SubClaim",
]


_LOGGER = logging.getLogger(__name__)


CITATION_MODULE_H = (
    "Lucid Module H — memory-corpus consistency verification via retrieval + "
    "two-stage classification. MedTrust-RAG framing: arxiv:2510.14400."
)
MODULE_NAME = ModuleName.H_MEMORY
EXTRACT_PROMPT_VERSION = "extract_v1"
CLASSIFY_PROMPT_VERSION = "classify_v1"
REFINE_PROMPT_VERSION = "refine_v1"
MODEL_OPUS = "claude-opus-4-7"
MODEL_SONNET = "claude-sonnet-4-6"
MAX_EXTRACT_OUTPUT_TOKENS = 1800
MAX_CLASSIFY_OUTPUT_TOKENS = 1100
MAX_REFINE_OUTPUT_TOKENS = 800
MAX_CONCURRENCY_DEFAULT = 10
INSUFFICIENT_THRESHOLD = 0.35
TOP_K_DEFAULT = 20
# Memory text is chunked for extraction — 3000 chars ≈ 850 tokens, large
# enough to carry a typical paragraph and small enough to keep the Opus
# call bounded. Memories rarely exceed a few chunks.
MEMORY_CHUNK_BUDGET_CHARS = 3000


CLAIM_CATEGORIES: tuple[str, ...] = (
    "work",
    "personal",
    "preference",
    "history",
    "belief",
    "skill",
)
ClaimCategory = Literal["work", "personal", "preference", "history", "belief", "skill"]


# ──────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=400)
    category: ClaimCategory
    source_span: str = Field(max_length=400)


class ExtractResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    claims: list[ExtractedClaim] = Field(default_factory=list)


class ClassifyScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    classification: MemorySupport
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quotes: list[str] = Field(default_factory=list)
    cited_excerpt_numbers: list[int] = Field(default_factory=list)


class SubClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=300)
    category: ClaimCategory
    aspect: str = Field(max_length=80)


class RefineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    sub_claims: list[SubClaim] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# Internal dataclasses
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Chunk:
    """One turn-pair chunk of the corpus, ready for embedding."""

    chunk_id: str  # content hash
    conversation_id: str
    anchor_turn_id: str  # id of the user (anchor) turn
    chunk_text: str


@dataclass(frozen=True, slots=True)
class Excerpt:
    """A retrieved excerpt (chunk + similarity score) for one claim."""

    conversation_id: str
    turn_id: str
    chunk_text: str
    similarity: float


@dataclass(frozen=True, slots=True)
class _ClassifyInput:
    """Everything Module H needs to call the classifier for one claim."""

    claim_text: str
    claim_category: ClaimCategory
    excerpts: list[Excerpt]


# ──────────────────────────────────────────────────────────────────────────
# Rendering + injection resistance
# ──────────────────────────────────────────────────────────────────────────


_DELIMITERS: tuple[str, ...] = (
    "<MEMORY_TEXT>",
    "</MEMORY_TEXT>",
    "<MEMORY_METADATA>",
    "</MEMORY_METADATA>",
    "<CLAIM>",
    "</CLAIM>",
    "<CLAIM_CATEGORY>",
    "</CLAIM_CATEGORY>",
    "<RETRIEVED_EXCERPTS>",
    "</RETRIEVED_EXCERPTS>",
    "<RETRIEVAL_METADATA>",
    "</RETRIEVAL_METADATA>",
    "<FIRST_PASS_METADATA>",
    "</FIRST_PASS_METADATA>",
)


def _escape_delimiters(content: str) -> str:
    out = content
    for d in _DELIMITERS:
        out = out.replace(d, d[0] + " " + d[1:])
    return out


def _render_extract_request(memory_text: str, source: str) -> str:
    return (
        f"<MEMORY_TEXT>\n{_escape_delimiters(memory_text)}\n</MEMORY_TEXT>\n\n"
        f"<MEMORY_METADATA>\n"
        f"source: {_escape_delimiters(source)}\n"
        f"</MEMORY_METADATA>\n"
    )


def _render_classify_request(inp: _ClassifyInput) -> str:
    top_sim = inp.excerpts[0].similarity if inp.excerpts else 0.0
    excerpt_lines = []
    for i, e in enumerate(inp.excerpts, start=1):
        header = f"[EXCERPT {i} score={e.similarity:.2f} conv={e.conversation_id} turn={e.turn_id}]"
        excerpt_lines.append(f"{header}\n{_escape_delimiters(e.chunk_text)}")
    excerpts_block = "\n\n".join(excerpt_lines) if excerpt_lines else "(no excerpts)"
    return (
        f"<CLAIM>\n{_escape_delimiters(inp.claim_text)}\n</CLAIM>\n\n"
        f"<CLAIM_CATEGORY>\n{inp.claim_category}\n</CLAIM_CATEGORY>\n\n"
        f"<RETRIEVED_EXCERPTS>\n{excerpts_block}\n</RETRIEVED_EXCERPTS>\n\n"
        f"<RETRIEVAL_METADATA>\n"
        f"top_similarity: {top_sim:.3f}\n"
        f"excerpt_count: {len(inp.excerpts)}\n"
        f"retrieval_model: voyage\n"
        f"</RETRIEVAL_METADATA>\n"
    )


def _render_refine_request(
    claim: ExtractedClaim,
    first_pass_top_sim: float,
    first_pass_top_preview: str,
) -> str:
    preview = first_pass_top_preview[:120]
    return (
        f"<CLAIM>\n{_escape_delimiters(claim.text)}\n</CLAIM>\n\n"
        f"<CLAIM_CATEGORY>\n{claim.category}\n</CLAIM_CATEGORY>\n\n"
        f"<FIRST_PASS_METADATA>\n"
        f"classification: insufficient-data\n"
        f"top_similarity: {first_pass_top_sim:.3f}\n"
        f'top_excerpt_preview: "{_escape_delimiters(preview)}"\n'
        f"</FIRST_PASS_METADATA>\n"
    )


# ──────────────────────────────────────────────────────────────────────────
# Corpus chunking
# ──────────────────────────────────────────────────────────────────────────


def _corpus_chunks(corpus: ModuleCorpus) -> list[_Chunk]:
    """Chunk the corpus into turn-pair chunks.

    For each user turn followed by an assistant turn, emit one chunk
    whose text is the concatenation of the two turn contents. Empty
    chunks (after stripping) are skipped. The ``anchor_turn_id`` is the
    user turn's id — that's the id we surface when a finding points at
    "the turn where this claim could have been verified".
    """
    chunks: list[_Chunk] = []
    for conv_id in corpus.conversations:
        turns = corpus.turns_by_conversation.get(conv_id, ())
        for i in range(len(turns) - 1):
            u = turns[i]
            a = turns[i + 1]
            if u.role is not Role.USER or a.role is not Role.ASSISTANT:
                continue
            text = f"{u.content}\n\n---\n\n{a.content}".strip()
            if not text:
                continue
            chunks.append(
                _Chunk(
                    chunk_id=chunk_id(text),
                    conversation_id=conv_id,
                    anchor_turn_id=u.id,
                    chunk_text=text,
                )
            )
    return chunks


def _memory_chunks(memory_text: str, *, budget: int = MEMORY_CHUNK_BUDGET_CHARS) -> list[str]:
    """Split a memory text into chunks within the extraction budget.

    Tries to cut on paragraph boundaries; falls back to char-budget cuts
    for very long paragraphs. An empty input returns an empty list.
    """
    text = (memory_text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        candidate = f"{buf}\n\n{p}" if buf else p
        if len(candidate) <= budget:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        # Paragraph itself exceeds budget — slice it.
        if len(p) > budget:
            for i in range(0, len(p), budget):
                chunks.append(p[i : i + budget])
        else:
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def _iter_memory_chunks(
    memories: Iterable[MemoryFile],
) -> list[tuple[str, str]]:
    """Flatten a list of MemoryFile into (source_tag, chunk_text) pairs.

    Source tag is ``conversations_memory`` or ``project_memories.<uuid>``
    so the Finding's metadata can trace each claim back to its origin.
    """
    out: list[tuple[str, str]] = []
    for mf in memories:
        if mf.conversations_memory:
            for chunk in _memory_chunks(mf.conversations_memory):
                out.append(("conversations_memory", chunk))
        for uuid, text in mf.project_memories.items():
            for chunk in _memory_chunks(text):
                out.append((f"project_memories.{uuid}", chunk))
    return out


_PROJECT_SOURCE_PREFIX = "project_memories."


def _project_uuid_from_source(memory_source: str) -> str | None:
    """Return the project UUID for a project-scoped memory source, else None."""
    if memory_source.startswith(_PROJECT_SOURCE_PREFIX):
        return memory_source[len(_PROJECT_SOURCE_PREFIX):]
    return None


def _build_project_attribution(
    corpus: ModuleCorpus,
    projects: Sequence[Project],
) -> dict[str, set[str]]:
    """Fuzzy-map project UUIDs → conversation ids in the audited corpus.

    Claude.ai exports don't carry ``project_uuid`` on conversations
    directly (per CLAUDE.md), so attribution is fuzzy: a project is
    considered "represented" in the corpus if any conversation's title
    or summary contains the project name (case-insensitive). Claude Code
    conversations have no project-UUID association at all — they always
    return empty here.

    Result is a dict keyed by project UUID. Absent keys mean the project
    has no conversations in the corpus (i.e. memories scoped to it are
    out-of-scope for this audit).
    """
    attribution: dict[str, set[str]] = {}
    for project in projects:
        name_lower = project.name.lower().strip()
        if not name_lower:
            continue
        matched_convs: set[str] = set()
        for conv_id, conv in corpus.conversations.items():
            haystack_parts: list[str] = []
            if conv.title:
                haystack_parts.append(conv.title.lower())
            if conv.summary:
                haystack_parts.append(conv.summary.lower())
            haystack = " \n ".join(haystack_parts)
            if not haystack:
                continue
            if name_lower in haystack:
                matched_convs.add(conv_id)
        if matched_convs:
            attribution[project.uuid] = matched_convs
    return attribution


# ──────────────────────────────────────────────────────────────────────────
# Finding construction + sub-claim rollup
# ──────────────────────────────────────────────────────────────────────────


# Order from "most decisive" (contradicted) to "least decisive"
# (insufficient-data). Parent rollup picks the first label present.
_SUPPORT_PRECEDENCE: tuple[MemorySupport, ...] = (
    MemorySupport.CONTRADICTED,
    MemorySupport.UNSUPPORTED,
    MemorySupport.WEAKLY_SUPPORTED,
    MemorySupport.WELL_SUPPORTED,
    MemorySupport.INSUFFICIENT_DATA,
)


def _rollup_subclaim_labels(
    labels: Sequence[MemorySupport],
) -> MemorySupport:
    """Roll up per-sub-claim classifications to a parent verdict.

    The parent inherits the most decisive label present. Intuition: if
    any sub-claim is contradicted, the parent is at best partially wrong
    and warrants contradicted. If none are contradicted but one is
    unsupported, the parent is partially unsupported → unsupported. And
    so on. insufficient-data sits last because it means "couldn't tell"
    which is the lowest-information label.
    """
    if not labels:
        return MemorySupport.INSUFFICIENT_DATA
    present = set(labels)
    for p in _SUPPORT_PRECEDENCE:
        if p in present:
            return p
    return MemorySupport.INSUFFICIENT_DATA


def _turn_ids_hash(turn_ids: Sequence[str]) -> str:
    return hashlib.sha256(",".join(sorted(turn_ids)).encode()).hexdigest()


def _finding_id(
    audit_run_id: str,
    memory_source: str,
    claim_id: str,
) -> str:
    raw = (f"{audit_run_id}:{MODULE_NAME.value}:{memory_source}:{claim_id}").encode()
    return hashlib.sha256(raw).hexdigest()


def _finding_for_claim(
    *,
    claim: ExtractedClaim,
    memory_source: str,
    classification: MemorySupport,
    confidence: float,
    evidence_quotes: Sequence[str],
    reasoning: str,
    top_similarity: float,
    excerpts: Sequence[Excerpt],
    sub_claim_details: list[dict[str, object]],
    audit_run_id: str,
    prompt_hash: str,
    detected_at: datetime,
) -> Finding:
    """Build a Module H Finding from a classified claim.

    Module H findings are cross-conversation. ``conversation_id`` on the
    Finding row is the top-excerpt's conversation (when classification
    cites corpus evidence) or ``None`` (when there were no excerpts or
    the claim was ``unsupported`` / ``insufficient-data`` with no
    anchor). The idempotency key is keyed on the claim id rather than
    the conversation, so one Finding per (run, memory_source, claim).
    """
    primary_conv = excerpts[0].conversation_id if excerpts else None
    turn_ids = [e.turn_id for e in excerpts[:3]]
    return Finding(
        id=_finding_id(audit_run_id, memory_source, claim.id),
        audit_run_id=audit_run_id,
        conversation_id=primary_conv,
        turn_ids=turn_ids,
        turn_ids_hash=_turn_ids_hash(turn_ids),
        module=MODULE_NAME,
        behavior=classification.value,
        intensity=None,  # Module H uses support labels, not the 1-3 scale.
        confidence=confidence,
        quote_user=claim.text,
        quote_assistant=None,
        evidence_quotes=list(evidence_quotes),
        explanation=(
            f"Memory claim ({claim.category}): {classification.value}. "
            f"Top retrieval similarity {top_similarity:.2f} over "
            f"{len(excerpts)} excerpts."
        ),
        citation=CITATION_MODULE_H,
        detected_by=[MODEL_OPUS],
        detected_at=detected_at,
        prompt_version=CLASSIFY_PROMPT_VERSION,
        prompt_hash=prompt_hash,
        metadata={
            "memory_source": memory_source,
            "claim_id": claim.id,
            "claim_category": claim.category,
            "claim_source_span": claim.source_span,
            "top_similarity": round(top_similarity, 4),
            "reasoning": reasoning,
            "sub_claim_details": sub_claim_details,
        },
    )


# ──────────────────────────────────────────────────────────────────────────
# Module class
# ──────────────────────────────────────────────────────────────────────────


class ModuleHMemory:
    """``CorpusModule`` implementation of memory-corpus consistency.

    Construct with the three inputs Module H needs: an Opus client (for
    extract + classify), a Sonnet client (for refine), an embedding
    provider (Voyage wrapper or a mock), and a store handle (to read
    memories + persist embeddings).

    Opus and Sonnet clients default to the same ``AsyncAnthropic``
    instance; the model parameter on each call steers routing. The
    separate parameter names surface the model dependency at the call
    site.
    """

    module_name: ModuleName = MODULE_NAME
    prompt_version: str = CLASSIFY_PROMPT_VERSION

    def __init__(
        self,
        *,
        opus_client: AsyncAnthropic,
        store: CorpusStore,
        embedding_provider: EmbeddingProvider,
        sonnet_client: AsyncAnthropic | None = None,
        max_concurrency: int = MAX_CONCURRENCY_DEFAULT,
        insufficient_threshold: float = INSUFFICIENT_THRESHOLD,
        top_k: int = TOP_K_DEFAULT,
        prompt_root: str | None = None,
    ) -> None:
        self._opus = opus_client
        self._sonnet = sonnet_client if sonnet_client is not None else opus_client
        self._store = store
        self._embedder = embedding_provider
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._insufficient_threshold = insufficient_threshold
        self._top_k = top_k

        kwargs: dict[str, object] = {}
        if prompt_root is not None:
            from pathlib import Path

            kwargs["root"] = Path(prompt_root)
        self._extract_prompt = load_prompt("h", EXTRACT_PROMPT_VERSION, **kwargs)  # type: ignore[arg-type]
        self._classify_prompt = load_prompt("h", CLASSIFY_PROMPT_VERSION, **kwargs)  # type: ignore[arg-type]
        self._refine_prompt = load_prompt("h", REFINE_PROMPT_VERSION, **kwargs)  # type: ignore[arg-type]

    async def run(self, corpus: ModuleCorpus) -> list[ModuleResult]:
        detected_at = datetime.now(UTC)

        memories = self._store.fetch_memory_files()
        memory_pairs = _iter_memory_chunks(memories)
        if not memory_pairs:
            _LOGGER.info("Module H: no memory text present in store; skipping run.")
            return []

        # Source-aware retrieval: project-scoped memories are verifiable
        # only when the audited corpus contains conversations from that
        # project. Otherwise the audit can't actually evaluate the claims
        # — trying to verify a project-X memory against a corpus made of
        # project-Y conversations produces spurious "unsupported" /
        # "contradicted" findings. We mark those out-of-scope instead.
        try:
            projects = self._store.fetch_all_projects()
        except Exception:
            projects = []
        project_attribution = _build_project_attribution(corpus, projects)
        _LOGGER.info(
            "Module H: project attribution maps %d of %d projects "
            "to audit-corpus conversations.",
            len(project_attribution),
            len(projects),
        )

        corpus_chunks = _corpus_chunks(corpus)
        # Partition memory sources into in-scope (need real verification)
        # and out-of-scope (emit OUT_OF_SCOPE after cheap extraction).
        in_scope_pairs: list[tuple[str, str]] = []
        out_of_scope_pairs: list[tuple[str, str]] = []
        for memory_source, memory_chunk in memory_pairs:
            project_uuid = _project_uuid_from_source(memory_source)
            if project_uuid is None:
                # conversations_memory (user-level) is always in-scope.
                in_scope_pairs.append((memory_source, memory_chunk))
                continue
            if project_uuid in project_attribution:
                in_scope_pairs.append((memory_source, memory_chunk))
            else:
                out_of_scope_pairs.append((memory_source, memory_chunk))

        results: list[ModuleResult] = []

        # Build the semantic index only when there is something to verify.
        index: _Index | None = None
        if in_scope_pairs:
            if not corpus_chunks:
                return [
                    ModuleError(
                        module=MODULE_NAME,
                        conversation_id=None,
                        error_type="empty_corpus",
                        message=(
                            "Module H has in-scope memory text but no "
                            "user-assistant turn pairs in the corpus; "
                            "verification cannot proceed."
                        ),
                    )
                ]
            try:
                index = await self._index_corpus(corpus_chunks)
            except Exception as exc:
                return [
                    ModuleError(
                        module=MODULE_NAME,
                        conversation_id=None,
                        error_type=f"index:{type(exc).__name__}",
                        message=str(exc)[:500],
                    )
                ]

        # Process out-of-scope memories first (cheap: extract, emit
        # OUT_OF_SCOPE, skip verify). These still get per-claim
        # granularity so the report can show exactly what wasn't audited.
        for memory_source, memory_chunk in out_of_scope_pairs:
            try:
                extraction = await self._call_extract(memory_chunk, memory_source)
            except Exception as exc:
                results.append(
                    ModuleError(
                        module=MODULE_NAME,
                        conversation_id=None,
                        error_type=f"extract:{type(exc).__name__}",
                        message=str(exc)[:500],
                    )
                )
                continue
            project_uuid = _project_uuid_from_source(memory_source) or "?"
            reason = (
                f"Memory belongs to project {project_uuid}, which has no "
                "conversations in this audit sample. The claim is not "
                "invalid — it simply cannot be verified against the "
                "audited corpus."
            )
            for claim in extraction.claims:
                try:
                    results.append(
                        _finding_for_claim(
                            claim=claim,
                            memory_source=memory_source,
                            classification=MemorySupport.OUT_OF_SCOPE,
                            confidence=1.0,
                            evidence_quotes=[],
                            reasoning=reason,
                            top_similarity=0.0,
                            excerpts=[],
                            sub_claim_details=[],
                            audit_run_id=corpus.audit_run_id,
                            prompt_hash=self._classify_prompt.body_hash,
                            detected_at=detected_at,
                        )
                    )
                except Exception as exc:
                    results.append(
                        ModuleError(
                            module=MODULE_NAME,
                            conversation_id=None,
                            error_type=f"finding_build:{type(exc).__name__}",
                            message=str(exc)[:500],
                        )
                    )

        # Now process in-scope memories the expensive way.
        for memory_source, memory_chunk in in_scope_pairs:
            try:
                extraction = await self._call_extract(memory_chunk, memory_source)
            except Exception as exc:
                results.append(
                    ModuleError(
                        module=MODULE_NAME,
                        conversation_id=None,
                        error_type=f"extract:{type(exc).__name__}",
                        message=str(exc)[:500],
                    )
                )
                continue

            if not extraction.claims:
                continue

            # index is guaranteed non-None when in_scope_pairs is non-empty
            # (we built it above or returned an error earlier).
            assert index is not None
            claim_results = await asyncio.gather(
                *(
                    self._verify_claim(
                        claim,
                        memory_source=memory_source,
                        index=index,
                        audit_run_id=corpus.audit_run_id,
                        detected_at=detected_at,
                    )
                    for claim in extraction.claims
                )
            )
            results.extend(claim_results)

        return results

    # ─────────────── Stage 1: index the corpus ────────────────────

    async def _index_corpus(self, chunks: Sequence[_Chunk]) -> _Index:
        """Embed chunks (with cache), build the numpy matrix, return the Index."""
        model = self._embedder.model
        dim = self._embedder.dim
        # Pull cache hits from persistent store.
        known = self._store.fetch_embeddings_for_conversations(
            [c.conversation_id for c in chunks],
            model=model,
        )
        cached_by_id: dict[str, bytes] = {row["id"]: row["vector_blob"] for row in known}

        to_embed_indices = [i for i, c in enumerate(chunks) if c.chunk_id not in cached_by_id]
        if to_embed_indices:
            texts = [chunks[i].chunk_text for i in to_embed_indices]
            result = await self._embedder.embed_batch(texts, input_type="document")
            new_rows = [
                (
                    chunks[orig_i].chunk_id,
                    chunks[orig_i].conversation_id,
                    chunks[orig_i].anchor_turn_id,
                    chunks[orig_i].chunk_text,
                    vector_to_blob(result.vectors[local_i]),
                    dim,
                    model,
                )
                for local_i, orig_i in enumerate(to_embed_indices)
            ]
            self._store.insert_embeddings(new_rows)
            # Update the in-memory map with new blobs so the matrix assembly
            # below reads from a single source.
            for local_i, orig_i in enumerate(to_embed_indices):
                cached_by_id[chunks[orig_i].chunk_id] = vector_to_blob(result.vectors[local_i])

        # Assemble the matrix in chunk order.
        matrix_rows: list[np.ndarray] = []
        chunk_order: list[_Chunk] = []
        for c in chunks:
            blob = cached_by_id.get(c.chunk_id)
            if blob is None:
                continue  # defensive; shouldn't happen
            matrix_rows.append(blob_to_vector(blob, dim=dim))
            chunk_order.append(c)
        if not matrix_rows:
            matrix = np.zeros((0, dim), dtype=np.float32)
        else:
            matrix = np.stack(matrix_rows, axis=0)
        return _Index(matrix=matrix, chunks=chunk_order, model=model, dim=dim)

    # ─────────────── Stage 2: extract claims from memory chunk ────

    async def _call_extract(
        self,
        memory_text: str,
        source: str,
    ) -> ExtractResult:
        content = await self._call_with_retry(
            self._opus,
            model=MODEL_OPUS,
            system_text=self._extract_prompt.padded_body,
            user_text=_render_extract_request(memory_text, source),
            max_tokens=MAX_EXTRACT_OUTPUT_TOKENS,
        )
        return ExtractResult.model_validate_json(extract_result_json(content))

    # ─────────────── Stage 3-6: verify one claim ──────────────────

    async def _verify_claim(
        self,
        claim: ExtractedClaim,
        *,
        memory_source: str,
        index: _Index,
        audit_run_id: str,
        detected_at: datetime,
    ) -> ModuleResult:
        async with self._semaphore:
            try:
                excerpts = await self._retrieve(claim.text, index)
            except Exception as exc:
                return ModuleError(
                    module=MODULE_NAME,
                    conversation_id=None,
                    error_type=f"retrieve:{type(exc).__name__}",
                    message=str(exc)[:500],
                )

            top_sim = excerpts[0].similarity if excerpts else 0.0

            # Short-circuit: low retrieval → insufficient-data without a classifier call.
            if top_sim < self._insufficient_threshold:
                try:
                    return _finding_for_claim(
                        claim=claim,
                        memory_source=memory_source,
                        classification=MemorySupport.INSUFFICIENT_DATA,
                        confidence=0.85,
                        evidence_quotes=[],
                        reasoning=(
                            f"Retrieval top-similarity {top_sim:.3f} is below "
                            f"{self._insufficient_threshold}; no excerpt is similar "
                            "enough to support verification."
                        ),
                        top_similarity=top_sim,
                        excerpts=excerpts,
                        sub_claim_details=[],
                        audit_run_id=audit_run_id,
                        prompt_hash=self._classify_prompt.body_hash,
                        detected_at=detected_at,
                    )
                except Exception as exc:
                    return ModuleError(
                        module=MODULE_NAME,
                        conversation_id=None,
                        error_type=f"finding_build:{type(exc).__name__}",
                        message=str(exc)[:500],
                    )

            try:
                classification = await self._call_classify(
                    _ClassifyInput(
                        claim_text=claim.text,
                        claim_category=claim.category,
                        excerpts=excerpts,
                    )
                )
            except Exception as exc:
                return ModuleError(
                    module=MODULE_NAME,
                    conversation_id=None,
                    error_type=f"classify:{type(exc).__name__}",
                    message=str(exc)[:500],
                )

            # Two-stage verification path: classifier said insufficient but we
            # have related context. Decompose + re-verify each sub-claim.
            sub_claim_details: list[dict[str, object]] = []
            final_label = classification.classification
            final_confidence = classification.confidence
            final_evidence = list(classification.evidence_quotes)
            final_reasoning = classification.reasoning

            if (
                classification.classification is MemorySupport.INSUFFICIENT_DATA
                and excerpts
                and top_sim >= self._insufficient_threshold
            ):
                try:
                    refine = await self._call_refine(
                        claim,
                        first_pass_top_sim=top_sim,
                        first_pass_top_preview=excerpts[0].chunk_text if excerpts else "",
                    )
                except Exception as exc:
                    # Refine failed — keep the first-pass verdict.
                    sub_claim_details.append(
                        {"refine_error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                    )
                else:
                    sub_labels: list[MemorySupport] = []
                    for sub in refine.sub_claims:
                        try:
                            sub_excerpts = await self._retrieve(sub.text, index)
                        except Exception as exc:
                            sub_claim_details.append(
                                {
                                    "sub_claim_id": sub.id,
                                    "error": (
                                        f"sub_retrieve:{type(exc).__name__}: {str(exc)[:200]}"
                                    ),
                                }
                            )
                            continue
                        sub_top = sub_excerpts[0].similarity if sub_excerpts else 0.0
                        if sub_top < self._insufficient_threshold:
                            sub_labels.append(MemorySupport.INSUFFICIENT_DATA)
                            sub_claim_details.append(
                                {
                                    "sub_claim_id": sub.id,
                                    "sub_claim_text": sub.text,
                                    "aspect": sub.aspect,
                                    "top_similarity": round(sub_top, 4),
                                    "classification": MemorySupport.INSUFFICIENT_DATA.value,
                                }
                            )
                            continue
                        try:
                            sub_score = await self._call_classify(
                                _ClassifyInput(
                                    claim_text=sub.text,
                                    claim_category=sub.category,
                                    excerpts=sub_excerpts,
                                )
                            )
                        except Exception as exc:
                            sub_claim_details.append(
                                {
                                    "sub_claim_id": sub.id,
                                    "error": (
                                        f"sub_classify:{type(exc).__name__}: {str(exc)[:200]}"
                                    ),
                                }
                            )
                            continue
                        sub_labels.append(sub_score.classification)
                        sub_claim_details.append(
                            {
                                "sub_claim_id": sub.id,
                                "sub_claim_text": sub.text,
                                "aspect": sub.aspect,
                                "top_similarity": round(sub_top, 4),
                                "classification": sub_score.classification.value,
                                "confidence": sub_score.confidence,
                                "evidence_quotes": list(sub_score.evidence_quotes),
                            }
                        )
                    if sub_labels:
                        rolled = _rollup_subclaim_labels(sub_labels)
                        # Only overwrite the first-pass verdict if the roll-up
                        # is more informative (anything other than insufficient).
                        if rolled is not MemorySupport.INSUFFICIENT_DATA:
                            final_label = rolled
                            final_confidence = max(0.55, min(0.9, final_confidence))
                            rolled_evidence: list[str] = []
                            for detail in sub_claim_details:
                                quotes = detail.get("evidence_quotes")
                                if not isinstance(quotes, list):
                                    continue
                                for q in quotes:
                                    if isinstance(q, str):
                                        rolled_evidence.append(q)
                            final_evidence = rolled_evidence[:3]
                            final_reasoning = (
                                f"{classification.reasoning} Two-stage refinement "
                                f"rolled up to {rolled.value} across "
                                f"{len(sub_labels)} sub-claims."
                            )

            try:
                return _finding_for_claim(
                    claim=claim,
                    memory_source=memory_source,
                    classification=final_label,
                    confidence=final_confidence,
                    evidence_quotes=final_evidence,
                    reasoning=final_reasoning,
                    top_similarity=top_sim,
                    excerpts=excerpts,
                    sub_claim_details=sub_claim_details,
                    audit_run_id=audit_run_id,
                    prompt_hash=self._classify_prompt.body_hash,
                    detected_at=detected_at,
                )
            except Exception as exc:
                return ModuleError(
                    module=MODULE_NAME,
                    conversation_id=None,
                    error_type=f"finding_build:{type(exc).__name__}",
                    message=str(exc)[:500],
                )

    # ─────────────── Retrieval ────────────────────────────────────

    async def _retrieve(
        self,
        query_text: str,
        index: _Index,
    ) -> list[Excerpt]:
        if index.matrix.shape[0] == 0:
            return []
        query_result = await self._embedder.embed_batch([query_text], input_type="query")
        query_vec = query_result.vectors[0]
        sims = cosine_similarity_matrix(
            query_vec,
            index.matrix,
            corpus_norm=index.corpus_norms,
        )
        top_idx = top_k_indices(sims, k=self._top_k)
        out: list[Excerpt] = []
        for i in top_idx.tolist():
            chunk = index.chunks[i]
            out.append(
                Excerpt(
                    conversation_id=chunk.conversation_id,
                    turn_id=chunk.anchor_turn_id,
                    chunk_text=chunk.chunk_text,
                    similarity=float(sims[i]),
                )
            )
        return out

    # ─────────────── LLM call wrappers ────────────────────────────

    async def _call_classify(self, inp: _ClassifyInput) -> ClassifyScore:
        content = await self._call_with_retry(
            self._opus,
            model=MODEL_OPUS,
            system_text=self._classify_prompt.padded_body,
            user_text=_render_classify_request(inp),
            max_tokens=MAX_CLASSIFY_OUTPUT_TOKENS,
        )
        return ClassifyScore.model_validate_json(extract_result_json(content))

    async def _call_refine(
        self,
        claim: ExtractedClaim,
        *,
        first_pass_top_sim: float,
        first_pass_top_preview: str,
    ) -> RefineResult:
        content = await self._call_with_retry(
            self._sonnet,
            model=MODEL_SONNET,
            system_text=self._refine_prompt.padded_body,
            user_text=_render_refine_request(
                claim,
                first_pass_top_sim=first_pass_top_sim,
                first_pass_top_preview=first_pass_top_preview,
            ),
            max_tokens=MAX_REFINE_OUTPUT_TOKENS,
        )
        return RefineResult.model_validate_json(extract_result_json(content))

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


# ──────────────────────────────────────────────────────────────────────────
# Index dataclass (private helper)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Index:
    """Corpus embedding matrix + metadata for retrieval.

    ``corpus_norms`` is pre-computed for the matrix so every claim's
    retrieval skips the O(N*dim) norm calculation.
    """

    matrix: np.ndarray  # (N, dim) float32
    chunks: list[_Chunk]
    model: str
    dim: int

    @property
    def corpus_norms(self) -> np.ndarray:
        norms: np.ndarray = np.linalg.norm(self.matrix, axis=1).astype(np.float32)
        return norms


# Unused import guard — `Turn` is imported for callers reading chunks in tests.
_ = Turn
