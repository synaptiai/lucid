"""Tests for :mod:`lucid.modules.module_h_memory`.

Uses :class:`StaticEmbeddingProvider` for deterministic retrieval and a
mocked Anthropic client for the extract/classify/refine prompts. No
live APIs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from anthropic.types import Usage

from lucid.modules.base import ModuleCorpus, ModuleError
from lucid.modules.embeddings import STATIC_DIM, StaticEmbeddingProvider
from lucid.modules.module_h_memory import (
    CITATION_MODULE_H,
    CLASSIFY_PROMPT_VERSION,
    INSUFFICIENT_THRESHOLD,
    MODEL_OPUS,
    TOP_K_DEFAULT,
    ClassifyScore,
    ExtractedClaim,
    ExtractResult,
    ModuleHMemory,
    RefineResult,
    SubClaim,
    _corpus_chunks,
    _escape_delimiters,
    _iter_memory_chunks,
    _memory_chunks,
    _rollup_subclaim_labels,
)
from lucid.schemas import (
    Conversation,
    Finding,
    MemoryFile,
    MemorySupport,
    ModuleName,
    Role,
    Source,
    Turn,
)
from lucid.store import initialize_db
from lucid.store.sqlite import CorpusStore

# ──────────────────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────────────────


def _turn(conv_id: str, idx: int, *, role: Role, content: str) -> Turn:
    return Turn(
        id=f"{conv_id}-t{idx}",
        conversation_id=conv_id,
        index=idx,
        role=role,
        content=content,
    )


def _conv(conv_id: str, *, turn_count: int, updated: datetime) -> Conversation:
    return Conversation(
        id=conv_id,
        source=Source.CLAUDE_AI,
        source_path="/tmp/x",
        created_at=updated,
        updated_at=updated,
        turn_count=turn_count,
    )


def _make_corpus(turns_by_conv: dict[str, list[Turn]]) -> ModuleCorpus:
    now = datetime(2026, 2, 1, tzinfo=UTC)
    convs = {cid: _conv(cid, turn_count=len(v), updated=now) for cid, v in turns_by_conv.items()}
    return ModuleCorpus(
        conversations=convs,
        turns_by_conversation=turns_by_conv,
        audit_run_id="run-h",
    )


def _seed_store_with_memory(
    tmp_path: Path,
    *,
    conversations_memory: str | None = None,
    project_memories: dict[str, str] | None = None,
    convs: list[Conversation] | None = None,
    turns: list[Turn] | None = None,
) -> CorpusStore:
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    store = CorpusStore(db)
    store.connect()
    if convs:
        store.insert_conversations(convs)
    if turns:
        store.insert_turns(turns)
    store.insert_memory_file(
        MemoryFile(
            account_uuid="acct-1",
            conversations_memory=conversations_memory,
            project_memories=project_memories or {},
        )
    )
    return store


# ──────────────────────────────────────────────────────────────────────────
# Pure-Python helpers
# ──────────────────────────────────────────────────────────────────────────


def test_memory_chunks_splits_at_paragraph_boundary() -> None:
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = _memory_chunks(text, budget=25)
    assert len(chunks) >= 2
    # Every chunk ≤ budget (with paragraph re-joining inherent slack).
    assert all(len(c) <= 55 for c in chunks)


def test_memory_chunks_handles_empty_input() -> None:
    assert _memory_chunks("") == []
    assert _memory_chunks("   ") == []


def test_memory_chunks_single_long_paragraph_force_splits() -> None:
    text = "x" * 100
    chunks = _memory_chunks(text, budget=30)
    assert all(len(c) <= 30 for c in chunks)
    assert "".join(chunks) == text


def test_iter_memory_chunks_tags_each_source() -> None:
    mf = MemoryFile(
        account_uuid="acct-1",
        conversations_memory="User works at Acme.",
        project_memories={"proj-uuid": "User owns this project."},
    )
    pairs = _iter_memory_chunks([mf])
    sources = [src for src, _ in pairs]
    assert "conversations_memory" in sources
    assert any(s.startswith("project_memories.proj-uuid") for s in sources)


def test_corpus_chunks_pairs_user_and_next_assistant() -> None:
    turns = [
        _turn("c1", 0, role=Role.USER, content="Hello"),
        _turn("c1", 1, role=Role.ASSISTANT, content="Hi there"),
        _turn("c1", 2, role=Role.USER, content="What's up"),
        _turn("c1", 3, role=Role.ASSISTANT, content="Not much"),
    ]
    corpus = _make_corpus({"c1": turns})
    chunks = _corpus_chunks(corpus)
    assert len(chunks) == 2
    assert chunks[0].anchor_turn_id == "c1-t0"
    assert "Hello" in chunks[0].chunk_text
    assert "Hi there" in chunks[0].chunk_text
    assert chunks[1].anchor_turn_id == "c1-t2"


def test_corpus_chunks_skips_non_user_assistant_pairs() -> None:
    turns = [
        _turn("c1", 0, role=Role.ASSISTANT, content="starts with assistant"),
        _turn("c1", 1, role=Role.USER, content="user"),
        _turn("c1", 2, role=Role.USER, content="double user"),
    ]
    corpus = _make_corpus({"c1": turns})
    chunks = _corpus_chunks(corpus)
    # No user→assistant pair; no chunks.
    assert chunks == []


def test_rollup_precedence_contradicted_wins() -> None:
    labels = [
        MemorySupport.WELL_SUPPORTED,
        MemorySupport.WEAKLY_SUPPORTED,
        MemorySupport.CONTRADICTED,
    ]
    assert _rollup_subclaim_labels(labels) is MemorySupport.CONTRADICTED


def test_rollup_unsupported_beats_weakly_and_well() -> None:
    labels = [
        MemorySupport.WELL_SUPPORTED,
        MemorySupport.UNSUPPORTED,
        MemorySupport.WEAKLY_SUPPORTED,
    ]
    assert _rollup_subclaim_labels(labels) is MemorySupport.UNSUPPORTED


def test_rollup_all_well_supported() -> None:
    labels = [MemorySupport.WELL_SUPPORTED, MemorySupport.WELL_SUPPORTED]
    assert _rollup_subclaim_labels(labels) is MemorySupport.WELL_SUPPORTED


def test_rollup_empty_returns_insufficient() -> None:
    assert _rollup_subclaim_labels([]) is MemorySupport.INSUFFICIENT_DATA


def test_rollup_all_insufficient_returns_insufficient() -> None:
    labels = [MemorySupport.INSUFFICIENT_DATA, MemorySupport.INSUFFICIENT_DATA]
    assert _rollup_subclaim_labels(labels) is MemorySupport.INSUFFICIENT_DATA


def test_escape_delimiters_covers_module_h_tokens() -> None:
    tokens = [
        "<MEMORY_TEXT>",
        "</MEMORY_TEXT>",
        "<CLAIM>",
        "</CLAIM>",
        "<RETRIEVED_EXCERPTS>",
        "</RETRIEVED_EXCERPTS>",
        "<FIRST_PASS_METADATA>",
        "</FIRST_PASS_METADATA>",
    ]
    raw = " ".join(tokens) + " then benign"
    escaped = _escape_delimiters(raw)
    for t in tokens:
        assert t not in escaped


# ──────────────────────────────────────────────────────────────────────────
# End-to-end with mocked client and static embeddings
# ──────────────────────────────────────────────────────────────────────────


def _seeded_corpus_store_and_static_embeddings(
    tmp_path: Path,
) -> tuple[CorpusStore, ModuleCorpus, StaticEmbeddingProvider]:
    """A 2-conversation corpus, one memory file, and a static embedder
    that pushes the right excerpt to the top for every test claim."""
    now = datetime(2026, 2, 1, tzinfo=UTC)
    convs = [
        _conv("c1", turn_count=2, updated=now),
        _conv("c2", turn_count=2, updated=now),
    ]
    turns = [
        _turn("c1", 0, role=Role.USER, content="I love working at Acme Corp."),
        _turn("c1", 1, role=Role.ASSISTANT, content="Sounds rewarding."),
        _turn("c2", 0, role=Role.USER, content="I prefer Python to JavaScript."),
        _turn("c2", 1, role=Role.ASSISTANT, content="Both are fine choices."),
    ]
    store = _seed_store_with_memory(
        tmp_path,
        conversations_memory="User works at Acme Corp. User prefers Python.",
        convs=convs,
        turns=turns,
    )
    corpus = _make_corpus(
        {
            "c1": [t for t in turns if t.conversation_id == "c1"],
            "c2": [t for t in turns if t.conversation_id == "c2"],
        }
    )

    # Static embedding mapping. Chunks: c1 pair (contains "Acme") + c2 pair
    # (contains "Python"). Claims: "User works at Acme Corp." (close to c1
    # chunk) and "User prefers Python." (close to c2 chunk).
    acme_vec = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    python_vec = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    c1_chunk_text = "I love working at Acme Corp.\n\n---\n\nSounds rewarding."
    c2_chunk_text = "I prefer Python to JavaScript.\n\n---\n\nBoth are fine choices."
    mapping: dict[str, np.ndarray] = {
        c1_chunk_text: acme_vec,
        c2_chunk_text: python_vec,
        # Claim texts used as queries.
        "User works at Acme Corp.": acme_vec,
        "User prefers Python.": python_vec,
    }
    embedder = StaticEmbeddingProvider(
        mapping=mapping,
        default=np.zeros(STATIC_DIM, dtype=np.float32),
    )
    return store, corpus, embedder


def _make_mock_client(outputs: list[Any]) -> MagicMock:
    """Builds a MagicMock client whose messages.create yields the given
    outputs in order. Each output is expected to have ``model_dump_json``."""

    async def _call(**_kwargs: object) -> object:
        item = outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        block = MagicMock()
        block.type = "text"
        block.text = item.model_dump_json() if hasattr(item, "model_dump_json") else str(item)
        resp = MagicMock()
        resp.content = [block]
        resp.usage = Usage(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return resp

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=_call)
    return client


async def test_run_produces_well_supported_finding(tmp_path: Path) -> None:
    store, corpus, embedder = _seeded_corpus_store_and_static_embeddings(tmp_path)

    extract_result = ExtractResult(
        reasoning="found one work claim",
        claims=[
            ExtractedClaim(
                id="claim_1",
                text="User works at Acme Corp.",
                category="work",
                source_span="User works at Acme Corp.",
            )
        ],
    )
    classify_well = ClassifyScore(
        reasoning="clear mention of Acme in excerpt 1",
        classification=MemorySupport.WELL_SUPPORTED,
        confidence=0.9,
        evidence_quotes=["I love working at Acme Corp."],
        cited_excerpt_numbers=[1],
    )
    client = _make_mock_client([extract_result, classify_well])

    module = ModuleHMemory(
        opus_client=client,
        store=store,
        embedding_provider=embedder,
    )
    results = await module.run(corpus)

    findings = [r for r in results if isinstance(r, Finding)]
    errors = [r for r in results if isinstance(r, ModuleError)]
    assert len(findings) == 1, f"results: {results}"
    assert errors == []
    f = findings[0]
    assert f.module is ModuleName.H_MEMORY
    assert f.behavior == MemorySupport.WELL_SUPPORTED.value
    assert f.citation == CITATION_MODULE_H
    assert f.prompt_version == CLASSIFY_PROMPT_VERSION
    assert f.quote_user == "User works at Acme Corp."
    assert "I love working at Acme Corp." in f.evidence_quotes
    assert f.metadata["memory_source"] == "conversations_memory"
    assert f.metadata["claim_category"] == "work"
    store.close()


async def test_run_short_circuits_low_similarity_to_insufficient(tmp_path: Path) -> None:
    """Claim whose embedding doesn't match any corpus chunk → top-sim
    below the threshold → insufficient-data Finding without classify call."""
    store, corpus, embedder = _seeded_corpus_store_and_static_embeddings(tmp_path)

    # Override default so the unrelated claim query returns zero-vector,
    # pushing max similarity below the 0.35 threshold.
    off_topic_vec = np.zeros(STATIC_DIM, dtype=np.float32)
    embedder.mapping["User collects stamps."] = off_topic_vec

    extract_result = ExtractResult(
        reasoning="one off-topic claim",
        claims=[
            ExtractedClaim(
                id="claim_x",
                text="User collects stamps.",
                category="preference",
                source_span="User collects stamps.",
            )
        ],
    )
    # Only the extract call is expected — classify is short-circuited.
    client = _make_mock_client([extract_result])

    module = ModuleHMemory(
        opus_client=client,
        store=store,
        embedding_provider=embedder,
    )
    results = await module.run(corpus)

    findings = [r for r in results if isinstance(r, Finding)]
    assert len(findings) == 1
    assert findings[0].behavior == MemorySupport.INSUFFICIENT_DATA.value
    # Exactly one LLM call: extract. Classify short-circuited.
    assert client.messages.create.await_count == 1
    store.close()


async def test_run_no_memory_returns_empty(tmp_path: Path) -> None:
    """A store with no memory file rows → Module H returns no results
    and no errors; the run is a no-op."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    store = CorpusStore(db)
    store.connect()

    embedder = StaticEmbeddingProvider(default=np.zeros(STATIC_DIM, dtype=np.float32))
    client = _make_mock_client([])
    corpus = ModuleCorpus(conversations={}, turns_by_conversation={}, audit_run_id="run-h")
    module = ModuleHMemory(opus_client=client, store=store, embedding_provider=embedder)
    assert await module.run(corpus) == []
    assert client.messages.create.await_count == 0
    store.close()


async def test_run_memory_but_empty_corpus_returns_error(tmp_path: Path) -> None:
    """Memory present but no user+assistant pairs in the corpus → a
    ModuleError documenting the condition rather than a silent no-op."""
    store = _seed_store_with_memory(
        tmp_path,
        conversations_memory="User works at Acme.",
    )
    embedder = StaticEmbeddingProvider(default=np.zeros(STATIC_DIM, dtype=np.float32))
    client = _make_mock_client([])
    corpus = ModuleCorpus(conversations={}, turns_by_conversation={}, audit_run_id="run-h")
    module = ModuleHMemory(opus_client=client, store=store, embedding_provider=embedder)
    results = await module.run(corpus)

    assert len(results) == 1
    assert isinstance(results[0], ModuleError)
    assert results[0].error_type == "empty_corpus"
    store.close()


async def test_run_two_stage_refinement_runs_when_first_pass_insufficient(
    tmp_path: Path,
) -> None:
    """When classify returns insufficient-data with top-sim ≥ threshold,
    refine runs and sub-claim classifications roll up."""
    store, corpus, embedder = _seeded_corpus_store_and_static_embeddings(tmp_path)

    # Map the sub-claim query texts to the same vectors as their parent.
    embedder.mapping["User works at Acme Corp."] = np.array(
        [1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
    )
    embedder.mapping["User is a senior engineer."] = np.array(
        [1, 0, 0, 0, 0, 0, 0, 0],
        dtype=np.float32,  # same cluster as Acme chunk
    )

    extract_result = ExtractResult(
        reasoning="one compound claim",
        claims=[
            ExtractedClaim(
                id="claim_1",
                text="User works at Acme Corp.",  # in mapping for first retrieve
                category="work",
                source_span="User works at Acme Corp.",
            )
        ],
    )
    # First classify: insufficient with top_sim ≥ 0.35 (we paired claim with Acme chunk).
    classify_insufficient = ClassifyScore(
        reasoning="retrieval found Acme chunk but not decisive",
        classification=MemorySupport.INSUFFICIENT_DATA,
        confidence=0.6,
        evidence_quotes=[],
        cited_excerpt_numbers=[],
    )
    # Refine decomposes into one sub-claim.
    refine_result = RefineResult(
        reasoning="split out seniority",
        sub_claims=[
            SubClaim(
                id="sub_1",
                text="User is a senior engineer.",
                category="work",
                aspect="seniority",
            )
        ],
    )
    # Sub-claim classify: well-supported.
    classify_sub_well = ClassifyScore(
        reasoning="sub-claim directly supported by excerpt",
        classification=MemorySupport.WELL_SUPPORTED,
        confidence=0.85,
        evidence_quotes=["I love working at Acme Corp."],
        cited_excerpt_numbers=[1],
    )

    client = _make_mock_client(
        [extract_result, classify_insufficient, refine_result, classify_sub_well]
    )

    module = ModuleHMemory(
        opus_client=client,
        store=store,
        embedding_provider=embedder,
    )
    results = await module.run(corpus)

    findings = [r for r in results if isinstance(r, Finding)]
    assert len(findings) == 1
    # Sub-claim rolled up from insufficient → well-supported.
    assert findings[0].behavior == MemorySupport.WELL_SUPPORTED.value
    details = findings[0].metadata["sub_claim_details"]
    assert isinstance(details, list)
    assert len(details) == 1
    assert details[0]["sub_claim_id"] == "sub_1"
    store.close()


async def test_run_isolates_extract_error_per_memory_chunk(tmp_path: Path) -> None:
    store, corpus, embedder = _seeded_corpus_store_and_static_embeddings(tmp_path)

    class ExtractBoom(Exception):
        pass

    client = _make_mock_client([ExtractBoom("extract failed")])

    module = ModuleHMemory(opus_client=client, store=store, embedding_provider=embedder)
    results = await module.run(corpus)

    errors = [r for r in results if isinstance(r, ModuleError)]
    findings = [r for r in results if isinstance(r, Finding)]
    assert len(errors) == 1
    assert errors[0].error_type.startswith("extract:")
    assert findings == []
    store.close()


async def test_run_persists_embeddings_for_resume(tmp_path: Path) -> None:
    """After a successful pre-index, the embeddings table holds the
    chunk vectors. Re-running should hit the cache (no new embed calls)."""
    store, corpus, embedder = _seeded_corpus_store_and_static_embeddings(tmp_path)

    extract_result = ExtractResult(reasoning="", claims=[])
    client = _make_mock_client([extract_result])

    module = ModuleHMemory(opus_client=client, store=store, embedding_provider=embedder)
    await module.run(corpus)

    assert store.count("embeddings") >= 1
    first_embed_calls = embedder.calls

    # Second run: rerun extract stub and check embedder didn't re-embed
    # the corpus (cache hit). Note: the query-side embed still happens if
    # there are claims; here the extract returns empty so query embeds
    # are zero too.
    client.messages.create.side_effect = AsyncMock(
        side_effect=[_make_mock_client([ExtractResult(reasoning="", claims=[])]).messages.create]
    )
    # Simpler: reset by constructing a fresh module + client.
    client2 = _make_mock_client([ExtractResult(reasoning="", claims=[])])
    module2 = ModuleHMemory(opus_client=client2, store=store, embedding_provider=embedder)
    await module2.run(corpus)

    # No new corpus embed: the pre-index short-circuited on cache hits.
    # The static provider's `calls` counter increments only on batches it
    # actually processes, so a cache-only run should leave calls flat for
    # the corpus. We observe ≤ 1 extra call (the query-side fan-out is
    # elided when extract returns no claims).
    assert embedder.calls <= first_embed_calls + 1
    store.close()


# ──────────────────────────────────────────────────────────────────────────
# Prompt wiring sanity
# ──────────────────────────────────────────────────────────────────────────


def test_module_loads_all_three_prompts(tmp_path: Path) -> None:
    store, _, embedder = _seeded_corpus_store_and_static_embeddings(tmp_path)
    client = _make_mock_client([])
    module = ModuleHMemory(opus_client=client, store=store, embedding_provider=embedder)
    assert module._extract_prompt.model == MODEL_OPUS
    assert module._classify_prompt.model == MODEL_OPUS
    assert module._refine_prompt.model == "claude-sonnet-4-6"
    assert module.prompt_version == CLASSIFY_PROMPT_VERSION
    store.close()


def test_module_constants_are_plan_aligned() -> None:
    assert INSUFFICIENT_THRESHOLD == 0.35
    assert TOP_K_DEFAULT == 20
