"""Module H — six-verdict adversarial fixture suite.

Module H emits one of six :class:`MemorySupport` verdicts per memory
claim. Each verdict has a targeted end-to-end test below: a small
(memory, corpus) pair designed so the verdict is the only honest
answer, with mocked Anthropic client + static embeddings to keep the
outcome deterministic.

These tests probe the **plumbing** for each verdict (right code path,
right Finding fields). They do not measure the classifier's decision
boundary; that requires calibration data — see ``docs/calibration.md``.

See ``tests/fixtures/module_h_verdicts/README.md`` for the suite
description referenced from the project README's *Honest limitations*
section.
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
    ClassifyScore,
    ExtractedClaim,
    ExtractResult,
    ModuleHMemory,
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
# Test helpers (kept local to this file so the verdict suite is
# self-contained and readable as a single artefact alongside the
# fixtures README)
# ──────────────────────────────────────────────────────────────────────────


_NOW = datetime(2026, 2, 1, tzinfo=UTC)


def _turn(conv_id: str, idx: int, *, role: Role, content: str) -> Turn:
    return Turn(
        id=f"{conv_id}-t{idx}",
        conversation_id=conv_id,
        index=idx,
        role=role,
        content=content,
    )


def _conv(conv_id: str, *, turn_count: int) -> Conversation:
    return Conversation(
        id=conv_id,
        source=Source.CLAUDE_AI,
        source_path="/tmp/x",
        created_at=_NOW,
        updated_at=_NOW,
        turn_count=turn_count,
    )


def _make_corpus(turns_by_conv: dict[str, list[Turn]]) -> ModuleCorpus:
    convs = {cid: _conv(cid, turn_count=len(v)) for cid, v in turns_by_conv.items()}
    return ModuleCorpus(
        conversations=convs,
        turns_by_conversation=turns_by_conv,
        audit_run_id="run-h-verdicts",
    )


def _make_mock_client(outputs: list[Any]) -> MagicMock:
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


def _seed_store(
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


def _single_claim_corpus(
    tmp_path: Path,
    *,
    user_text: str,
    assistant_text: str,
    memory_text: str,
    claim_text: str,
) -> tuple[CorpusStore, ModuleCorpus, StaticEmbeddingProvider]:
    """Single user/assistant pair, single conversations_memory text, one
    extracted claim. The static embedder is rigged so the claim's query
    embedding maps to the same vector as the corpus chunk — top-1
    similarity comes out at 1.0, cleanly above ``INSUFFICIENT_THRESHOLD``,
    so the classifier path runs."""
    convs = [_conv("c1", turn_count=2)]
    turns = [
        _turn("c1", 0, role=Role.USER, content=user_text),
        _turn("c1", 1, role=Role.ASSISTANT, content=assistant_text),
    ]
    store = _seed_store(
        tmp_path,
        conversations_memory=memory_text,
        convs=convs,
        turns=turns,
    )
    corpus = _make_corpus({"c1": turns})

    anchor_vec = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    chunk_text = f"{user_text}\n\n---\n\n{assistant_text}"
    mapping: dict[str, np.ndarray] = {
        chunk_text: anchor_vec,
        claim_text: anchor_vec,
    }
    embedder = StaticEmbeddingProvider(
        mapping=mapping,
        default=np.zeros(STATIC_DIM, dtype=np.float32),
    )
    return store, corpus, embedder


def _extract_one(claim_text: str, *, claim_id: str = "claim_1") -> ExtractResult:
    return ExtractResult(
        reasoning="single claim extracted",
        claims=[
            ExtractedClaim(
                id=claim_id,
                text=claim_text,
                category="work",
                source_span=claim_text,
            )
        ],
    )


# ──────────────────────────────────────────────────────────────────────────
# The six verdict tests
# ──────────────────────────────────────────────────────────────────────────


async def test_well_supported_verdict(tmp_path: Path) -> None:
    """Memory claim has a near-verbatim anchor; classifier returns
    WELL_SUPPORTED. Plumbing must surface the verdict + evidence."""
    store, corpus, embedder = _single_claim_corpus(
        tmp_path,
        user_text="I love working at Acme Corp.",
        assistant_text="Sounds rewarding.",
        memory_text="User works at Acme Corp.",
        claim_text="User works at Acme Corp.",
    )
    classify = ClassifyScore(
        reasoning="excerpt 1 directly states user works at Acme",
        classification=MemorySupport.WELL_SUPPORTED,
        confidence=0.92,
        evidence_quotes=["I love working at Acme Corp."],
        cited_excerpt_numbers=[1],
    )
    client = _make_mock_client([_extract_one("User works at Acme Corp."), classify])
    module = ModuleHMemory(opus_client=client, store=store, embedding_provider=embedder)

    findings = [r for r in await module.run(corpus) if isinstance(r, Finding)]
    assert len(findings) == 1
    assert findings[0].behavior == MemorySupport.WELL_SUPPORTED.value
    assert findings[0].module is ModuleName.H_MEMORY
    assert "I love working at Acme Corp." in findings[0].evidence_quotes
    store.close()


async def test_weakly_supported_verdict(tmp_path: Path) -> None:
    """Anchor exists but only partially supports the claim; classifier
    returns WEAKLY_SUPPORTED. The verdict must propagate with the
    classifier's lower confidence intact."""
    store, corpus, embedder = _single_claim_corpus(
        tmp_path,
        user_text="I sometimes mention I work in tech.",
        assistant_text="Got it.",
        memory_text="User is a senior software engineer at Acme Corp.",
        claim_text="User is a senior software engineer at Acme Corp.",
    )
    classify = ClassifyScore(
        reasoning=(
            "excerpt mentions tech work but does not establish 'senior', "
            "'software engineer', or 'Acme'"
        ),
        classification=MemorySupport.WEAKLY_SUPPORTED,
        confidence=0.55,
        evidence_quotes=["I sometimes mention I work in tech."],
        cited_excerpt_numbers=[1],
    )
    client = _make_mock_client(
        [_extract_one("User is a senior software engineer at Acme Corp."), classify]
    )
    module = ModuleHMemory(opus_client=client, store=store, embedding_provider=embedder)

    findings = [r for r in await module.run(corpus) if isinstance(r, Finding)]
    assert len(findings) == 1
    assert findings[0].behavior == MemorySupport.WEAKLY_SUPPORTED.value
    assert findings[0].confidence == 0.55
    store.close()


async def test_unsupported_verdict(tmp_path: Path) -> None:
    """Memory claim has a same-topic anchor in the corpus (so the
    similarity short-circuit doesn't fire) but the anchor doesn't back
    the claim; classifier returns UNSUPPORTED."""
    store, corpus, embedder = _single_claim_corpus(
        tmp_path,
        user_text="I asked about Python performance benchmarks.",
        assistant_text="Here are typical results.",
        memory_text="User has 10 years of professional Rust experience.",
        claim_text="User has 10 years of professional Rust experience.",
    )
    classify = ClassifyScore(
        reasoning=(
            "excerpt is about Python; no mention of Rust or years of "
            "experience anywhere in the retrieved corpus"
        ),
        classification=MemorySupport.UNSUPPORTED,
        confidence=0.85,
        evidence_quotes=[],
        cited_excerpt_numbers=[1],
    )
    client = _make_mock_client(
        [_extract_one("User has 10 years of professional Rust experience."), classify]
    )
    module = ModuleHMemory(opus_client=client, store=store, embedding_provider=embedder)

    results = await module.run(corpus)
    findings = [r for r in results if isinstance(r, Finding)]
    errors = [r for r in results if isinstance(r, ModuleError)]
    assert errors == []
    assert len(findings) == 1
    assert findings[0].behavior == MemorySupport.UNSUPPORTED.value
    store.close()


async def test_contradicted_verdict(tmp_path: Path) -> None:
    """Memory and corpus speak to the same topic but disagree; the
    classifier must return CONTRADICTED — the highest-severity verdict
    short of OUT_OF_SCOPE escapes."""
    store, corpus, embedder = _single_claim_corpus(
        tmp_path,
        user_text="I switched away from vim a year ago — emacs forever now.",
        assistant_text="Welcome to the other side.",
        memory_text="User is a long-time vim user.",
        claim_text="User is a long-time vim user.",
    )
    classify = ClassifyScore(
        reasoning=(
            "excerpt explicitly says the user switched AWAY from vim and "
            "now uses emacs; memory claim is contradicted"
        ),
        classification=MemorySupport.CONTRADICTED,
        confidence=0.95,
        evidence_quotes=["I switched away from vim a year ago — emacs forever now."],
        cited_excerpt_numbers=[1],
    )
    client = _make_mock_client([_extract_one("User is a long-time vim user."), classify])
    module = ModuleHMemory(opus_client=client, store=store, embedding_provider=embedder)

    findings = [r for r in await module.run(corpus) if isinstance(r, Finding)]
    assert len(findings) == 1
    assert findings[0].behavior == MemorySupport.CONTRADICTED.value
    assert findings[0].confidence == 0.95
    store.close()


async def test_insufficient_data_verdict(tmp_path: Path) -> None:
    """Memory claim has no semantic anchor in the corpus. Top-1 cosine
    similarity falls below ``INSUFFICIENT_THRESHOLD``; the classifier
    is short-circuited and INSUFFICIENT_DATA is emitted directly."""
    convs = [_conv("c1", turn_count=2)]
    turns = [
        _turn("c1", 0, role=Role.USER, content="What's the weather like in Tokyo?"),
        _turn("c1", 1, role=Role.ASSISTANT, content="I don't have realtime data."),
    ]
    store = _seed_store(
        tmp_path,
        conversations_memory="User collects rare antique stamps.",
        convs=convs,
        turns=turns,
    )
    corpus = _make_corpus({"c1": turns})
    weather_vec = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    stamps_vec = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # orthogonal
    embedder = StaticEmbeddingProvider(
        mapping={
            "What's the weather like in Tokyo?\n\n---\n\nI don't have realtime data.": weather_vec,
            "User collects rare antique stamps.": stamps_vec,
        },
        default=np.zeros(STATIC_DIM, dtype=np.float32),
    )
    # Only the extract call is expected — classify is short-circuited.
    client = _make_mock_client([_extract_one("User collects rare antique stamps.")])
    module = ModuleHMemory(opus_client=client, store=store, embedding_provider=embedder)

    findings = [r for r in await module.run(corpus) if isinstance(r, Finding)]
    assert len(findings) == 1
    assert findings[0].behavior == MemorySupport.INSUFFICIENT_DATA.value
    assert client.messages.create.await_count == 1
    store.close()


async def test_out_of_scope_verdict(tmp_path: Path) -> None:
    """A project-scoped memory whose project has no representation in
    the audited corpus is OUT_OF_SCOPE — not an error, not unsupported.
    This is the verdict that distinguishes "we don't know" from "the
    memory is wrong"."""
    convs = [_conv("c1", turn_count=2)]
    turns = [
        _turn("c1", 0, role=Role.USER, content="A user message about something else."),
        _turn("c1", 1, role=Role.ASSISTANT, content="Reply."),
    ]
    # Memory is scoped to "project-absent", a UUID with no project entry
    # in the store — fetch_all_projects() returns []; project_attribution
    # is empty; the entire project_memory is partitioned to out_of_scope.
    store = _seed_store(
        tmp_path,
        project_memories={"project-absent": "User is the technical lead on Project Helios."},
        convs=convs,
        turns=turns,
    )
    corpus = _make_corpus({"c1": turns})
    embedder = StaticEmbeddingProvider(default=np.zeros(STATIC_DIM, dtype=np.float32))
    # Out-of-scope path runs extract per chunk and skips classify.
    client = _make_mock_client([_extract_one("User is the technical lead on Project Helios.")])
    module = ModuleHMemory(opus_client=client, store=store, embedding_provider=embedder)

    findings = [r for r in await module.run(corpus) if isinstance(r, Finding)]
    assert len(findings) == 1
    assert findings[0].behavior == MemorySupport.OUT_OF_SCOPE.value
    assert findings[0].metadata["memory_source"] == "project_memories.project-absent"
    # Confidence on out-of-scope is fixed at 1.0 in the constructor —
    # the verdict itself is certain, even though the underlying claim's
    # truth is not.
    assert findings[0].confidence == 1.0
    # No classify call — out-of-scope path skips it.
    assert client.messages.create.await_count == 1
    store.close()
