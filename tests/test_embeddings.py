"""Tests for :mod:`lucid.modules.embeddings` and CorpusStore embedding helpers.

All tests run against :class:`StaticEmbeddingProvider` or mocked Voyage
clients — no live API calls. The real VoyageEmbeddingProvider is
exercised only through a mocked ``AsyncClient``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lucid.modules.embeddings import (
    CHARS_PER_TOKEN_ESTIMATE,
    MAX_TEXTS_PER_BATCH,
    STATIC_DIM,
    VOYAGE_DEFAULT_MODEL,
    VOYAGE_DIM,
    EmbeddingProvider,
    EmbeddingProviderError,
    StaticEmbeddingProvider,
    VoyageEmbeddingProvider,
    batch_texts,
    blob_to_vector,
    chunk_id,
    cosine_similarity_matrix,
    top_k_indices,
    vector_to_blob,
)
from lucid.schemas import Conversation, Role, Source, TextBlock, Turn
from lucid.store import initialize_db
from lucid.store.sqlite import CorpusStore

# ──────────────────────────────────────────────────────────────────────────
# chunk_id / vector_to_blob / blob_to_vector
# ──────────────────────────────────────────────────────────────────────────


def test_chunk_id_is_deterministic() -> None:
    assert chunk_id("hello") == chunk_id("hello")
    assert chunk_id("hello") != chunk_id("world")
    # sha256 is 64 hex chars
    assert len(chunk_id("hello")) == 64


def test_vector_to_blob_round_trip() -> None:
    vec = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    blob = vector_to_blob(vec)
    recovered = blob_to_vector(blob)
    assert recovered.dtype == np.float32
    assert np.allclose(recovered, vec, atol=1e-7)


def test_vector_to_blob_rejects_2d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        vector_to_blob(np.zeros((2, 3)))


def test_blob_to_vector_dim_mismatch_raises() -> None:
    vec = np.ones(4, dtype=np.float32)
    blob = vector_to_blob(vec)
    with pytest.raises(ValueError, match="expected dim"):
        blob_to_vector(blob, dim=8)


def test_blob_to_vector_dim_match_passes() -> None:
    vec = np.ones(4, dtype=np.float32)
    assert blob_to_vector(vector_to_blob(vec), dim=4).shape == (4,)


# ──────────────────────────────────────────────────────────────────────────
# batch_texts
# ──────────────────────────────────────────────────────────────────────────


def test_batch_texts_empty_input_returns_empty() -> None:
    assert batch_texts([]) == []


def test_batch_texts_respects_text_count_cap() -> None:
    texts = [f"short text {i}" for i in range(300)]
    batches = batch_texts(texts, max_texts_per_batch=50)
    # Every batch ≤ 50 items.
    assert all(len(b) <= 50 for b in batches)
    # Union covers every index exactly once.
    indices = [i for b in batches for i in b]
    assert indices == list(range(300))


def test_batch_texts_respects_token_budget() -> None:
    # One very long text eats the entire token budget alone.
    long_text = "x" * int(50_000 * CHARS_PER_TOKEN_ESTIMATE)
    short = ["a", "b", "c"]
    texts = [*short, long_text, *short]
    batches = batch_texts(texts, max_tokens_per_batch=10_000)
    # Long text must be in its own batch (the preceding short items fit
    # but the long one overflows; then short items resume in a new batch).
    long_batch = next(b for b in batches if 3 in b)
    assert long_batch == [3]


def test_batch_texts_default_caps_are_sensible() -> None:
    # Rough sanity: with default caps, 500 short texts split into ≤ 4 batches.
    texts = [f"t{i}" for i in range(500)]
    batches = batch_texts(texts)
    assert len(batches) <= 500 // MAX_TEXTS_PER_BATCH + 1


# ──────────────────────────────────────────────────────────────────────────
# StaticEmbeddingProvider
# ──────────────────────────────────────────────────────────────────────────


async def test_static_provider_returns_mapped_vectors() -> None:
    provider = StaticEmbeddingProvider(
        mapping={
            "hello": np.ones(STATIC_DIM, dtype=np.float32),
            "world": np.zeros(STATIC_DIM, dtype=np.float32),
        }
    )
    result = await provider.embed_batch(["hello", "world"], input_type="document")
    assert result.vectors.shape == (2, STATIC_DIM)
    assert result.vectors[0].tolist() == [1.0] * STATIC_DIM
    assert result.vectors[1].tolist() == [0.0] * STATIC_DIM


async def test_static_provider_missing_key_with_no_default_raises() -> None:
    provider = StaticEmbeddingProvider()
    with pytest.raises(KeyError):
        await provider.embed_batch(["unmapped"], input_type="query")


async def test_static_provider_default_covers_missing_keys() -> None:
    provider = StaticEmbeddingProvider(
        mapping={"known": np.ones(STATIC_DIM, dtype=np.float32)},
        default=np.full(STATIC_DIM, 0.5, dtype=np.float32),
    )
    result = await provider.embed_batch(
        ["known", "unknown"], input_type="document"
    )
    assert result.vectors[1].tolist() == [0.5] * STATIC_DIM


async def test_static_provider_counts_calls_for_assertions() -> None:
    provider = StaticEmbeddingProvider(default=np.zeros(STATIC_DIM, dtype=np.float32))
    assert provider.calls == 0
    await provider.embed_batch(["a"], input_type="query")
    await provider.embed_batch(["b"], input_type="query")
    assert provider.calls == 2


def test_static_provider_satisfies_embedding_provider_protocol() -> None:
    provider = StaticEmbeddingProvider()
    assert isinstance(provider, EmbeddingProvider)


# ──────────────────────────────────────────────────────────────────────────
# VoyageEmbeddingProvider (mocked client)
# ──────────────────────────────────────────────────────────────────────────


def _make_voyage_response(vectors: list[list[float]], *, total_tokens: int = 0) -> Any:
    response = MagicMock()
    response.embeddings = vectors
    response.total_tokens = total_tokens
    return response


async def test_voyage_provider_single_batch_call() -> None:
    fake_client = MagicMock()
    fake_client.embed = AsyncMock(
        return_value=_make_voyage_response(
            [[0.0] * VOYAGE_DIM, [1.0] * VOYAGE_DIM],
            total_tokens=42,
        )
    )
    provider = VoyageEmbeddingProvider(client=fake_client)
    result = await provider.embed_batch(["hello", "world"], input_type="document")
    assert result.vectors.shape == (2, VOYAGE_DIM)
    assert result.input_tokens == 42
    assert fake_client.embed.await_count == 1
    kwargs = fake_client.embed.await_args.kwargs
    assert kwargs["model"] == VOYAGE_DEFAULT_MODEL
    assert kwargs["input_type"] == "document"
    assert kwargs["truncation"] is True


async def test_voyage_provider_handles_empty_input_without_api_call() -> None:
    fake_client = MagicMock()
    fake_client.embed = AsyncMock(return_value=_make_voyage_response([]))
    provider = VoyageEmbeddingProvider(client=fake_client)

    result = await provider.embed_batch([], input_type="query")

    assert result.vectors.shape == (0, VOYAGE_DIM)
    assert fake_client.embed.await_count == 0


async def test_voyage_provider_splits_large_input_into_multiple_batches() -> None:
    fake_client = MagicMock()

    async def _serve(texts: list[str], **_kwargs: Any) -> Any:
        return _make_voyage_response(
            [[float(i)] * VOYAGE_DIM for i in range(len(texts))], total_tokens=0
        )

    fake_client.embed = AsyncMock(side_effect=_serve)
    provider = VoyageEmbeddingProvider(client=fake_client, max_concurrency=2)

    texts = [f"t{i}" for i in range(300)]
    result = await provider.embed_batch(texts, input_type="document")

    assert result.vectors.shape == (300, VOYAGE_DIM)
    # With the default per-batch cap of 128, 300 texts should need 3 calls.
    assert fake_client.embed.await_count == 3


async def test_voyage_provider_rejects_wrong_dim_from_api() -> None:
    fake_client = MagicMock()
    fake_client.embed = AsyncMock(
        return_value=_make_voyage_response([[0.0] * 512])  # wrong dim
    )
    provider = VoyageEmbeddingProvider(client=fake_client)
    with pytest.raises(EmbeddingProviderError, match="returned dim"):
        await provider.embed_batch(["a"], input_type="document")


def test_voyage_provider_without_key_or_client_raises() -> None:
    with pytest.raises(EmbeddingProviderError, match="VOYAGE_API_KEY"):
        VoyageEmbeddingProvider()


def test_voyage_provider_accepts_explicit_api_key(monkeypatch: Any) -> None:
    """A real Voyage client will try to connect on first call — but we
    only instantiate here and never call, so no network I/O happens.
    This tests that the key is accepted without raising."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    provider = VoyageEmbeddingProvider(api_key="v-fake-key")
    assert provider.model == VOYAGE_DEFAULT_MODEL
    assert provider.dim == VOYAGE_DIM


# ──────────────────────────────────────────────────────────────────────────
# Cosine similarity + top_k
# ──────────────────────────────────────────────────────────────────────────


def test_cosine_similarity_identical_vectors_equals_one() -> None:
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    corpus = np.stack([v, v], axis=0)
    sims = cosine_similarity_matrix(v, corpus)
    assert sims.shape == (2,)
    assert np.allclose(sims, 1.0, atol=1e-5)


def test_cosine_similarity_orthogonal_vectors_equals_zero() -> None:
    q = np.array([1.0, 0.0], dtype=np.float32)
    corpus = np.array([[0.0, 1.0], [0.0, 5.0]], dtype=np.float32)
    sims = cosine_similarity_matrix(q, corpus)
    assert np.allclose(sims, 0.0, atol=1e-6)


def test_cosine_similarity_zero_query_returns_all_zero() -> None:
    q = np.zeros(3, dtype=np.float32)
    corpus = np.ones((2, 3), dtype=np.float32)
    sims = cosine_similarity_matrix(q, corpus)
    assert np.all(sims == 0.0)


def test_cosine_similarity_zero_corpus_row_does_not_nan() -> None:
    q = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    corpus = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    sims = cosine_similarity_matrix(q, corpus)
    assert not np.any(np.isnan(sims))
    # Second row is the query itself → similarity 1.
    assert sims[1] == pytest.approx(1.0, abs=1e-5)
    assert sims[0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_dim_mismatch_raises() -> None:
    q = np.ones(3, dtype=np.float32)
    corpus = np.ones((2, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="dim"):
        cosine_similarity_matrix(q, corpus)


def test_top_k_returns_descending_indices() -> None:
    sims = np.array([0.1, 0.9, 0.5, 0.3], dtype=np.float32)
    top = top_k_indices(sims, k=3)
    assert top.tolist() == [1, 2, 3]


def test_top_k_k_larger_than_n_returns_all() -> None:
    sims = np.array([0.1, 0.9], dtype=np.float32)
    top = top_k_indices(sims, k=10)
    assert sorted(top.tolist()) == [0, 1]


def test_top_k_zero_returns_empty() -> None:
    sims = np.array([0.1, 0.9], dtype=np.float32)
    assert top_k_indices(sims, k=0).shape == (0,)


# ──────────────────────────────────────────────────────────────────────────
# CorpusStore embedding helpers
# ──────────────────────────────────────────────────────────────────────────


def _seed_store(tmp_path: Path) -> CorpusStore:
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    store = CorpusStore(db)
    store.connect()
    now = datetime(2026, 4, 21, tzinfo=UTC)
    store.insert_conversations(
        [
            Conversation(
                id="c-1",
                source=Source.CLAUDE_CODE,
                source_path="/tmp",
                created_at=now,
                updated_at=now,
                turn_count=1,
            )
        ]
    )
    store.insert_turns(
        [
            Turn(
                id="t-0",
                conversation_id="c-1",
                index=0,
                role=Role.USER,
                content="hello",
                blocks=[TextBlock(text="hello")],
            )
        ]
    )
    return store


def test_store_insert_and_fetch_embedding(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    blob = vector_to_blob(vec)
    store.insert_embedding(
        id=chunk_id("hello world"),
        conversation_id="c-1",
        turn_id="t-0",
        chunk_text="hello world",
        vector_blob=blob,
        dim=3,
        model="test-model",
    )
    rows = store.fetch_embeddings_for_conversations(["c-1"])
    assert len(rows) == 1
    assert rows[0]["chunk_text"] == "hello world"
    recovered = blob_to_vector(rows[0]["vector_blob"], dim=3)
    assert np.allclose(recovered, vec, atol=1e-7)
    store.close()


def test_store_insert_embeddings_batch(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    rows_in = [
        (
            chunk_id(f"text-{i}"),
            "c-1",
            "t-0",
            f"text-{i}",
            vector_to_blob(np.full(4, i, dtype=np.float32)),
            4,
            "m1",
        )
        for i in range(3)
    ]
    inserted = store.insert_embeddings(rows_in)
    assert inserted == 3
    rows_out = store.fetch_embeddings_for_conversations(["c-1"])
    assert len(rows_out) == 3
    store.close()


def test_store_insert_embedding_is_idempotent(tmp_path: Path) -> None:
    """INSERT OR REPLACE means a re-embed with the same id doesn't raise;
    the row is overwritten with the latest vector."""
    store = _seed_store(tmp_path)
    id_ = chunk_id("repeated chunk")
    vec_a = vector_to_blob(np.array([1.0, 2.0], dtype=np.float32))
    vec_b = vector_to_blob(np.array([9.9, 8.8], dtype=np.float32))
    store.insert_embedding(
        id=id_, conversation_id="c-1", turn_id="t-0",
        chunk_text="repeated chunk", vector_blob=vec_a, dim=2, model="m1",
    )
    # Re-insert with same id but different blob.
    store.insert_embedding(
        id=id_, conversation_id="c-1", turn_id="t-0",
        chunk_text="repeated chunk", vector_blob=vec_b, dim=2, model="m1",
    )
    rows = store.fetch_embeddings_for_conversations(["c-1"])
    assert len(rows) == 1
    recovered = blob_to_vector(rows[0]["vector_blob"], dim=2)
    assert np.allclose(recovered, [9.9, 8.8], atol=1e-5)
    store.close()


def test_store_fetch_embedding_ids_cache_hit(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    a = chunk_id("cached-a")
    b = chunk_id("cached-b")
    c = chunk_id("uncached-c")
    store.insert_embeddings(
        [
            (a, "c-1", "t-0", "cached-a", vector_to_blob(np.ones(2, np.float32)), 2, "m1"),
            (b, "c-1", "t-0", "cached-b", vector_to_blob(np.zeros(2, np.float32)), 2, "m1"),
        ]
    )
    hit = store.fetch_embedding_ids([a, b, c])
    assert hit == {a, b}
    store.close()


def test_store_fetch_embeddings_filter_by_model(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    store.insert_embedding(
        id=chunk_id("x"), conversation_id="c-1", turn_id="t-0",
        chunk_text="x", vector_blob=vector_to_blob(np.ones(2, np.float32)),
        dim=2, model="voyage-3-large",
    )
    store.insert_embedding(
        id=chunk_id("y"), conversation_id="c-1", turn_id="t-0",
        chunk_text="y", vector_blob=vector_to_blob(np.zeros(2, np.float32)),
        dim=2, model="static-test",
    )
    voyage_rows = store.fetch_embeddings_for_conversations(
        ["c-1"], model="voyage-3-large"
    )
    assert len(voyage_rows) == 1
    assert voyage_rows[0]["chunk_text"] == "x"
    store.close()
