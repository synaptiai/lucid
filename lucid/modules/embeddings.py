"""Voyage AI embeddings wrapper for Module H (memory-corpus consistency).

Protocol-typed so Module H can inject a mock in tests without touching
voyageai. The real provider (:class:`VoyageEmbeddingProvider`) wraps the
async Voyage client, batches inputs within API-level size caps, retries
on transient errors via tenacity, and returns ``numpy.ndarray`` of
float32 values.

**Why Voyage.** ``voyage-3-large`` (1024-dim) has strong retrieval
performance at competitive cost; the hackathon plan pins the provider
here and treats OpenAI ``text-embedding-3-small`` as a documented-but-
unshipped fallback. If the user lacks a ``VOYAGE_API_KEY``, this module
raises a clear :class:`EmbeddingProviderError` rather than silently
failing or falling through to a different provider. Post-hackathon the
fallback can be wired behind an ``EmbeddingProvider`` implementation.

**Batching.** The Voyage API caps a single call at ~128 texts or 120k
input tokens, whichever is lower. :func:`batch_texts` chunks inputs
into sub-batches respecting both caps. Because we don't call
``messages.count_tokens`` for embeddings, we use a conservative
characters-per-token heuristic (~3.5 chars / token) for the token
budget.

**Persistence.** Content addressing via sha256 of the chunk text
(``chunk_id``) makes the embedding cache idempotent: re-embedding the
same chunk in a later audit hits the cache instead of burning an API
call. The SQLite ``embeddings`` table (see ``store/schema.sql``) is
the cache store; helpers to read/write live on :class:`CorpusStore`.

**Testing.** The :class:`EmbeddingProvider` protocol is fulfilled by
simple stubs; tests construct a :class:`StaticEmbeddingProvider` that
returns pre-baked vectors and never touches the network. See
``tests/test_embeddings.py`` for the contract tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

if TYPE_CHECKING:
    from voyageai.client_async import AsyncClient

__all__ = [
    "CHARS_PER_TOKEN_ESTIMATE",
    "MAX_TEXTS_PER_BATCH",
    "MAX_TOKENS_PER_BATCH",
    "STATIC_DIM",
    "VOYAGE_DEFAULT_MODEL",
    "VOYAGE_DIM",
    "VOYAGE_INPUT_TYPE_DOCUMENT",
    "VOYAGE_INPUT_TYPE_QUERY",
    "EmbeddingBatchResult",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "StaticEmbeddingProvider",
    "VoyageEmbeddingProvider",
    "batch_texts",
    "blob_to_vector",
    "chunk_id",
    "vector_to_blob",
]


_LOGGER = logging.getLogger(__name__)


VOYAGE_DEFAULT_MODEL = "voyage-3-large"
VOYAGE_DIM = 1024
VOYAGE_INPUT_TYPE_DOCUMENT = "document"
VOYAGE_INPUT_TYPE_QUERY = "query"
MAX_TEXTS_PER_BATCH = 128
MAX_TOKENS_PER_BATCH = 120_000
# Rough char → token ratio for the batching budget. Voyage tokenizer is
# BPE-ish; 3.5 is a conservative upper bound that keeps batches under the
# API cap without a real count_tokens pass.
CHARS_PER_TOKEN_ESTIMATE = 3.5

STATIC_DIM = 8  # small dim for the test-only StaticEmbeddingProvider


InputType = Literal["document", "query"]


class EmbeddingProviderError(RuntimeError):
    """Raised for provider configuration errors (missing key, unsupported model)."""


@dataclass(frozen=True, slots=True)
class EmbeddingBatchResult:
    """Output of :meth:`EmbeddingProvider.embed_batch`.

    ``vectors`` is ``(N, dim)`` float32. ``input_tokens`` is the total
    tokens the provider billed for this batch (if it reports; 0 for
    providers that don't).
    """

    vectors: np.ndarray
    input_tokens: int = 0


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Async embedding provider contract used by Module H."""

    model: str
    dim: int

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        input_type: InputType,
    ) -> EmbeddingBatchResult: ...


# ──────────────────────────────────────────────────────────────────────────
# Chunk id + vector serialization
# ──────────────────────────────────────────────────────────────────────────


def chunk_id(text: str) -> str:
    """Content-addressed id for a chunk (used as embeddings.id).

    sha256 over the chunk text in UTF-8 bytes. Deterministic across
    processes, platforms, and Python versions.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def vector_to_blob(vector: np.ndarray) -> bytes:
    """Serialize a 1-D float vector to SQLite-storable bytes.

    Casts to float32 (saves 50% space vs float64 and is the precision
    Voyage returns anyway) and takes raw bytes. Callers recover with
    :func:`blob_to_vector` for bit-identical round-trips.
    """
    arr = np.asarray(vector).astype(np.float32, copy=False)
    if arr.ndim != 1:
        raise ValueError(f"vector must be 1-D, got shape {arr.shape}")
    return arr.tobytes()


def blob_to_vector(blob: bytes, *, dim: int | None = None) -> np.ndarray:
    """Deserialize a float32 vector from stored bytes.

    ``dim`` is optional; when provided, a shape mismatch raises
    ``ValueError`` immediately rather than returning a vector of
    unexpected length.
    """
    vec = np.frombuffer(blob, dtype=np.float32)
    if dim is not None and vec.shape[0] != dim:
        raise ValueError(f"expected dim {dim}, got {vec.shape[0]}")
    return vec


# ──────────────────────────────────────────────────────────────────────────
# Batching
# ──────────────────────────────────────────────────────────────────────────


def batch_texts(
    texts: Sequence[str],
    *,
    max_texts_per_batch: int = MAX_TEXTS_PER_BATCH,
    max_tokens_per_batch: int = MAX_TOKENS_PER_BATCH,
) -> list[list[int]]:
    """Group input indices into sub-batches respecting size caps.

    Returns a list of batches where each batch is a list of indices
    into ``texts``. A batch grows until adding the next text would
    exceed either the text count cap or the token budget estimate;
    then a new batch starts.

    Very long single texts (>= ``max_tokens_per_batch`` estimated
    tokens) are placed into solo-batches; the Voyage API will handle
    truncation when ``truncation=True`` is passed on the call.
    """
    if max_texts_per_batch < 1:
        raise ValueError("max_texts_per_batch must be >= 1")
    if max_tokens_per_batch < 1:
        raise ValueError("max_tokens_per_batch must be >= 1")

    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for i, text in enumerate(texts):
        est_tokens = max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))
        if current and (
            len(current) >= max_texts_per_batch
            or current_tokens + est_tokens > max_tokens_per_batch
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(i)
        current_tokens += est_tokens
    if current:
        batches.append(current)
    return batches


# ──────────────────────────────────────────────────────────────────────────
# Voyage implementation
# ──────────────────────────────────────────────────────────────────────────


def _voyage_transient_exceptions() -> tuple[type[BaseException], ...]:
    """Return the exception classes worth retrying on a Voyage call.

    Always includes the asyncio / stdlib transport classes
    (:class:`asyncio.TimeoutError`, builtin :class:`TimeoutError`,
    :class:`ConnectionError`). When ``voyageai.error`` is importable
    we additionally include Voyage's RateLimitError, ServerError,
    ServiceUnavailableError, APIConnectionError, Timeout, and
    TryAgain — every transient signal Voyage emits, but never
    AuthenticationError / InvalidRequestError / MalformedRequestError
    (those are configuration bugs that retrying would only delay).

    Built lazily so a test environment without ``voyageai`` installed
    still gets the stdlib classes and so import-time errors in
    voyageai don't kill the rest of Module H.
    """
    transient: list[type[BaseException]] = [
        asyncio.TimeoutError,
        TimeoutError,
        ConnectionError,
    ]
    try:
        from voyageai import error as voyage_error
    except Exception:
        return tuple(transient)
    for name in (
        "RateLimitError",
        "ServerError",
        "ServiceUnavailableError",
        "APIConnectionError",
        "Timeout",
        "TryAgain",
    ):
        cls = getattr(voyage_error, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            transient.append(cls)
    return tuple(transient)


class VoyageEmbeddingProvider:
    """``EmbeddingProvider`` backed by the async Voyage AI client.

    The provider is async-only; there is no sync path. Embedding
    ~10K chunks takes 30–60s with a warm pool and eight concurrent
    sub-batches.

    The client is injected at construction (or pulled from the
    ``voyageai`` module on-demand) so tests don't need to monkey-patch.
    """

    model: str
    dim: int

    def __init__(
        self,
        *,
        client: AsyncClient | None = None,
        api_key: str | None = None,
        model: str = VOYAGE_DEFAULT_MODEL,
        dim: int = VOYAGE_DIM,
        max_concurrency: int = 8,
    ) -> None:
        if client is None:
            resolved_key = api_key or os.environ.get("VOYAGE_API_KEY")
            if not resolved_key:
                raise EmbeddingProviderError(
                    "VOYAGE_API_KEY is not set. Provide api_key= explicitly, "
                    "set the VOYAGE_API_KEY environment variable, or inject a "
                    "mocked client= for tests."
                )
            # Defer the import so tests that never build the real provider
            # don't pay the voyageai startup cost.
            from voyageai.client_async import AsyncClient as _VoyageAsyncClient

            client = _VoyageAsyncClient(api_key=resolved_key)
        self._client = client
        self.model = model
        self.dim = dim
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        input_type: InputType,
    ) -> EmbeddingBatchResult:
        """Embed ``texts`` with automatic sub-batching.

        ``input_type`` steers Voyage's per-document vs per-query
        embeddings (Voyage models are asymmetric on that axis). For
        memory-chunk indexing pass ``"document"``; for at-query-time
        claim embedding pass ``"query"``.
        """
        if not texts:
            return EmbeddingBatchResult(
                vectors=np.zeros((0, self.dim), dtype=np.float32),
                input_tokens=0,
            )

        batch_indices = batch_texts(list(texts))
        all_vectors: list[np.ndarray] = [np.empty(0, dtype=np.float32)] * len(texts)
        total_tokens = 0

        async def _run_one(idx_batch: Sequence[int]) -> tuple[Sequence[int], EmbeddingBatchResult]:
            sub_texts = [texts[i] for i in idx_batch]
            async with self._semaphore:
                result = await self._call_embed(sub_texts, input_type=input_type)
            return idx_batch, result

        results = await asyncio.gather(*(_run_one(b) for b in batch_indices))
        for idx_batch, sub in results:
            for local_i, original_i in enumerate(idx_batch):
                all_vectors[original_i] = sub.vectors[local_i]
            total_tokens += sub.input_tokens

        stacked = (
            np.stack(all_vectors, axis=0)
            if all_vectors
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        return EmbeddingBatchResult(vectors=stacked, input_tokens=total_tokens)

    async def _call_embed(
        self,
        texts: Sequence[str],
        *,
        input_type: InputType,
    ) -> EmbeddingBatchResult:
        """One Voyage API call with tenacity retry on transient errors.

        Retries cover both transport failures (timeouts, connection
        drops) and Voyage server-side throttling
        (:class:`voyageai.error.RateLimitError`,
        :class:`~ServerError`, :class:`~ServiceUnavailableError`).
        Permanent failures (auth, malformed request) propagate
        immediately so misconfiguration surfaces in the first audit
        rather than after six futile retries.

        Six attempts with exponential backoff up to 90s gives the
        free Voyage tier (3 RPM) room to recover between calls — at
        the worst case (~20s wait + 90s cap) we cover a full
        rate-limit window without giving up.
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(6),
            wait=wait_random_exponential(multiplier=1, max=90),
            retry=retry_if_exception_type(_voyage_transient_exceptions()),
            reraise=True,
        ):
            with attempt:
                obj = await self._client.embed(
                    list(texts),
                    model=self.model,
                    input_type=input_type,
                    truncation=True,
                    output_dtype="float",
                )
        vectors = np.asarray(obj.embeddings, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise EmbeddingProviderError(
                f"Voyage returned dim {vectors.shape[1]} but provider was "
                f"configured for dim {self.dim}. Did you change the model?"
            )
        return EmbeddingBatchResult(
            vectors=vectors,
            input_tokens=int(obj.total_tokens),
        )


# ──────────────────────────────────────────────────────────────────────────
# Static provider — test-only
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class StaticEmbeddingProvider:
    """Deterministic embedding provider for tests.

    Maps each text to a fixed vector. If ``default`` is set and the text
    isn't in ``mapping``, returns ``default`` (typically a zero vector
    — useful for "unknown content" in similarity tests). Otherwise
    missing texts raise ``KeyError``.

    Not an ``EmbeddingProvider`` fallback for production: real embeddings
    carry semantic similarity; static vectors do not.
    """

    model: str = "static-test"
    dim: int = STATIC_DIM
    mapping: dict[str, np.ndarray] = field(default_factory=dict)
    default: np.ndarray | None = None
    calls: int = 0

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        input_type: InputType,
    ) -> EmbeddingBatchResult:
        self.calls += 1
        rows: list[np.ndarray] = []
        for t in texts:
            vec = self.mapping.get(t)
            if vec is None:
                if self.default is not None:
                    vec = self.default
                else:
                    raise KeyError(f"StaticEmbeddingProvider has no mapping for {t!r}")
            rows.append(vec.astype(np.float32, copy=False))
        stacked = np.stack(rows, axis=0) if rows else np.zeros((0, self.dim), dtype=np.float32)
        return EmbeddingBatchResult(vectors=stacked, input_tokens=0)


# ──────────────────────────────────────────────────────────────────────────
# Retrieval helpers (numpy)
# ──────────────────────────────────────────────────────────────────────────


def cosine_similarity_matrix(
    query_vec: np.ndarray,
    corpus_matrix: np.ndarray,
    *,
    corpus_norm: np.ndarray | None = None,
) -> np.ndarray:
    """Cosine similarity between one query vector and a stack of corpus vectors.

    Parameters
    ----------
    query_vec : (dim,) float32
    corpus_matrix : (N, dim) float32
    corpus_norm : (N,) float32 or ``None``
        Pre-computed L2 norms of the corpus rows. Passing it avoids
        recomputing the full norm per query — the caller can compute
        once and reuse across all claims.

    Returns
    -------
    (N,) float32 array of cosine similarities in [-1, 1].
    """
    if query_vec.ndim != 1:
        raise ValueError(f"query_vec must be 1-D, got shape {query_vec.shape}")
    if corpus_matrix.ndim != 2:
        raise ValueError(f"corpus_matrix must be 2-D, got shape {corpus_matrix.shape}")
    if query_vec.shape[0] != corpus_matrix.shape[1]:
        raise ValueError(
            f"query dim {query_vec.shape[0]} does not match corpus dim {corpus_matrix.shape[1]}"
        )

    if corpus_norm is None:
        corpus_norm = np.linalg.norm(corpus_matrix, axis=1).astype(np.float32)
    q_norm = float(np.linalg.norm(query_vec))
    # Guard against zero-norm vectors (would propagate NaN).
    if q_norm == 0.0:
        return np.zeros(corpus_matrix.shape[0], dtype=np.float32)
    safe_corpus = np.where(corpus_norm == 0, 1.0, corpus_norm).astype(np.float32)
    dots = corpus_matrix @ query_vec.astype(np.float32)
    sims = dots / (safe_corpus * q_norm)
    # Zero out rows whose original corpus vector was zero — they carry
    # no signal and should not bubble up as "high similarity".
    sims = np.where(corpus_norm == 0, 0.0, sims).astype(np.float32)
    return sims


def top_k_indices(
    similarities: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Return the indices of the top-k similarity scores, in descending order.

    ``k`` greater than the number of items returns every index. ``k``
    of zero returns an empty array.
    """
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if similarities.ndim != 1:
        raise ValueError(f"similarities must be 1-D, got shape {similarities.shape}")
    n = similarities.shape[0]
    k_effective = min(k, n)
    # argpartition is O(n); argsort-on-slice recovers descending order.
    partitioned = np.argpartition(-similarities, kth=k_effective - 1)[:k_effective]
    return partitioned[np.argsort(-similarities[partitioned])]
