"""Orchestrator tool-handler tests — no transport layer involved."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lucid.orchestrator.tools import build_tool_registry, to_tool_result_content
from lucid.schemas import Conversation, Role, Source, TextBlock, Turn
from lucid.store import initialize_db
from lucid.store.sqlite import CorpusStore


def _seed_store(tmp_path: Path) -> tuple[CorpusStore, str]:
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    store = CorpusStore(db)
    store.connect()

    # Seed an AuditRun via raw SQL so we don't need the full Pydantic model here.
    store.connect().execute(
        """
        INSERT INTO audit_runs (
            id, sources_json, source_paths_json, started_at, corpus_stats_json,
            token_usage_json, sampling_config_json, status, corpus_fingerprint,
            prompt_versions_json, schema_version, skipped_modules_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run-1",
            '["claude-code"]',
            '{"claude-code": "/tmp"}',
            datetime(2026, 4, 21, tzinfo=UTC).isoformat(),
            "{}",
            "{}",
            "{}",
            "running",
            "fingerprint",
            "{}",
            1,
            "[]",
        ),
    )
    now = datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC)
    convs = [
        Conversation(
            id="c-1",
            source=Source.CLAUDE_CODE,
            source_path="/tmp",
            created_at=now,
            updated_at=now,
            title="Test conversation",
            turn_count=3,
            project_slug="proj-1",
        ),
    ]
    store.insert_conversations(convs)
    store.insert_turns(
        [
            Turn(
                id=f"t-{i}",
                conversation_id="c-1",
                index=i,
                role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
                content=f"content for turn {i}",
                blocks=[TextBlock(text=f"turn {i}")],
            )
            for i in range(3)
        ]
    )
    return store, "run-1"


# ----- query_corpus ----------------------------------------------------


async def test_query_corpus_returns_seeded_conversation(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    handler = registry.get("query_corpus")
    assert handler is not None
    result = await handler.handler({})
    assert result["count"] == 1
    assert result["conversations"][0]["id"] == "c-1"
    store.close()


async def test_query_corpus_filters_by_source(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    result_ai = await registry.get("query_corpus").handler({"source": "claude-ai"})  # type: ignore[union-attr]
    result_cc = await registry.get("query_corpus").handler({"source": "claude-code"})  # type: ignore[union-attr]
    assert result_ai["count"] == 0
    assert result_cc["count"] == 1
    store.close()


# ----- get_conversation -----------------------------------------------


async def test_get_conversation_returns_metadata(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    result = await registry.get("get_conversation").handler(  # type: ignore[union-attr]
        {"conversation_id": "c-1"}
    )
    assert result["id"] == "c-1"
    assert result["title"] == "Test conversation"
    store.close()


async def test_get_conversation_missing_returns_not_found(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    result = await registry.get("get_conversation").handler(  # type: ignore[union-attr]
        {"conversation_id": "no-such-id"}
    )
    assert result["error"] == "not_found"
    store.close()


# ----- get_turn_window ------------------------------------------------


async def test_get_turn_window_returns_slice(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    result = await registry.get("get_turn_window").handler(  # type: ignore[union-attr]
        {"conversation_id": "c-1", "start_index": 1, "count": 2}
    )
    assert result["conversation_id"] == "c-1"
    assert len(result["turns"]) == 2
    assert result["turns"][0]["index"] == 1
    store.close()


# ----- invoke_module --------------------------------------------------


async def test_invoke_module_returns_no_client_without_anthropic(tmp_path: Path) -> None:
    """When the registry is built without an anthropic_client, LLM-backed
    modules return ``no_client`` so the orchestrator can log and continue
    rather than crashing. Real audits inject the client."""
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    result = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "A", "conversation_ids": ["c-1"]}
    )
    assert result["status"] == "no_client"
    assert result["module"] == "A"
    store.close()


async def test_invoke_module_unknown_module_returns_error(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    result = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "Z", "conversation_ids": ["c-1"]}
    )
    assert result["error"] == "unknown_module"
    store.close()


async def test_invoke_module_d_skipped_without_opt_in(tmp_path: Path) -> None:
    """Module D is opt-in; without ``allow_module_d`` the dispatcher
    short-circuits with ``skipped`` and makes no LLM call."""
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    result = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "D", "conversation_ids": ["c-1"]}
    )
    assert result["status"] == "skipped"
    assert "--include-module-d" in result["message"]
    store.close()


async def test_invoke_module_h_without_embedding_provider_is_skipped(
    tmp_path: Path,
) -> None:
    """Module H requires an EmbeddingProvider. Without one the dispatcher
    returns ``no_embedding_provider`` and makes no LLM or embedding call."""
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    result = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "H", "conversation_ids": ["c-1"]}
    )
    assert result["status"] == "no_embedding_provider"
    assert "Voyage" in result["message"] or "EmbeddingProvider" in result["message"]
    store.close()


async def test_invoke_module_h_with_provider_but_no_client_returns_no_client(
    tmp_path: Path,
) -> None:
    """Module H also requires an anthropic_client. With an embedding
    provider but no client the dispatcher still short-circuits with
    ``no_client`` — the client check runs after the provider check."""
    import numpy as np

    from lucid.modules.embeddings import STATIC_DIM, StaticEmbeddingProvider

    store, run_id = _seed_store(tmp_path)
    provider = StaticEmbeddingProvider(
        default=np.zeros(STATIC_DIM, dtype=np.float32)
    )
    registry = build_tool_registry(
        store=store, audit_run_id=run_id, embedding_provider=provider
    )
    result = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "H", "conversation_ids": ["c-1"]}
    )
    assert result["status"] == "no_client"
    store.close()


async def test_invoke_module_g_runs_without_client(tmp_path: Path) -> None:
    """Module G (attribution) is deterministic and does not require a
    client; the dispatcher should run it end-to-end and persist findings."""
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    result = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "G", "conversation_ids": ["c-1"]}
    )
    assert result["status"] == "completed"
    assert result["findings_stored"] >= 1
    assert result["module_errors"] == []

    # Verify the finding actually landed in the DB.
    findings = await registry.get("get_findings").handler({"module": "G"})  # type: ignore[union-attr]
    assert findings["count"] == result["findings_stored"]
    store.close()


async def test_invoke_module_g_idempotent_on_second_call(tmp_path: Path) -> None:
    """Running Module G twice against the same conversation should leave
    the findings count unchanged — the UNIQUE key blocks the second
    insert and the dispatcher counts it as an idempotent collision, not
    an error."""
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    first = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "G", "conversation_ids": ["c-1"]}
    )
    second = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "G", "conversation_ids": ["c-1"]}
    )
    assert first["findings_stored"] >= 1
    assert second["findings_stored"] == 0
    assert second["idempotent_skips"] == first["findings_stored"]
    store.close()


async def test_invoke_module_d_with_opt_in_still_skipped_without_client(
    tmp_path: Path,
) -> None:
    """allow_module_d=True unlocks the opt-in gate, but Module D still
    needs a client; without it the dispatcher returns ``no_client``."""
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(
        store=store, audit_run_id=run_id, allow_module_d=True
    )
    result = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "D", "conversation_ids": ["c-1"]}
    )
    assert result["status"] == "no_client"
    store.close()


async def test_invoke_module_empty_corpus_still_succeeds(tmp_path: Path) -> None:
    """Empty conversation_ids → empty corpus → module runs cleanly with
    zero findings. This is the pre-flight-check shape the orchestrator
    uses."""
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    result = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "G", "conversation_ids": []}
    )
    assert result["status"] == "completed"
    assert result["findings_stored"] == 0
    store.close()


async def test_invoke_module_a_end_to_end_with_mocked_client(tmp_path: Path) -> None:
    """End-to-end: dispatcher runs Module A with a mocked LLM, persists
    findings to the store, and returns a completed status."""
    from unittest.mock import AsyncMock, MagicMock

    from anthropic.types import Usage

    # Seed a longer conversation so Module A has a window to score.
    store, run_id = _seed_store(tmp_path)
    # Add a few more turns so there are ≥ 4 alternating turns (1 window).
    store.insert_turns(
        [
            Turn(
                id=f"extra-t-{i}",
                conversation_id="c-1",
                index=3 + i,
                role=Role.USER if (3 + i) % 2 == 0 else Role.ASSISTANT,
                content=f"extra turn {i}",
                blocks=[TextBlock(text=f"x{i}")],
            )
            for i in range(3)
        ]
    )

    # Build a mocked client that returns one SpiralBenchScore with a
    # single pushback incident at turn_index=0 (the first assistant turn).
    from lucid.modules.module_a_spiralbench import (
        BehaviorIncident,
        SpiralBenchIncidents,
        SpiralBenchScore,
    )

    score = SpiralBenchScore(
        reasoning="found one pushback incident",
        incidents=SpiralBenchIncidents(
            pushback=[
                BehaviorIncident(snippet="let me push back", intensity=2, turn_index=0)
            ]
        ),
    )

    async def _call(**_kwargs: object) -> object:
        block = MagicMock()
        block.type = "text"
        block.text = score.model_dump_json()
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

    registry = build_tool_registry(
        store=store, audit_run_id=run_id, anthropic_client=client
    )
    result = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "A", "conversation_ids": ["c-1"]}
    )

    assert result["status"] == "completed", result
    assert result["findings_stored"] >= 1
    assert result["module_errors"] == []

    # Verify the finding actually persisted.
    persisted = await registry.get("get_findings").handler({"module": "A"})  # type: ignore[union-attr]
    assert persisted["count"] == result["findings_stored"]
    store.close()


async def test_invoke_module_h_end_to_end_with_mocked_client_and_provider(
    tmp_path: Path,
) -> None:
    """End-to-end: dispatcher runs Module H with mocked Opus + Voyage
    stubs, persists findings, returns completed status."""
    from unittest.mock import AsyncMock, MagicMock

    import numpy as np
    from anthropic.types import Usage

    from lucid.modules.embeddings import STATIC_DIM, StaticEmbeddingProvider
    from lucid.modules.module_h_memory import (
        ClassifyScore,
        ExtractedClaim,
        ExtractResult,
    )
    from lucid.schemas import MemoryFile, MemorySupport

    store, run_id = _seed_store(tmp_path)
    # Seed: a memory file + a user-assistant pair the memory claims about.
    store.insert_memory_file(
        MemoryFile(
            account_uuid="acct-1",
            conversations_memory="User works at Acme Corp.",
            project_memories={},
        )
    )
    # The seeded conversation has turns t-0 / t-1 / t-2 alternating user/
    # assistant/user. Module H only pairs user→assistant, so t-0/t-1 is a
    # chunk.

    # Static embeddings: the one chunk text and the one claim query both
    # map to the same vector, so similarity = 1.0 and retrieval surfaces
    # the chunk.
    # Chunk text = "content for turn 0\n\n---\n\ncontent for turn 1"
    same_vec = np.ones(STATIC_DIM, dtype=np.float32)
    chunk_text = "content for turn 0\n\n---\n\ncontent for turn 1"
    provider = StaticEmbeddingProvider(
        mapping={
            chunk_text: same_vec,
            "User works at Acme Corp.": same_vec,
        },
        default=np.zeros(STATIC_DIM, dtype=np.float32),
    )

    extract_result = ExtractResult(
        reasoning="one work claim",
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
        reasoning="excerpt references Acme directly",
        classification=MemorySupport.WELL_SUPPORTED,
        confidence=0.9,
        evidence_quotes=["content for turn 0"],
        cited_excerpt_numbers=[1],
    )

    outputs = [extract_result, classify_well]

    async def _call(**_kwargs: object) -> object:
        item = outputs.pop(0)
        block = MagicMock()
        block.type = "text"
        block.text = item.model_dump_json()
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

    registry = build_tool_registry(
        store=store,
        audit_run_id=run_id,
        anthropic_client=client,
        embedding_provider=provider,
    )
    result = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "H", "conversation_ids": ["c-1"]}
    )

    assert result["status"] == "completed", result
    assert result["findings_stored"] == 1
    assert result["module_errors"] == []

    persisted = await registry.get("get_findings").handler({"module": "H"})  # type: ignore[union-attr]
    assert persisted["count"] == 1
    store.close()


# ----- store_finding --------------------------------------------------


def _valid_finding_args(run_id: str) -> dict[str, object]:
    return {
        "id": "f-1",
        "audit_run_id": run_id,
        "conversation_id": "c-1",
        "turn_ids": ["t-0", "t-1"],
        "module": "A",
        "behavior": "safe-redirection",
        "intensity": 2,
        "confidence": 0.8,
        "explanation": "Module identified a redirection to a safer topic.",
        "citation": "Spiral-Bench",
        "detected_by": ["claude-opus-4-7"],
        "detected_at": datetime.now(tz=UTC).isoformat(),
        "prompt_version": "v1",
        "prompt_hash": "h" * 16,
    }


async def test_store_finding_happy_path(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    result = await registry.get("store_finding").handler(  # type: ignore[union-attr]
        _valid_finding_args(run_id)
    )
    assert result["stored"] is True
    assert result["finding_id"] == "f-1"
    store.close()


async def test_store_finding_idempotency_returns_integrity_error(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    first = await registry.get("store_finding").handler(_valid_finding_args(run_id))  # type: ignore[union-attr]
    second = await registry.get("store_finding").handler(  # type: ignore[union-attr]
        {**_valid_finding_args(run_id), "id": "f-duplicate"}
    )
    assert first["stored"] is True
    assert second["error"] == "integrity_error"
    store.close()


async def test_store_finding_validation_error_missing_required(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    bad = _valid_finding_args(run_id)
    del bad["explanation"]
    result = await registry.get("store_finding").handler(bad)  # type: ignore[union-attr]
    assert result["error"] == "validation_error"
    store.close()


async def test_store_finding_autocomputes_turn_ids_hash(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    args = _valid_finding_args(run_id)
    args.pop("turn_ids_hash", None)  # ensure it's missing
    result = await registry.get("store_finding").handler(args)  # type: ignore[union-attr]
    assert result["stored"] is True
    # Confirmed stored with a hash.
    row = store.fetchall("SELECT turn_ids_hash FROM findings WHERE id = ?", ("f-1",))[0]
    assert len(row["turn_ids_hash"]) == 64  # sha256 hex
    store.close()


# ----- get_findings ---------------------------------------------------


async def test_get_findings_lists_stored_findings(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    await registry.get("store_finding").handler(_valid_finding_args(run_id))  # type: ignore[union-attr]
    result = await registry.get("get_findings").handler({})  # type: ignore[union-attr]
    assert result["count"] == 1
    assert result["findings"][0]["behavior"] == "safe-redirection"
    store.close()


# ----- log_progress ---------------------------------------------------


async def test_log_progress_calls_callback(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    captured: list[tuple[str, str]] = []

    def _progress(level: str, message: str) -> None:
        captured.append((level, message))

    registry = build_tool_registry(store=store, audit_run_id=run_id, progress_log=_progress)
    result = await registry.get("log_progress").handler(  # type: ignore[union-attr]
        {"level": "WARNING", "message": "budget tight"}
    )
    assert result["ok"] is True
    assert captured == [("WARNING", "budget tight")]
    store.close()


# ----- estimate_remaining_cost ---------------------------------------


async def test_estimate_remaining_cost_uses_budget(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id, remaining_budget_usd=5.0)
    result = await registry.get("estimate_remaining_cost").handler({})  # type: ignore[union-attr]
    assert result["budget_usd"] == 5.0
    assert result["accrued_usd"] == 0.0
    assert result["remaining_usd"] == 5.0
    store.close()


# ----- registry metadata ----------------------------------------------


async def test_registry_registers_all_eight_tools(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    expected = {
        "query_corpus",
        "get_conversation",
        "get_turn_window",
        "invoke_module",
        "store_finding",
        "get_findings",
        "log_progress",
        "estimate_remaining_cost",
    }
    assert set(registry.names) == expected
    store.close()


def test_to_tool_result_content_serializes_dict() -> None:
    payload = {"a": 1, "b": [1, 2, 3]}
    content = to_tool_result_content(payload)
    assert json.loads(content) == payload


# ----- registry double-registration rejected -------------------------


async def test_registry_rejects_duplicate_name(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    # Re-registering the same tool surfaces a clear error.
    from lucid.orchestrator.tools import CustomTool

    duplicate = CustomTool(
        name="query_corpus",
        description="dup",
        input_schema={"type": "object"},
        handler=lambda args: asyncio_stub(args),
    )
    with pytest.raises(ValueError, match="already registered"):
        registry.register(duplicate)
    store.close()


async def asyncio_stub(args: dict[str, object]) -> dict[str, object]:
    return {"ok": True, "args": args}
