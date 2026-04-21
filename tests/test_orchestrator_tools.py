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


async def test_invoke_module_stub_signals_not_implemented(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    result = await registry.get("invoke_module").handler(  # type: ignore[union-attr]
        {"module": "A", "conversation_ids": ["c-1"]}
    )
    assert result["status"] == "not_implemented"
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

    registry = build_tool_registry(
        store=store, audit_run_id=run_id, progress_log=_progress
    )
    result = await registry.get("log_progress").handler(  # type: ignore[union-attr]
        {"level": "WARNING", "message": "budget tight"}
    )
    assert result["ok"] is True
    assert captured == [("WARNING", "budget tight")]
    store.close()


# ----- estimate_remaining_cost ---------------------------------------


async def test_estimate_remaining_cost_uses_budget(tmp_path: Path) -> None:
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(
        store=store, audit_run_id=run_id, remaining_budget_usd=5.0
    )
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
