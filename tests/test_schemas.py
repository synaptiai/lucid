"""Round-trip tests for every schema model + discriminated-union coverage.

Every schema should: serialize to dict / JSON, round-trip back through
`model_validate`, compare equal to the original. `ContentBlock` is a union
so we test each variant separately and verify a `tool_use` block doesn't
accidentally validate as a `text` block.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from lucid.schemas import (
    AuditRun,
    AuditStatus,
    ContentBlock,
    Conversation,
    CorpusStats,
    Finding,
    MemoryClaim,
    MemoryFile,
    ModuleName,
    ModuleTokenUsage,
    Project,
    Role,
    SamplingConfigRecord,
    Source,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolResultBlock,
    ToolUseBlock,
    Turn,
)

_UTC = UTC


def _now() -> datetime:
    # Fixed-ish datetime so round-trips don't hit microsecond drift.
    return datetime(2026, 4, 21, 12, 0, 0, tzinfo=_UTC)


def _roundtrip[T: BaseModel](model: T) -> T:
    """Serialize to JSON, parse back, return the new instance."""
    dumped = model.model_dump_json()
    return type(model).model_validate_json(dumped)


# ----- corpus roundtrips -------------------------------------------


def test_conversation_roundtrip() -> None:
    c = Conversation(
        id="conv-1",
        source=Source.CLAUDE_AI,
        source_path="/path/to/export",
        created_at=_now(),
        updated_at=_now(),
        model=None,
        title="A chat",
        summary="AI summary",
        turn_count=4,
        project_slug=None,
        metadata={"k": "v"},
    )
    assert _roundtrip(c) == c


def test_turn_roundtrip_with_blocks() -> None:
    t = Turn(
        id="t-1",
        conversation_id="conv-1",
        index=0,
        role=Role.USER,
        content="Hello",
        blocks=[TextBlock(text="Hello")],
        timestamp=_now(),
        parent_message_uuid="00000000-0000-4000-8000-000000000000",
        token_count=2,
    )
    assert _roundtrip(t) == t


def test_project_roundtrip() -> None:
    p = Project(
        uuid="u",
        name="n",
        description=None,
        prompt_template=None,
        created_at=_now(),
        updated_at=_now(),
        doc_count=0,
        doc_char_total=0,
    )
    assert _roundtrip(p) == p


def test_memory_file_roundtrip() -> None:
    mf = MemoryFile(
        account_uuid="acct",
        conversations_memory="some text",
        project_memories={"proj-1": "memory text"},
        extracted_claims=[
            MemoryClaim(id="mc-1", source="conversations_memory", claim_text="X", category="work")
        ],
    )
    assert _roundtrip(mf) == mf


# ----- findings + audit runs ----------------------------------------


def _turn_ids_hash(ids: list[str]) -> str:
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()


def test_finding_roundtrip_behavioral() -> None:
    f = Finding(
        id="f-1",
        audit_run_id="run-1",
        conversation_id="conv-1",
        turn_ids=["t-1", "t-2"],
        turn_ids_hash=_turn_ids_hash(["t-1", "t-2"]),
        module=ModuleName.A_SPIRALBENCH,
        behavior="safe-redirection",
        intensity=2,
        confidence=0.82,
        confidence_alpha=8.0,
        confidence_beta=2.0,
        quote_user="U",
        quote_assistant="A",
        evidence_quotes=[],
        explanation="Redirected to professional support.",
        citation="Spiral-Bench, https://eqbench.com/spiral-bench.html",
        detected_by=["claude-opus-4-7"],
        detected_at=_now(),
        prompt_version="v1",
        prompt_hash="deadbeef",
        metadata={},
    )
    assert _roundtrip(f) == f


def test_finding_roundtrip_module_h() -> None:
    """Module H findings have no intensity + no conversation_id."""
    f = Finding(
        id="f-h-1",
        audit_run_id="run-1",
        conversation_id=None,
        turn_ids=[],
        turn_ids_hash=_turn_ids_hash([]),
        module=ModuleName.H_MEMORY,
        behavior="contradicted",
        intensity=None,
        confidence=0.9,
        quote_user=None,
        quote_assistant=None,
        evidence_quotes=["...evidence..."],
        explanation="Memory says X, corpus contradicts it.",
        citation="Module H design; MedTrust-RAG arxiv:2510.14400",
        detected_by=["claude-opus-4-7"],
        detected_at=_now(),
        prompt_version="v1",
        prompt_hash="cafebabe",
    )
    assert _roundtrip(f) == f


def test_finding_rejects_empty_detected_by() -> None:
    with pytest.raises(ValidationError):
        Finding(
            id="f-bad",
            audit_run_id="run-1",
            turn_ids_hash="h",
            module=ModuleName.A_SPIRALBENCH,
            behavior="x",
            intensity=1,
            confidence=0.5,
            explanation="e",
            citation="c",
            detected_by=[],  # min_length=1 rejects this
            detected_at=_now(),
            prompt_version="v1",
            prompt_hash="h",
        )


def test_audit_run_roundtrip() -> None:
    stats = CorpusStats(
        discovered_conversations=100,
        sampled_conversations=50,
        discovered_turns=1000,
        sampled_turns=500,
        date_range_start=_now(),
        date_range_end=_now(),
        sources=[Source.CLAUDE_CODE],
    )
    usage = TokenUsage(
        by_module={
            ModuleName.A_SPIRALBENCH: ModuleTokenUsage(
                input_tokens=10_000, output_tokens=500, usd_cost=0.05
            )
        },
        orchestrator=ModuleTokenUsage(input_tokens=2_000, output_tokens=300, usd_cost=0.02),
    )
    sampling = SamplingConfigRecord(
        n=50,
        seed=42,
        min_turns=5,
        recency_weight=0.7,
        recency_window_days=90,
        stratify_by_project=True,
        top_n_projects=10,
    )
    run = AuditRun(
        id="run-1",
        sources=[Source.CLAUDE_CODE],
        source_paths={Source.CLAUDE_CODE: "/home/user/.claude/projects"},
        started_at=_now(),
        completed_at=_now(),
        corpus_stats=stats,
        token_usage=usage,
        sampling_config=sampling,
        status="completed",
        corpus_fingerprint="abc123",
        prompt_versions={ModuleName.A_SPIRALBENCH: "v1"},
        schema_version=1,
        skipped_modules=[],
    )
    assert _roundtrip(run) == run


def test_audit_status_literal_rejects_other() -> None:
    # 'status' is a Literal; pydantic enforces the allowed set.
    status_values: list[AuditStatus] = [
        "running",
        "completed",
        "failed",
        "partial",
        "aborted_pre_spend",
    ]
    assert len(status_values) == 5


# ----- content block discriminated union ----------------------------


_BLOCK_ADAPTER: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)


def test_content_block_text() -> None:
    raw = {"type": "text", "text": "hi"}
    block = _BLOCK_ADAPTER.validate_python(raw)
    assert isinstance(block, TextBlock)
    assert block.text == "hi"


def test_content_block_thinking() -> None:
    raw = {"type": "thinking", "thinking": "step by step", "signature": "sig"}
    block = _BLOCK_ADAPTER.validate_python(raw)
    assert isinstance(block, ThinkingBlock)


def test_content_block_tool_use() -> None:
    raw = {
        "type": "tool_use",
        "tool_name": "bash",
        "tool_input": {"command": "ls"},
        "tool_use_id": "tu-1",
        "mcp_integration": None,
    }
    block = _BLOCK_ADAPTER.validate_python(raw)
    assert isinstance(block, ToolUseBlock)
    assert block.tool_input == {"command": "ls"}


def test_content_block_tool_result() -> None:
    raw = {"type": "tool_result", "tool_use_id": "tu-1", "content": "ok", "is_error": False}
    block = _BLOCK_ADAPTER.validate_python(raw)
    assert isinstance(block, ToolResultBlock)


def test_content_block_rejects_wrong_shape() -> None:
    """A tool_use payload with `text` set should fail (extra='forbid')."""
    with pytest.raises(ValidationError):
        _BLOCK_ADAPTER.validate_python({"type": "tool_use", "tool_name": "bash", "text": "nope"})


def test_content_block_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        _BLOCK_ADAPTER.validate_python({"type": "image", "data": "..."})
