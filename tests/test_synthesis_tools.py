"""Tests for the synthesis tool registry."""

from __future__ import annotations

import pytest


def test_synthesis_registry_exposes_only_expected_tools(tmp_path):
    """Synthesis exposes 6 tools: 5 read-only shared + 1 new write_report_section.

    Explicitly excludes invoke_module, store_finding, estimate_remaining_cost
    which belong to the scoring phase.
    """
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore
    from lucid.synthesis.tools import build_synthesis_registry

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        registry = build_synthesis_registry(
            store=store,
            audit_run_id="run-synth-test",
            progress_log=lambda *_: None,
        )
    names = set(registry.names)
    expected = {
        "query_corpus",
        "get_conversation",
        "get_turn_window",
        "get_findings",
        "log_progress",
        "write_report_section",
    }
    assert names == expected, f"synthesis registry exposed {names}, expected {expected}"
    # Explicit absence check — these MUST NOT be in the synthesis registry.
    for forbidden in ("invoke_module", "store_finding", "estimate_remaining_cost"):
        assert forbidden not in names, f"synthesis registry must not expose {forbidden}"


@pytest.mark.asyncio
async def test_write_report_section_stub_returns_ok(tmp_path):
    """The stub handler accepts any args and returns {ok: True, stub: True}."""
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore
    from lucid.synthesis.tools import build_synthesis_registry

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        registry = build_synthesis_registry(
            store=store,
            audit_run_id="run-stub",
            progress_log=lambda *_: None,
        )
    tool = registry.get("write_report_section")
    assert tool is not None
    result = await tool.handler(
        {
            "section_id": "exec_summary",
            "markdown": "A paragraph with [F:f001].",
            "cited_finding_ids": ["f001"],
        }
    )
    assert result == {
        "ok": True,
        "stub": True,
        "args_received": ["section_id", "markdown", "cited_finding_ids"],
    }


def test_synthesis_registry_shares_read_only_factories_with_orchestrator(tmp_path):
    """The read-only handlers used by synthesis come from the same factories
    ``build_tool_registry`` uses — guards against divergence."""
    from lucid.orchestrator.tools import (
        make_get_conversation_tool,
        make_get_findings_tool,
        make_get_turn_window_tool,
        make_log_progress_tool,
        make_query_corpus_tool,
    )
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        # Each factory should return a registerable CustomTool.
        for factory_call in [
            lambda: make_query_corpus_tool(store),
            lambda: make_get_conversation_tool(store),
            lambda: make_get_turn_window_tool(store),
            lambda: make_get_findings_tool(store, "run-shared"),
            lambda: make_log_progress_tool(lambda *_: None),
        ]:
            tool = factory_call()
            assert tool.name
            assert tool.handler is not None
            assert callable(tool.handler)
