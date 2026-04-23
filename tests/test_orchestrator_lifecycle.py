"""Tests for :mod:`lucid.orchestrator.lifecycle` — version-keyed agent reuse."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from lucid.orchestrator import lifecycle
from lucid.orchestrator.lifecycle import (
    LUCID_ORCHESTRATOR_PREFIX,
    archive_agents,
    classify_agent,
    current_orchestrator_agent_name,
    get_or_create_orchestrator_agent,
    iter_lucid_agents,
    prune_stale_orchestrator_agents,
    wipe_all_lucid_agents,
)
from lucid.orchestrator.system_prompt import PROMPT_VERSION


def _agent(name: str, agent_id: str = "") -> MagicMock:
    a = MagicMock()
    a.name = name
    a.id = agent_id or f"ag_{name.replace('-', '_')}"
    return a


def _client_with_agents(agents: list[Any], archive_raises_on: set[str] | None = None) -> MagicMock:
    raises_on = archive_raises_on or set()
    client = MagicMock()

    def _list(*, include_archived: bool = False) -> list[Any]:
        # include_archived=True would in real life surface archived rows; the
        # fake just ignores it — tests never ask for archived agents.
        _ = include_archived
        return list(agents)

    def _archive(agent_id: str, **_kwargs: Any) -> Any:
        if agent_id in raises_on:
            raise RuntimeError(f"simulated archive failure for {agent_id}")
        return MagicMock(id=agent_id, archived=True)

    def _create(*, name: str, **_kwargs: Any) -> Any:
        created = MagicMock()
        created.id = f"ag_created_{name}"
        created.name = name
        agents.append(created)
        return created

    client.beta.agents.list.side_effect = _list
    client.beta.agents.archive.side_effect = _archive
    client.beta.agents.create.side_effect = _create
    return client


# ---------------------------------------------------------------------------
# classify_agent + iter_lucid_agents
# ---------------------------------------------------------------------------


def test_current_orchestrator_name_uses_prompt_version() -> None:
    assert current_orchestrator_agent_name() == f"{LUCID_ORCHESTRATOR_PREFIX}{PROMPT_VERSION}"


def test_classify_agent_current_stale_foreign() -> None:
    current = _agent(current_orchestrator_agent_name())
    stale = _agent(f"{LUCID_ORCHESTRATOR_PREFIX}v0")
    foreign = _agent("lucid-smoke-1")
    assert classify_agent(current) == "current"
    assert classify_agent(stale) == "stale"
    assert classify_agent(foreign) == "foreign"


def test_iter_lucid_agents_filters_non_lucid_names() -> None:
    agents = [
        _agent("lucid-orchestrator-v2"),
        _agent("some-other-agent"),
        _agent("lucid-smoke-run-1"),
    ]
    client = _client_with_agents(agents)
    names = sorted(a.name for a in iter_lucid_agents(client))
    assert names == ["lucid-orchestrator-v2", "lucid-smoke-run-1"]


def test_iter_lucid_agents_tolerates_agents_without_name() -> None:
    anon = MagicMock()
    anon.name = None
    anon.id = "ag_anon"
    client = _client_with_agents([anon, _agent("lucid-orchestrator-v2")])
    names = [a.name for a in iter_lucid_agents(client)]
    assert names == ["lucid-orchestrator-v2"]


# ---------------------------------------------------------------------------
# get_or_create_orchestrator_agent
# ---------------------------------------------------------------------------


def test_get_or_create_reuses_existing_agent() -> None:
    existing = _agent(current_orchestrator_agent_name(), agent_id="ag_existing")
    client = _client_with_agents([existing])
    agent_id = get_or_create_orchestrator_agent(
        client, model="claude-sonnet-4-6", system_prompt="hi", tools=[]
    )
    assert agent_id == "ag_existing"
    assert client.beta.agents.create.call_count == 0


def test_get_or_create_creates_when_no_match() -> None:
    client = _client_with_agents([_agent("lucid-orchestrator-v0")])
    agent_id = get_or_create_orchestrator_agent(
        client, model="claude-sonnet-4-6", system_prompt="hi", tools=[]
    )
    assert agent_id == f"ag_created_{current_orchestrator_agent_name()}"
    client.beta.agents.create.assert_called_once()
    kwargs = client.beta.agents.create.call_args.kwargs
    assert kwargs["name"] == current_orchestrator_agent_name()
    assert kwargs["model"] == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# prune_stale_orchestrator_agents
# ---------------------------------------------------------------------------


def test_prune_stale_archives_only_old_orchestrator_versions() -> None:
    agents = [
        _agent(current_orchestrator_agent_name(), agent_id="ag_current"),
        _agent(f"{LUCID_ORCHESTRATOR_PREFIX}v0", agent_id="ag_v0"),
        _agent(f"{LUCID_ORCHESTRATOR_PREFIX}v1", agent_id="ag_v1"),
        _agent("lucid-smoke-1", agent_id="ag_smoke"),
    ]
    client = _client_with_agents(agents)
    result = prune_stale_orchestrator_agents(client)
    assert set(result.keys()) == {"ag_v0", "ag_v1"}
    assert all(status == "ok" for status in result.values())
    archived_ids = {call.args[0] for call in client.beta.agents.archive.call_args_list}
    assert archived_ids == {"ag_v0", "ag_v1"}


def test_prune_stale_is_noop_when_only_current_and_foreign() -> None:
    agents = [
        _agent(current_orchestrator_agent_name(), agent_id="ag_current"),
        _agent("lucid-smoke-1", agent_id="ag_smoke"),
    ]
    client = _client_with_agents(agents)
    result = prune_stale_orchestrator_agents(client)
    assert result == {}
    assert client.beta.agents.archive.call_count == 0


def test_prune_stale_collects_archive_errors() -> None:
    agents = [
        _agent(f"{LUCID_ORCHESTRATOR_PREFIX}v0", agent_id="ag_v0"),
        _agent(f"{LUCID_ORCHESTRATOR_PREFIX}v1", agent_id="ag_v1"),
    ]
    client = _client_with_agents(agents, archive_raises_on={"ag_v1"})
    result = prune_stale_orchestrator_agents(client)
    assert result["ag_v0"] == "ok"
    assert result["ag_v1"].startswith("error: RuntimeError")


# ---------------------------------------------------------------------------
# wipe_all_lucid_agents
# ---------------------------------------------------------------------------


def test_wipe_all_archives_every_lucid_prefixed_agent() -> None:
    agents = [
        _agent(current_orchestrator_agent_name(), agent_id="ag_current"),
        _agent(f"{LUCID_ORCHESTRATOR_PREFIX}v0", agent_id="ag_v0"),
        _agent("lucid-smoke-1", agent_id="ag_smoke"),
        _agent("unrelated-agent", agent_id="ag_unrelated"),
    ]
    client = _client_with_agents(agents)
    result = wipe_all_lucid_agents(client, dry_run=False)
    assert set(result.keys()) == {"ag_current", "ag_v0", "ag_smoke"}
    assert all(row["status"] == "ok" for row in result.values())
    archived_ids = {call.args[0] for call in client.beta.agents.archive.call_args_list}
    assert archived_ids == {"ag_current", "ag_v0", "ag_smoke"}


def test_wipe_all_dry_run_does_not_mutate() -> None:
    agents = [
        _agent(current_orchestrator_agent_name(), agent_id="ag_current"),
        _agent("lucid-smoke-1", agent_id="ag_smoke"),
    ]
    client = _client_with_agents(agents)
    result = wipe_all_lucid_agents(client, dry_run=True)
    assert all(row["status"] == "would-archive" for row in result.values())
    assert client.beta.agents.archive.call_count == 0


def test_wipe_all_empty_returns_empty_dict() -> None:
    client = _client_with_agents([])
    assert wipe_all_lucid_agents(client, dry_run=False) == {}
    assert wipe_all_lucid_agents(client, dry_run=True) == {}


# ---------------------------------------------------------------------------
# archive_agents (direct)
# ---------------------------------------------------------------------------


def test_archive_agents_returns_per_id_status() -> None:
    client = _client_with_agents([], archive_raises_on={"ag_bad"})
    result = archive_agents(client, ["ag_good", "ag_bad"])
    assert result["ag_good"] == "ok"
    assert result["ag_bad"].startswith("error: RuntimeError")


def test_archive_agents_empty_input_noop() -> None:
    client = _client_with_agents([])
    assert archive_agents(client, []) == {}
    assert client.beta.agents.archive.call_count == 0


# ---------------------------------------------------------------------------
# CLI integration — a thin smoke to verify wiring + confirmation path.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_client_with_agents(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Patch the CLI's client factory to return a controlled client."""
    agents: list[Any] = [
        _agent(current_orchestrator_agent_name(), agent_id="ag_current"),
        _agent(f"{LUCID_ORCHESTRATOR_PREFIX}v0", agent_id="ag_v0"),
        _agent("lucid-smoke-1", agent_id="ag_smoke"),
    ]
    client = _client_with_agents(agents)

    import lucid.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "_build_anthropic_clients_or_exit",
        lambda: (client, object()),
    )
    # Bypass the conftest env-strip so the CLI doesn't bail early on
    # missing ANTHROPIC_API_KEY — the fake factory above renders the
    # real one unreachable.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return agents


def test_cli_cleanup_agents_dry_run_lists_without_mutating(
    fake_client_with_agents: list[Any],
) -> None:
    from typer.testing import CliRunner

    from lucid.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["cleanup-agents", "--dry-run"])
    assert result.exit_code == 0, result.stdout
    assert "would-archive" in result.stdout or "Dry-run" in result.stdout


def test_cli_cleanup_agents_all_with_yes_archives_everything(
    fake_client_with_agents: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    from lucid.cli import app

    # Track archives through the patched lifecycle module to be sure the
    # CLI routes through `wipe_all_lucid_agents` on --all (and not the
    # stale-only path).
    calls: list[str] = []
    original = lifecycle.wipe_all_lucid_agents

    def _spy(client: Any, *, dry_run: bool = False) -> Any:
        calls.append("wipe_all")
        return original(client, dry_run=dry_run)

    monkeypatch.setattr("lucid.cli.wipe_all_lucid_agents", _spy, raising=False)
    # The CLI imports lazily inside the command body, so patch the source.
    monkeypatch.setattr(lifecycle, "wipe_all_lucid_agents", _spy)

    runner = CliRunner()
    result = runner.invoke(app, ["cleanup-agents", "--all", "--yes"])
    assert result.exit_code == 0, result.stdout
    assert "Archived 3/3" in result.stdout
    assert calls == ["wipe_all"]
