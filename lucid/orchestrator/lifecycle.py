"""Managed Agents lifecycle: versioned reuse + stale pruning.

One long-lived agent per :data:`PROMPT_VERSION` is the steady state. Each
audit run calls :func:`get_or_create_orchestrator_agent` — find-first by
exact name, create only if absent. Any ``lucid-orchestrator-*`` agents
whose version suffix is older than the current ``PROMPT_VERSION`` are
archived by :func:`prune_stale_orchestrator_agents` so the Anthropic
console auto-prunes on every run.

Archiving is the SDK's soft-delete: :py:meth:`agents.archive` removes the
agent from the default ``agents.list()`` response (``include_archived``
defaults to ``False``). There is no hard-delete on the agent surface as
of anthropic-sdk 0.96.

The public surface also exposes :func:`wipe_all_lucid_agents` for a
one-shot cleanup from the CLI (``lucid cleanup-agents``).

.. note::
   ``PROMPT_VERSION`` used to live in ``lucid.orchestrator.system_prompt``
   alongside the retired Managed-Agents orchestrator prompt. The scoring
   loop replaces the orchestrator session, so ``system_prompt.py`` is
   deleted; the version constant is inlined here solely to drive the
   agent-naming scheme until this module migrates to
   ``lucid/synthesis/`` in Phase 3.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

_LOGGER = logging.getLogger(__name__)


# Kept as a module-level constant (not read from system_prompt.py, which
# is deleted) so the existing ``lucid-orchestrator-v<N>`` agents on the
# Anthropic console remain reachable for pruning in Phase 3.
PROMPT_VERSION = "v3"

LUCID_AGENT_NAME_PREFIX = "lucid-"
LUCID_ORCHESTRATOR_PREFIX = "lucid-orchestrator-"


def current_orchestrator_agent_name() -> str:
    """Return the name the current code uses when creating an orchestrator agent."""
    return f"{LUCID_ORCHESTRATOR_PREFIX}{PROMPT_VERSION}"


def iter_lucid_agents(client: Any, *, include_archived: bool = False) -> Iterator[Any]:
    """Yield each agent whose name starts with ``lucid-``.

    Pagination is handled by the SDK's ``SyncPageCursor`` — iterating the
    result of ``agents.list()`` walks pages transparently.
    """
    for agent in client.beta.agents.list(include_archived=include_archived):
        name = getattr(agent, "name", None) or ""
        if isinstance(name, str) and name.startswith(LUCID_AGENT_NAME_PREFIX):
            yield agent


def classify_agent(agent: Any) -> str:
    """Classify a ``lucid-*`` agent as ``current``, ``stale``, or ``foreign``.

    ``current``  - name matches the current orchestrator name (reuse target).
    ``stale``    - starts with ``lucid-orchestrator-`` but version differs.
    ``foreign``  - other ``lucid-*`` agent (experimental, smoke, custom).
    """
    name = getattr(agent, "name", "") or ""
    if name == current_orchestrator_agent_name():
        return "current"
    if name.startswith(LUCID_ORCHESTRATOR_PREFIX):
        return "stale"
    return "foreign"


def archive_agents(client: Any, agent_ids: list[str]) -> dict[str, str]:
    """Archive each agent by id; return ``{id: 'ok' | error_string}``.

    Catches and stringifies any exception so one bad id doesn't block the
    rest of the sweep. The caller (typically :func:`wipe_all_lucid_agents`
    or :func:`prune_stale_orchestrator_agents`) logs the summary.
    """
    results: dict[str, str] = {}
    for agent_id in agent_ids:
        try:
            client.beta.agents.archive(agent_id)
        except Exception as err:
            results[agent_id] = f"error: {type(err).__name__}: {err}"
            _LOGGER.warning("Failed to archive agent %s: %s", agent_id, err)
        else:
            results[agent_id] = "ok"
    return results


def get_or_create_orchestrator_agent(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    tools: list[dict[str, Any]],
) -> str:
    """Find-first-by-exact-name; create only when no existing agent matches.

    Returns the agent id. After a :data:`PROMPT_VERSION` bump the old
    name no longer matches, so a fresh agent is created; the stale ones
    should be pruned with :func:`prune_stale_orchestrator_agents`
    immediately after.
    """
    want_name = current_orchestrator_agent_name()
    for agent in iter_lucid_agents(client):
        if getattr(agent, "name", None) == want_name:
            _LOGGER.info("Reusing existing orchestrator agent %s (%s)", agent.id, want_name)
            return str(agent.id)
    agent = client.beta.agents.create(
        name=want_name,
        model=model,
        system=system_prompt,
        tools=tools,
    )
    _LOGGER.info("Created new orchestrator agent %s (%s)", agent.id, want_name)
    return str(agent.id)


def prune_stale_orchestrator_agents(client: Any) -> dict[str, str]:
    """Archive ``lucid-orchestrator-*`` agents whose version != current.

    Called after :func:`get_or_create_orchestrator_agent` on every audit
    run so the console steady state is one agent per active PROMPT_VERSION.
    Returns ``{id: 'ok' | error}`` for any stale agents found. Foreign
    ``lucid-*`` agents are left alone — use :func:`wipe_all_lucid_agents`
    for a full sweep.
    """
    stale_ids = [
        str(agent.id) for agent in iter_lucid_agents(client) if classify_agent(agent) == "stale"
    ]
    if not stale_ids:
        return {}
    _LOGGER.info("Pruning %d stale orchestrator agent(s)", len(stale_ids))
    return archive_agents(client, stale_ids)


def wipe_all_lucid_agents(client: Any, *, dry_run: bool = False) -> dict[str, dict[str, str]]:
    """Archive every ``lucid-*`` agent. Returns per-id summary rows.

    Shape: ``{agent_id: {"name": ..., "status": "ok"|"error: ..."|"would-archive"}}``.
    When ``dry_run=True`` nothing is archived; the status is
    ``"would-archive"`` for every row so callers can preview.
    """
    agents = list(iter_lucid_agents(client))
    if not agents:
        return {}
    rows: dict[str, dict[str, str]] = {
        str(a.id): {"name": str(getattr(a, "name", "")), "status": "would-archive"} for a in agents
    }
    if dry_run:
        return rows
    statuses = archive_agents(client, list(rows.keys()))
    for agent_id, row in rows.items():
        row["status"] = statuses.get(agent_id, "unknown")
    return rows


__all__ = [
    "LUCID_AGENT_NAME_PREFIX",
    "LUCID_ORCHESTRATOR_PREFIX",
    "archive_agents",
    "classify_agent",
    "current_orchestrator_agent_name",
    "get_or_create_orchestrator_agent",
    "iter_lucid_agents",
    "prune_stale_orchestrator_agents",
    "wipe_all_lucid_agents",
]
