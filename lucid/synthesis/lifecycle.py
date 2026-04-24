"""Managed Agents lifecycle: versioned reuse + stale pruning.

One long-lived agent per synthesis prompt version is the steady state.
Each audit run calls :func:`get_or_create_synthesis_agent` with the
current ``prompt_version`` — find-first by exact name, create only if
absent. Any agents whose name prefix matches either the current
synthesis scheme or a legacy one (see :data:`LUCID_LEGACY_PREFIXES`)
but whose version suffix differs from the current version are archived
by :func:`prune_stale_synthesis_agents` so the Anthropic console
auto-prunes on every run.

Archiving is the SDK's soft-delete: :py:meth:`agents.archive` removes
the agent from the default ``agents.list()`` response
(``include_archived`` defaults to ``False``). There is no hard-delete
on the agent surface as of anthropic-sdk 0.96.

The public surface also exposes :func:`wipe_all_lucid_agents` for a
one-shot cleanup from the CLI (``lucid cleanup-agents``).

.. note::
   The agent-naming functions take ``prompt_version`` as a parameter
   rather than closing over a module-level constant. Phase 4.2 will
   introduce ``SYNTHESIS_PROMPT_VERSION`` at the prompt-loading
   boundary; until then, callers (CLI, upcoming ``synthesis.session``)
   pass the string literal explicitly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

_LOGGER = logging.getLogger(__name__)


LUCID_AGENT_NAME_PREFIX = "lucid-"
LUCID_SYNTHESIS_PREFIX = "lucid-synthesis-"

# Prefixes of legacy agents that should be archived alongside the
# current synthesis agents. When Phase 3 shipped, Anthropic consoles
# had ``lucid-orchestrator-v1..v3`` agents from the removed
# orchestrator. Pruning must still reach them so the console doesn't
# accumulate abandoned rows. Future prefix changes go here too.
LUCID_LEGACY_PREFIXES: tuple[str, ...] = ("lucid-orchestrator-",)


def current_synthesis_agent_name(prompt_version: str) -> str:
    """Return the canonical name for the synthesis agent at this prompt version.

    Example: ``current_synthesis_agent_name("v1")`` → ``"lucid-synthesis-v1"``.
    """
    return f"{LUCID_SYNTHESIS_PREFIX}{prompt_version}"


def iter_lucid_agents(client: Any, *, include_archived: bool = False) -> Iterator[Any]:
    """Yield each agent whose name starts with ``lucid-``.

    Pagination is handled by the SDK's ``SyncPageCursor`` — iterating the
    result of ``agents.list()`` walks pages transparently.
    """
    for agent in client.beta.agents.list(include_archived=include_archived):
        name = getattr(agent, "name", None) or ""
        if isinstance(name, str) and name.startswith(LUCID_AGENT_NAME_PREFIX):
            yield agent


def classify_agent(agent: Any, *, current_prompt_version: str) -> str:
    """Classify a ``lucid-*`` agent as ``current``, ``stale``, or ``foreign``.

    ``current``  - name matches the current synthesis name (reuse target).
    ``stale``    - starts with the current synthesis prefix or any legacy
                   prefix in :data:`LUCID_LEGACY_PREFIXES` but the version
                   suffix differs from the current one.
    ``foreign``  - other ``lucid-*`` agent (experimental, smoke, custom).
    """
    name = getattr(agent, "name", "") or ""
    if name == current_synthesis_agent_name(current_prompt_version):
        return "current"
    stale_prefixes = (LUCID_SYNTHESIS_PREFIX, *LUCID_LEGACY_PREFIXES)
    if any(name.startswith(prefix) for prefix in stale_prefixes):
        return "stale"
    return "foreign"


def archive_agents(client: Any, agent_ids: list[str]) -> dict[str, str]:
    """Archive each agent by id; return ``{id: 'ok' | error_string}``.

    Catches and stringifies any exception so one bad id doesn't block
    the rest of the sweep. The caller (typically
    :func:`wipe_all_lucid_agents` or
    :func:`prune_stale_synthesis_agents`) logs the summary.
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


def get_or_create_synthesis_agent(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    tools: list[dict[str, Any]],
    prompt_version: str,
) -> str:
    """Find or create the synthesis agent for this prompt version.

    Agents are named ``lucid-synthesis-v<prompt_version>`` so one agent
    serves every run at the same prompt version — the warm internal
    cache survives across audits. A prompt-version bump lands in a
    different name; :func:`prune_stale_synthesis_agents` archives older
    ones on subsequent runs.

    Returns the agent id.
    """
    want_name = current_synthesis_agent_name(prompt_version)
    for agent in iter_lucid_agents(client):
        if getattr(agent, "name", None) == want_name:
            _LOGGER.info("Reusing existing synthesis agent %s (%s)", agent.id, want_name)
            return str(agent.id)
    agent = client.beta.agents.create(
        name=want_name,
        model=model,
        system=system_prompt,
        tools=tools,
    )
    _LOGGER.info("Created new synthesis agent %s (%s)", agent.id, want_name)
    return str(agent.id)


def prune_stale_synthesis_agents(client: Any, *, current_prompt_version: str) -> dict[str, str]:
    """Archive stale synthesis agents and any legacy orchestrator agents.

    Keeps exactly the one agent matching
    ``current_synthesis_agent_name(current_prompt_version)``. Archives
    anything else whose name starts with :data:`LUCID_SYNTHESIS_PREFIX`
    or any prefix in :data:`LUCID_LEGACY_PREFIXES`. Foreign ``lucid-*``
    agents are left alone — use :func:`wipe_all_lucid_agents` for a
    full sweep.

    Failures per-agent are logged and continue — prune is best-effort.
    Returns ``{id: 'ok' | error}`` for any stale agents found.
    """
    stale_ids = [
        str(agent.id)
        for agent in iter_lucid_agents(client)
        if classify_agent(agent, current_prompt_version=current_prompt_version) == "stale"
    ]
    if not stale_ids:
        return {}
    _LOGGER.info("Pruning %d stale synthesis/legacy agent(s)", len(stale_ids))
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
    "LUCID_LEGACY_PREFIXES",
    "LUCID_SYNTHESIS_PREFIX",
    "archive_agents",
    "classify_agent",
    "current_synthesis_agent_name",
    "get_or_create_synthesis_agent",
    "iter_lucid_agents",
    "prune_stale_synthesis_agents",
    "wipe_all_lucid_agents",
]
