"""Contracts every detection module implements.

Two shapes:

- :class:`CorpusModule` — reads a ``ModuleCorpus`` and produces findings.
  Modules A, B, C, D, E, F, H all fit here.
- :class:`FindingsModule` — reads existing findings (plus corpus for
  timestamps) and produces derived findings. Module G (deterministic
  time/model attribution) fits here.

A module run returns ``list[ModuleResult]`` where ``ModuleResult = Finding |
ModuleError``. Per-conversation failures are wrapped as ``ModuleError``
entries rather than raised, so one bad conversation never aborts a whole
module pass. Module-level crashes (transport death, malformed prompt file)
still raise — the orchestrator decides whether to mark the run ``partial``.

:func:`run_with_bounded_concurrency` wraps :func:`asyncio.gather` with a
:class:`asyncio.Semaphore` cap. Used by every LLM-backed module to keep
concurrent requests within the ``asyncio.Semaphore(10)`` ceiling documented
in the plan. Exceptions propagate — the module is responsible for catching
expected per-coroutine failures and converting them into ``ModuleError``
entries before handing coroutines to the helper.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from lucid.schemas import Conversation, Finding, ModuleName, Turn

__all__ = [
    "CorpusModule",
    "FindingsModule",
    "ModuleCorpus",
    "ModuleError",
    "ModuleResult",
    "extract_result_json",
    "run_with_bounded_concurrency",
]


@dataclass(frozen=True, slots=True)
class ModuleCorpus:
    """Read-only view of the sampled corpus handed to a module.

    ``conversations`` is keyed by conversation id for O(1) lookup during
    cross-referencing (Module H's memory-to-corpus matching, Module G's
    attribution lookup). ``turns_by_conversation`` preserves turn order per
    conversation — modules should never re-sort.

    ``audit_run_id`` propagates into every emitted ``Finding.audit_run_id``
    without the module having to thread it through its own state.
    """

    conversations: Mapping[str, Conversation]
    turns_by_conversation: Mapping[str, Sequence[Turn]]
    audit_run_id: str


@dataclass(frozen=True, slots=True)
class ModuleError:
    """A recoverable, per-conversation (or module-global) failure.

    Frozen so the orchestrator can pass error lists across tasks without
    worrying about aliasing. ``conversation_id`` is ``None`` for
    module-global errors (e.g. Module H's embedding endpoint down — affects
    the whole module run, not one conversation).

    ``message`` must never contain raw user-conversation content — callers
    are responsible for redacting. ``SafeFormatter`` in ``lucid.logging``
    catches the common case, but a bug that puts a corpus excerpt into
    ``message`` would persist through the DB if not caught here.
    """

    module: ModuleName
    conversation_id: str | None
    error_type: str
    message: str


ModuleResult = Finding | ModuleError


class CorpusModule(Protocol):
    """A detection module that reads corpus and emits findings.

    Implementers must expose ``module_name`` and ``prompt_version`` as
    attributes — the orchestrator reads them to populate provenance fields
    on persisted findings without calling the module.
    """

    module_name: ModuleName
    prompt_version: str

    async def run(self, corpus: ModuleCorpus) -> list[ModuleResult]: ...


class FindingsModule(Protocol):
    """A module that reads findings produced by other modules.

    Used by Module G (time/model attribution): it runs *after* behaviour
    modules and augments their findings with deterministic bucketing.
    """

    module_name: ModuleName
    prompt_version: str

    async def run(
        self,
        corpus: ModuleCorpus,
        findings: Sequence[Finding],
    ) -> list[ModuleResult]: ...


async def run_with_bounded_concurrency[T](
    coros: Iterable[Coroutine[Any, Any, T]],
    max_concurrency: int,
) -> list[T]:
    """Run many coroutines with a concurrency cap, preserving input order.

    Semantics match :func:`asyncio.gather`: results are returned in
    submission order, and the first exception from any coroutine cancels
    pending work and re-raises. Callers that want per-coroutine error
    isolation must wrap their coroutines in ``try/except`` before passing
    in (converting expected failures to ``ModuleError`` entries).
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _guarded(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_guarded(c) for c in coros))


_RESULT_MARKER = re.compile(r"\bRESULT\b\s*", re.MULTILINE)
_FENCE_OPEN = re.compile(r"^\s*```(?:json)?\s*\n", re.MULTILINE)
_FENCE_CLOSE = re.compile(r"\n\s*```\s*$", re.MULTILINE)


def extract_result_json(content: str) -> str:
    """Recover the JSON blob from a Lucid module's two-section response.

    Every Lucid module prompt asks the model to emit an optional
    ``REASONING`` section followed by a ``RESULT`` section that contains a
    single JSON object. Real model output drifts across three shapes we
    have to tolerate:

    1. Model returned bare JSON — already parses cleanly. Return as-is.
    2. Model returned ``REASONING … RESULT\\n<json>`` with or without
       markdown fences around the JSON. Strip everything up to and
       including ``RESULT``, then peel fences.
    3. Model returned prose plus a JSON blob somewhere inside. Locate the
       outermost ``{ … }`` span and return that.

    Raises nothing; returns the best-effort candidate. Downstream
    ``model_validate_json`` surfacing a ``ValidationError`` is how the
    module signals "model returned garbage" — the retry loop handles it.
    """
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    marker = _RESULT_MARKER.search(content)
    if marker is not None:
        rest = content[marker.end() :].strip()
        rest = _FENCE_OPEN.sub("", rest, count=1)
        rest = _FENCE_CLOSE.sub("", rest, count=1)
        rest = rest.strip()
        if rest:
            return rest

    first = content.find("{")
    last = content.rfind("}")
    if first != -1 and last > first:
        return content[first : last + 1].strip()

    return stripped
