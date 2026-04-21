"""Shared pytest fixtures.

**Session-level env isolation:** we strip ``ANTHROPIC_API_KEY``,
``VOYAGE_API_KEY``, and ``LUCID_ALLOW_UNATTENDED`` from ``os.environ`` at
collection time so tests never make live API calls or trip over shell
state. Tests that WANT the key (integration-style) should set it
explicitly via ``monkeypatch.setenv(...)``.

**mock_anthropic_client:** factory-style fixture that returns an
``AsyncMock`` emulating ``client.messages.parse``. Tests build a stub
response object with ``.parsed_output`` (the pydantic model the SDK would
return), ``.usage`` (real ``anthropic.types.Usage``), and the handful of
other fields ``ParsedMessage`` carries. Keeping the shape close to the
real SDK guards against regressions when the SDK bumps and also avoids
needing the full HTTP stack for unit tests.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic.types import Usage


@pytest.fixture(autouse=True, scope="session")
def _isolate_api_env() -> None:
    for var in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "LUCID_ALLOW_UNATTENDED"):
        os.environ.pop(var, None)


def _default_usage() -> Usage:
    """A realistic Usage payload. Non-zero cache read so cache-hit assertions
    against the mock don't silently pass because the fixture looks like a
    cold start."""
    return Usage(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=4200,
    )


@pytest.fixture
def mock_anthropic_client() -> Callable[..., MagicMock]:
    """Return a factory that builds a mocked AsyncAnthropic client.

    Usage::

        def test_something(mock_anthropic_client):
            outputs = [SpiralBenchScore(...), SpiralBenchScore(...)]
            client = mock_anthropic_client(parse_outputs=outputs)
            module = ModuleASpiralBench(client=client)
            ...

    The factory's ``parse_outputs`` sequence is consumed in order — one
    entry per expected ``messages.parse`` call. Tests that make more
    calls than outputs raise ``IndexError`` (fail loudly rather than
    silently reuse the last response).
    """

    def _factory(
        *,
        parse_outputs: Sequence[Any] = (),
        usage: Usage | None = None,
    ) -> MagicMock:
        usage_payload = usage or _default_usage()

        responses = [
            _make_response(output, usage_payload) for output in parse_outputs
        ]

        async def _parse(**_kwargs: object) -> Any:
            if not responses:
                raise IndexError(
                    "mock_anthropic_client: messages.parse called but no "
                    "parse_outputs remain"
                )
            return responses.pop(0)

        client = MagicMock()
        client.messages = MagicMock()
        client.messages.parse = AsyncMock(side_effect=_parse)
        return client

    return _factory


def _make_response(parsed_output: Any, usage: Usage) -> MagicMock:
    """Build a lightweight stand-in for ``ParsedMessage[T]``.

    We expose the attributes the module actually reads — ``parsed_output``
    and ``usage`` — and nothing else. Adding the full ``content`` +
    ``model`` + ``role`` surface would not help tests and would couple
    them to the SDK's internal shape.
    """
    response = MagicMock()
    response.parsed_output = parsed_output
    response.usage = usage
    return response
