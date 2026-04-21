"""Shared pytest fixtures.

Session-level env isolation: we strip `ANTHROPIC_API_KEY`,
`VOYAGE_API_KEY`, and `LUCID_ALLOW_UNATTENDED` from `os.environ` at
collection time so tests never make live API calls or trip over shell
state. Tests that WANT the key (integration-style) should set it
explicitly via `monkeypatch.setenv(...)`.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_api_env() -> None:
    for var in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "LUCID_ALLOW_UNATTENDED"):
        os.environ.pop(var, None)
