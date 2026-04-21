"""Smoke test: package imports and CLI wires up."""

from __future__ import annotations

from typer.testing import CliRunner

from lucid import __version__
from lucid.cli import app


def test_version_constant_is_semver_like() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_cli_version_subcommand() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Lucid" in result.stdout
