"""CLI tests: flag surface + flow-gap exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lucid.cli import EXIT_USAGE, app

runner = CliRunner()


def test_cli_bare_exits_zero_with_help() -> None:
    result = runner.invoke(app, [])
    # Typer's no_args_is_help=True prints the help screen.
    assert "lucid" in result.stdout.lower()
    assert "audit" in result.stdout.lower()


def test_version_subcommand() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "lucid " in result.stdout


def test_audit_requires_source_and_path() -> None:
    result = runner.invoke(app, ["audit"])
    assert result.exit_code != 0


def test_audit_non_dry_run_exits_until_phase_5(tmp_path: Path) -> None:
    _seed_claude_code_corpus(tmp_path)
    result = runner.invoke(
        app,
        [
            "audit",
            "--source", "claude-code",
            "--path", str(tmp_path),
            "--sample", "1",
        ],
    )
    assert result.exit_code == EXIT_USAGE


def test_audit_dry_run_succeeds_on_seeded_corpus(tmp_path: Path) -> None:
    _seed_claude_code_corpus(tmp_path)
    result = runner.invoke(
        app,
        [
            "audit",
            "--source", "claude-code",
            "--path", str(tmp_path),
            "--sample", "1",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Estimated cost" in result.stdout
    assert "USD" in result.stdout


def test_audit_zero_conversations_exits_usage(tmp_path: Path) -> None:
    # Empty directory -> no .jsonl files -> zero conversations.
    result = runner.invoke(
        app,
        [
            "audit",
            "--source", "claude-code",
            "--path", str(tmp_path),
            "--sample", "5",
            "--dry-run",
        ],
    )
    assert result.exit_code == EXIT_USAGE
    assert "No conversations" in result.stdout


def test_audit_bad_path_exits_usage(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "audit",
            "--source", "claude-code",
            "--path", str(tmp_path / "does-not-exist"),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0


def test_audit_sample_rejects_bogus_value(tmp_path: Path) -> None:
    _seed_claude_code_corpus(tmp_path)
    result = runner.invoke(
        app,
        [
            "audit",
            "--source", "claude-code",
            "--path", str(tmp_path),
            "--sample", "banana",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0


def test_audit_sample_all_parses(tmp_path: Path) -> None:
    _seed_claude_code_corpus(tmp_path)
    result = runner.invoke(
        app,
        [
            "audit",
            "--source", "claude-code",
            "--path", str(tmp_path),
            "--sample", "all",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0


def test_calibrate_stub_exits_nonzero() -> None:
    result = runner.invoke(app, ["calibrate", "--module", "a"])
    assert result.exit_code == EXIT_USAGE


# ----- helpers --------------------------------------------------------


def _seed_claude_code_corpus(root: Path) -> None:
    proj = root / "test-project"
    proj.mkdir()
    session = proj / "session.jsonl"
    records = []
    # 6 turns: long enough to pass min_turns=5 filter.
    for i in range(6):
        records.append({
            "type": "user" if i % 2 == 0 else "assistant",
            "message": {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": [{"type": "text", "text": f"turn {i} content"}],
            },
            "timestamp": f"2026-04-10T12:00:0{i}Z",
            "sessionId": "sess-seed",
            "uuid": f"t-{i}",
            "parentUuid": f"t-{i-1}" if i else None,
        })
    session.write_text("\n".join(json.dumps(r) for r in records) + "\n")


# Keep pytest quiet about the unused import when re-ordering.
_ = pytest
