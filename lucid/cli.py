"""Lucid CLI entry point.

Phase 4 wires `lucid audit --dry-run` end-to-end: discover -> sample ->
cost-estimate via `count_tokens` -> print summary -> exit. Real Managed
Agents orchestration lands in Phase 5; `lucid calibrate` lands in Phase 6.

Flag surface (locked by plan):

    lucid audit --source {claude-code,claude-ai,all} --path <p>
               [--sample N | --sample all | --projects p1,p2,...]
               [--dry-run] [--resume <run-id>]
               [--include-module-d]
               [--yes-i-authorize-spend-up-to N]
               [--log-level LEVEL]

    lucid calibrate --module {a,h} [--prompt-version VN]
"""

from __future__ import annotations

import os
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

# Load .env.local / .env into os.environ at import time so the CLI picks up
# API keys without requiring the user to shell-export them. config.load_settings
# does the same thing, but the CLI's non-dry-run path reads ANTHROPIC_API_KEY
# directly (not via Settings) so we invoke the loader here too.
from lucid.config import _load_dotenv_files

_load_dotenv_files()

from lucid import __version__  # noqa: E402
from lucid.cost import (  # noqa: E402
    COST_GATE_USD,
    DEFAULT_MODULES,
    CostEstimate,
    CostEstimator,
    TokenCounter,
    heuristic_counter,
    make_anthropic_counter,
)
from lucid.ingest.base import IngestError, fingerprint_corpus  # noqa: E402
from lucid.ingest.claude_ai import ClaudeAiAdapter  # noqa: E402
from lucid.ingest.claude_code import ClaudeCodeAdapter  # noqa: E402
from lucid.logging import configure_logging  # noqa: E402
from lucid.orchestrator.smoke import (  # noqa: E402
    SMOKE_KICKOFF_MESSAGE,
    SMOKE_PROMPT_VERSION,
    SMOKE_SYSTEM_PROMPT,
)
from lucid.run import AuditInputs, LockHeldError, run_audit  # noqa: E402
from lucid.sampling import SamplingConfig, sample_conversations  # noqa: E402
from lucid.schemas import Conversation, ModuleName, Source, Turn  # noqa: E402

app = typer.Typer(
    name="lucid",
    help=f"Lucid {__version__} — epistemic audit for personal AI conversation history.",
    no_args_is_help=True,
    add_completion=False,
)


_CONSOLE = Console()

# Exit codes follow BSD sysexits conventions where applicable:
#   0  OK
#   2  usage / configuration / input error
#   3  cost-gate rejection (user didn't authorize)
#   4  concurrent-audit lock collision
EXIT_USAGE = 2
EXIT_COST_GATE = 3
EXIT_LOCK = 4


# ──────────────────────────────────────────────────────────────────────────
# Shared argument types
# ──────────────────────────────────────────────────────────────────────────


class SourceChoice(StrEnum):
    CLAUDE_CODE = "claude-code"
    CLAUDE_AI = "claude-ai"
    ALL = "all"


class CalibrateModule(StrEnum):
    A = "a"
    H = "h"


def _parse_sample(raw: str) -> int | None:
    """`--sample 100` -> 100; `--sample all` -> None (meaning 'no cap')."""
    if raw.lower() == "all":
        return None
    try:
        n = int(raw)
    except ValueError as err:
        raise typer.BadParameter(f"--sample must be an int or 'all', got {raw!r}") from err
    if n < 1:
        raise typer.BadParameter("--sample must be >= 1 (or 'all')")
    return n


def _parse_projects(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    if not parts:
        raise typer.BadParameter("--projects was empty after trimming")
    return parts


# ──────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────


@app.command()
def version() -> None:
    """Print Lucid's version and exit."""
    typer.echo(f"lucid {__version__}")


@app.command()
def audit(
    source: Annotated[
        SourceChoice,
        typer.Option("--source", help="Corpus source to audit."),
    ],
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            help="Source directory. For claude-code: ~/.claude/projects. For claude-ai: unzipped export.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    sample: Annotated[
        str,
        typer.Option("--sample", help="Sample size. Integer or 'all'.", show_default=True),
    ] = "100",
    projects: Annotated[
        str | None,
        typer.Option("--projects", help="Comma-separated project slugs/uuids to keep."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Parse + sample + estimate; skip LLM calls."),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Resume an earlier audit by run id."),
    ] = None,
    include_module_d: Annotated[
        bool,
        typer.Option(
            "--include-module-d",
            help="Opt in to Module D (Jain perspective sycophancy). Off by default.",
        ),
    ] = False,
    yes_authorize: Annotated[
        int | None,
        typer.Option(
            "--yes-i-authorize-spend-up-to",
            help="Pre-authorize spend up to $N. Required with LUCID_ALLOW_UNATTENDED=1.",
        ),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level: DEBUG, INFO, WARNING, ERROR."),
    ] = "INFO",
) -> None:
    """Run an audit on a conversation corpus."""
    configure_logging(log_level)

    sample_n = _parse_sample(sample)
    project_filter = _parse_projects(projects)

    # Resume isn't plumbed yet; land in Phase 6 when calibration runs persist
    # enough state to be worth picking up.
    if resume is not None:
        _CONSOLE.print(
            "[yellow]--resume is not yet wired (Phase 6). "
            "Run without --resume for a fresh audit.[/yellow]"
        )
        raise typer.Exit(EXIT_USAGE)

    # ----- Dispatch adapters ----------------------------------------
    try:
        convs_by_id, turns_by_id = _ingest(source, path)
    except IngestError as err:
        _CONSOLE.print(f"[red]Ingest failed:[/red] {err}")
        raise typer.Exit(EXIT_USAGE) from err
    except FileNotFoundError as err:
        _CONSOLE.print(f"[red]Path not found:[/red] {err}")
        raise typer.Exit(EXIT_USAGE) from err

    if not convs_by_id:
        _CONSOLE.print(
            f"[yellow]No conversations discovered at {path}. "
            "Check the --path is correct and that you have at least one "
            "session of 5+ turns. See docs/BUILD_GUIDE.md §3 for the "
            "expected layout.[/yellow]"
        )
        raise typer.Exit(EXIT_USAGE)

    # ----- Sample --------------------------------------------------
    sampling_config = SamplingConfig(
        n=sample_n if sample_n is not None else len(convs_by_id),
        project_filter=project_filter,
    )
    sampled = sample_conversations(list(convs_by_id.values()), sampling_config)
    if sample_n is not None and len(sampled) < sample_n:
        _CONSOLE.print(
            f"[yellow]Requested --sample {sample_n} but only {len(sampled)} "
            "conversations matched filters; clamping to the available count.[/yellow]"
        )

    # ----- Cost estimate ------------------------------------------
    enabled = list(DEFAULT_MODULES)
    if include_module_d and ModuleName.D_PERSPECTIVE not in enabled:
        enabled.append(ModuleName.D_PERSPECTIVE)

    counter = _build_counter()
    estimator = CostEstimator(count_tokens=counter)
    estimate = estimator.estimate(
        convs=sampled,
        turns_by_conv={c.id: turns_by_id.get(c.id, []) for c in sampled},
        enabled_modules=enabled,
    )

    _render_summary(
        source=source,
        path=path,
        sampled=sampled,
        discovered=len(convs_by_id),
        estimate=estimate,
        enabled_modules=enabled,
    )

    if dry_run:
        return

    # ----- Cost gate ----------------------------------------------
    authorized_budget = _authorized_budget(estimate, yes_authorize)

    # ----- Hand off to runner ------------------------------------
    source_paths: dict[Source, Path] = {}
    if source in (SourceChoice.CLAUDE_CODE, SourceChoice.ALL):
        source_paths[Source.CLAUDE_CODE] = path
    if source in (SourceChoice.CLAUDE_AI, SourceChoice.ALL):
        source_paths[Source.CLAUDE_AI] = path

    fingerprint = fingerprint_corpus(
        (c.id, _hash_content(c, turns_by_id.get(c.id, []))) for c in sampled
    )
    inputs = AuditInputs(
        source_paths=source_paths,
        sampled=sampled,
        turns_by_conv={c.id: turns_by_id.get(c.id, []) for c in sampled},
        corpus_fingerprint=fingerprint,
        sampling_config=sampling_config,
        estimate=estimate,
        enabled_modules=enabled,
        authorized_budget_usd=authorized_budget,
    )

    # Phase 5B smoke kickoff. Phase 7 swaps to SYSTEM_PROMPT + audit-workflow kickoff.
    prompt_versions: dict[ModuleName, str] = {}

    data_dir = Path(".lucid")
    try:
        client = _anthropic_client_or_exit()
    except ImportError as err:
        _CONSOLE.print(f"[red]Anthropic SDK not installed:[/red] {err}")
        raise typer.Exit(EXIT_USAGE) from err

    _CONSOLE.print(
        f"[cyan]Launching Managed Agents session (Phase 5B smoke). "
        f"Budget ceiling: ${authorized_budget:.2f}. This will cost real money.[/cyan]"
    )

    def _progress(level: str, message: str) -> None:
        _CONSOLE.print(f"[dim]\\[agent][/dim] \\[{level}] {message}")

    try:
        result = run_audit(
            inputs=inputs,
            data_dir=data_dir,
            client=client,
            system_prompt=SMOKE_SYSTEM_PROMPT,
            kickoff_message=SMOKE_KICKOFF_MESSAGE,
            prompt_versions=prompt_versions,
            progress_log=_progress,
        )
    except LockHeldError as err:
        _CONSOLE.print(f"[red]Lock held:[/red] {err}")
        raise typer.Exit(EXIT_LOCK) from err

    _CONSOLE.print(
        f"[green]Audit {result.run_id} finished:[/green] status={result.status}, "
        f"reason={result.reason}, findings_written={result.findings_written}"
    )
    if result.outcome is not None:
        outcome = result.outcome
        _CONSOLE.print(
            f"[dim]events={outcome.events_received} tool_calls={outcome.tool_calls} "
            f"cache_read_tokens={outcome.cache_read_tokens} "
            f"cache_write_tokens={outcome.cache_write_tokens}[/dim]"
        )
    if result.status != "completed":
        raise typer.Exit(EXIT_USAGE)

    _ = SMOKE_PROMPT_VERSION  # exported for future prompt-version tracking


@app.command()
def calibrate(
    module: Annotated[
        CalibrateModule,
        typer.Option("--module", help="Which module to calibrate against ground truth."),
    ],
    human_labels: Annotated[
        Path | None,
        typer.Option(
            "--human-labels",
            help="JSONL of LabeledTurn rows from the human labeler (ground truth).",
        ),
    ] = None,
    judge_labels: Annotated[
        Path | None,
        typer.Option(
            "--judge-labels",
            help="JSONL of LabeledTurn rows produced by the LLM judge on the same turns.",
        ),
    ] = None,
    test_frac: Annotated[
        float,
        typer.Option(
            "--test-frac",
            help="Fraction of shared turns held out for reported metrics (rest is training).",
        ),
    ] = 0.3,
    seed: Annotated[
        int,
        typer.Option("--seed", help="RNG seed for the train/test split + bootstrap."),
    ] = 42,
    n_bootstrap: Annotated[
        int,
        typer.Option("--n-bootstrap", help="Resamples for BCa bootstrap CIs."),
    ] = 2000,
    behaviors_csv: Annotated[
        str | None,
        typer.Option(
            "--behaviors",
            help="Comma-separated subset of behaviors to score. Default: all 17.",
        ),
    ] = None,
    prompt_version: Annotated[
        str | None,
        typer.Option("--prompt-version", help="Pin to a specific prompt version (e.g. 'v2')."),
    ] = None,
    write_markdown: Annotated[
        Path | None,
        typer.Option(
            "--write-markdown",
            help="If set, write a markdown report to this path (append-safe).",
        ),
    ] = None,
) -> None:
    """Compute inter-annotator agreement between human and judge labels.

    Phase 6A scope: compares two pre-computed ``LabeledTurn`` JSONL files
    (human ground truth and LLM-judge predictions) and reports per-behavior
    Krippendorff α, Gwet AC1, Cohen κ, and QWK on intensity — each with a
    95% BCa bootstrap CI. Does not call the LLM. Phase 6B will add an
    ``--auto-judge`` flag that runs Module A on the held-out split and
    generates the judge labels in-place (behind the cost gate).
    """
    if module != CalibrateModule.A:
        _CONSOLE.print(
            f"[yellow]Calibration for Module {module.value.upper()} lands in a later phase "
            "(Phase 8 for H; other modules do not have calibration targets).[/yellow]"
        )
        raise typer.Exit(EXIT_USAGE)

    if human_labels is None or judge_labels is None:
        _CONSOLE.print(
            "[red]`lucid calibrate --module a` requires "
            "--human-labels PATH and --judge-labels PATH.[/red]"
        )
        _CONSOLE.print(
            "[dim]Each file is JSONL of LabeledTurn rows "
            "(see lucid/calibration/data.py for schema).[/dim]"
        )
        raise typer.Exit(EXIT_USAGE)

    _run_calibrate_module_a(
        human_labels_path=human_labels,
        judge_labels_path=judge_labels,
        test_frac=test_frac,
        seed=seed,
        n_bootstrap=n_bootstrap,
        behaviors_csv=behaviors_csv,
        prompt_version_override=prompt_version,
        write_markdown=write_markdown,
    )


def _run_calibrate_module_a(
    *,
    human_labels_path: Path,
    judge_labels_path: Path,
    test_frac: float,
    seed: int,
    n_bootstrap: int,
    behaviors_csv: str | None,
    prompt_version_override: str | None,
    write_markdown: Path | None,
) -> None:
    """Module A calibration flow. Pulled out of the Typer command for
    readability + direct unit-test coverage without CliRunner plumbing."""
    from lucid.calibration.data import load_hand_labels, train_test_split
    from lucid.calibration.report import compute_calibration, render_markdown, render_rich_table
    from lucid.modules.module_a_spiralbench import BEHAVIORS as MODULE_A_BEHAVIORS
    from lucid.modules.module_a_spiralbench import PROMPT_VERSION as MODULE_A_PROMPT_VERSION

    try:
        human = load_hand_labels(human_labels_path)
        judge = load_hand_labels(judge_labels_path)
    except (ValueError, OSError) as err:
        _CONSOLE.print(f"[red]Failed to load labels:[/red] {err}")
        raise typer.Exit(EXIT_USAGE) from err

    if not human or not judge:
        _CONSOLE.print("[red]One or both label files are empty after parsing.[/red]")
        raise typer.Exit(EXIT_USAGE)

    human_keys = {(lt.conversation_id, lt.turn_id) for lt in human}
    judge_keys = {(lt.conversation_id, lt.turn_id) for lt in judge}
    shared = sorted(human_keys & judge_keys)
    if not shared:
        _CONSOLE.print(
            "[red]No overlapping (conversation_id, turn_id) pairs between the two label files.[/red]"
        )
        raise typer.Exit(EXIT_USAGE)

    _, test_keys = train_test_split(shared, test_frac=test_frac, seed=seed)
    if len(test_keys) < 2:
        _CONSOLE.print(
            f"[red]Held-out split has {len(test_keys)} items — need ≥ 2 for bootstrap.[/red]"
        )
        raise typer.Exit(EXIT_USAGE)

    test_set = set(test_keys)
    human_in_test = [lt for lt in human if (lt.conversation_id, lt.turn_id) in test_set]
    judge_in_test = [lt for lt in judge if (lt.conversation_id, lt.turn_id) in test_set]

    if behaviors_csv:
        requested = [b.strip() for b in behaviors_csv.split(",") if b.strip()]
        unknown = [b for b in requested if b not in MODULE_A_BEHAVIORS]
        if unknown:
            _CONSOLE.print(f"[red]Unknown behaviors: {unknown}.[/red]")
            raise typer.Exit(EXIT_USAGE)
        behaviors_to_score = requested
    else:
        behaviors_to_score = list(MODULE_A_BEHAVIORS)

    try:
        report = compute_calibration(
            labels_by_rater={"human": human_in_test, "judge": judge_in_test},
            behaviors=behaviors_to_score,
            turn_keys=test_keys,
            module="A",
            prompt_version=prompt_version_override or MODULE_A_PROMPT_VERSION,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    except ValueError as err:
        _CONSOLE.print(f"[red]Calibration failed:[/red] {err}")
        raise typer.Exit(EXIT_USAGE) from err

    render_rich_table(report, _CONSOLE)

    if write_markdown is not None:
        write_markdown.parent.mkdir(parents=True, exist_ok=True)
        existing = write_markdown.read_text(encoding="utf-8") if write_markdown.is_file() else ""
        with write_markdown.open("w", encoding="utf-8") as f:
            if existing:
                f.write(existing.rstrip() + "\n\n")
            f.write(render_markdown(report) + "\n")
        _CONSOLE.print(f"[green]Wrote markdown report to[/green] {write_markdown}")


# ──────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────


def _ingest(
    source: SourceChoice, path: Path
) -> tuple[dict[str, Conversation], dict[str, list[Turn]]]:
    """Dispatch to the selected adapter(s); return (conv_by_id, turns_by_id)."""
    from lucid.ingest.base import ParsedConversation

    convs_by_id: dict[str, Conversation] = {}
    turns_by_id: dict[str, list[Turn]] = {}

    def _merge(parsed_list: list[ParsedConversation]) -> None:
        for p in parsed_list:
            convs_by_id[p.conversation.id] = p.conversation
            turns_by_id[p.conversation.id] = p.turns

    if source in (SourceChoice.CLAUDE_CODE, SourceChoice.ALL):
        _merge(ClaudeCodeAdapter().parse_all(path))
    if source in (SourceChoice.CLAUDE_AI, SourceChoice.ALL):
        _merge(ClaudeAiAdapter().parse_all(path))

    return convs_by_id, turns_by_id


def _build_counter() -> TokenCounter:
    """Prefer the real Anthropic SDK counter when the key is present.

    Falls back to the offline heuristic counter if no key is configured
    (useful for smoke tests; never shipped in a real audit).
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return heuristic_counter

    try:
        # Local import — the SDK is a heavy dep we don't want to load when
        # running `lucid --help`.
        from anthropic import Anthropic
    except ImportError:
        return heuristic_counter

    return make_anthropic_counter(Anthropic(api_key=key))


def _anthropic_client_or_exit() -> object:
    """Return a real `anthropic.Anthropic()` client or raise typer.Exit."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        _CONSOLE.print(
            "[red]ANTHROPIC_API_KEY is not set. Set it in .env.local or export it.[/red]"
        )
        raise typer.Exit(EXIT_USAGE)
    from anthropic import Anthropic

    return Anthropic(api_key=key)


def _authorized_budget(estimate: CostEstimate, yes_authorize: int | None) -> float:
    """Apply the $20 gate + per-run authorize flag.

    Returns the effective budget ceiling in USD, or raises typer.Exit with
    code `EXIT_COST_GATE` (3) if the estimate fails the gate.
    """
    # Under the default $20 gate, no authorize flag required.
    if not estimate.exceeds_gate():
        return max(estimate.total_usd * 1.5, COST_GATE_USD)

    if yes_authorize is None:
        _CONSOLE.print(
            f"[red]Estimated cost ${estimate.total_usd:.2f} exceeds the "
            f"${COST_GATE_USD:.0f} gate.[/red] Re-run with "
            f"`--yes-i-authorize-spend-up-to N` (and LUCID_ALLOW_UNATTENDED=1 "
            f"for unattended use) to authorize."
        )
        raise typer.Exit(EXIT_COST_GATE)

    if yes_authorize < estimate.total_usd:
        _CONSOLE.print(
            f"[red]Estimated cost ${estimate.total_usd:.2f} exceeds your "
            f"`--yes-i-authorize-spend-up-to {yes_authorize}` ceiling.[/red]"
        )
        raise typer.Exit(EXIT_COST_GATE)

    if os.environ.get("LUCID_ALLOW_UNATTENDED") != "1":
        _CONSOLE.print(
            "[red]--yes-i-authorize-spend-up-to requires LUCID_ALLOW_UNATTENDED=1 "
            "in the environment (prevents accidental unattended spend).[/red]"
        )
        raise typer.Exit(EXIT_COST_GATE)

    return float(yes_authorize)


def _hash_content(conv: Conversation, turns: list[Turn]) -> str:
    """Tiny hash-of-turns helper reused across CLI + run.py."""
    from lucid.ingest.base import content_hash_for

    return content_hash_for(conv, turns)


def _render_summary(
    *,
    source: SourceChoice,
    path: Path,
    sampled: list[Conversation],
    discovered: int,
    estimate: CostEstimate,
    enabled_modules: list[ModuleName],
) -> None:
    """Print the dry-run summary to the console."""
    header = Table.grid(padding=(0, 2))
    header.add_row("[bold]Source[/bold]", source.value)
    header.add_row("[bold]Path[/bold]", str(path))
    header.add_row("[bold]Conversations[/bold]", f"{discovered} discovered, {len(sampled)} sampled")
    header.add_row("[bold]Modules[/bold]", ", ".join(m.value for m in enabled_modules))
    _CONSOLE.print(header)

    tbl = Table(title="Estimated cost (dry-run)")
    tbl.add_column("Module")
    tbl.add_column("Model")
    tbl.add_column("Input tokens", justify="right")
    tbl.add_column("Output tokens", justify="right")
    tbl.add_column("Cache hit", justify="right")
    tbl.add_column("USD", justify="right")
    for mc in estimate.per_module:
        cached_pct = (
            f"{(mc.cached_input_tokens / mc.input_tokens * 100):.0f}%" if mc.input_tokens else "—"
        )
        tbl.add_row(
            mc.module.value,
            mc.model,
            f"{mc.input_tokens:,}",
            f"{mc.output_tokens:,}",
            cached_pct,
            f"${mc.usd:,.4f}",
        )
    tbl.add_section()
    tbl.add_row(
        "[bold]TOTAL[/bold]",
        "",
        "",
        "",
        "",
        f"[bold]${estimate.total_usd:,.4f}[/bold]",
    )
    _CONSOLE.print(tbl)

    if estimate.exceeds_gate():
        _CONSOLE.print(
            f"[red]Estimated cost ${estimate.total_usd:,.2f} exceeds the "
            f"${COST_GATE_USD:.0f} cost gate. A real run would require "
            "--yes-i-authorize-spend-up-to N + LUCID_ALLOW_UNATTENDED=1.[/red]"
        )


if __name__ == "__main__":
    app()


# Suppress unused-import warnings for attr helpers we re-export at runtime.
_ = sys
