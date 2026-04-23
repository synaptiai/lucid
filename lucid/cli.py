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
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover
    from anthropic import Anthropic, AsyncAnthropic

    from lucid.modules.embeddings import EmbeddingProvider

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
from lucid.ingest.claude_ai import ClaudeAiAdapter, parse_memory_file, parse_projects  # noqa: E402
from lucid.ingest.claude_code import ClaudeCodeAdapter  # noqa: E402
from lucid.logging import configure_logging  # noqa: E402
from lucid.orchestrator.system_prompt import SYSTEM_PROMPT  # noqa: E402
from lucid.run import AuditInputs, LockHeldError, generate_run_id, run_audit  # noqa: E402
from lucid.sampling import SamplingConfig, sample_conversations  # noqa: E402
from lucid.schemas import Conversation, ModuleName, Source, Turn  # noqa: E402
from lucid.store import initialize_db  # noqa: E402
from lucid.store.sqlite import CorpusStore  # noqa: E402

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

    prompt_versions = _collect_prompt_versions(enabled)
    allow_module_d = ModuleName.D_PERSPECTIVE in enabled

    data_dir = Path(".lucid")
    try:
        sync_client, async_client = _build_anthropic_clients_or_exit()
    except ImportError as err:
        _CONSOLE.print(f"[red]Anthropic SDK not installed:[/red] {err}")
        raise typer.Exit(EXIT_USAGE) from err
    embedding_provider = _build_embedding_provider_or_warn()

    # memories.json (Claude.ai exports only) seeds Module H. Persist
    # before run_audit so the dispatcher can read it inside the session.
    memories_persisted = _persist_memories_for_sources(data_dir, source_paths)
    if memories_persisted:
        _CONSOLE.print(f"[dim]Loaded {memories_persisted} memory file(s) for Module H.[/dim]")

    run_id = generate_run_id()
    kickoff_message = _build_kickoff_message(run_id=run_id, inputs=inputs)

    _CONSOLE.print(
        f"[cyan]Launching Managed Agents session for {run_id}. "
        f"Budget ceiling: ${authorized_budget:.2f}. This will cost real money.[/cyan]"
    )

    def _progress(level: str, message: str) -> None:
        _CONSOLE.print(f"[dim]\\[agent][/dim] \\[{level}] {message}")

    try:
        result = run_audit(
            inputs=inputs,
            data_dir=data_dir,
            client=sync_client,
            async_client=async_client,
            embedding_provider=embedding_provider,
            allow_module_d=allow_module_d,
            system_prompt=SYSTEM_PROMPT,
            kickoff_message=kickoff_message,
            prompt_versions=prompt_versions,
            progress_log=_progress,
            run_id=run_id,
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
    if result.report_path is not None:
        _CONSOLE.print(f"[green]Report:[/green] {result.report_path}")
    if result.status != "completed":
        raise typer.Exit(EXIT_USAGE)


@app.command()
def calibrate(
    module: Annotated[
        CalibrateModule,
        typer.Option("--module", help="Which module to calibrate against ground truth."),
    ],
    # Phase 6A mode: compare two pre-computed label JSONLs.
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
    # Phase 6B mode: end-to-end auto-judge pipeline.
    auto_judge: Annotated[
        bool,
        typer.Option(
            "--auto-judge",
            help=(
                "Run the full Phase 6B pipeline: fetch SpiralBench, run Module A + Ollama "
                "judges, compute IAA, export disagreements for human review. Requires "
                "--yes-i-authorize-spend-up-to and ANTHROPIC_API_KEY."
            ),
        ),
    ] = False,
    sb_models: Annotated[
        str,
        typer.Option(
            "--sb-models",
            help="CSV of SpiralBench target-model filenames (without .json suffix).",
        ),
    ] = "claude-sonnet-4.5,gpt-5-2025-08-07,kimi-k2",
    ollama_models: Annotated[
        str,
        typer.Option(
            "--ollama-models",
            help="CSV of Ollama model identifiers. Pass empty string to disable Ollama.",
        ),
    ] = "kimi-k2.6:cloud,gemma4:31b-cloud,glm-5.1:cloud",
    chunk_sizes: Annotated[
        str,
        typer.Option(
            "--chunk-sizes",
            help="CSV of Module A chunk sizes to run (default both 10 and 2).",
        ),
    ] = "10,2",
    conversations_per_sb_model: Annotated[
        int | None,
        typer.Option(
            "--conversations-per-sb-model",
            help="Cap conversations per SpiralBench target model (None = full 30).",
        ),
    ] = None,
    include_synthetic: Annotated[
        bool,
        typer.Option(
            "--include-synthetic/--no-include-synthetic",
            help="Include the 60-turn synthetic gold corpus.",
        ),
    ] = True,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Base directory for calibration run artifacts."),
    ] = Path("calibration-runs"),
    disagreements_top_n: Annotated[
        int,
        typer.Option(
            "--disagreements-top-n",
            help="Cap on top-N disagreements exported to the reviewer JSONL.",
        ),
    ] = 50,
    yes_authorize_usd: Annotated[
        int,
        typer.Option(
            "--yes-i-authorize-spend-up-to",
            help=(
                "Required for --auto-judge. Minimum 50 per methodology.md §10 "
                "(~$46 projected spend at both chunk sizes)."
            ),
        ),
    ] = 0,
    # Phase 6B mode: post-audit finalisation.
    import_verified: Annotated[
        Path | None,
        typer.Option(
            "--import-verified",
            help=(
                "Re-run IAA with human-audit overrides from this reviewer JSONL. "
                "--output-dir must point at the completed --auto-judge run."
            ),
        ),
    ] = None,
    # Shared knobs.
    test_frac: Annotated[
        float,
        typer.Option(
            "--test-frac",
            help="Fraction of shared turns held out for reported metrics (Phase 6A mode).",
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

    # Three modes, one command. Select based on the flags present.
    modes_set = sum(
        [
            auto_judge,
            import_verified is not None,
            human_labels is not None or judge_labels is not None,
        ]
    )
    if modes_set != 1:
        _CONSOLE.print(
            "[red]Pick exactly one mode:[/red]\n"
            "  Phase 6A (compare): --human-labels PATH --judge-labels PATH\n"
            "  Phase 6B (run):     --auto-judge --yes-i-authorize-spend-up-to 50\n"
            "  Phase 6B (audit):   --import-verified PATH --output-dir <prev run dir>"
        )
        raise typer.Exit(EXIT_USAGE)

    if import_verified is not None:
        _run_calibrate_import_verified(
            verified_path=import_verified,
            output_dir=output_dir,
            behaviors_csv=behaviors_csv,
            n_bootstrap=n_bootstrap,
            seed=seed,
            write_markdown=write_markdown,
        )
        return

    if auto_judge:
        _run_calibrate_auto_judge(
            sb_models_csv=sb_models,
            ollama_models_csv=ollama_models,
            chunk_sizes_csv=chunk_sizes,
            conversations_per_sb_model=conversations_per_sb_model,
            include_synthetic=include_synthetic,
            output_dir=output_dir,
            disagreements_top_n=disagreements_top_n,
            yes_authorize_usd=yes_authorize_usd,
            behaviors_csv=behaviors_csv,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        return

    # Phase 6A mode
    if human_labels is None or judge_labels is None:
        _CONSOLE.print(
            "[red]Phase 6A compare mode requires both --human-labels and --judge-labels.[/red]"
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


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    return tuple(int(s) for s in _parse_csv(raw))


def _run_calibrate_auto_judge(
    *,
    sb_models_csv: str,
    ollama_models_csv: str,
    chunk_sizes_csv: str,
    conversations_per_sb_model: int | None,
    include_synthetic: bool,
    output_dir: Path,
    disagreements_top_n: int,
    yes_authorize_usd: int,
    behaviors_csv: str | None,
    n_bootstrap: int,
    seed: int,
) -> None:
    """Phase 6B auto-judge mode — end-to-end calibration pipeline."""
    import asyncio as _asyncio

    from lucid.calibration.auto_judge import (
        MINIMUM_COST_GATE_USD,
        AutoJudgeConfig,
        run_auto_judge,
    )
    from lucid.modules.module_a_spiralbench import BEHAVIORS as MODULE_A_BEHAVIORS

    if yes_authorize_usd < MINIMUM_COST_GATE_USD:
        _CONSOLE.print(
            f"[red]--auto-judge requires --yes-i-authorize-spend-up-to "
            f">= {MINIMUM_COST_GATE_USD} (got {yes_authorize_usd}).[/red]"
        )
        _CONSOLE.print(
            "[dim]See docs/methodology.md §10 — projected spend is ~$46 for "
            "3 SB targets × 2 chunk sizes on Opus 4.7.[/dim]"
        )
        raise typer.Exit(EXIT_COST_GATE)

    client = None
    if chunk_sizes_csv.strip():
        client = _async_anthropic_client_or_exit()

    try:
        behaviors = (
            tuple(b.strip() for b in behaviors_csv.split(",") if b.strip())
            if behaviors_csv
            else tuple(MODULE_A_BEHAVIORS)
        )
        unknown = [b for b in behaviors if b not in MODULE_A_BEHAVIORS]
        if unknown:
            _CONSOLE.print(f"[red]Unknown behaviors: {unknown}.[/red]")
            raise typer.Exit(EXIT_USAGE)

        config = AutoJudgeConfig(
            sb_target_models=_parse_csv(sb_models_csv),
            ollama_models=_parse_csv(ollama_models_csv),
            chunk_sizes=_parse_int_csv(chunk_sizes_csv),
            include_synthetic=include_synthetic,
            conversations_per_sb_model=conversations_per_sb_model,
            behaviors=behaviors,
            n_bootstrap=n_bootstrap,
            seed=seed,
            disagreements_top_n=disagreements_top_n,
            output_dir=output_dir,
        )
    except (ValueError, typer.Exit):
        raise
    except Exception as err:
        _CONSOLE.print(f"[red]Invalid configuration:[/red] {err}")
        raise typer.Exit(EXIT_USAGE) from err

    try:
        _asyncio.run(
            run_auto_judge(
                config,
                anthropic_client=client,  # type: ignore[arg-type]
                yes_authorize_usd=yes_authorize_usd,
                console=_CONSOLE,
            )
        )
    except ValueError as err:
        _CONSOLE.print(f"[red]{err}[/red]")
        raise typer.Exit(EXIT_COST_GATE) from err
    except RuntimeError as err:
        _CONSOLE.print(f"[red]auto-judge failed:[/red] {err}")
        raise typer.Exit(EXIT_USAGE) from err


def _run_calibrate_import_verified(
    *,
    verified_path: Path,
    output_dir: Path,
    behaviors_csv: str | None,
    n_bootstrap: int,
    seed: int,
    write_markdown: Path | None,
) -> None:
    """Apply human-audit overrides from a completed reviewer JSONL."""
    from lucid.calibration.auto_judge import AutoJudgeConfig, import_and_finalize
    from lucid.modules.module_a_spiralbench import BEHAVIORS as MODULE_A_BEHAVIORS

    if not verified_path.is_file():
        _CONSOLE.print(f"[red]--import-verified path does not exist:[/red] {verified_path}")
        raise typer.Exit(EXIT_USAGE)

    # When --output-dir is the top-level base (calibration-runs/), try to
    # find the most recent auto-* subdirectory.
    resolved_run_dir = output_dir
    if (output_dir / "judgements").is_dir():
        resolved_run_dir = output_dir
    else:
        candidates = sorted(output_dir.glob("auto-*") if output_dir.is_dir() else [])
        if candidates:
            resolved_run_dir = candidates[-1]
            _CONSOLE.print(f"[dim]Using most-recent run directory {resolved_run_dir}[/dim]")
        else:
            _CONSOLE.print(f"[red]No calibration run artifacts found under {output_dir}.[/red]")
            raise typer.Exit(EXIT_USAGE)

    behaviors = (
        tuple(b.strip() for b in behaviors_csv.split(",") if b.strip())
        if behaviors_csv
        else tuple(MODULE_A_BEHAVIORS)
    )

    config = AutoJudgeConfig(
        behaviors=behaviors,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    try:
        import_and_finalize(
            verified_path=verified_path,
            output_dir=resolved_run_dir,
            config=config,
            write_markdown=write_markdown,
            console=_CONSOLE,
        )
    except FileNotFoundError as err:
        _CONSOLE.print(f"[red]{err}[/red]")
        raise typer.Exit(EXIT_USAGE) from err


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


def _build_anthropic_clients_or_exit() -> tuple[Anthropic, AsyncAnthropic]:
    """Return ``(sync_client, async_client)`` or raise ``typer.Exit``.

    Lucid needs both shapes for one audit:

    * **Sync** ``Anthropic()`` drives :class:`ManagedAgentsSession` —
      the SDK's ``beta.agents.create`` / ``events.send`` /
      ``events.stream`` surface is synchronous and the driver bridges
      to async internally.
    * **Async** ``AsyncAnthropic()`` is handed to every detection
      module so ``await client.messages.create(...)`` runs as a real
      coroutine. Passing a sync client to an ``await`` site does not
      raise — it returns a coroutine-like object that, when awaited,
      executes a blocking HTTP call on the event loop. Visibly this
      looks like a hang (0% CPU, one open socket, no log output). The
      separate async client prevents this entire failure mode.

    A single :envvar:`ANTHROPIC_API_KEY` covers both clients.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        _CONSOLE.print(
            "[red]ANTHROPIC_API_KEY is not set. Set it in .env.local or export it.[/red]"
        )
        raise typer.Exit(EXIT_USAGE)
    from anthropic import Anthropic, AsyncAnthropic

    return Anthropic(api_key=key), AsyncAnthropic(api_key=key, timeout=120.0)


def _build_embedding_provider_or_warn() -> EmbeddingProvider | None:
    """Build a :class:`VoyageEmbeddingProvider` if ``VOYAGE_API_KEY`` is
    set, otherwise return ``None`` and tell the user Module H is
    skipped.

    Module H is the only consumer; without an embedding provider the
    dispatcher returns ``no_embedding_provider`` and the audit
    proceeds with the other modules.
    """
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        _CONSOLE.print(
            "[yellow]VOYAGE_API_KEY not set; Module H "
            "(memory-corpus consistency) will be skipped.[/yellow]"
        )
        return None
    from lucid.modules.embeddings import VoyageEmbeddingProvider

    return VoyageEmbeddingProvider(api_key=key)


def _persist_memories_for_sources(data_dir: Path, source_paths: dict[Source, Path]) -> int:
    """Parse and persist ``memories.json`` for every Claude.ai source.

    Returns the count of memory files persisted (zero on Claude Code
    paths or if no memories.json is present). Module H reads from the
    ``memory_files`` table during the audit; without this step Module
    H produces no findings even when a real export is supplied.

    Persistence is incremental — re-running an audit on the same DB
    skips already-persisted files (the table is keyed by
    ``account_uuid``).
    """
    persisted = 0
    db_path = data_dir / "lucid.sqlite3"
    data_dir.mkdir(parents=True, exist_ok=True)
    initialize_db(db_path)

    with CorpusStore(db_path) as store:
        existing_uuids = {
            row["account_uuid"] for row in store.fetchall("SELECT account_uuid FROM memory_files")
        }
        existing_project_uuids = {
            row["uuid"] for row in store.fetchall("SELECT uuid FROM projects")
        }
        for source, path in source_paths.items():
            if source is not Source.CLAUDE_AI:
                continue
            memories_path = path / "memories.json"
            projects_path = path / "projects.json"
            if memories_path.is_file():
                try:
                    memory_file = parse_memory_file(memories_path)
                except IngestError as err:
                    _CONSOLE.print(
                        f"[yellow]Skipping {memories_path}:[/yellow] {err}"
                    )
                else:
                    if memory_file.account_uuid not in existing_uuids:
                        store.insert_memory_file(memory_file)
                        existing_uuids.add(memory_file.account_uuid)
                        persisted += 1
            # projects.json — required by Module H's source-aware retrieval
            # so project-scoped memory claims can be fuzzy-matched to the
            # conversations that actually belong to that project. Without
            # this the projects table is empty, every project UUID fails
            # attribution, and every project memory lands in out-of-scope.
            if projects_path.is_file():
                try:
                    projects = parse_projects(projects_path)
                except IngestError as err:
                    _CONSOLE.print(
                        f"[yellow]Skipping {projects_path}:[/yellow] {err}"
                    )
                else:
                    for project in projects:
                        if project.uuid in existing_project_uuids:
                            continue
                        store.insert_project(project)
                        existing_project_uuids.add(project.uuid)
    return persisted


def _build_kickoff_message(*, run_id: str, inputs: AuditInputs) -> str:
    """Compose the orchestrator kickoff message.

    The kickoff is intentionally directive: it enumerates every
    invoke_module call the orchestrator must make. Sonnet 4.6 reads
    the system prompt's "Workflow" section as advisory and stops
    after `query_corpus` if the kickoff doesn't hand it a concrete
    sequence to execute. Spelling out the exact module letters in
    the kickoff is what gets the agent to actually run them.

    Not cache-stable across runs — ``run_id`` and per-run conversation
    ids are baked in — but the orchestrator system prompt is. Only
    the system prompt benefits from the 1-hour cache TTL.
    """
    enabled_letters = sorted(m.value for m in inputs.enabled_modules)
    sample_n = len(inputs.sampled)
    all_ids = [c.id for c in inputs.sampled]
    # Render the conversation ids inline. Module-G safety net runs
    # post-session so we always include G in the explicit step list
    # — the orchestrator should still run it, but a no-op-if-already-
    # done is cheap.
    behaviour_letters = [letter for letter in enabled_letters if letter != "G"]
    invoke_steps = "\n".join(
        f'    Step {2 + i}: invoke_module(module="{letter}", conversation_ids={all_ids!r})'
        for i, letter in enumerate(behaviour_letters)
    )
    g_step_number = 2 + len(behaviour_letters)
    finalize_step_number = g_step_number + 1
    summary_step_number = finalize_step_number + 1
    return (
        f"Begin audit run {run_id}.\n\n"
        f"Corpus: {sample_n} sampled conversations.\n"
        f"Enabled modules: {', '.join(enabled_letters)}.\n"
        f"Budget ceiling: ${inputs.authorized_budget_usd:.2f}.\n\n"
        "You MUST execute every numbered step below in order. Do not\n"
        "stop early. Do not skip any module. The pre-flight estimate\n"
        f"covers all of them.\n\n"
        "    Step 1: query_corpus(limit=200) to confirm the corpus.\n"
        f"{invoke_steps}\n"
        f'    Step {g_step_number}: invoke_module(module="G", '
        f"conversation_ids={all_ids!r})\n"
        f"    Step {finalize_step_number}: get_findings() to verify total coverage.\n"
        f"    Step {summary_step_number}: log_progress with a one-line summary,\n"
        "             then stop (the dispatcher already persisted findings).\n\n"
        "Reminder: invoke_module persists findings automatically. Do not\n"
        "call store_finding. If a module returns status=skipped /\n"
        "no_client / no_embedding_provider, log_progress that and continue\n"
        "to the next step — never abort the run.\n"
    )


def _module_primary_prompt_version(module: ModuleName) -> str:
    """Return the primary judge-prompt version for ``module``.

    For multi-prompt modules (B, E, F, H) this is the version of the
    prompt whose findings actually land in the report. Sub-prompt
    versions (Module B's extract pass, Module E's topics pass, …)
    are tracked inside their module files; the audit-run column is
    a coarse pointer to "which version of the framework did I use",
    not a complete inventory.
    """
    match module:
        case ModuleName.A_SPIRALBENCH:
            from lucid.modules.module_a_spiralbench import PROMPT_VERSION

            return PROMPT_VERSION
        case ModuleName.B_SHARMA:
            from lucid.modules.module_b_sharma import FEEDBACK_PROMPT_VERSION

            return FEEDBACK_PROMPT_VERSION
        case ModuleName.C_SYCEVAL:
            from lucid.modules.module_c_syceval import PROMPT_VERSION

            return PROMPT_VERSION
        case ModuleName.D_PERSPECTIVE:
            from lucid.modules.module_d_perspective import PROMPT_VERSION

            return PROMPT_VERSION
        case ModuleName.E_BELIEFSHIFT:
            from lucid.modules.module_e_beliefshift import DRIFT_PROMPT_VERSION

            return DRIFT_PROMPT_VERSION
        case ModuleName.F_ITP:
            from lucid.modules.module_f_itp import CLASSIFY_PROMPT_VERSION

            return CLASSIFY_PROMPT_VERSION
        case ModuleName.G_ATTRIBUTION:
            from lucid.modules.module_g_attribution import PROMPT_VERSION

            return PROMPT_VERSION
        case ModuleName.H_MEMORY:
            from lucid.modules.module_h_memory import CLASSIFY_PROMPT_VERSION

            return CLASSIFY_PROMPT_VERSION


def _collect_prompt_versions(enabled: list[ModuleName]) -> dict[ModuleName, str]:
    """Recorded on ``audit_runs.prompt_versions_json`` for
    calibration reproducibility.

    Module G's version is always pinned because it runs as a
    post-session safety net even when not in the user's ``enabled``
    set.
    """
    versions = {module: _module_primary_prompt_version(module) for module in enabled}
    if ModuleName.G_ATTRIBUTION not in versions:
        versions[ModuleName.G_ATTRIBUTION] = _module_primary_prompt_version(
            ModuleName.G_ATTRIBUTION
        )
    return versions


def _async_anthropic_client_or_exit() -> object:
    """Return an `anthropic.AsyncAnthropic()` client or raise typer.Exit.

    **Why the sync/async split matters:** Module A runs inside an
    asyncio event loop (Semaphore-bounded concurrency). Passing a
    synchronous ``Anthropic()`` client to code that ``await``s
    ``client.messages.parse(...)`` does not raise — it returns a
    coroutine-like object that, when awaited, executes a blocking HTTP
    request on the event loop, serialising every concurrent call.
    Visibly this looks like a hang (0% CPU, 1 open connection, no log
    progress). Wiring an async client fixes the concurrency.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        _CONSOLE.print(
            "[red]ANTHROPIC_API_KEY is not set. Set it in .env.local or export it.[/red]"
        )
        raise typer.Exit(EXIT_USAGE)
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(api_key=key, timeout=120.0)


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
