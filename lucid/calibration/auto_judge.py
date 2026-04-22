"""End-to-end auto-judge calibration pipeline.

Orchestrates the full Phase 6B flow:

1. Build a calibration corpus from SpiralBench target-model files + the
   synthetic gold corpus.
2. Build the rater pool: Module A (one per chunk size) + SpiralBench
   judges (already-recorded, three per target model) + Ollama judges
   (one per configured model) + synthetic gold (if synthetic corpus
   included).
3. Run every judge concurrently — each emits a ``list[LabeledTurn]``.
4. Persist every rater's labels as a JSONL in ``output_dir`` for
   reproducibility and re-import.
5. Compute pre-audit IAA via :mod:`lucid.calibration.report`.
6. Export top-N disagreements to ``disagreements.jsonl`` for human
   review.

The caller is expected to then run the human audit out-of-band, fill in
``verified_label`` on each row, and re-invoke :func:`import_and_finalize`
to compute the post-audit IAA and write the final ``calibration.md``.

**Cost safety:** ``yes_authorize_usd`` must be explicitly passed by the
CLI (no default). The plan's Phase 6B cost gate is $50; this function
raises before any LLM call if the gate is below the projected minimum.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.calibration.audit import (
    compute_disagreements,
    export_for_review,
)
from lucid.calibration.data import LabeledTurn
from lucid.calibration.judges import (
    Judge,
    ModuleAJudge,
    SpiralBenchFileJudge,
)
from lucid.calibration.judges.ollama import OllamaJudge
from lucid.calibration.report import (
    CalibrationReport,
    compute_calibration,
    render_markdown,
    render_rich_table,
)
from lucid.calibration.spiralbench import (
    SpiralBenchCorpusData,
    fetch_spiralbench_model,
    parse_spiralbench_file,
)
from lucid.calibration.synthetic import build_synthetic_corpus
from lucid.modules.base import ModuleCorpus
from lucid.modules.module_a_spiralbench import BEHAVIORS as MODULE_A_BEHAVIORS
from lucid.modules.module_a_spiralbench import PROMPT_VERSION as MODULE_A_PROMPT_VERSION
from lucid.schemas import Conversation, Turn

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from rich.console import Console

__all__ = [
    "MINIMUM_COST_GATE_USD",
    "AutoJudgeConfig",
    "AutoJudgeResult",
    "build_corpus",
    "build_raters",
    "import_and_finalize",
    "run_auto_judge",
]

log = logging.getLogger(__name__)


# Plan §6B mandates $50 for the full both-chunks-three-targets run.
# Keep as a documented constant so the CLI can cite it in gate errors.
MINIMUM_COST_GATE_USD = 50

DEFAULT_SB_MODELS = ("claude-sonnet-4.5", "gpt-5-2025-08-07", "kimi-k2")
DEFAULT_OLLAMA_MODELS = (
    "kimi-k2.6:cloud",
    "gemma4:31b-cloud",
    "glm-5.1:cloud",
)
DEFAULT_CHUNK_SIZES = (10, 2)


@dataclass(frozen=True, slots=True)
class AutoJudgeConfig:
    """Knobs for a full auto-judge calibration run."""

    sb_target_models: tuple[str, ...] = DEFAULT_SB_MODELS
    ollama_models: tuple[str, ...] = DEFAULT_OLLAMA_MODELS
    chunk_sizes: tuple[int, ...] = DEFAULT_CHUNK_SIZES
    include_synthetic: bool = True
    conversations_per_sb_model: int | None = None  # None = all (30)
    behaviors: tuple[str, ...] = field(default_factory=lambda: tuple(MODULE_A_BEHAVIORS))
    n_bootstrap: int = 2000
    seed: int = 42
    disagreements_top_n: int = 50
    output_dir: Path = Path("calibration-runs")


@dataclass(frozen=True, slots=True)
class AutoJudgeResult:
    """Artifacts produced by a completed run."""

    output_dir: Path
    corpus: ModuleCorpus
    labels_by_rater: dict[str, list[LabeledTurn]]
    report: CalibrationReport
    disagreements_path: Path
    markdown_path: Path
    rater_names: tuple[str, ...]


def _combine_corpora(
    pieces: Sequence[tuple[ModuleCorpus, list[LabeledTurn]]],
    *,
    audit_run_id: str,
) -> tuple[ModuleCorpus, list[LabeledTurn]]:
    """Merge conversations + turns + pre-existing labels from multiple sources."""
    conversations: dict[str, Conversation] = {}
    turns: dict[str, list[Turn]] = {}
    labels: list[LabeledTurn] = []
    for corpus, corpus_labels in pieces:
        for conv_id, conv in corpus.conversations.items():
            if conv_id in conversations:
                raise ValueError(f"duplicate conversation id across sources: {conv_id}")
            conversations[conv_id] = conv
            turns[conv_id] = list(corpus.turns_by_conversation.get(conv_id, ()))
        labels.extend(corpus_labels)
    return (
        ModuleCorpus(
            conversations=conversations,
            turns_by_conversation=turns,
            audit_run_id=audit_run_id,
        ),
        labels,
    )


def build_corpus(
    config: AutoJudgeConfig,
    *,
    audit_run_id: str,
) -> tuple[ModuleCorpus, list[SpiralBenchCorpusData], list[LabeledTurn]]:
    """Build the combined ModuleCorpus from SpiralBench + synthetic sources.

    Returns ``(corpus, sb_data_list, preloaded_labels)``. ``sb_data_list``
    keeps the per-target SpiralBench records around so downstream code
    can build the :class:`SpiralBenchFileJudge` rater set; they are NOT
    merged label-wise yet because those judges are attached in
    :func:`build_raters`. ``preloaded_labels`` is the synthetic-gold
    labels (if synthetic corpus is included); SB judgements attach in
    :func:`build_raters`.
    """
    pieces: list[tuple[ModuleCorpus, list[LabeledTurn]]] = []
    sb_datas: list[SpiralBenchCorpusData] = []

    for target in config.sb_target_models:
        path = fetch_spiralbench_model(target)
        sb_data = parse_spiralbench_file(
            path,
            target_model=target,
            conversation_limit=config.conversations_per_sb_model,
        )
        sb_datas.append(sb_data)
        piece_corpus = ModuleCorpus(
            conversations={c.id: c for c in sb_data.conversations},
            turns_by_conversation=sb_data.turns,
            audit_run_id=audit_run_id,
        )
        # SpiralBench judges' LabeledTurns are attached in build_raters,
        # not here, so we pass empty labels for the "preloaded" bucket.
        pieces.append((piece_corpus, []))

    preloaded: list[LabeledTurn] = []
    if config.include_synthetic:
        syn_corpus, syn_labels = build_synthetic_corpus(audit_run_id=audit_run_id)
        pieces.append((syn_corpus, syn_labels))
        preloaded.extend(syn_labels)

    corpus, _unused = _combine_corpora(pieces, audit_run_id=audit_run_id)
    return corpus, sb_datas, preloaded


def build_raters(
    config: AutoJudgeConfig,
    sb_datas: Sequence[SpiralBenchCorpusData],
    *,
    anthropic_client: AsyncAnthropic | None,
) -> list[Judge]:
    """Assemble the judge pool. Ordering is meaningful for CLI output only."""
    judges: list[Judge] = []

    if anthropic_client is not None:
        for cs in config.chunk_sizes:
            judges.append(ModuleAJudge(client=anthropic_client, chunk_size=cs))

    # One SpiralBenchFileJudge per (target × rater). Label names are
    # disambiguated with the target model so "sb_sonnet45 on sonnet-4.5"
    # and "sb_sonnet45 on gpt-5" are different raters in the final IAA.
    for sb in sb_datas:
        for rater in sb.rater_names:
            tagged = f"{rater}@{sb.target_model}"
            judges.append(SpiralBenchFileJudge(sb, tagged, source_rater=rater))

    for model in config.ollama_models:
        judges.append(OllamaJudge(model))

    if config.include_synthetic:
        # SyntheticGoldJudge reads from a JSONL; for auto-judge we build
        # the corpus in-memory, so we inject the labels directly through
        # a special path: use build_synthetic_corpus + filter. Cheaper
        # than round-tripping to disk.
        judges.append(_InMemorySyntheticGoldJudge())

    return judges


class _InMemorySyntheticGoldJudge:
    """SyntheticGoldJudge equivalent that skips disk-round-trip.

    Implements the same ``Judge`` protocol but reads labels directly
    from :func:`build_synthetic_corpus` so the auto-judge flow doesn't
    need a committed JSONL at runtime.
    """

    rater_name: str = "synthetic_gold"

    async def run(self, corpus: ModuleCorpus) -> list[LabeledTurn]:
        _, labels = build_synthetic_corpus()
        conv_ids = set(corpus.conversations.keys())
        return [lt for lt in labels if lt.conversation_id in conv_ids]


async def _run_one_judge(
    judge: Judge,
    corpus: ModuleCorpus,
    *,
    console: Console | None,
) -> tuple[str, list[LabeledTurn]]:
    name = judge.rater_name
    try:
        labels = await judge.run(corpus)
    except Exception as err:
        log.error("judge %s failed: %s: %s", name, type(err).__name__, err)
        if console is not None:
            console.print(f"[red]✗[/red] {name}: {type(err).__name__}: {err}")
        return name, []
    if console is not None:
        console.print(f"[green]✓[/green] {name}: {len(labels)} labeled turns")
    return name, labels


async def _run_all_judges(
    judges: Sequence[Judge],
    corpus: ModuleCorpus,
    *,
    console: Console | None,
) -> dict[str, list[LabeledTurn]]:
    outputs = await asyncio.gather(*(_run_one_judge(j, corpus, console=console) for j in judges))
    result: dict[str, list[LabeledTurn]] = {}
    for name, labels in outputs:
        if labels:
            result[name] = labels
    return result


def _persist_labels(
    labels_by_rater: dict[str, list[LabeledTurn]],
    output_dir: Path,
) -> dict[str, Path]:
    """Write each rater's labels to its own JSONL under ``output_dir/judgements/``."""
    dest = output_dir / "judgements"
    dest.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for rater, labels in labels_by_rater.items():
        # Sanitise rater name for filesystem safety (already ASCII-safe per
        # our conventions but be defensive on user-supplied aliases).
        safe = rater.replace("/", "_").replace(":", "_")
        path = dest / f"{safe}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for lt in labels:
                f.write(lt.model_dump_json() + "\n")
        paths[rater] = path
    return paths


async def run_auto_judge(
    config: AutoJudgeConfig,
    *,
    anthropic_client: AsyncAnthropic | None = None,
    yes_authorize_usd: int,
    console: Console | None = None,
) -> AutoJudgeResult:
    """Execute the full auto-judge pipeline. Raises if cost gate < $50."""
    if yes_authorize_usd < MINIMUM_COST_GATE_USD:
        raise ValueError(
            f"auto-judge calibration requires --yes-i-authorize-spend-up-to "
            f">= {MINIMUM_COST_GATE_USD}; got {yes_authorize_usd}. "
            "See docs/methodology.md §10 for cost model."
        )

    if console is not None:
        console.print(
            f"[bold]Auto-judge calibration[/bold]: "
            f"{len(config.sb_target_models)} SB targets × "
            f"{len(config.chunk_sizes)} chunk sizes × "
            f"{len(config.ollama_models)} Ollama models"
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = config.output_dir / f"auto-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_run_id = f"calibration-{timestamp}"

    if console is not None:
        console.print("[dim]Building corpus (SpiralBench fetch + synthetic) …[/dim]")
    corpus, sb_datas, _preloaded = build_corpus(config, audit_run_id=audit_run_id)
    if console is not None:
        console.print(
            f"[dim]  {len(corpus.conversations)} conversations, "
            f"{sum(len(ts) for ts in corpus.turns_by_conversation.values())} turns[/dim]"
        )

    judges = build_raters(config, sb_datas, anthropic_client=anthropic_client)
    if console is not None:
        console.print(f"[dim]Running {len(judges)} judges …[/dim]")

    labels_by_rater = await _run_all_judges(judges, corpus, console=console)
    if not labels_by_rater:
        raise RuntimeError("no judges produced any labels — check API keys + Ollama daemon")

    _persist_labels(labels_by_rater, output_dir)

    if len(labels_by_rater) < 2:
        raise RuntimeError(
            f"only {len(labels_by_rater)} rater(s) produced labels — IAA needs ≥ 2. "
            "Configure more raters (sb_target_models, ollama_models, include_synthetic)."
        )

    if console is not None:
        console.print("[dim]Computing inter-annotator agreement …[/dim]")

    shared = _shared_turn_keys(labels_by_rater)
    if not shared:
        raise RuntimeError(
            "no turns are rated by every rater. This usually means the rater pool "
            "spans disjoint corpora (e.g. SpiralBench judges only cover SB conversations "
            "while synthetic_gold only covers synthetic). Run SB and synthetic corpora "
            "in separate auto-judge invocations for cleanly-overlapping raters."
        )

    report = compute_calibration(
        labels_by_rater=labels_by_rater,
        behaviors=list(config.behaviors),
        turn_keys=shared,
        module="A",
        prompt_version=MODULE_A_PROMPT_VERSION,
        n_bootstrap=config.n_bootstrap,
        seed=config.seed,
    )

    if console is not None:
        render_rich_table(report, console)

    markdown_path = output_dir / "report.md"
    markdown_path.write_text(render_markdown(report) + "\n", encoding="utf-8")

    if console is not None:
        console.print("[dim]Ranking disagreements …[/dim]")

    disagreements = compute_disagreements(
        labels_by_rater,
        corpus,
        behaviors=list(config.behaviors),
    )
    disagreements_path = output_dir / "disagreements.jsonl"
    exported = export_for_review(
        disagreements, disagreements_path, top_n=config.disagreements_top_n
    )

    if console is not None:
        console.print(f"[green]Wrote {exported} disagreements to {disagreements_path}[/green]")
        console.print(
            "[dim]Next: review the JSONL (fill in `verified_label`), then:\n"
            f"  lucid calibrate --module a --import-verified {disagreements_path} "
            f"--output-dir {output_dir} --write-markdown docs/calibration.md[/dim]"
        )

    return AutoJudgeResult(
        output_dir=output_dir,
        corpus=corpus,
        labels_by_rater=labels_by_rater,
        report=report,
        disagreements_path=disagreements_path,
        markdown_path=markdown_path,
        rater_names=tuple(sorted(labels_by_rater.keys())),
    )


def _shared_turn_keys(
    labels_by_rater: dict[str, list[LabeledTurn]],
) -> list[tuple[str, str]]:
    """Intersection of (conv, turn) keys rated by every rater."""
    sets = [
        {(lt.conversation_id, lt.turn_id) for lt in labels} for labels in labels_by_rater.values()
    ]
    if not sets:
        return []
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(common)


def import_and_finalize(
    verified_path: Path,
    output_dir: Path,
    *,
    config: AutoJudgeConfig,
    write_markdown: Path | None = None,
    console: Console | None = None,
) -> CalibrationReport:
    """Apply human-audit overrides and recompute the post-audit IAA.

    Reads back every judgement JSONL written during the previous
    :func:`run_auto_judge` (``output_dir/judgements/*.jsonl``), applies
    the human overrides as a new ``human_audit`` rater, and recomputes
    the IAA table.
    """
    from lucid.calibration.audit import apply_verified_overrides, import_verified
    from lucid.calibration.data import load_hand_labels

    judgements_dir = output_dir / "judgements"
    if not judgements_dir.is_dir():
        raise FileNotFoundError(
            f"{judgements_dir} does not exist — run `lucid calibrate --module a "
            f"--auto-judge` first, then re-run with --import-verified."
        )

    labels_by_rater: dict[str, list[LabeledTurn]] = {}
    for path in sorted(judgements_dir.glob("*.jsonl")):
        labels = load_hand_labels(path)
        if labels:
            # Rater name is restored from the first label; all labels in
            # one file share a rater by construction.
            labels_by_rater[labels[0].labeler] = labels

    verified = import_verified(verified_path)
    overrides = apply_verified_overrides(verified)
    if overrides:
        labels_by_rater["human_audit"] = overrides

    shared = _shared_turn_keys(labels_by_rater)

    report = compute_calibration(
        labels_by_rater=labels_by_rater,
        behaviors=list(config.behaviors),
        turn_keys=shared,
        module="A",
        prompt_version=MODULE_A_PROMPT_VERSION,
        n_bootstrap=config.n_bootstrap,
        seed=config.seed,
    )

    if console is not None:
        render_rich_table(report, console)

    if write_markdown is not None:
        write_markdown.parent.mkdir(parents=True, exist_ok=True)
        existing = write_markdown.read_text(encoding="utf-8") if write_markdown.is_file() else ""
        with write_markdown.open("w", encoding="utf-8") as f:
            if existing:
                f.write(existing.rstrip() + "\n\n")
            f.write(render_markdown(report) + "\n")
        if console is not None:
            console.print(f"[green]Wrote {write_markdown}[/green]")

    return report
