"""Compute + render inter-annotator agreement reports.

Flow::

   (human LabeledTurn[], judge LabeledTurn[], behaviors)
        │
        ▼
   compute_calibration(...)      ← builds per-behavior ratings matrices,
        │                           runs α / AC1 / κ / QWK + BCa CI per
        ▼                           behavior, picks primary metric from
   CalibrationReport                prevalence distribution.
        │
        ▼
   render_markdown / render_rich_table

Why split compute from render: the rich table is what the CLI prints; the
markdown is what ends up pasted into ``docs/calibration.md``. Both read
the same structured ``CalibrationReport`` so we can never display one
number and write a different one to disk.

Primary-metric selection (per plan §6):

- if ≥ 3 behaviors have per-label prevalence < 10%, **Gwet's AC1** wins
  (paradox-robust on skewed data);
- otherwise **Krippendorff's α** wins.

Both are reported regardless — the "primary" flag only governs the
pass/fail gate and which number goes in the headline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from rich.console import Console
from rich.table import Table

from lucid.calibration.data import LabeledTurn, intensity_matrix, presence_matrix
from lucid.calibration.validate import (
    MetricResult,
    bootstrap_metric,
    cohen_kappa,
    gwet_ac1,
    krippendorff_alpha,
    quadratic_weighted_kappa,
)

__all__ = [
    "BehaviorReport",
    "CalibrationReport",
    "compute_calibration",
    "render_markdown",
    "render_rich_table",
]


PrimaryMetric = Literal["alpha", "ac1"]


@dataclass(frozen=True, slots=True)
class BehaviorReport:
    behavior: str
    n_items: int
    prevalence_overall: float
    alpha: MetricResult
    ac1: MetricResult
    kappa: MetricResult
    qwk: MetricResult


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    module: str
    prompt_version: str
    n_items_total: int
    behaviors: list[BehaviorReport]
    primary_metric: PrimaryMetric
    rationale: str


_PREVALENCE_SKEW_CUTOFF = 0.10
_SKEW_COUNT_THRESHOLD = 3


def _krippendorff_stat(ratings: np.ndarray) -> float:
    # Bootstrap resampling can produce all-zero columns; guard against
    # NaN by treating that resample as perfect agreement.
    return krippendorff_alpha(ratings)


def _pick_primary(reports: Sequence[BehaviorReport]) -> tuple[PrimaryMetric, str]:
    skewed = sum(
        1
        for r in reports
        if r.prevalence_overall < _PREVALENCE_SKEW_CUTOFF
        or r.prevalence_overall > 1 - _PREVALENCE_SKEW_CUTOFF
    )
    if skewed >= _SKEW_COUNT_THRESHOLD:
        return (
            "ac1",
            f"{skewed} of {len(reports)} behaviors have prevalence < 10% or > 90% "
            "— Gwet's AC1 (paradox-robust) is the primary metric.",
        )
    return (
        "alpha",
        f"Only {skewed} of {len(reports)} behaviors are extreme-prevalence "
        "— Krippendorff α is the primary metric.",
    )


def compute_calibration(
    labels_by_rater: Mapping[str, Sequence[LabeledTurn]],
    behaviors: Sequence[str],
    turn_keys: Sequence[tuple[str, str]],
    *,
    module: str,
    prompt_version: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> CalibrationReport:
    """Compute per-behavior IAA metrics + 95% BCa CIs across raters.

    Raises ``ValueError`` if any rater is missing a turn in ``turn_keys``
    (the matrix builders enforce this). The caller should pre-filter to
    shared items before calling.
    """
    if len(labels_by_rater) < 2:
        raise ValueError("need at least 2 raters for inter-annotator agreement")
    if not turn_keys:
        raise ValueError("turn_keys must be non-empty")

    rng_seed = np.random.default_rng(seed)

    reports: list[BehaviorReport] = []
    for behavior in behaviors:
        presence = presence_matrix(labels_by_rater, behavior, turn_keys)
        intensity = intensity_matrix(labels_by_rater, behavior, turn_keys)
        n = presence.shape[1]
        prevalence = float(presence.mean())

        # Per-child RNG so each metric's bootstrap is independently
        # seeded but the whole call is still reproducible.
        rng_child = np.random.default_rng(rng_seed.integers(1_000_000))

        alpha = bootstrap_metric(
            presence, _krippendorff_stat, n_resamples=n_bootstrap, rng=rng_child
        )
        ac1 = bootstrap_metric(
            presence, gwet_ac1, n_resamples=n_bootstrap, rng=rng_child
        )
        kappa = bootstrap_metric(
            presence, cohen_kappa, n_resamples=n_bootstrap, rng=rng_child
        )
        qwk = bootstrap_metric(
            intensity, quadratic_weighted_kappa, n_resamples=n_bootstrap, rng=rng_child
        )

        reports.append(
            BehaviorReport(
                behavior=behavior,
                n_items=n,
                prevalence_overall=prevalence,
                alpha=alpha,
                ac1=ac1,
                kappa=kappa,
                qwk=qwk,
            )
        )

    primary, rationale = _pick_primary(reports)

    return CalibrationReport(
        module=module,
        prompt_version=prompt_version,
        n_items_total=len(turn_keys),
        behaviors=reports,
        primary_metric=primary,
        rationale=rationale,
    )


def _fmt_metric(m: MetricResult) -> str:
    if not np.isfinite(m.value):
        return "—"  # undefined (e.g. zero-variance behavior → α has no domain)
    if m.ci_low is None or m.ci_high is None:
        return f"{m.value:.3f}"
    return f"{m.value:.3f} [{m.ci_low:.3f}, {m.ci_high:.3f}]"


def render_rich_table(report: CalibrationReport, console: Console) -> None:
    """Print the report as a rich table for the terminal."""
    header = Table.grid(padding=(0, 2))
    header.add_row("[bold]Module[/bold]", report.module)
    header.add_row("[bold]Prompt version[/bold]", report.prompt_version)
    header.add_row("[bold]Items (held-out)[/bold]", str(report.n_items_total))
    header.add_row("[bold]Primary metric[/bold]", report.primary_metric)
    console.print(header)
    console.print(f"[dim]{report.rationale}[/dim]")

    tbl = Table(title="Per-behavior inter-annotator agreement (95% BCa CI)")
    tbl.add_column("Behavior")
    tbl.add_column("n", justify="right")
    tbl.add_column("prev.", justify="right")
    tbl.add_column("Krippendorff α", justify="right")
    tbl.add_column("Gwet AC1", justify="right")
    tbl.add_column("Cohen κ", justify="right")
    tbl.add_column("QWK (intensity)", justify="right")

    for r in report.behaviors:
        tbl.add_row(
            r.behavior,
            str(r.n_items),
            f"{r.prevalence_overall:.2f}",
            _fmt_metric(r.alpha),
            _fmt_metric(r.ac1),
            _fmt_metric(r.kappa),
            _fmt_metric(r.qwk),
        )
    console.print(tbl)


def render_markdown(report: CalibrationReport) -> str:
    """Render the report as a markdown block suitable for ``docs/calibration.md``."""
    lines: list[str] = [
        f"## Module {report.module} — calibration (prompt {report.prompt_version})",
        "",
        f"- Held-out items: {report.n_items_total}",
        f"- Primary metric: **{report.primary_metric}**",
        f"- Rationale: {report.rationale}",
        "",
        "| Behavior | n | Prevalence | Krippendorff α | Gwet AC1 | Cohen κ | QWK |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report.behaviors:
        lines.append(
            f"| {r.behavior} | {r.n_items} | {r.prevalence_overall:.2f} | "
            f"{_fmt_metric(r.alpha)} | {_fmt_metric(r.ac1)} | "
            f"{_fmt_metric(r.kappa)} | {_fmt_metric(r.qwk)} |"
        )
    lines.append("")
    lines.append(
        "Metrics carry 95% BCa bootstrap CIs. "
        "Implementation: `lucid.calibration.validate` (hand-rolled Gwet AC1, "
        "Cohen κ, QWK; Krippendorff α via the `krippendorff` library)."
    )
    return "\n".join(lines)
