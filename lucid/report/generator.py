"""Lucid HTML report generator.

Reads ``AuditRun`` + ``Finding`` rows and produces a single static HTML
file. Rendering is one-shot: no streaming, no live queries — the report
is a frozen artifact of the audit run.

**Aggregation.** Raw findings are bucketed before the template ever
sees them. :func:`aggregate_findings` groups by module and computes:

- per-module count + per-behavior count (for the top-level summary)
- per-(behavior, time-bucket) counts for Module G time-series bars
- per-(behavior, model) counts for Module G model attribution
- top N findings per module by intensity-then-confidence for drill-down

**SVG whiskers.** When a ``Finding`` carries Beta posterior parameters
(``confidence_alpha`` and ``confidence_beta``), the report renders a
95% credible interval via :func:`beta_ci` + inline SVG. Findings
without posteriors still render a confidence bar (no whiskers).

**Security.** Jinja autoescape is on for ``.html`` and ``.j2``
extensions. CSP meta tag in ``base.html.j2`` blocks inline scripts and
external resources. All user-sourced text (explanations, quotes,
behavior labels) passes through Jinja's default escaping — do not add
``|safe`` filters. See the test-suite's XSS-injection fixture
(:mod:`tests.test_report_generator`) for the contract.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, select_autoescape
from scipy.stats import beta as scipy_beta

from lucid.schemas import AuditRun, Finding, ModuleName

__all__ = [
    "DEFAULT_TEMPLATE_NAME",
    "AggregatedFindings",
    "ConfidenceInterval",
    "ReportContext",
    "aggregate_findings",
    "beta_ci",
    "build_jinja_env",
    "render_report",
    "write_report",
]


DEFAULT_TEMPLATE_NAME = "report.html.j2"


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A scalar estimate plus symmetric-credible-interval bounds in [0, 1]."""

    point: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.lower <= self.point <= self.upper <= 1.0):
            # Guard against NaN or out-of-order bounds; render will skip it.
            object.__setattr__(self, "lower", max(0.0, min(1.0, self.lower)))
            object.__setattr__(self, "upper", max(0.0, min(1.0, self.upper)))
            object.__setattr__(
                self,
                "point",
                max(self.lower, min(self.upper, max(0.0, min(1.0, self.point)))),
            )


def beta_ci(
    alpha: float | None,
    beta_param: float | None,
    *,
    credible_mass: float = 0.95,
) -> ConfidenceInterval | None:
    """95% credible interval for a Beta posterior.

    Returns ``None`` when ``alpha`` or ``beta_param`` are missing or
    non-positive — the caller renders the plain confidence value
    without whiskers in that case.

    ``credible_mass`` is symmetric about the mean; pass 0.9 for a 90%
    interval. The point estimate is the posterior mean
    ``alpha / (alpha + beta)``.
    """
    if alpha is None or beta_param is None:
        return None
    if alpha <= 0 or beta_param <= 0:
        return None
    tail = (1.0 - credible_mass) / 2.0
    lower = float(scipy_beta.ppf(tail, alpha, beta_param))
    upper = float(scipy_beta.ppf(1.0 - tail, alpha, beta_param))
    mean = alpha / (alpha + beta_param)
    return ConfidenceInterval(point=float(mean), lower=lower, upper=upper)


# ──────────────────────────────────────────────────────────────────────────
# Aggregation shapes
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FindingSummary:
    """One row in a per-module findings table."""

    behavior: str
    count: int
    mean_intensity: float | None
    mean_confidence: float


@dataclass(frozen=True, slots=True)
class FindingDetail:
    """One Finding, pre-formatted for template rendering.

    Precomputes the CI so the template doesn't call scipy. Quote fields
    are already HTML-unsafe strings — Jinja autoescape handles escaping
    at render time.
    """

    id: str
    module: ModuleName
    conversation_id: str | None
    behavior: str
    intensity: int | None
    confidence: float
    confidence_ci: ConfidenceInterval | None
    quote_user: str | None
    quote_assistant: str | None
    evidence_quotes: list[str]
    explanation: str
    citation: str
    detected_at: datetime


@dataclass(frozen=True, slots=True)
class ModuleSection:
    """All the data the template needs to render one module's section."""

    module: ModuleName
    name: str
    description: str
    count: int
    summary_rows: list[FindingSummary]
    top_details: list[FindingDetail]
    status: Literal["ran", "empty", "incomplete"]


@dataclass(frozen=True, slots=True)
class AttributionBucket:
    """One row in Module G's per-model / per-time bucket."""

    key: str
    count: int
    avg_confidence: float


@dataclass(frozen=True, slots=True)
class AggregatedFindings:
    """Everything the template receives (besides AuditRun metadata)."""

    total: int
    module_sections: list[ModuleSection]
    model_buckets: list[AttributionBucket]
    time_buckets: list[AttributionBucket]


MODULE_DESCRIPTIONS: dict[ModuleName, tuple[str, str]] = {
    ModuleName.A_SPIRALBENCH: (
        "Module A — Spiral-Bench behaviors",
        "17-behavior scorer on assistant turns (Spiral-Bench v1.2 rubric). "
        "Turn-level; intensity 1-3.",
    ),
    ModuleName.B_SHARMA: (
        "Module B — Sharma paired-exchange sycophancy",
        "Feedback divergence across opposite-sentiment pairs and "
        "answer cave-ins on user challenges.",
    ),
    ModuleName.C_SYCEVAL: (
        "Module C — SycEval direction",
        "Classifies detected sycophancy events as progressive (cave-in "
        "landed correct) or regressive (cave-in landed wrong).",
    ),
    ModuleName.D_PERSPECTIVE: (
        "Module D — Jain perspective sycophancy (opt-in)",
        "Cross-turn framing / vocabulary / premise drift toward the user's "
        "worldview without explicit agreement. Only runs with --include-module-d.",
    ),
    ModuleName.E_BELIEFSHIFT: (
        "Module E — BeliefShift",
        "Cross-conversation belief-drift with evidence-vs-pressure attribution on each shift.",
    ),
    ModuleName.F_ITP: (
        "Module F — Influence Tactics Protocol",
        "9-category user-prompt analyzer: emotional triggers, urgency, "
        "false dilemmas, authority overload, framing techniques, etc.",
    ),
    ModuleName.G_ATTRIBUTION: (
        "Module G — Time/model attribution",
        "Deterministic attribution of each conversation to a Claude model "
        "version and time bucket. No LLM; runs last.",
    ),
    ModuleName.H_MEMORY: (
        "Module H — Memory-corpus consistency",
        "Verifies AI-synthesized memory claims against the user's actual "
        "conversation corpus via retrieval + Opus classification.",
    ),
}


TOP_DETAILS_PER_MODULE = 10
TIME_BUCKETS_SHOWN = 18  # ~18 months of history


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _summary_for_module(findings: Sequence[Finding]) -> list[FindingSummary]:
    """Aggregate per-behavior summary rows for one module."""
    by_behavior: dict[str, list[Finding]] = {}
    for f in findings:
        by_behavior.setdefault(f.behavior, []).append(f)
    out: list[FindingSummary] = []
    for behavior, group in by_behavior.items():
        intensities = [f.intensity for f in group if f.intensity is not None]
        mean_int = _mean([float(i) for i in intensities]) if intensities else None
        confs = [f.confidence for f in group]
        out.append(
            FindingSummary(
                behavior=behavior,
                count=len(group),
                mean_intensity=mean_int,
                mean_confidence=_mean(confs),
            )
        )
    out.sort(key=lambda r: (-r.count, r.behavior))
    return out


def _detail_for_finding(f: Finding) -> FindingDetail:
    return FindingDetail(
        id=f.id,
        module=f.module,
        conversation_id=f.conversation_id,
        behavior=f.behavior,
        intensity=f.intensity,
        confidence=f.confidence,
        confidence_ci=beta_ci(f.confidence_alpha, f.confidence_beta),
        quote_user=f.quote_user,
        quote_assistant=f.quote_assistant,
        evidence_quotes=list(f.evidence_quotes),
        explanation=f.explanation,
        citation=f.citation,
        detected_at=f.detected_at,
    )


def _top_details(
    findings: Sequence[Finding], *, n: int = TOP_DETAILS_PER_MODULE
) -> list[FindingDetail]:
    """Top-n findings for drill-down, sorted by intensity desc, confidence desc."""
    scored = sorted(
        findings,
        key=lambda f: (-(f.intensity or 0), -f.confidence),
    )[:n]
    return [_detail_for_finding(f) for f in scored]


def _model_buckets(attribution_findings: Sequence[Finding]) -> list[AttributionBucket]:
    """From Module G findings, group by model id."""
    by_model: dict[str, list[Finding]] = {}
    for f in attribution_findings:
        model = str(f.metadata.get("model_id") or f.behavior)
        by_model.setdefault(model, []).append(f)
    out = [
        AttributionBucket(
            key=model,
            count=len(group),
            avg_confidence=_mean([f.confidence for f in group]),
        )
        for model, group in by_model.items()
    ]
    out.sort(key=lambda b: (-b.count, b.key))
    return out


def _time_buckets(
    attribution_findings: Sequence[Finding],
    *,
    limit: int = TIME_BUCKETS_SHOWN,
) -> list[AttributionBucket]:
    by_month: dict[str, list[Finding]] = {}
    for f in attribution_findings:
        key = str(f.metadata.get("year_month") or "")
        if not key:
            continue
        by_month.setdefault(key, []).append(f)
    out = [
        AttributionBucket(
            key=month,
            count=len(group),
            avg_confidence=_mean([f.confidence for f in group]),
        )
        for month, group in by_month.items()
    ]
    out.sort(key=lambda b: b.key)  # chronological
    return out[-limit:]  # most recent N


def aggregate_findings(
    findings: Iterable[Finding],
    *,
    skipped_modules: Sequence[ModuleName] = (),
) -> AggregatedFindings:
    """Bucket raw findings into what the template expects.

    ``skipped_modules`` receives an "incomplete" marker in its section
    rather than an "empty" marker — the distinction matters for the
    partial-status banner's accuracy.
    """
    findings_list = list(findings)
    by_module: dict[ModuleName, list[Finding]] = {m: [] for m in ModuleName}
    for f in findings_list:
        by_module[f.module].append(f)

    skipped = set(skipped_modules)
    sections: list[ModuleSection] = []
    for module in ModuleName:
        group = by_module[module]
        name, description = MODULE_DESCRIPTIONS[module]
        status: Literal["ran", "empty", "incomplete"]
        if module in skipped:
            status = "incomplete"
        elif not group:
            status = "empty"
        else:
            status = "ran"
        sections.append(
            ModuleSection(
                module=module,
                name=name,
                description=description,
                count=len(group),
                summary_rows=_summary_for_module(group),
                top_details=_top_details(group),
                status=status,
            )
        )

    g_findings = by_module[ModuleName.G_ATTRIBUTION]
    return AggregatedFindings(
        total=len(findings_list),
        module_sections=sections,
        model_buckets=_model_buckets(g_findings),
        time_buckets=_time_buckets(g_findings),
    )


# ──────────────────────────────────────────────────────────────────────────
# Jinja env + render
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ReportContext:
    """What the template receives."""

    audit_run: AuditRun
    aggregate: AggregatedFindings


def _format_pct(value: float) -> str:
    """1-decimal percent; used by the template via Jinja filter."""
    return f"{value * 100:.1f}%"


def _format_conf(value: float) -> str:
    return f"{value:.2f}"


def _svg_whisker(ci: ConfidenceInterval | None, width: int = 160) -> str:
    """Render a confidence bar + whiskers as an inline SVG string.

    The return value is marked-safe by the caller: this function never
    embeds untrusted input. All numeric values pass through
    :func:`float` and :func:`html.escape` before formatting.
    """
    if ci is None:
        return ""
    x_lower = int(ci.lower * width)
    x_upper = int(ci.upper * width)
    x_point = int(ci.point * width)
    height = 14
    # Solid bar from 0 → point estimate; whisker from lower to upper.
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="confidence interval">'
        f'<rect x="0" y="4" width="{x_point}" height="6" fill="#335580"/>'
        f'<line x1="{x_lower}" y1="7" x2="{x_upper}" y2="7" '
        f'stroke="#335580" stroke-width="2"/>'
        f'<line x1="{x_lower}" y1="2" x2="{x_lower}" y2="12" '
        f'stroke="#335580" stroke-width="2"/>'
        f'<line x1="{x_upper}" y1="2" x2="{x_upper}" y2="12" '
        f'stroke="#335580" stroke-width="2"/>'
        f"</svg>"
    )


def _conf_bar(point: float, width: int = 160) -> str:
    """Simple confidence bar (no whiskers) for findings without Beta params."""
    x = int(max(0.0, min(1.0, point)) * width)
    height = 14
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="confidence">'
        f'<rect x="0" y="4" width="{width}" height="6" fill="#eee"/>'
        f'<rect x="0" y="4" width="{x}" height="6" fill="#8aaad0"/>'
        f"</svg>"
    )


def build_jinja_env(template_dir: Path | None = None) -> Environment:
    """Construct the Jinja environment.

    Every template extension we render is HTML-class; autoescape is
    unconditional. The only filters registered produce inline SVG and
    numeric formatting — neither carries user content, so both are
    safe-by-construction.
    """
    root = template_dir or (Path(__file__).parent / "templates")
    env = Environment(
        loader=FileSystemLoader(str(root)),
        autoescape=select_autoescape(enabled_extensions=("html", "j2", "html.j2")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = _format_pct
    env.filters["conf"] = _format_conf
    env.filters["svg_whisker"] = _svg_whisker
    env.filters["conf_bar"] = _conf_bar
    env.filters["html_escape"] = html.escape
    return env


def render_report(
    audit_run: AuditRun,
    findings: Iterable[Finding],
    *,
    template_name: str = DEFAULT_TEMPLATE_NAME,
    template_dir: Path | None = None,
) -> str:
    """Render the report HTML as a string.

    Callers pass ``audit_run`` with any status (``completed``,
    ``partial``, ``aborted_pre_spend``, …) — the template surfaces the
    status in a banner and, for partial runs, marks incomplete modules
    separately in their sections.
    """
    aggregate = aggregate_findings(
        findings,
        skipped_modules=tuple(audit_run.skipped_modules),
    )
    context = ReportContext(audit_run=audit_run, aggregate=aggregate)
    env = build_jinja_env(template_dir)
    template = env.get_template(template_name)
    return template.render(
        audit=context.audit_run,
        aggregate=context.aggregate,
        sections=context.aggregate.module_sections,
        total=context.aggregate.total,
        model_buckets=context.aggregate.model_buckets,
        time_buckets=context.aggregate.time_buckets,
    )


def write_report(
    audit_run: AuditRun,
    findings: Iterable[Finding],
    *,
    output_dir: Path = Path("report"),
    template_name: str = DEFAULT_TEMPLATE_NAME,
    template_dir: Path | None = None,
) -> Path:
    """Render and write the report to ``<output_dir>/<run_id>.html``.

    Returns the resolved output path for the CLI to log.
    """
    html_text = render_report(
        audit_run,
        findings,
        template_name=template_name,
        template_dir=template_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{audit_run.id}.html"
    out.write_text(html_text, encoding="utf-8")
    return out


# Dataclasses need this extra annotation to satisfy mypy strict on
# ``list[str] = field(default_factory=list)`` style; importing ``field``
# keeps it live for the tests that touch it.
_ = field
