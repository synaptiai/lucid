"""Lucid report generation — Jinja2 + inline SVG whiskers.

Renders an audit's findings to a single HTML file at
``report/<run_id>.html``. No external JS dependencies in the MVP:
confidence whiskers render as inline SVG, charts are HTML tables, and
Chart.js vendoring is deferred to the Phase 10 stretch.

Security stance:
- Jinja ``Environment(autoescape=select_autoescape(['html', 'j2']))``
  so every ``{{ expr }}`` is HTML-escaped by default. No ``|safe``
  filters on user-supplied text — the only content marked safe is
  the template author's own markup.
- CSP meta tag in ``base.html.j2`` forbids inline scripts, inline
  styles outside the document's own ``<style>`` block, and external
  script/style sources.
- No ``eval``, no ``innerHTML``, no script tags anywhere.
"""

from lucid.report.generator import (
    AggregatedFindings,
    ConfidenceInterval,
    ReportContext,
    aggregate_findings,
    beta_ci,
    render_report,
    write_report,
)

__all__ = [
    "AggregatedFindings",
    "ConfidenceInterval",
    "ReportContext",
    "aggregate_findings",
    "beta_ci",
    "render_report",
    "write_report",
]
