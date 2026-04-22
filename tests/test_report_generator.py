"""Tests for :mod:`lucid.report.generator`.

Covers aggregation correctness, Beta-CI computation, SVG whisker
rendering, end-to-end Jinja render, XSS resistance on user-sourced
strings, CSP meta presence, partial-status banner, empty-state
markers, and report size budget.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lucid.report.generator import (
    TOP_DETAILS_PER_MODULE,
    aggregate_findings,
    beta_ci,
    render_report,
    write_report,
)
from lucid.schemas import (
    AuditRun,
    CorpusStats,
    Finding,
    ModuleName,
    SamplingConfigRecord,
    Source,
    TokenUsage,
)

# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _audit_run(
    *,
    run_id: str = "run-report-1",
    status: str = "completed",
    skipped: list[ModuleName] | None = None,
) -> AuditRun:
    return AuditRun(
        id=run_id,
        sources=[Source.CLAUDE_AI],
        source_paths={Source.CLAUDE_AI: "/tmp/export"},
        started_at=datetime(2026, 4, 22, 10, 0, 0, tzinfo=UTC),
        completed_at=datetime(2026, 4, 22, 10, 12, 0, tzinfo=UTC),
        corpus_stats=CorpusStats(
            discovered_conversations=120,
            sampled_conversations=50,
            discovered_turns=2400,
            sampled_turns=900,
            sources=[Source.CLAUDE_AI],
        ),
        token_usage=TokenUsage(by_module={}, orchestrator=None),
        sampling_config=SamplingConfigRecord(
            n=50,
            seed=42,
            min_turns=5,
            recency_weight=0.3,
            recency_window_days=90,
            stratify_by_project=True,
            top_n_projects=20,
        ),
        status=status,  # type: ignore[arg-type]
        corpus_fingerprint="abcdef" * 10,
        prompt_versions={},
        schema_version=1,
        skipped_modules=skipped or [],
    )


def _finding(
    *,
    id_: str,
    module: ModuleName,
    behavior: str,
    intensity: int | None = 2,
    confidence: float = 0.82,
    confidence_alpha: float | None = None,
    confidence_beta: float | None = None,
    quote_user: str | None = None,
    quote_assistant: str | None = None,
    explanation: str = "A behaviour was detected in this turn.",
    evidence_quotes: list[str] | None = None,
    citation: str = "test citation",
    conversation_id: str | None = "c-42",
) -> Finding:
    return Finding(
        id=id_,
        audit_run_id="run-report-1",
        conversation_id=conversation_id,
        turn_ids=[f"{id_}-t0"],
        turn_ids_hash="hash-" + id_,
        module=module,
        behavior=behavior,
        intensity=intensity,
        confidence=confidence,
        confidence_alpha=confidence_alpha,
        confidence_beta=confidence_beta,
        quote_user=quote_user,
        quote_assistant=quote_assistant,
        evidence_quotes=evidence_quotes or [],
        explanation=explanation,
        citation=citation,
        detected_by=["claude-opus-4-7"],
        detected_at=datetime(2026, 4, 22, 10, 5, 0, tzinfo=UTC),
        prompt_version="v1",
        prompt_hash="0" * 64,
    )


def _sample_findings() -> list[Finding]:
    return [
        _finding(id_="a1", module=ModuleName.A_SPIRALBENCH, behavior="pushback", intensity=2, confidence=0.85),
        _finding(id_="a2", module=ModuleName.A_SPIRALBENCH, behavior="pushback", intensity=3, confidence=0.92),
        _finding(id_="a3", module=ModuleName.A_SPIRALBENCH, behavior="sycophancy", intensity=1, confidence=0.60),
        _finding(id_="b1", module=ModuleName.B_SHARMA, behavior="feedback-sycophancy", intensity=2, confidence=0.75),
        _finding(
            id_="g1",
            module=ModuleName.G_ATTRIBUTION,
            behavior="model=claude-sonnet-4-5",
            intensity=None,
            confidence=0.6,
        ),
        _finding(
            id_="g2",
            module=ModuleName.G_ATTRIBUTION,
            behavior="model=claude-sonnet-4-5",
            intensity=None,
            confidence=0.6,
        ),
        _finding(
            id_="g3",
            module=ModuleName.G_ATTRIBUTION,
            behavior="model=claude-sonnet-4-6",
            intensity=None,
            confidence=0.6,
        ),
    ]


# Attach metadata to G findings so the aggregation has year-month + model_id.
def _sample_findings_with_g_metadata() -> list[Finding]:
    findings = _sample_findings()
    augmented: list[Finding] = []
    for f in findings:
        if f.module is ModuleName.G_ATTRIBUTION:
            model_id = f.behavior.removeprefix("model=")
            # 2 findings 2026-02 for sonnet-4-5; 1 finding 2026-04 for sonnet-4-6
            ym = "2026-02" if "4-5" in model_id else "2026-04"
            augmented.append(
                f.model_copy(
                    update={
                        "metadata": {
                            "model_id": model_id,
                            "year_month": ym,
                            "source": "inferred",
                        }
                    }
                )
            )
        else:
            augmented.append(f)
    return augmented


# ──────────────────────────────────────────────────────────────────────────
# beta_ci
# ──────────────────────────────────────────────────────────────────────────


def test_beta_ci_returns_none_for_missing_params() -> None:
    assert beta_ci(None, None) is None
    assert beta_ci(1.0, None) is None
    assert beta_ci(None, 1.0) is None


def test_beta_ci_rejects_non_positive() -> None:
    assert beta_ci(0.0, 1.0) is None
    assert beta_ci(1.0, -1.0) is None


def test_beta_ci_symmetric_around_half() -> None:
    ci = beta_ci(10.0, 10.0)
    assert ci is not None
    assert ci.point == pytest.approx(0.5, abs=1e-6)
    # symmetric CI
    assert ci.upper - 0.5 == pytest.approx(0.5 - ci.lower, abs=1e-3)


def test_beta_ci_high_alpha_pushes_toward_one() -> None:
    ci = beta_ci(80.0, 20.0)
    assert ci is not None
    assert ci.point == pytest.approx(0.8, abs=1e-6)
    assert ci.upper > ci.point > ci.lower


# ──────────────────────────────────────────────────────────────────────────
# aggregate_findings
# ──────────────────────────────────────────────────────────────────────────


def test_aggregate_total_count() -> None:
    findings = _sample_findings()
    agg = aggregate_findings(findings)
    assert agg.total == len(findings)


def test_aggregate_every_module_has_a_section() -> None:
    agg = aggregate_findings(_sample_findings())
    modules = {section.module for section in agg.module_sections}
    assert modules == set(ModuleName)


def test_aggregate_empty_module_marked_empty() -> None:
    agg = aggregate_findings(_sample_findings())
    # ModuleName.H_MEMORY has no findings in the sample set → status=empty
    h_section = next(s for s in agg.module_sections if s.module is ModuleName.H_MEMORY)
    assert h_section.status == "empty"
    assert h_section.count == 0


def test_aggregate_skipped_module_marked_incomplete() -> None:
    agg = aggregate_findings(
        _sample_findings(),
        skipped_modules=[ModuleName.D_PERSPECTIVE],
    )
    d_section = next(s for s in agg.module_sections if s.module is ModuleName.D_PERSPECTIVE)
    assert d_section.status == "incomplete"


def test_aggregate_per_module_count_and_summary_rows() -> None:
    agg = aggregate_findings(_sample_findings())
    a_section = next(s for s in agg.module_sections if s.module is ModuleName.A_SPIRALBENCH)
    assert a_section.status == "ran"
    assert a_section.count == 3
    behaviors = {row.behavior for row in a_section.summary_rows}
    assert behaviors == {"pushback", "sycophancy"}
    pushback_row = next(r for r in a_section.summary_rows if r.behavior == "pushback")
    assert pushback_row.count == 2
    assert pushback_row.mean_intensity == pytest.approx(2.5)


def test_aggregate_top_details_sorted_by_intensity() -> None:
    """Higher-intensity findings appear first in top_details."""
    agg = aggregate_findings(_sample_findings())
    a_section = next(s for s in agg.module_sections if s.module is ModuleName.A_SPIRALBENCH)
    top = a_section.top_details
    assert len(top) <= TOP_DETAILS_PER_MODULE
    # First two should be intensity 3 then 2 for pushback.
    assert top[0].intensity == 3


def test_aggregate_model_buckets_groups_g_findings() -> None:
    agg = aggregate_findings(_sample_findings_with_g_metadata())
    keys = [b.key for b in agg.model_buckets]
    assert "claude-sonnet-4-5" in keys
    sonnet_45 = next(b for b in agg.model_buckets if b.key == "claude-sonnet-4-5")
    assert sonnet_45.count == 2


def test_aggregate_time_buckets_chronological_and_keys() -> None:
    agg = aggregate_findings(_sample_findings_with_g_metadata())
    keys = [b.key for b in agg.time_buckets]
    # Chronological order
    assert keys == sorted(keys)
    assert "2026-02" in keys
    assert "2026-04" in keys


# ──────────────────────────────────────────────────────────────────────────
# render_report — structural checks
# ──────────────────────────────────────────────────────────────────────────


def test_render_contains_run_id() -> None:
    html_text = render_report(_audit_run(), _sample_findings())
    assert "run-report-1" in html_text


def test_render_csp_meta_tag_present() -> None:
    html_text = render_report(_audit_run(), [])
    assert "Content-Security-Policy" in html_text
    # default-src 'none' keeps every source class explicit
    assert "default-src 'none'" in html_text


def test_render_includes_partial_banner_when_status_partial() -> None:
    html_text = render_report(_audit_run(status="partial"), _sample_findings())
    assert 'class="partial-banner"' in html_text
    assert "Partial run" in html_text


def test_render_omits_partial_banner_when_completed() -> None:
    html_text = render_report(_audit_run(status="completed"), _sample_findings())
    assert 'class="partial-banner"' not in html_text


def test_render_shows_incomplete_badge_for_skipped_module() -> None:
    html_text = render_report(
        _audit_run(status="partial", skipped=[ModuleName.D_PERSPECTIVE]),
        _sample_findings(),
    )
    # At minimum the D section should have the incomplete badge.
    section_d = _extract_section(html_text, ModuleName.D_PERSPECTIVE)
    assert 'class="status-badge status-incomplete"' in section_d


def test_render_shows_empty_state_for_modules_with_no_findings() -> None:
    html_text = render_report(_audit_run(), _sample_findings())
    section_h = _extract_section(html_text, ModuleName.H_MEMORY)
    assert 'class="empty-state"' in section_h


def test_render_shows_whisker_svg_when_beta_params_present() -> None:
    finding = _finding(
        id_="a1",
        module=ModuleName.A_SPIRALBENCH,
        behavior="pushback",
        intensity=2,
        confidence=0.8,
        confidence_alpha=40,
        confidence_beta=10,
    )
    html_text = render_report(_audit_run(), [finding])
    # Inline SVG whisker renders with the aria-label.
    assert 'aria-label="confidence interval"' in html_text
    assert "<rect" in html_text and "<line" in html_text


def test_render_shows_plain_bar_when_no_beta_params() -> None:
    finding = _finding(
        id_="a1",
        module=ModuleName.A_SPIRALBENCH,
        behavior="pushback",
        intensity=2,
        confidence=0.8,
    )
    html_text = render_report(_audit_run(), [finding])
    assert 'aria-label="confidence"' in html_text


# ──────────────────────────────────────────────────────────────────────────
# XSS resistance
# ──────────────────────────────────────────────────────────────────────────


def _xss_payload() -> str:
    return '<script>alert(1)</script><img src=x onerror=alert(2)>'


def test_xss_payload_escaped_in_explanation() -> None:
    payload = _xss_payload()
    finding = _finding(
        id_="a1",
        module=ModuleName.A_SPIRALBENCH,
        behavior="pushback",
        explanation=payload,
    )
    html_text = render_report(_audit_run(), [finding])
    assert payload not in html_text
    # The escaped form should appear instead.
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_text


def test_xss_payload_escaped_in_quote_fields() -> None:
    payload = _xss_payload()
    finding = _finding(
        id_="a1",
        module=ModuleName.A_SPIRALBENCH,
        behavior="pushback",
        quote_user=payload,
        quote_assistant=payload,
        evidence_quotes=[payload],
    )
    html_text = render_report(_audit_run(), [finding])
    assert payload not in html_text
    # Escaped occurrences should appear multiple times — one per surface.
    assert html_text.count("&lt;script&gt;alert(1)&lt;/script&gt;") >= 3


def test_xss_payload_escaped_in_behavior_label() -> None:
    """A behaviour string like '<img src=x onerror=alert(1)>' — as could
    happen if an LLM emitted a malicious label — must render escaped.

    Assertion focuses on the angle brackets: once ``<`` / ``>`` are
    escaped, the payload is text rather than a tag. The literal
    ``onerror=alert(1)`` substring surviving the render is fine because
    browsers only execute it when it sits inside an actual tag's
    attribute list.
    """
    payload = "<img src=x onerror=alert(1)>"
    finding = _finding(
        id_="a1",
        module=ModuleName.A_SPIRALBENCH,
        behavior=payload,
    )
    html_text = render_report(_audit_run(), [finding])
    assert payload not in html_text
    assert "&lt;img src=x onerror=alert(1)&gt;" in html_text


def test_xss_payload_in_citation_escaped() -> None:
    payload = '"><script>alert(1)</script>'
    finding = _finding(
        id_="a1",
        module=ModuleName.A_SPIRALBENCH,
        behavior="pushback",
        citation=payload,
    )
    html_text = render_report(_audit_run(), [finding])
    assert "<script>alert(1)</script>" not in html_text


def test_svg_whisker_is_not_user_influenced() -> None:
    """The SVG whisker markup is marked-safe, but it only ever embeds
    numeric fields computed from Beta(α, β). Injecting a malicious
    'behavior' string still escapes — the safe-mark does not extend to
    surrounding template expressions."""
    finding = _finding(
        id_="a1",
        module=ModuleName.A_SPIRALBENCH,
        behavior="<x>",
        confidence_alpha=10,
        confidence_beta=10,
    )
    html_text = render_report(_audit_run(), [finding])
    assert "<x>" not in html_text
    assert "&lt;x&gt;" in html_text


# ──────────────────────────────────────────────────────────────────────────
# Report size budget
# ──────────────────────────────────────────────────────────────────────────


def test_report_size_under_2mb_for_sample_findings() -> None:
    html_text = render_report(_audit_run(), _sample_findings())
    assert len(html_text.encode("utf-8")) < 2 * 1024 * 1024


# ──────────────────────────────────────────────────────────────────────────
# write_report
# ──────────────────────────────────────────────────────────────────────────


def test_write_report_writes_file_and_returns_path(tmp_path: Path) -> None:
    out = write_report(
        _audit_run(run_id="run-write-1"),
        _sample_findings(),
        output_dir=tmp_path / "report",
    )
    assert out.is_file()
    assert out.name == "run-write-1.html"
    content = out.read_text(encoding="utf-8")
    assert "run-write-1" in content
    assert "Content-Security-Policy" in content


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────


def _extract_section(html_text: str, module: ModuleName) -> str:
    """Return the section markup for one module (best-effort regex cut)."""
    start_pattern = f'<section id="mod-{module.value}">'
    end_pattern = "</section>"
    m = re.search(
        re.escape(start_pattern) + r"(.*?)" + re.escape(end_pattern),
        html_text,
        re.DOTALL,
    )
    assert m is not None, f"section for {module.value} not found in rendered HTML"
    return m.group(0)
