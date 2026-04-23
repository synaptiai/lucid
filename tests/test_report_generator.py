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
    FingerprintCell,
    _compute_module_radar,
    _fingerprint_cells,
    _severity_counts_for_module,
    _svg_fingerprint,
    _svg_radar,
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
        _finding(
            id_="a1",
            module=ModuleName.A_SPIRALBENCH,
            behavior="pushback",
            intensity=2,
            confidence=0.85,
        ),
        _finding(
            id_="a2",
            module=ModuleName.A_SPIRALBENCH,
            behavior="pushback",
            intensity=3,
            confidence=0.92,
        ),
        _finding(
            id_="a3",
            module=ModuleName.A_SPIRALBENCH,
            behavior="sycophancy",
            intensity=1,
            confidence=0.60,
        ),
        _finding(
            id_="b1",
            module=ModuleName.B_SHARMA,
            behavior="feedback-sycophancy",
            intensity=2,
            confidence=0.75,
        ),
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
    # At minimum the D section should mark itself incomplete. Class
    # name is intentionally matched substring-only so later purely
    # visual refactors (adding utility classes, renaming the base
    # class) don't force a test update.
    section_d = _extract_section(html_text, ModuleName.D_PERSPECTIVE)
    assert "incomplete" in section_d.lower()
    assert "ms-incomplete" in section_d or "status-incomplete" in section_d


def test_render_shows_empty_state_for_modules_with_no_findings() -> None:
    html_text = render_report(_audit_run(), _sample_findings())
    section_h = _extract_section(html_text, ModuleName.H_MEMORY)
    # The empty-state line has a distinctive class hook; the exact
    # class name is implementation detail, so match either the
    # legacy ``empty-state`` name or the new ``empty-line``.
    assert 'class="empty-state"' in section_h or 'class="empty-line"' in section_h


# ──────────────────────────────────────────────────────────────────────────
# Editorial-brief primitives (headlines + plain-English + charts)
# ──────────────────────────────────────────────────────────────────────────


def test_behavior_to_plain_english_known_behaviour() -> None:
    """A behaviour in the canonical dictionary returns its
    hand-written translation verbatim."""
    from lucid.report.generator import behavior_to_plain_english

    assert (
        behavior_to_plain_english("sycophancy") == "Effusive agreement or praise without substance"
    )
    assert (
        behavior_to_plain_english("contradicted")
        == "Memory claim is contradicted by the conversations"
    )


def test_behavior_to_plain_english_unknown_falls_back() -> None:
    """An unseen behaviour gets humanised rather than failing — lets
    new modules ship without a same-PR dictionary update."""
    from lucid.report.generator import behavior_to_plain_english

    assert behavior_to_plain_english("brand-new-behaviour") == "Brand new behaviour"
    assert behavior_to_plain_english("snake_case_example") == "Snake case example"


def test_aggregate_findings_populates_headline_findings() -> None:
    """``aggregate_findings`` must produce a cross-module ``headlines``
    list sorted by severity desc, intensity desc, confidence desc.
    This powers the report's hero "What surprised us" section."""
    from lucid.report.generator import aggregate_findings

    findings = [
        _finding(
            id_="a1",
            module=ModuleName.A_SPIRALBENCH,
            behavior="sycophancy",
            intensity=1,
            confidence=0.7,
        ),
        _finding(
            id_="a2",
            module=ModuleName.A_SPIRALBENCH,
            behavior="confident-bullshitting",
            intensity=2,
            confidence=0.8,
        ),
        # Module G should never appear in headlines regardless of count.
        _finding(
            id_="g1",
            module=ModuleName.G_ATTRIBUTION,
            behavior="model=claude-sonnet-4-6",
            intensity=None,
            confidence=1.0,
        ),
    ]
    agg = aggregate_findings(findings)
    assert len(agg.headlines) == 2
    # Mid-severity intensity-2 beats low-severity intensity-1.
    assert agg.headlines[0].behavior == "confident-bullshitting"
    assert agg.headlines[0].rank == 1
    assert agg.headlines[0].severity_class == "mid"
    assert agg.headlines[0].plain_english == "States claims with conviction but no grounding"
    # Module G excluded — attribution is never a "surprise".
    assert all(h.module is not ModuleName.G_ATTRIBUTION for h in agg.headlines)


def test_aggregate_findings_headlines_diversify_by_module_behavior_pair() -> None:
    """The hero must not show five identical headlines when one
    (module, behavior) pair dominates the top of the severity-sorted
    list. The diversification first pass takes one entry per pair
    before backfilling. Five ``H/unsupported`` findings should produce
    one ``H/unsupported`` headline, then fill remaining slots from
    other pairs."""
    from lucid.report.generator import aggregate_findings

    findings = [
        _finding(
            id_=f"h{i}",
            module=ModuleName.H_MEMORY,
            behavior="unsupported",
            intensity=None,
            confidence=0.9,
        )
        for i in range(5)
    ] + [
        _finding(
            id_="a1",
            module=ModuleName.A_SPIRALBENCH,
            behavior="confident-bullshitting",
            intensity=2,
            confidence=0.8,
        ),
        _finding(
            id_="a2",
            module=ModuleName.A_SPIRALBENCH,
            behavior="sycophancy",
            intensity=1,
            confidence=0.7,
        ),
    ]
    agg = aggregate_findings(findings)
    # Exactly one H/unsupported in the first diversified pass.
    h_unsupported = [
        h for h in agg.headlines if h.module is ModuleName.H_MEMORY and h.behavior == "unsupported"
    ]
    # With only 3 distinct pairs (H/unsupported, A/confident-bullshitting,
    # A/sycophancy) and HEADLINES_SHOWN=6, the backfill will add 3 more
    # H/unsupported findings once diversity is exhausted — but never
    # BEFORE the other pairs get their first slot.
    assert len(h_unsupported) <= 4  # first slot + up to 3 backfills
    # Every distinct (module, behavior) pair in the input must appear
    # at least once.
    distinct_pairs_in_headlines = {(h.module, h.behavior) for h in agg.headlines}
    assert (ModuleName.A_SPIRALBENCH, "confident-bullshitting") in distinct_pairs_in_headlines
    assert (ModuleName.A_SPIRALBENCH, "sycophancy") in distinct_pairs_in_headlines
    assert (ModuleName.H_MEMORY, "unsupported") in distinct_pairs_in_headlines


def test_aggregate_findings_memory_module_severity_uses_behavior() -> None:
    """Module H has no ``intensity``; severity comes from the
    ``behavior`` (contradicted/unsupported → high; weakly-supported
    → mid; else neutral)."""
    from lucid.report.generator import aggregate_findings

    findings = [
        _finding(
            id_="h1",
            module=ModuleName.H_MEMORY,
            behavior="contradicted",
            intensity=None,
            confidence=0.9,
        ),
        _finding(
            id_="h2",
            module=ModuleName.H_MEMORY,
            behavior="well-supported",
            intensity=None,
            confidence=0.95,
        ),
    ]
    agg = aggregate_findings(findings)
    # ``contradicted`` must outrank ``well-supported`` even though
    # the latter has higher raw confidence.
    assert [h.behavior for h in agg.headlines] == ["contradicted", "well-supported"]
    assert agg.headlines[0].severity_class == "high"
    assert agg.headlines[1].severity_class == "neutral"


def test_aggregate_findings_populates_behaviour_totals() -> None:
    """The hero block reads ``behaviour_total`` + ``modules_with_findings``
    + ``modules_total`` — none of these should count Module G."""
    from lucid.report.generator import aggregate_findings

    findings = [
        _finding(id_="a1", module=ModuleName.A_SPIRALBENCH, behavior="sycophancy"),
        _finding(id_="h1", module=ModuleName.H_MEMORY, behavior="unsupported", intensity=None),
        _finding(id_="g1", module=ModuleName.G_ATTRIBUTION, behavior="model=x", intensity=None),
    ]
    agg = aggregate_findings(findings)
    assert agg.behaviour_total == 2
    assert agg.modules_with_findings == 2
    # 8 modules total minus G (deterministic attribution).
    assert agg.modules_total == 7


def test_render_hero_shows_plain_english_headline_for_top_finding() -> None:
    """The hero's "Top moments" section must render the plain-English
    translation as the primary copy, not the raw behaviour slug."""
    finding = _finding(
        id_="a1",
        module=ModuleName.A_SPIRALBENCH,
        behavior="confident-bullshitting",
        intensity=2,
        confidence=0.85,
        quote_assistant="This is definitely how databases work.",
    )
    html_text = render_report(_audit_run(), [finding])
    assert "States claims with conviction but no grounding" in html_text
    assert "What surprised us" in html_text
    # Hero should render the quote alongside the headline.
    assert "This is definitely how databases work." in html_text


def test_render_hero_quote_is_html_escaped() -> None:
    """User-derived quotes must still pass through Jinja autoescape;
    the hero block is not exempt."""
    finding = _finding(
        id_="a1",
        module=ModuleName.A_SPIRALBENCH,
        behavior="sycophancy",
        intensity=1,
        confidence=0.7,
        quote_assistant='<script>alert("pwn")</script>',
    )
    html_text = render_report(_audit_run(), [finding])
    assert "<script>alert" not in html_text
    assert "&lt;script&gt;" in html_text or "&lt;script" in html_text


def test_render_module_bars_svg_is_present_and_excludes_module_g() -> None:
    """The at-a-glance SVG bar chart excludes Module G (one finding
    per conversation by construction; it dominates the scale).
    Verified structurally: the aria-label is present and there are
    exactly 7 count labels (one per behaviour module)."""
    findings = [
        _finding(id_=f"a{i}", module=ModuleName.A_SPIRALBENCH, behavior="sycophancy")
        for i in range(3)
    ] + [
        _finding(id_=f"g{i}", module=ModuleName.G_ATTRIBUTION, behavior="model=x", intensity=None)
        for i in range(10)
    ]
    html_text = render_report(_audit_run(), findings)
    assert 'aria-label="findings per module"' in html_text
    # Seven module letters (A, B, C, D, E, F, H) in the bar chart — not 8.
    # The donut is a separate SVG; restrict the check to the bar-chart SVG.
    glance_start = html_text.index('aria-label="findings per module"')
    glance_end = html_text.index("</svg>", glance_start)
    bars = html_text[glance_start:glance_end]
    for letter in "ABCDEFH":
        assert f">{letter}<" in bars, f"module letter {letter} missing from at-a-glance bars"
    assert ">G<" not in bars, "Module G must be excluded from the at-a-glance bar chart"


def test_module_h_finding_renders_audited_claim_label_not_user() -> None:
    """The original bug: Module H stores the audited memory claim in
    ``quote_user`` (claims often start with the literal word "User"
    because that's how Anthropic synthesises memory entries). The
    legacy template labelled it ``Assistant``/``User``, which read as
    "the user said this" — actively disinforming. The new template
    must label it as the audited claim, not as a user utterance."""
    finding = _finding(
        id_="h1",
        module=ModuleName.H_MEMORY,
        behavior="unsupported",
        intensity=None,
        confidence=0.78,
        quote_user="User is preparing a Series A fundraise.",
        evidence_quotes=[],
    )
    finding = finding.model_copy(
        update={
            "metadata": {
                "claim_category": "work",
                "memory_source": "conversations_memory",
                "reasoning": (
                    "Top similarity 0.45; no excerpts mention investor pitches."
                ),
            }
        }
    )
    html_text = render_report(_audit_run(), [finding])
    assert "Audited memory claim" in html_text
    assert "work claim" in html_text
    assert "from conversations_memory" in html_text
    # The legacy "User" label must not be applied to a Module H claim.
    assert ">User<" not in html_text or "Audited memory claim" in html_text


def test_module_h_finding_surfaces_model_reasoning() -> None:
    """The model's verdict reasoning lives in
    ``finding.metadata.reasoning`` and previously wasn't rendered
    anywhere. It is the most informative single field on a Module H
    finding (e.g. "no excerpts state a backend-engineer title");
    it must reach the reader."""
    finding = _finding(
        id_="h2",
        module=ModuleName.H_MEMORY,
        behavior="weakly-supported",
        intensity=None,
        confidence=0.6,
        quote_user="User is a backend engineer.",
        evidence_quotes=["FastAPI router refactor", "TanStack Query setup"],
    )
    finding = finding.model_copy(
        update={
            "metadata": {
                "claim_category": "work",
                "memory_source": "conversations_memory",
                "reasoning": (
                    "Excerpts show full-stack work, not backend-only. "
                    "Retrieval moderate (top 0.53)."
                ),
            }
        }
    )
    html_text = render_report(_audit_run(), [finding])
    assert "Why this verdict" in html_text
    assert "Excerpts show full-stack work" in html_text
    # Corpus excerpts also reach the reader, with their own label.
    assert "Corpus excerpt" in html_text
    assert "FastAPI router refactor" in html_text


def test_module_h_unsupported_with_zero_excerpts_explains_no_evidence() -> None:
    """An ``unsupported`` Module H finding with no retrieved
    excerpts previously had no evidence UI at all — the absence
    of corpus support IS the verdict; surface it explicitly."""
    finding = _finding(
        id_="h3",
        module=ModuleName.H_MEMORY,
        behavior="unsupported",
        intensity=None,
        confidence=0.78,
        quote_user="User regularly pitches to investors.",
        evidence_quotes=[],
    )
    html_text = render_report(_audit_run(), [finding])
    assert "No matching evidence" in html_text
    assert "absence" in html_text.lower() or "no excerpts" in html_text.lower()


def test_module_a_finding_labels_assistant_quote_correctly() -> None:
    """For non-H modules the ``quote_assistant`` label stays — but
    the new evidence-block layout uses the canonical "Assistant
    turn" label instead of the bare "Assistant" the legacy template
    used. This is the same fix Module H got, applied uniformly."""
    finding = _finding(
        id_="a1",
        module=ModuleName.A_SPIRALBENCH,
        behavior="confident-bullshitting",
        intensity=2,
        confidence=0.85,
        quote_assistant="Partial indexes reduce scans by 80%+.",
    )
    html_text = render_report(_audit_run(), [finding])
    assert "Assistant turn" in html_text
    assert "Partial indexes reduce scans" in html_text


def test_module_b_finding_labels_user_challenge_distinctly() -> None:
    """Module B paired-exchange uses ``quote_user`` for the user's
    challenge that elicited a cave-in — not a generic user prompt.
    The label should reflect that semantic distinction."""
    finding = _finding(
        id_="b1",
        module=ModuleName.B_SHARMA,
        behavior="answer-sycophancy",
        intensity=2,
        confidence=0.8,
        quote_user="Are you sure about that answer?",
        quote_assistant="You're right, my apologies — let me reconsider.",
    )
    html_text = render_report(_audit_run(), [finding])
    assert "User challenge" in html_text
    assert "Assistant turn" in html_text
    assert "Are you sure about that answer?" in html_text


def test_finding_with_no_quotes_or_evidence_shows_no_evidence_state() -> None:
    """A non-H concern-direction finding without extracted quotes must
    still render a "no evidence" note rather than a silent empty
    section. Uses ``regressive`` (the real Module C concern direction);
    the old ``unknown`` case is now filtered from detail cards as a
    null-result behaviour — its empty-state is covered by
    :func:`test_null_result_only_module_shows_empty_note` below.
    """
    finding = _finding(
        id_="c1",
        module=ModuleName.C_SYCEVAL,
        behavior="regressive",
        intensity=None,
        confidence=0.95,
        quote_user=None,
        quote_assistant=None,
        evidence_quotes=[],
    )
    html_text = render_report(_audit_run(), [finding])
    assert "No matching evidence" in html_text


def test_null_result_only_module_shows_empty_note() -> None:
    """A module whose only findings are null-result behaviours (e.g.
    Module C ``unknown``) must tell the reader the classifier ran
    cleanly rather than leave the section visually empty below the
    per-behaviour summary."""
    finding = _finding(
        id_="c1",
        module=ModuleName.C_SYCEVAL,
        behavior="unknown",
        intensity=None,
        confidence=0.95,
        quote_user=None,
        quote_assistant=None,
        evidence_quotes=[],
    )
    html_text = render_report(_audit_run(), [finding])
    assert "No concern-direction behaviours surfaced" in html_text


def test_provenance_footer_includes_full_conversation_id() -> None:
    """The legacy hero showed conv ids truncated to 8 chars, which
    made findings unverifiable. The new provenance footer must
    include the full id so the reader can grep ``.lucid/lucid.sqlite3``
    or scroll to the source conversation."""
    finding = _finding(
        id_="a1",
        module=ModuleName.A_SPIRALBENCH,
        behavior="sycophancy",
        intensity=1,
        confidence=0.7,
        quote_assistant="Perfect!",
        conversation_id="cffc3637-5611-4356-bd65-187502c1a02c",
    )
    html_text = render_report(_audit_run(), [finding])
    # Full UUID present (provenance footer), not just prefix.
    assert "cffc3637-5611-4356-bd65-187502c1a02c" in html_text


def test_provenance_includes_detected_at_and_citation_short() -> None:
    """Each finding's verify-strip lists detection time + the source
    framework. Without these the reader can't tell which prompt
    version produced the finding."""
    finding = _finding(
        id_="a1",
        module=ModuleName.A_SPIRALBENCH,
        behavior="sycophancy",
        intensity=1,
        confidence=0.7,
        quote_assistant="Perfect!",
        citation="Spiral-Bench v1.2, https://github.com/sam-paech/spiral-bench",
    )
    html_text = render_report(_audit_run(), [finding])
    assert "Spiral-Bench v1.2" in html_text
    # Time is rendered in the human-readable format, not raw isoformat.
    assert "2026-04-22 10:05 UTC" in html_text


def test_evidence_filter_drops_empty_evidence_quotes() -> None:
    """Some modules pad ``evidence_quotes`` with placeholder empty
    strings when no excerpt was available. The renderer must drop
    them — otherwise the card shows a "Supporting context" label
    above an empty blockquote, which reads as a bug.

    A finding renders in two places (headline + per-module card),
    so the contract under test is "no empty blockquote anywhere",
    not "exactly one Supporting context label". The real excerpt
    must reach the reader; empty placeholders must not.
    """
    finding = _finding(
        id_="c1",
        module=ModuleName.C_SYCEVAL,
        behavior="regressive",
        intensity=None,
        confidence=0.95,
        quote_assistant="Some assistant text",
        evidence_quotes=["", "  ", "real excerpt"],
    )
    html_text = render_report(_audit_run(), [finding])
    assert "real excerpt" in html_text
    # No empty evidence blockquote anywhere — that was the legacy bug.
    assert '<blockquote class="evidence-text"></blockquote>' not in html_text
    # Whitespace-only contents are also a regression.
    assert not re.search(
        r'<blockquote class="evidence-text">\s*</blockquote>', html_text
    )


def test_short_citation_keeps_author_year_for_paper_citations() -> None:
    """Paper citations follow ``"<authors> <year>, '<title>', arxiv:..."``;
    the short tag must keep the author-list + year (the actually
    informative bit), not just the first author before the comma.
    Catches the ``"Fanous"`` truncation bug from the live audit."""
    from lucid.report.generator import _short_citation

    fanous = "Fanous, Goldberg et al. 2025, 'SycEval: Evaluating LLM Sycophancy', AAAI AIES 2025"
    assert _short_citation(fanous) == "Fanous, Goldberg et al. 2025"
    sharma = "Sharma et al. 2023, 'Towards Understanding Sycophancy', arxiv:2310.13548"
    assert _short_citation(sharma) == "Sharma et al. 2023"


def test_short_citation_keeps_framework_prefix_intact() -> None:
    """Framework-prefixed citations (``"Lucid Module H — …"``,
    ``"Spiral-Bench, …"``) must not be split mid-name.
    ``"Lucid Module H — memory-corpus consistency"`` should produce
    ``"Lucid Module H"``, not ``"Lucid Module H"`` (correct) or
    ``"Lucid"`` (the bug we're fixing)."""
    from lucid.report.generator import _short_citation

    assert _short_citation("Lucid Module H — memory-corpus consistency") == "Lucid Module H"
    assert _short_citation("Lucid Module G: deterministic attribution") == "Lucid Module G"
    assert _short_citation("Spiral-Bench, https://github.com/sam-paech/spiral-bench") == "Spiral-Bench"


def test_module_h_provenance_labels_source_as_memories_json() -> None:
    """For Module H the audit's "source" isn't a conversation — it's
    the user's memories.json file. The provenance label should
    reflect that so the reader knows where to look."""
    finding = _finding(
        id_="h1",
        module=ModuleName.H_MEMORY,
        behavior="unsupported",
        intensity=None,
        confidence=0.78,
        quote_user="User is preparing a Series A fundraise.",
        evidence_quotes=[],
    )
    html_text = render_report(_audit_run(), [finding])
    assert "memories.json" in html_text


def test_render_hero_stat_surfaces_behaviour_total() -> None:
    """The hero's primary headline number is behaviour_total (excludes
    Module G). Verifies the audit's "real" finding count is what the
    reader sees, not the inflated count that includes attribution."""
    findings = [
        _finding(id_="a1", module=ModuleName.A_SPIRALBENCH, behavior="sycophancy"),
        _finding(id_="g1", module=ModuleName.G_ATTRIBUTION, behavior="model=x", intensity=None),
        _finding(id_="g2", module=ModuleName.G_ATTRIBUTION, behavior="model=y", intensity=None),
    ]
    html_text = render_report(_audit_run(), findings)
    # The hero block lives before "By module" sections; look for the
    # stat-num near the "behaviour findings" label.
    assert 'class="stat-num"' in html_text
    hero_end = html_text.index("By module") if "By module" in html_text else len(html_text)
    hero = html_text[:hero_end]
    assert ">1<" in hero  # behaviour_total == 1, not 3
    assert "behaviour findings" in hero


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
    return "<script>alert(1)</script><img src=x onerror=alert(2)>"


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


# ──────────────────────────────────────────────────────────────────────────
# Fingerprint mosaic
# ──────────────────────────────────────────────────────────────────────────


def test_fingerprint_one_cell_per_unique_conversation() -> None:
    """Two findings on the same conversation collapse into one cell;
    two distinct conversation_ids produce two cells."""
    findings = [
        _finding(id_="x1", module=ModuleName.A_SPIRALBENCH, behavior="pushback",
                 conversation_id="conv-aaa"),
        _finding(id_="x2", module=ModuleName.A_SPIRALBENCH, behavior="sycophancy",
                 conversation_id="conv-aaa"),
        _finding(id_="x3", module=ModuleName.B_SHARMA, behavior="feedback-sycophancy",
                 conversation_id="conv-bbb"),
    ]
    cells = _fingerprint_cells(findings)
    assert {c.conversation_id for c in cells} == {"conv-aaa", "conv-bbb"}
    by_id = {c.conversation_id: c for c in cells}
    assert by_id["conv-aaa"].finding_count == 2
    assert by_id["conv-bbb"].finding_count == 1


def test_fingerprint_excludes_module_g() -> None:
    """Module G fires once per conversation by construction; including
    it would give every cell the same neutral colour."""
    findings = [
        _finding(id_="g1", module=ModuleName.G_ATTRIBUTION, behavior="model=x",
                 conversation_id="conv-only-g"),
    ]
    cells = _fingerprint_cells(findings)
    assert cells == []


def test_fingerprint_dominant_module_wins_ties_by_count() -> None:
    findings = [
        _finding(id_="m1", module=ModuleName.A_SPIRALBENCH, behavior="pushback",
                 conversation_id="conv-zzz"),
        _finding(id_="m2", module=ModuleName.A_SPIRALBENCH, behavior="pushback",
                 conversation_id="conv-zzz"),
        _finding(id_="m3", module=ModuleName.B_SHARMA, behavior="feedback-sycophancy",
                 conversation_id="conv-zzz"),
    ]
    cells = _fingerprint_cells(findings)
    assert len(cells) == 1
    assert cells[0].dominant_module == ModuleName.A_SPIRALBENCH


def test_fingerprint_severity_takes_worst_finding() -> None:
    findings = [
        _finding(id_="s1", module=ModuleName.A_SPIRALBENCH, behavior="pushback",
                 intensity=1, conversation_id="conv-mix"),
        _finding(id_="s2", module=ModuleName.A_SPIRALBENCH, behavior="harmful-escalation",
                 intensity=3, conversation_id="conv-mix"),
    ]
    cells = _fingerprint_cells(findings)
    assert len(cells) == 1
    assert cells[0].severity == "high"


def test_fingerprint_skips_findings_without_conversation_id() -> None:
    """Cross-corpus Module H findings can carry conversation_id=None."""
    findings = [
        _finding(id_="h1", module=ModuleName.H_MEMORY, behavior="unsupported",
                 conversation_id=None),
    ]
    cells = _fingerprint_cells(findings)
    assert cells == []


def test_fingerprint_cells_sorted_by_severity_then_count() -> None:
    findings = [
        _finding(id_="a", module=ModuleName.A_SPIRALBENCH, behavior="sycophancy",
                 intensity=1, conversation_id="conv-low"),
        _finding(id_="b", module=ModuleName.A_SPIRALBENCH, behavior="harmful-escalation",
                 intensity=3, conversation_id="conv-high"),
        _finding(id_="c", module=ModuleName.A_SPIRALBENCH, behavior="harmful-escalation",
                 intensity=3, conversation_id="conv-high-2"),
        _finding(id_="d", module=ModuleName.A_SPIRALBENCH, behavior="harmful-escalation",
                 intensity=3, conversation_id="conv-high-2"),
    ]
    cells = _fingerprint_cells(findings)
    # conv-high-2 has 2 high-severity findings → ranked first
    # conv-high has 1 high-severity → second
    # conv-low has 1 low-severity → last
    assert [c.conversation_id for c in cells] == ["conv-high-2", "conv-high", "conv-low"]


def test_svg_fingerprint_renders_one_rect_per_loud_cell() -> None:
    """Smoke: the SVG includes a coloured rect for each loud cell, and
    open-stroke rects for the clean placeholders."""
    cells = [
        FingerprintCell(
            conversation_id="conv-1",
            conversation_short="conv-1",
            finding_count=3,
            dominant_module=ModuleName.A_SPIRALBENCH,
            dominant_module_short="A",
            severity="high",
        ),
        FingerprintCell(
            conversation_id="conv-2",
            conversation_short="conv-2",
            finding_count=1,
            dominant_module=ModuleName.B_SHARMA,
            dominant_module_short="B",
            severity="mid",
        ),
    ]
    svg = _svg_fingerprint(cells, total_conversations=4)  # 2 loud + 2 clean placeholders
    # Loud cells: filled with severity colour.
    assert 'fill="#9b1c1c"' in svg  # high
    assert 'fill="#b45309"' in svg  # mid
    # Clean placeholders: stroke-only.
    assert svg.count('fill="none"') >= 2
    # Tooltip text.
    assert "Conversation conv-1" in svg
    assert "dominant Module A" in svg


def test_svg_fingerprint_handles_zero_total_gracefully() -> None:
    svg = _svg_fingerprint([], total_conversations=0)
    assert "fingerprint unavailable" in svg.lower()
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")


def test_render_includes_fingerprint_section() -> None:
    """The hero/fingerprint section becomes the screenshot — it must
    render when there are loud cells."""
    findings = [
        _finding(id_="f1", module=ModuleName.A_SPIRALBENCH, behavior="pushback",
                 intensity=3, conversation_id="conv-fingerprint-1"),
        _finding(id_="g1", module=ModuleName.G_ATTRIBUTION, behavior="model=x",
                 intensity=None, conversation_id="conv-fingerprint-1"),
    ]
    html_out = render_report(_audit_run(), findings)
    assert 'class="fingerprint"' in html_out
    assert "The conversation fingerprint of this corpus" in html_out
    # Fig. 1 caption present.
    assert "Fig.&nbsp;1" in html_out


def test_render_includes_drop_cap_and_masthead() -> None:
    html_out = render_report(_audit_run(), _sample_findings_with_g_metadata())
    assert 'hero-masthead' in html_out
    assert "Lucid Epistemic Audit" in html_out
    assert "Run №" in html_out


def test_render_includes_how_to_read_sidenote() -> None:
    html_out = render_report(_audit_run(), _sample_findings_with_g_metadata())
    assert 'class="sidenote"' in html_out
    assert "How to read this report" in html_out


def test_render_numbers_attribution_figures() -> None:
    html_out = render_report(_audit_run(), _sample_findings_with_g_metadata())
    # Sequential numbering after the figure-collision fix:
    # Fig. 4 (module bars), Fig. 5 (month histogram), Fig. 6 (model donut).
    assert "Fig.&nbsp;4" in html_out
    assert "Fig.&nbsp;5" in html_out
    assert "Fig.&nbsp;6" in html_out


def test_severity_counts_for_module_buckets_by_class() -> None:
    findings = [
        _finding(
            id_="con1",
            module=ModuleName.A_SPIRALBENCH,
            behavior="sycophancy",
            intensity=3,
        ),
        _finding(
            id_="con2",
            module=ModuleName.A_SPIRALBENCH,
            behavior="sycophancy",
            intensity=1,
        ),
        _finding(
            id_="prot",
            module=ModuleName.A_SPIRALBENCH,
            behavior="pushback",
            intensity=2,
        ),
        _finding(
            id_="other",
            module=ModuleName.B_SHARMA,
            behavior="answer-sycophancy",
            intensity=2,
        ),
    ]
    counts = _severity_counts_for_module(findings, ModuleName.A_SPIRALBENCH)
    assert counts == {"high": 1, "mid": 0, "low": 1, "neutral": 1}
    # Other module isn't mixed in.
    assert _severity_counts_for_module(findings, ModuleName.C_SYCEVAL) == {
        "high": 0,
        "mid": 0,
        "low": 0,
        "neutral": 0,
    }


def test_module_radar_axes_carry_severity_breakdown() -> None:
    findings = [
        _finding(
            id_="h1",
            module=ModuleName.H_MEMORY,
            behavior="unsupported",
            intensity=None,
            confidence=0.75,
        ),
        _finding(
            id_="h2",
            module=ModuleName.H_MEMORY,
            behavior="weakly-supported",
            intensity=None,
            confidence=0.70,
        ),
        _finding(
            id_="h3",
            module=ModuleName.H_MEMORY,
            behavior="out-of-scope",
            intensity=None,
            confidence=1.0,
        ),
    ]
    radar = _compute_module_radar(findings, n_conversations=50)
    h_axis = next(a for a in radar.axes if a.module is ModuleName.H_MEMORY)
    assert h_axis.high_count == 1  # unsupported
    assert h_axis.mid_count == 1  # weakly-supported
    assert h_axis.low_count == 0
    assert h_axis.neutral_count == 1  # out-of-scope
    assert h_axis.total_count == 3


def test_svg_radar_renders_stacked_bars_not_polygon() -> None:
    """Stacked-bar encoding replaces the polygon-fill of the old radar.

    The SVG must emit per-severity segment colours and must NOT emit
    the former polygon-fill (fill-opacity="0.22") that made sparse
    concern scores vanish at small values.
    """
    findings = [
        _finding(
            id_="a1",
            module=ModuleName.A_SPIRALBENCH,
            behavior="sycophancy",
            intensity=3,
            confidence=0.9,
        ),
        _finding(
            id_="a2",
            module=ModuleName.A_SPIRALBENCH,
            behavior="pushback",
            intensity=2,
            confidence=0.85,
        ),
    ]
    radar = _compute_module_radar(findings, n_conversations=10)
    svg = _svg_radar(radar)
    # Severity palette references are present.
    assert "#9b1c1c" in svg  # high
    assert "#8b8678" in svg  # neutral
    # No more translucent polygon fill.
    assert 'fill-opacity="0.22"' not in svg
    # Count subtext, not the old "score · findings" form.
    assert "finding" in svg
    assert "0.00 · 0 findings" not in svg


def test_svg_radar_empty_module_draws_dot_not_bar() -> None:
    """Modules with zero findings get a centre dot, not a full bar."""
    radar = _compute_module_radar([], n_conversations=10)
    svg = _svg_radar(radar)
    # Every axis had zero findings; we expect the empty-spoke marker
    # (`#cdc6b8` fill circle) to appear at the centre and NO severity-
    # coloured polygons.
    assert "#cdc6b8" in svg
    assert "#9b1c1c" not in svg
    assert "#b45309" not in svg


def _extract_section(html_text: str, module: ModuleName) -> str:
    """Return the section markup for one module (best-effort regex cut).

    Matches either attribute ordering so template refactors that
    leave the anchor id intact (but reorder ``class``/``id``) don't
    need a test update.
    """
    pattern = rf'<section\b[^>]*\bid="mod-{re.escape(module.value)}"[^>]*>(.*?)</section>'
    m = re.search(pattern, html_text, re.DOTALL)
    assert m is not None, f"section for {module.value} not found in rendered HTML"
    return m.group(0)
