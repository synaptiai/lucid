"""Tests for synthesis-section rendering in the HTML report.

Covers Tasks 6.1 + 6.2 of the synthesis-agent refactor:

* ``markdown_with_citations`` Jinja filter — empty input, ``[F:id]`` /
  ``[T:id]`` token substitution, HTML escaping (prompt-injection
  defence), and paragraph-break handling.
* End-to-end ``write_report`` hook — the ``exec_summary`` section
  renders into the HTML when present; the deterministic Actionable
  Summary remains the sole lead-in when it is absent; declined
  sections surface their ``decline_reason`` without leaking a stale
  body.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lucid.report.generator import markdown_with_citations, write_report
from lucid.schemas import (
    AuditRun,
    CorpusStats,
    ReportSection,
    SamplingConfigRecord,
    Source,
    TokenUsage,
)

# ──────────────────────────────────────────────────────────────────────
# markdown_with_citations filter
# ──────────────────────────────────────────────────────────────────────


def test_markdown_with_citations_empty_input_returns_empty_markup() -> None:
    """Empty / None / whitespace-only input renders as empty markup."""
    assert str(markdown_with_citations("")) == ""
    assert str(markdown_with_citations(None)) == ""
    assert str(markdown_with_citations("   \n\n  ")) == ""


def test_markdown_with_citations_converts_finding_tokens() -> None:
    result = str(markdown_with_citations("Pattern observed [F:abc123] repeatedly."))
    assert "#finding-abc123" in result
    assert "citation-finding" in result
    # Raw token must be gone after substitution.
    assert "[F:abc123]" not in result
    # Visible anchor text is `[F]`.
    assert ">[F]<" in result


def test_markdown_with_citations_converts_turn_tokens() -> None:
    result = str(markdown_with_citations("Turn content [T:xyz_9] shows a shift."))
    assert "#turn-xyz_9" in result
    assert "citation-turn" in result
    assert "[T:xyz_9]" not in result
    assert ">[T]<" in result


def test_markdown_with_citations_handles_both_token_types() -> None:
    result = str(markdown_with_citations("Finding [F:f1] backed by turn [T:t1]."))
    assert "#finding-f1" in result
    assert "#turn-t1" in result
    # Both visible anchors present.
    assert result.count("citation-link") == 2


def test_markdown_with_citations_escapes_html_before_substitution() -> None:
    """Prompt-injected HTML in the prose must be defanged, not rendered.

    This is the key defence-in-depth contract: agent-authored markdown
    gets escaped first, and only *then* does the filter re-introduce
    anchor tags via the citation-token regex. An attacker who somehow
    smuggles raw markup past the synthesis validators still sees it
    rendered as text.
    """
    result = str(markdown_with_citations("A <script>evil()</script> pattern."))
    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    assert "&lt;/script&gt;" in result


def test_markdown_with_citations_paragraph_breaks() -> None:
    """Blank-line separated blocks become distinct <p> elements."""
    result = str(markdown_with_citations("First paragraph.\n\nSecond paragraph."))
    assert result.count("<p>") == 2
    assert result.count("</p>") == 2
    assert "First paragraph." in result
    assert "Second paragraph." in result


def test_markdown_with_citations_single_newline_becomes_br() -> None:
    """Within a paragraph, single newlines become <br> soft breaks."""
    result = str(markdown_with_citations("Line one.\nLine two."))
    assert "<br>" in result
    # One paragraph, not two.
    assert result.count("<p>") == 1


# ──────────────────────────────────────────────────────────────────────
# End-to-end write_report hook
# ──────────────────────────────────────────────────────────────────────


def _build_minimal_audit_run() -> AuditRun:
    """Construct a minimal valid AuditRun for render-path smoke tests."""
    return AuditRun(
        id="run-render-test",
        sources=[Source.CLAUDE_CODE],
        source_paths={Source.CLAUDE_CODE: "/tmp/test"},
        started_at=datetime(2026, 4, 24, 10, 0, tzinfo=UTC),
        completed_at=datetime(2026, 4, 24, 10, 5, tzinfo=UTC),
        corpus_stats=CorpusStats(
            discovered_conversations=10,
            sampled_conversations=5,
            discovered_turns=200,
            sampled_turns=100,
            date_range_start=datetime(2026, 1, 1, tzinfo=UTC),
            date_range_end=datetime(2026, 4, 24, tzinfo=UTC),
            sources=[Source.CLAUDE_CODE],
        ),
        token_usage=TokenUsage(by_module={}, orchestrator=None),
        sampling_config=SamplingConfigRecord(
            n=5,
            seed=42,
            min_turns=5,
            recency_weight=0.3,
            recency_window_days=90,
            stratify_by_project=True,
            top_n_projects=10,
        ),
        status="completed",
        corpus_fingerprint="fp" * 32,
        prompt_versions={},
        schema_version=1,
        skipped_modules=[],
    )


def test_write_report_inlines_exec_summary_when_present(tmp_path: pytest.TempPathFactory) -> None:
    """An exec_summary in report_sections renders into the HTML."""
    audit_run = _build_minimal_audit_run()
    section = ReportSection(
        audit_run_id=audit_run.id,
        section_id="exec_summary",
        markdown="Across the corpus, you showed a pushback pattern [F:f001].",
        cited_finding_ids=["f001"],
        cited_turn_ids=[],
        insufficient_evidence=False,
        decline_reason=None,
        created_at=datetime(2026, 4, 24, 10, 3, tzinfo=UTC),
    )
    path = write_report(
        audit_run,
        [],
        output_dir=tmp_path,  # type: ignore[arg-type]
        report_sections=[section],
    )
    html_text = path.read_text(encoding="utf-8")
    # The agent-prose wrapper is present.
    assert 'class="exec-summary agent-prose"' in html_text
    # The prose body rendered.
    assert "Across the corpus" in html_text
    # The citation token resolved to an anchor link.
    assert "#finding-f001" in html_text
    assert "citation-finding" in html_text
    # Attribution line references the synthesis author.
    assert "Claude Opus" in html_text
    # Declined variant's section wrapper is not present.
    # (``agent-declined`` as a CSS class selector always appears in
    # the inline stylesheet; we assert on the fully-qualified section
    # wrapper instead.)
    assert 'class="exec-summary agent-declined"' not in html_text


def test_write_report_falls_back_when_no_sections(tmp_path: pytest.TempPathFactory) -> None:
    """Without report_sections the agent-prose hook is silent.

    The existing Actionable Summary block remains the lead-in; the
    agent-prose wrapper must not appear. This is the graceful-
    degradation contract for demo runs and pre-synthesis reruns.
    """
    audit_run = _build_minimal_audit_run()
    path = write_report(
        audit_run,
        [],
        output_dir=tmp_path,  # type: ignore[arg-type]
        report_sections=None,
    )
    html_text = path.read_text(encoding="utf-8")
    assert "exec-summary agent-prose" not in html_text
    assert "exec-summary agent-declined" not in html_text


def test_write_report_falls_back_on_empty_sections_list(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """An explicit empty list is equivalent to None for the template hook."""
    audit_run = _build_minimal_audit_run()
    path = write_report(
        audit_run,
        [],
        output_dir=tmp_path,  # type: ignore[arg-type]
        report_sections=[],
    )
    html_text = path.read_text(encoding="utf-8")
    assert "exec-summary agent-prose" not in html_text


def test_write_report_inlines_top_3_actions_when_present(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """top_3_actions in report_sections renders agent prose, overriding
    the deterministic top_actions block."""
    audit_run = _build_minimal_audit_run()
    section = ReportSection(
        audit_run_id=audit_run.id,
        section_id="top_3_actions",
        markdown="**1.** Review conversation [F:f001].\n\n**2.** Note pattern [F:f002].",
        cited_finding_ids=["f001", "f002"],
        cited_turn_ids=[],
        insufficient_evidence=False,
        decline_reason=None,
        created_at=datetime(2026, 4, 24, 10, 3, tzinfo=UTC),
    )
    path = write_report(
        audit_run,
        [],
        output_dir=tmp_path,  # type: ignore[arg-type]
        report_sections=[section],
    )
    html_text = path.read_text(encoding="utf-8")
    assert 'class="actionable-summary agent-prose"' in html_text
    assert "Review conversation" in html_text


def test_write_report_inlines_per_module_narrative(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """module_a_narrative renders inside the Module A section."""
    from lucid.schemas import Finding, ModuleName

    audit_run = _build_minimal_audit_run()
    # Seed a finding so the Module A section appears in sections loop.
    f1 = Finding(
        id="f-mod-a",
        audit_run_id=audit_run.id,
        conversation_id="c1",
        turn_ids=["t1"],
        turn_ids_hash="x" * 64,
        module=ModuleName.A_SPIRALBENCH,
        behavior="sycophancy",
        intensity=2,
        confidence=0.8,
        explanation="e",
        citation="Spiral-Bench v1.2",
        detected_by=["claude-opus-4-7"],
        detected_at=datetime.now(tz=UTC),
        prompt_version="v1",
        prompt_hash="h",
    )
    narrative = ReportSection(
        audit_run_id=audit_run.id,
        section_id="module_a_narrative",
        markdown="Module A observed sycophancy at intensity 2 in [F:f-mod-a].",
        cited_finding_ids=["f-mod-a"],
        cited_turn_ids=[],
        insufficient_evidence=False,
        decline_reason=None,
        created_at=datetime.now(tz=UTC),
    )
    path = write_report(
        audit_run,
        [f1],
        output_dir=tmp_path,  # type: ignore[arg-type]
        report_sections=[narrative],
    )
    html_text = path.read_text(encoding="utf-8")
    assert "Module A observed sycophancy" in html_text
    assert 'class="module-narrative agent-prose"' in html_text


def test_write_report_shows_missing_synthesis_note_when_no_sections(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Without report_sections, a small banner explains why narrative is absent."""
    audit_run = _build_minimal_audit_run()
    path = write_report(
        audit_run,
        [],
        output_dir=tmp_path,  # type: ignore[arg-type]
        report_sections=None,
    )
    html_text = path.read_text(encoding="utf-8")
    assert 'class="synthesis-missing-note"' in html_text
    assert "no agent-written narrative" in html_text


def test_write_report_hides_missing_note_when_any_section_present(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """The missing-synthesis banner does NOT show when at least one section exists."""
    audit_run = _build_minimal_audit_run()
    section = ReportSection(
        audit_run_id=audit_run.id,
        section_id="exec_summary",
        markdown="Some prose [F:f1].",
        cited_finding_ids=["f1"],
        cited_turn_ids=[],
        insufficient_evidence=False,
        decline_reason=None,
        created_at=datetime.now(tz=UTC),
    )
    path = write_report(
        audit_run,
        [],
        output_dir=tmp_path,  # type: ignore[arg-type]
        report_sections=[section],
    )
    html_text = path.read_text(encoding="utf-8")
    assert 'class="synthesis-missing-note"' not in html_text


def test_write_report_shows_declined_section(tmp_path: pytest.TempPathFactory) -> None:
    """insufficient_evidence sections render a 'Section skipped' message.

    The populated agent-prose block must NOT render when the agent
    declined, and the decline_reason string is the only prose that
    surfaces in that path.
    """
    audit_run = _build_minimal_audit_run()
    section = ReportSection(
        audit_run_id=audit_run.id,
        section_id="exec_summary",
        markdown="",
        cited_finding_ids=[],
        cited_turn_ids=[],
        insufficient_evidence=True,
        decline_reason="Fewer than 5 qualifying findings.",
        created_at=datetime(2026, 4, 24, 10, 3, tzinfo=UTC),
    )
    path = write_report(
        audit_run,
        [],
        output_dir=tmp_path,  # type: ignore[arg-type]
        report_sections=[section],
    )
    html_text = path.read_text(encoding="utf-8")
    assert "Section skipped" in html_text
    assert "Fewer than 5 qualifying findings" in html_text
    assert 'class="exec-summary agent-declined"' in html_text
    # The populated variant must NOT render.
    assert "exec-summary agent-prose" not in html_text
