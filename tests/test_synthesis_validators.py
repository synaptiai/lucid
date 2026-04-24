"""Tests for synthesis post-generation validators."""

from __future__ import annotations

from datetime import UTC, datetime

from lucid.schemas import Finding, ModuleName, ReportSection
from lucid.synthesis.validators import (
    THIN_EVIDENCE_THRESHOLD,
    validate_aggregate_claims,
    validate_superlatives,
    validate_uncited_high_intensity,
)

# ----- aggregate claims -------------------------------------------------


def test_aggregate_claim_backed_by_tool_call():
    prose = "Across 42 conversations, you showed pushback."
    errors = validate_aggregate_claims(prose, {"conversation": 42})
    assert errors == []


def test_aggregate_claim_unbacked():
    prose = "Across 40 conversations, you showed pushback."
    errors = validate_aggregate_claims(prose, {})
    assert len(errors) == 1
    assert errors[0].kind == "aggregate_unsupported"


def test_aggregate_claim_mismatch():
    prose = "Across 100 sessions, you showed pushback."
    errors = validate_aggregate_claims(prose, {"session": 50})
    assert len(errors) == 1
    assert errors[0].kind == "aggregate_mismatch"
    assert "50" in errors[0].detail


def test_aggregate_no_claims_no_errors():
    prose = "You sometimes showed pushback."
    errors = validate_aggregate_claims(prose, {})
    assert errors == []


# ----- superlatives -----------------------------------------------------


def test_superlative_with_sufficient_evidence():
    """'consistently' on a behavior with >= 5 findings — no error."""
    prose = "You consistently showed pushback across sessions."
    errors = validate_superlatives(prose, {"pushback": 8})
    assert errors == []


def test_superlative_with_thin_evidence():
    """'consistently' on a behavior with < 5 findings — error."""
    prose = "You consistently showed pushback."
    errors = validate_superlatives(prose, {"pushback": 3})
    assert len(errors) == 1
    assert errors[0].kind == "superlative_thin_evidence"
    assert "pushback" in errors[0].detail


def test_no_superlatives_no_errors():
    prose = "You occasionally showed pushback in a handful of cases."
    errors = validate_superlatives(prose, {"pushback": 2})
    assert errors == []


# ----- uncited high-intensity ------------------------------------------


def _finding(finding_id: str, intensity: int | None) -> Finding:
    return Finding(
        id=finding_id,
        audit_run_id="run-v",
        conversation_id="conv-1",
        turn_ids=["t1"],
        turn_ids_hash="x" * 64,
        module=ModuleName.A_SPIRALBENCH,
        behavior="sycophancy",
        intensity=intensity,
        confidence=0.8,
        explanation="e",
        citation="Spiral-Bench v1.2",
        detected_by=["claude-opus-4-7"],
        detected_at=datetime.now(tz=UTC),
        prompt_version="v1",
        prompt_hash="h",
    )


def _section(section_id: str, cited_ids: list[str]) -> ReportSection:
    return ReportSection(
        audit_run_id="run-v",
        section_id=section_id,
        markdown=f"body citing {cited_ids}" if cited_ids else "",
        cited_finding_ids=cited_ids,
        cited_turn_ids=[],
        insufficient_evidence=not cited_ids,
        decline_reason="empty" if not cited_ids else None,
        created_at=datetime.now(tz=UTC),
    )


def test_high_intensity_all_cited_returns_empty():
    findings = [
        _finding("f-hi-1", intensity=3),
        _finding("f-hi-2", intensity=2),
    ]
    sections = [_section("exec_summary", cited_ids=["f-hi-1", "f-hi-2"])]
    uncited = validate_uncited_high_intensity(findings, sections)
    assert uncited == []


def test_high_intensity_uncited_is_returned():
    findings = [
        _finding("f-hi-1", intensity=3),
        _finding("f-hi-2", intensity=2),
    ]
    sections = [_section("exec_summary", cited_ids=["f-hi-1"])]
    uncited = validate_uncited_high_intensity(findings, sections)
    assert [f.id for f in uncited] == ["f-hi-2"]


def test_low_intensity_not_returned_even_if_uncited():
    findings = [
        _finding("f-lo-1", intensity=1),
        _finding("f-hi-1", intensity=3),
    ]
    sections = [_section("exec_summary", cited_ids=["f-hi-1"])]
    uncited = validate_uncited_high_intensity(findings, sections)
    assert uncited == []


def test_none_intensity_not_returned():
    """Findings with intensity=None (e.g. Module G) should never appear."""
    findings = [_finding("f-g-1", intensity=None)]
    sections: list[ReportSection] = []
    uncited = validate_uncited_high_intensity(findings, sections)
    assert uncited == []


# ----- threshold exposed via module-level constant ---------------------


def test_thin_evidence_threshold_default():
    """Guard against accidental threshold changes."""
    assert THIN_EVIDENCE_THRESHOLD == 5
